import os
import json
import logging
from functools import lru_cache
from typing import Any, Dict, Tuple, List, cast

from openai import OpenAI, AsyncOpenAI
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

from tools import get_tool_schemas, run_tool_call

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

CHAT_MODEL_NAME = os.getenv('CHAT_MODEL_NAME', 'Qwen/Qwen3.5')
LLM_BASE_URL = os.getenv('LLM_BASE_URL', 'https://chatapi.starlake.tech/v1')
# CHAT_MODEL_NAME = os.getenv('OR_CHAT_MODEL_NAME', 'qwen/qwen3.6-plus-preview:free')
# LLM_BASE_URL = os.getenv('OR_LLM_BASE_URL', 'https://openrouter.ai/api/v1')

MAX_USER_QUERY_CHARS = 100

SYSTEM_PROMPT = """
你是一个双模式中文助手，支持：
- 模式A：Hacker News（HN）日报知识问答。
- 模式B：普通聊天问答。

请严格遵守以下决策与回答规则：
1. 先判断用户问题是否与 Hacker News 相关。
2. 如果相关，必须先调用工具获取参考信息，再给出答案；在调用工具前不要直接下结论。
3. 如果工具返回 has_relevant=false，请明确回复：“根据我目前的检索，无法提供准确答案”。
4. 如果问题与 HN 无关，不要调用任何工具，直接给出正常聊天回复。
5. 使用 Markdown，回答简洁清晰；当使用工具结果时，尽量引用其中的日期或标题。
6. 禁止编造工具结果中不存在的 HN 事实。
""".strip()


@lru_cache(maxsize=1)
def _get_LLM_client() -> OpenAI:
    api_key = os.getenv('LLM_API_KEY')
    # api_key = os.getenv('OR_LLM_API_KEY')
    if not api_key:
        raise RuntimeError('LLM_API_KEY 未配置，无法调用大模型。')

    return OpenAI(base_url=LLM_BASE_URL, api_key=api_key)


@lru_cache(maxsize=1)
def _get_async_LLM_client() -> AsyncOpenAI:
    api_key = os.getenv('LLM_API_KEY')
    # api_key = os.getenv('OR_LLM_API_KEY')
    if not api_key:
        raise RuntimeError('LLM_API_KEY 未配置，无法调用大模型。')

    return AsyncOpenAI(base_url=LLM_BASE_URL, api_key=api_key)


def _normalize_tool_arguments(raw_arguments: str) -> Dict[str, Any]:
    """Validate and parse the raw JSON string from tool call by LLM into a dict of arguments."""
    if not raw_arguments:
        return {}

    try:
        parsed = json.loads(raw_arguments)
    except json.JSONDecodeError:
        logging.error('Failed to decode tool arguments: %s', raw_arguments)
        raise json.JSONDecodeError('Tool arguments must be a valid JSON string.', raw_arguments, 0)

    if not isinstance(parsed, dict):
        logging.error('Tool arguments must decode to a JSON object.')
        raise ValueError('Tool arguments must decode to a JSON object.')

    return parsed


async def _agent_loop(messages):
    """Core agentic loop: send messages to LLM, handle tool calls, and yield final answer."""

    # Limit the number of turns to prevent infinite loops of tool calling without resolution.
    MAX_TURNS = 5
    turn_count = 0

    client = _get_async_LLM_client()

    try:
        while turn_count < MAX_TURNS:
            turn_count += 1
            logging.info(f'--- Agentic loop turn {turn_count} ---')

            response = await client.chat.completions.create(
                model=CHAT_MODEL_NAME,
                messages=messages,
                tools=cast(Any, get_tool_schemas()),
                tool_choice="auto",
                temperature=0.2,
                timeout=90,
            )

            assistant_msg = response.choices[0].message
            messages.append(assistant_msg)

            if not assistant_msg.tool_calls:
                content = assistant_msg.content
                if isinstance(content, str) and content.strip():
                    yield content.strip()
                else:
                    yield '⚠️ 我暂时无法生成稳定回答，请稍后重试。'
                # Agent has finished without calling more tools, end the loop.
                return

            yield f'🔧 正在执行工具调用 (第 {turn_count} 轮)...'

            for tool_call in assistant_msg.tool_calls:
                function = getattr(tool_call, 'function', None)
                tool_name = getattr(function, 'name', '')
                raw_arguments = getattr(function, 'arguments', '{}')

                try:
                    parsed_arguments = _normalize_tool_arguments(raw_arguments or '{}')
                except Exception as e:
                    error_feedback = f'工具调用参数解析或校验失败：{e}。请修正参数格式后重试。'
                    messages.append({
                        'role': 'tool',
                        'tool_call_id': tool_call.id,
                        'content': error_feedback,
                    })
                    continue

                tool_output = run_tool_call(tool_name, parsed_arguments)
                messages.append({
                    'role': 'tool',
                    'tool_call_id': tool_call.id,
                    'content': str(tool_output),
                })

        yield '⚠️ Agent 工具调用已达最大轮数限制，未能生成最终回答。请尝试简化问题或检查工具调用结果。'

    except Exception as e:
        logging.exception(f'Error during agent loop: {e}')
        yield '⚠️ 处理请求时出现异常，暂时无法完成问答。\n请稍后重试。'


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    if update.message is None:
        logging.warning('Received /start without message payload.')
        return

    await update.message.reply_text(
        '您好，我是你的 HN 日报知识助手。\n'
        '我会先判断是否需要查询 HN 数据库：HN 相关问题会自动检索后回答，其他问题则直接聊天。\n'
        f'单条消息最多处理 {MAX_USER_QUERY_CHARS} 个字符。'
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle user text with tool-calling (HN retrieval + normal chat)."""
    if update.message is None:
        logging.warning('Received update without message payload.')
        return

    user_query = (update.message.text or '').strip()
    if not user_query:
        await update.message.reply_text('我收到了空消息，请输入您的咨询。')
        return

    if len(user_query) > MAX_USER_QUERY_CHARS:
        user_query = user_query[:MAX_USER_QUERY_CHARS]
        await update.message.reply_text(f'您的问题较长，我已截断到前 {MAX_USER_QUERY_CHARS} 个字符进行处理。')

    logging.info(f'Received query from user: {user_query}')

    processing_msg = await update.message.reply_text('🤖 Agent 正在思考中...')

    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_query},
    ]

    async for output in _agent_loop(messages):
        await processing_msg.edit_text(output, parse_mode='Markdown')


# Run in DO server
def main():
    token = os.getenv('TG_BOT_TOKEN')
    if not token:
        logging.critical('Missing TG_BOT_TOKEN environment variable.')
        return

    # Build bot application
    application = ApplicationBuilder().token(token).build()

    # Register command and message handlers
    application.add_handler(CommandHandler('start', start_command))
    # Capture all text messages that are not commands and pass them to handle_message
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logging.info('Starting bot in Long Polling mode...')
    # Start the bot, which will run indefinitely until manually stopped.
    application.run_polling()


if __name__ == '__main__':
    main()
