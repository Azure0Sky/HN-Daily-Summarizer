import os
import json
import logging
from functools import lru_cache
from typing import Any, Dict, cast

from openai import OpenAI, AsyncOpenAI
from telegram import Update
from telegram.error import BadRequest
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
你是一个严谨的中文助手。
在处理用户问题时，请严格遵守以下决策：
1. 根据用户请求，判断用户意图，选择合适的工具辅助你的回答。
2. 若问题包含相对时间表达（如：昨天、这两天、最近、本周等），必须首先获取时间锚点，再将相对时间转换成明确日期，然后才进行下一步。
3. 若问题与 Hacker News 社区内容相关，必须优先进行数据库检索，并严格遵循返回结果进行回答；如果检索没有结果，请明确回复：“根据我目前的检索，无法提供准确答案”。
4. 若问题需要外部实时信息、公开网页事实或超出 HN 数据库范围，可调用工具进行联网检索；使用工具返回的摘要证据辅助回答，禁止编造。
5. 使用任何检索工具后，回答末尾需附“参考来源”小节。
6. 使用 Markdown 格式，回答简洁清晰。
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
            
                logging.info(f'loop {turn_count} - Processing tool call: {tool_name}')

                try:
                    parsed_arguments = _normalize_tool_arguments(raw_arguments or '{}')
                except Exception as e:
                    logging.warning(f'Failed to parse tool arguments for tool "{tool_name}" with raw args: {raw_arguments}')

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
        '您好，我是你的Agent助手。\n'
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
        try:
            await processing_msg.edit_text(output, parse_mode='Markdown')

        except BadRequest as e:
            if "Can't parse entities" in str(e):
                logging.warning(f'Failed to parse Markdown entities: {e}. Falling back to plain text.')
                await processing_msg.edit_text(output)
            else:
                raise


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
