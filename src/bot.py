import os
import json
import time
import logging
import zoneinfo
import textwrap
from pathlib import Path
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, cast

from openai import OpenAI, AsyncOpenAI
from telegram import Update
from telegram.error import BadRequest
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters
from telegramify_markdown import convert

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
CHAT_HISTORY_KEY = 'history'

TRANSCRIPT_DIR = Path("./transcripts")
TRANSCRIPT_DIR.mkdir(exist_ok=True)

SYSTEM_PROMPT = """
你是一个严谨的中文助手。
在处理用户问题时，请严格遵守以下决策：
1. 根据用户请求，判断用户意图，选择合适的工具辅助你的回答。
2. 若问题与 Hacker News 社区内容相关，必须优先进行数据库检索，并严格遵循返回结果进行回答；如果检索没有结果，请明确回复：“根据我目前的检索，无法提供准确答案”。
3. 若问题需要外部实时信息、公开网页事实或超出 HN 数据库范围，可调用工具进行联网检索；使用工具返回的摘要证据辅助回答，禁止编造。
4. 若使用任何检索工具，回答末尾需附“参考来源”小节。
5. 使用 Markdown 格式，回答简洁清晰。
""".strip()


def _get_current_time() -> str:
    """Return the current datetime in UTC+8."""
    current_time = datetime.now(zoneinfo.ZoneInfo('Asia/Shanghai'))

    formatted_time = current_time.strftime('%Y-%m-%d %H:%M %A')
    return f"当前系统时间: {formatted_time} (UTC+8)"


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


def micro_compact_history(history: list, keep_recent: int = 1):
    """
    Clean up the conversation history by keeping only the most recent N tool calls 
    and replacing older ones with a placeholder.
    In-place modification of the history list for now.
    """
    tool_results_count = 0

    # Iterate through history in reverse to keep the most recent tool calls intact, and compact older ones.
    for msg in reversed(history):
        if msg.get('role') == 'tool':
            tool_results_count += 1
            if tool_results_count > keep_recent:
                # For older tool calls beyond the most recent N, replace content with a placeholder to save space.
                tool_name = msg.get('tool_name', 'unknown_tool')
                msg['content'] = f'[执行历史工具 {tool_name} 调用，结果已折叠]'


async def auto_compact_history(chat_id: str, history: list):
    # TODO: Apply this strategy when history exceeds certain token count
    chat_transcript_dir = TRANSCRIPT_DIR / chat_id
    chat_transcript_dir.mkdir(exist_ok=True)

    # Write the full history to a timestamped transcript file for this chat session
    chat_transcript_dir.joinpath(f'{int(time.time())}.jsonl').write_text(
        '\n'.join(json.dumps(msg, ensure_ascii=False) for msg in history),
        encoding='utf-8'
    )
    
    # Strip tool call results from older messages
    pure_dialogue = [m for m in history if m['role'] in {'user', 'assistant'} and not m.get('tool_calls')]
    conversation_text = json.dumps(pure_dialogue, ensure_ascii=False)

    client = _get_async_LLM_client()
    summary_prompt = '请用简练的中文总结以下对话的上下文、用户的核心关注点以及已经得出的结论。禁止包含任何客套话。'

    try:
        response = await client.chat.completions.create(
            model=CHAT_MODEL_NAME,
            messages=[
                {'role': 'system', 'content': summary_prompt},
                {'role': 'user', 'content': conversation_text}
            ],
            temperature=0.1
        )
        summary = response.choices[0].message.content

        return [
            {'role': 'user', 'content': f'[历史对话已压缩]\n此前对话总结：{summary}'},
            {'role': 'assistant', 'content': '已了解之前的上下文。请继续提问。'}
        ]

    except Exception as e:
        logging.exception(f'Auto compaction failed: {e}')
        # return history[-10:]


async def _agent_loop(messages):
    """Core agentic loop: send messages to LLM, handle tool calls, and yield final answer."""

    # Limit the number of turns to prevent infinite loops of tool calling without resolution.
    MAX_TURNS = 7
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

                    messages.append({
                        'role': 'tool',
                        'tool_call_id': tool_call.id,
                        'tool_name': tool_name,
                        'content': f'工具调用参数解析或校验失败：{e}。请修正参数格式后重试。',
                    })
                    continue

                tool_output = run_tool_call(tool_name, parsed_arguments)
                messages.append({
                    'role': 'tool',
                    'tool_call_id': tool_call.id,
                    'tool_name': tool_name,
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


async def end_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /end command and clear current chat conversation history."""
    if update.message is None:
        logging.warning('Received /end without message payload.')
        return

    chat_data = context.chat_data
    if chat_data is None:
        logging.warning('chat_data is unavailable when handling /end.')
        await update.message.reply_text('当前会话上下文不可用，暂时无法清理历史。')
        return

    if CHAT_HISTORY_KEY in chat_data:
        chat_data[CHAT_HISTORY_KEY] = []
    await update.message.reply_text('🧹 对话历史已清空，我们将开始全新的对话。')


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle user text with tool-calling (HN retrieval + normal chat)."""
    if update.message is None:
        logging.warning('Received update without message payload.')
        return

    chat_id = str(update.message.chat_id)

    user_query = (update.message.text or '').strip()
    if not user_query:
        await update.message.reply_text('我收到了空消息，请输入您的咨询。')
        return

    chat_data = context.chat_data
    if chat_data is None:
        logging.warning('chat_data is unavailable when handling user message.')
        await update.message.reply_text('⚠️ 当前会话上下文不可用，请稍后重试。')
        return

    if len(user_query) > MAX_USER_QUERY_CHARS:
        user_query = user_query[:MAX_USER_QUERY_CHARS]
        await update.message.reply_text(f'您的问题较长，我已截断到前 {MAX_USER_QUERY_CHARS} 个字符进行处理。')

    logging.info(f'Received query from user: {user_query}')

    processing_msg = await update.message.reply_text('🤖 Agent 正在思考中...')

    if CHAT_HISTORY_KEY not in chat_data:
        chat_data[CHAT_HISTORY_KEY] = []
    history = chat_data[CHAT_HISTORY_KEY]

    micro_compact_history(history)  # in-place

    history.append({'role': 'user', 'content': user_query})
    system_prompt = textwrap.dedent(f"""
    [环境信息]
    当前系统时间：{_get_current_time()}

    [核心指令]
    {SYSTEM_PROMPT}
    """)

    messages = [
        {'role': 'system', 'content': system_prompt},
        *history
    ]

    async for output in _agent_loop(messages):
        try:
            text, entities = convert(output)
            await processing_msg.edit_text(text, entities=[cast(Any, e.to_dict()) for e in entities])
            # await processing_msg.edit_text(output, parse_mode='Markdown')

        except BadRequest as e:
            if "Can't parse entities" in str(e):
                logging.warning(f'Failed to parse Markdown entities: {e}. Falling back to plain text.')
                await processing_msg.edit_text(output)
            else:
                raise

    chat_data[CHAT_HISTORY_KEY] = messages[1:]  # Exclude system prompt from stored history


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
    application.add_handler(CommandHandler('end', end_command))
    # Capture all text messages that are not commands and pass them to handle_message
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logging.info('Starting bot in Long Polling mode...')
    # Start the bot, which will run indefinitely until manually stopped.
    application.run_polling()


if __name__ == '__main__':
    main()


# from datetime import datetime
# import pytz # 建议明确时区，比如你的服务器在 DO，但用户可能在中国

# async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     # ... 前置代码 ...

#     # 1. 获取当前时间（精确到分钟或天即可，没必要精确到秒）
#     # 假设你的目标用户主要关注北京时间 (UTC+8) 的 HN 总结
#     tz = pytz.timezone('Asia/Shanghai')
#     current_time_str = datetime.now(tz).strftime('%Y-%m-%d %H:%M %A')

#     # 2. 动态组装包含时间的 System Prompt
#     dynamic_system_prompt = f"""
#     [环境信息]
#     当前系统时间：{current_time_str}
    
#     [核心指令]
#     {SYSTEM_PROMPT}
#     """

#     # 3. 压入上下文 (注意前文讨论的 History 隔离与压缩逻辑)
#     messages = [{'role': 'system', 'content': dynamic_system_prompt}] + history + [{'role': 'user', 'content': user_query}]
    
#     # ... 传给 LLM ...