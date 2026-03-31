import os
import asyncio
import logging
from functools import lru_cache
from typing import List, Tuple
from openai import OpenAI
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

from database import get_chroma_collection

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

CHAT_MODEL_NAME = os.getenv('CHAT_MODEL_NAME', 'Qwen/Qwen3.5')
LLM_BASE_URL = os.getenv('LLM_BASE_URL', 'https://chatapi.starlake.tech/v1')
# CHAT_MODEL_NAME = os.getenv('OR_CHAT_MODEL_NAME', 'qwen/qwen3.6-plus-preview:free')
# LLM_BASE_URL = os.getenv('OR_LLM_BASE_URL', 'https://openrouter.ai/api/v1')

DEFAULT_RETRIEVAL_TOP_K = 5
DEFAULT_DISTANCE_THRESHOLD = 1.0
MAX_USER_QUERY_CHARS = 150
MAX_CONTEXT_CHARS = 2000


@lru_cache(maxsize=1)
def _get_LLM_client() -> OpenAI:
    api_key = os.getenv('LLM_API_KEY')
    # api_key = os.getenv('OR_LLM_API_KEY')
    if not api_key:
        raise RuntimeError('LLM_API_KEY 未配置，无法调用大模型。')

    return OpenAI(base_url=LLM_BASE_URL, api_key=api_key)  # TODO: May change to async client to support multiple concurrent user queries in the future


def _truncate_context(text: str) -> str:
    return text if len(text) <= MAX_CONTEXT_CHARS else text[:MAX_CONTEXT_CHARS] + '\n\n[上下文过长，已截断]'


def _retrieve_relevant_context(user_query: str) -> Tuple[str, bool]:
    """Retrieve relevant context from Chroma and apply distance-threshold filtering."""
    collection = get_chroma_collection()

    query_result = collection.query(
        query_texts=[user_query],
        n_results=DEFAULT_RETRIEVAL_TOP_K,
        include=['documents', 'metadatas', 'distances']
    )

    documents = (query_result.get('documents') or [[]])[0] or []
    metadatas = (query_result.get('metadatas') or [[]])[0] or []
    distances = (query_result.get('distances') or [[]])[0] or []

    # documents: List[str] = [doc for doc in raw_documents if isinstance(doc, str)]
    # metadatas: List[Dict[str, Any]] = [meta if isinstance(meta, dict) else {} for meta in raw_metadatas]
    # distances: List[Optional[float]] = [
    #     float(distance) if isinstance(distance, (int, float)) else None
    #     for distance in raw_distances
    # ]

    if not documents:
        return '', False

    filtered_chunks: List[str] = []

    for idx, doc in enumerate(documents):
        distance = distances[idx] if idx < len(distances) else None
        metadata = metadatas[idx] if idx < len(metadatas) and isinstance(metadatas[idx], dict) else {}

        # Skip weak matches beyond the threshold
        if distance is not None and distance > DEFAULT_DISTANCE_THRESHOLD:
            continue

        # title = metadata.get('title', '未知标题')
        date_value = metadata.get('date', '未知日期')

        filtered_chunks.append(
            f'候选片段 {idx + 1}（日期: {date_value}）\n{doc.strip()}'
        )

    if not filtered_chunks:
        return '', False

    merged_context = '\n\n'.join(filtered_chunks)
    merged_context = _truncate_context(merged_context)
    return merged_context, True


def _generate_rag_answer(user_query: str, context_text: str) -> str:
    """Generate final answer from retrieved context with a Chinese system prompt."""
    client = _get_LLM_client()

    system_prompt = """
        你是一个严谨的科技资讯 Agent。你的任务是根据提供的【参考记忆库】回答用户的问题。

        严格遵守以下规则：
        1. 只能使用【参考记忆库】中提供的信息来回答。
        2. 如果参考信息无法回答该问题，请明确回复“根据我目前的抓取记录，无法提供准确答案”，严禁依靠预训练知识进行编造。
        3. 回答时，请引用具体的日期或新闻标题作为信息来源。
        4. 保持回答精炼，使用 Markdown 排版。
        """

    user_prompt = (
        f'用户问题：{user_query}\n\n'
        f'【参考记忆库】：\n{context_text}'
    )

    response = client.chat.completions.create(
        model=CHAT_MODEL_NAME,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ],
        temperature=0.1,
        timeout=90
    )

    content = response.choices[0].message.content
    if not content or not isinstance(content, str):
        return '我暂时无法生成稳定回答，请稍后重试。'

    return content.strip()


def _answer_user_query(user_query: str) -> str:
    """Run retrieval + generation"""
    context_text, has_relevant = _retrieve_relevant_context(user_query)

    if not has_relevant:
        return '💦 抱歉，我在当前记忆库中没有检索到足够相关的内容，暂时无法给出可靠回答。'

    return _generate_rag_answer(user_query, context_text)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    if update.message is None:
        logging.warning('Received /start without message payload.')
        return

    await update.message.reply_text(
        '您好，我是你的 HN 日报知识助手。\n'
        f'您可以直接提问，我会基于已入库的日报内容进行检索并回答（仅支持 {MAX_USER_QUERY_CHARS} 个字符以内的问题）。'
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle user text and respond with retrieval-augmented answers."""
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

    processing_msg = await update.message.reply_text('🔍正在检索记忆库，请稍候...')

    try:
        # Move blocking I/O work to a thread to avoid stalling the async event loop.
        answer_text = await asyncio.to_thread(_answer_user_query, user_query)
        await processing_msg.edit_text(answer_text, parse_mode='Markdown')

    except Exception as exc:
        logging.exception(f'Failed to handle user query: {exc}')
        await processing_msg.edit_text(
            '处理请求时出现异常，暂时无法完成问答。\n'
            '请稍后重试。'
        )


# Run in DO sever
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
