import os
import logging
from typing import Any, cast
from pathlib import Path
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters
from telegramify_markdown import convert

from src.config.constants import TG_CHAT_HISTORY_KEY
from src.agent.engine import chat_with_agent

MAX_USER_QUERY_CHARS = 100


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

    if TG_CHAT_HISTORY_KEY in chat_data:
        chat_data[TG_CHAT_HISTORY_KEY] = []
    await update.message.reply_text('🧹 对话历史已清空，我们将开始全新的对话。')


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle user text with tool-calling (HN retrieval + normal chat)."""
    if update.message is None:
        logging.warning('Received update without message payload.')
        return

    chat_id = str(update.message.chat_id)

    user_input = (update.message.text or '').strip()
    if not user_input:
        await update.message.reply_text('我收到了空消息，请输入您的咨询。')
        return

    chat_data = context.chat_data
    if chat_data is None:
        logging.warning('chat_data is unavailable when handling user message.')
        await update.message.reply_text('⚠️ 当前会话上下文不可用，请稍后重试。')
        return

    if len(user_input) > MAX_USER_QUERY_CHARS:
        user_input = user_input[:MAX_USER_QUERY_CHARS]
        await update.message.reply_text(f'您的问题较长，我已截断到前 {MAX_USER_QUERY_CHARS} 个字符进行处理。')

    logging.info(f'Received query from user: {user_input}')

    processing_msg = await update.message.reply_text('🤖 Agent 正在思考中...')

    if TG_CHAT_HISTORY_KEY not in chat_data:
        chat_data[TG_CHAT_HISTORY_KEY] = []

    history = chat_data[TG_CHAT_HISTORY_KEY]
    history.append({'role': 'user', 'content': user_input})

    async for output in chat_with_agent(history):
        try:
            text, entities = convert(output)
            await processing_msg.edit_text(text, entities=[cast(Any, e.to_dict()) for e in entities])

        except Exception as e:
            if "Can't parse entities" in str(e):
                logging.warning(f'Failed to parse Markdown entities: {e}. Falling back to plain text.')
                await processing_msg.edit_text(output)
            else:
                raise
