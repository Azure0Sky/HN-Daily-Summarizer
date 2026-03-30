import os
import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    await update.message.reply_text(
        "🚀 核心控制节点已上线。\n"
        "目前我是一个空壳 Agent，等待接入 RAG 记忆库和 HN 抓取工具。"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle user's plain text messages (this is the entry point for the future Agent to receive commands)"""
    user_text = update.message.text
    logging.info(f'Received message from user: {user_text}')

    # Simple echo response for now, to confirm the bot is receiving messages correctly.
    response_text = f"收到指令: '{user_text}'。\n(Agent 路由模块尚未实装)"
    await update.message.reply_text(response_text)


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
