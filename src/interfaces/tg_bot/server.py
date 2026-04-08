import logging
import src.interfaces.tg_bot.handlers as handlers
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, PicklePersistence, filters
from src.config.settings import settings


def run_tg_bot():
    if not settings.tg_bot_token:
        logging.critical('Missing TG_BOT_TOKEN environment variable in settings.')
        return

    persistence = PicklePersistence(filepath='bot_chat_data.pkl')
    application = ApplicationBuilder().token(settings.tg_bot_token).persistence(persistence).build()

    application.add_handler(CommandHandler('start', handlers.start_command))
    application.add_handler(CommandHandler('end', handlers.end_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handlers.handle_message))

    logging.info('Starting Telegram Bot in Long Polling mode...')
    application.run_polling()
