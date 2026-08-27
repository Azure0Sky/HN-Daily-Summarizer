import logging

from src.config.settings import settings
from src.infrastructure.qq_client import QQBotClient
from src.interfaces.qq_bot.openid_store import load_qq_openid


def get_qq_openid() -> str | None:
    return settings.qq_bot_user_openid or load_qq_openid()


def is_qq_push_configured() -> bool:
    return bool(settings.qq_bot_app_id and settings.qq_bot_app_secret and get_qq_openid())


def send_qq_message(text: str) -> bool:
    """Send a proactive C2C message while a short-lived Gateway session keeps the bot online."""
    openid = get_qq_openid()
    if not settings.qq_bot_app_id or not settings.qq_bot_app_secret or not openid:
        logging.warning(
            'QQ Bot AppID, AppSecret or target OpenID is not configured. '
            'Skipping QQ push.'
        )
        return False

    try:
        client = QQBotClient(settings.qq_bot_app_id, settings.qq_bot_app_secret)
        with client.connect_gateway():
            return client.send_c2c_message(openid, text)
    except Exception as e:
        logging.error(f'Failed to send QQ message: {e}')
        return False
