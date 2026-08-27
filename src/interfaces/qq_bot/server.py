import logging
import time

from src.config.settings import settings
from src.infrastructure.qq_client import QQBotClient, QQBotError, QQGatewayReconnect
from src.interfaces.qq_bot.openid_store import load_qq_openid, register_qq_openid

RECONNECT_DELAY = 5


def _handle_c2c_message(client: QQBotClient, event: dict):
    data = event.get('d', {})
    openid = data.get('author', {}).get('user_openid')
    message_id = data.get('id')

    if not openid or not message_id:
        logging.warning(f'Received malformed QQ C2C message event: {event}')
        return

    existing_openid = settings.qq_bot_user_openid or load_qq_openid()
    if existing_openid and existing_openid != openid:
        logging.warning('Ignored QQ OpenID registration attempt from a different user.')
        client.send_c2c_message(
            openid,
            '该机器人已绑定推送接收人，无法覆盖现有绑定。',
            msg_id=message_id,
        )
        return

    if not register_qq_openid(openid):
        logging.warning('QQ OpenID registration was rejected because another user is already registered.')
        return

    client.send_c2c_message(
        openid,
        '✅ 已登记为每日 Hacker News 摘要推送接收人。',
        msg_id=message_id,
    )


def run_qq_bot():
    """Listen for C2C messages so the first user OpenID can be registered locally."""
    if not settings.qq_bot_app_id or not settings.qq_bot_app_secret:
        logging.critical('Missing QQ_BOT_APP_ID or QQ_BOT_APP_SECRET environment variable in settings.')
        return

    if settings.qq_bot_user_openid:
        logging.info('QQ_BOT_OPENID is already configured; incoming users cannot replace it.')
    elif load_qq_openid():
        logging.info(
            f'QQ Bot target OpenID already exists in {settings.qq_bot_user_openid_file}. '
            'Delete the file manually to register a different user.'
        )
    else:
        logging.info('Send a C2C message to the QQ Bot to register the target OpenID.')

    client = QQBotClient(settings.qq_bot_app_id, settings.qq_bot_app_secret)

    while True:
        try:
            with client.connect_gateway() as gateway:
                logging.info('QQ Bot listener is ready for C2C messages.')
                for event in gateway.iter_events():
                    if event.get('t') == 'C2C_MESSAGE_CREATE':
                        _handle_c2c_message(client, event)

        except KeyboardInterrupt:
            raise
        except (QQGatewayReconnect, QQBotError, OSError) as e:
            logging.warning(f'QQ Bot listener disconnected: {e}. Reconnecting in {RECONNECT_DELAY}s...')
            time.sleep(RECONNECT_DELAY)
        except Exception as e:
            logging.error(f'Unexpected QQ Bot listener error: {e}', exc_info=True)
            time.sleep(RECONNECT_DELAY)
