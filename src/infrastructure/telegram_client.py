import time
import logging
import requests
from src.config.settings import settings

# Telegram has a maximum message length of 4096 characters
MAX_LENGTH = 4000

SEPERATOR = '\n\n---\n\n'  # separator for splitting long messages


def send_telegram_message(text):
    """Send a message to a Telegram chat using the Bot API. Handles long messages by splitting them into parts."""
    if not text:
        logging.warning('No text to send.')
        return False

    url = f'https://api.telegram.org/bot{settings.tg_bot_token}/sendMessage'

    # If the text is too long, split it into parts based on double newlines (typically separators between news items).
    msg_parts = []
    if len(text) > MAX_LENGTH:
        logging.info('Message exceeds maximum length. Splitting into parts.')
        curr_message = ''

        for paragraph in text.split(SEPERATOR):
            if len(paragraph) > MAX_LENGTH:
                logging.warning('A paragraph exceeds the maximum length.')

            if len(curr_message) + len(paragraph) + len(SEPERATOR) + 10 < MAX_LENGTH:
                curr_message += paragraph + SEPERATOR
            else:
                msg_parts.append(curr_message)
                curr_message = '（续上）\n\n' + paragraph + SEPERATOR

        if curr_message:
            msg_parts.append(curr_message)

    else:
        msg_parts.append(text)

    for msg in msg_parts:
        payload = {
            'chat_id': settings.tg_chat_id,
            'text': msg,
            'parse_mode': 'Markdown',
            'disable_web_page_preview': True  # disable link previews to prevent clutter
        }

        try:
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 400 and "can\'t parse entities" in response.text:
                logging.warning('Markdown parse error. Retrying as plain text.')
                payload['parse_mode'] = ''  # fallback to plain text
                response = requests.post(url, json=payload, timeout=10)

            response.raise_for_status()
            logging.info(f'Message part sent successfully.')

            time.sleep(1)  # slight delay to avoid hitting rate limits

        except Exception as e:
            logging.error(f"Failed to send Telegram message part: {e}\nResponse: {response.text if 'response' in locals() else 'No response'}")
            return False

    return True
