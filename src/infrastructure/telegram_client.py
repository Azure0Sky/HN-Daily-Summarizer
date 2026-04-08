import time
import logging
import requests
from src.config.settings import settings

# Telegram has a maximum message length of 4096 characters
MAX_LENGTH = 4000


def send_telegram_message(text):
    """Send a message to a Telegram chat using the Bot API. Handles long messages by splitting them into parts."""
    if not text:
        logging.warning('No text to send.')
        return False

    url = f'https://api.telegram.org/bot{settings.tg_bot_token}/sendMessage'

    # If the text is too long, split it into parts based on double newlines (typically separators between news items).
    # message_parts = []
    # if len(text) <= MAX_LENGTH:
    #     message_parts.append(text)

    # else:
    #     paragraphs = text.split('\n\n---\n\n')
    #     current_part = ''
    #     for p in paragraphs:
    #         if len(current_part) + len(p) + 10 < MAX_LENGTH:
    #             current_part += p + '\n\n---\n\n'
    #         else:
    #             message_parts.append(current_part)
    #             current_part = p + '\n\n---\n\n'

    #     if current_part:
    #         message_parts.append(current_part)

    payload = {
        'chat_id': settings.tg_chat_id,
        'text': text,
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
        logging.info(f'Message sent successfully.')

        # Telegram API rate limits: To be safe, add a short delay between messages if sending multiple parts
        time.sleep(1.5) 
        
    except Exception as e:
        logging.error(f"Failed to send Telegram message: {e}\nResponse: {response.text if 'response' in locals() else 'No response'}")
        return False

    return True
