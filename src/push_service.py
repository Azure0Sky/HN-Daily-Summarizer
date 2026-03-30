import logging
from datetime import date
from typing import Dict, List, Optional

import requests

API_KEY_NAME = 'X-Action-Secret'
DEFAULT_WEBHOOK_PATH = '/api/webhook/daily_news'


def push_to_do_server(
    digest_date: date,
    summaries: List[Dict],
    webhook_url: Optional[str],
    api_secret: Optional[str]
) -> bool:
    """Push the daily digest payload to the FastAPI data server.

    The payload and headers are aligned with src/data_server.py:
    - Header key: X-Action-Secret
    - Payload shape: {"d": "YYYY-MM-DD", "summaries": [...]} 
    """

    if not webhook_url or not api_secret:
        logging.warning('DO server push skipped: missing webhook URL or API secret.')
        return False

    target_url = webhook_url.rstrip('/')
    if not target_url.endswith(DEFAULT_WEBHOOK_PATH):
        target_url = f'{target_url}{DEFAULT_WEBHOOK_PATH}'

    headers = {
        API_KEY_NAME: api_secret,
        'Content-Type': 'application/json'
    }
    payload = {
        'd': digest_date.isoformat(),
        'summaries': summaries
    }

    try:
        response = requests.post(target_url, json=payload, headers=headers, timeout=20)
        response.raise_for_status()
        logging.info(f'DO server push succeeded with status {response.status_code}.')
        return True

    except Exception as exc:
        logging.error(f'DO server push failed: {exc}')
        return False
