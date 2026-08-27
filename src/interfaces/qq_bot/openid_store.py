import json
import logging
from pathlib import Path

from src.config.settings import settings


def load_qq_openid() -> str | None:
    path = Path(settings.qq_bot_user_openid_file)
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        openid = data.get('openid')
        return openid if isinstance(openid, str) and openid else None
    except Exception as e:
        logging.error(f'Failed to load QQ Bot OpenID from {path}: {e}')
        return None


def register_qq_openid(openid: str) -> bool:
    """Persist the first OpenID and never let a different user overwrite it."""
    existing_openid = load_qq_openid()
    if existing_openid:
        return existing_openid == openid

    path = Path(settings.qq_bot_user_openid_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + '.tmp')
    temp_path.write_text(
        json.dumps({'openid': openid}, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    temp_path.replace(path)
    logging.info(f'Registered QQ Bot target OpenID in {path}.')
    return True
