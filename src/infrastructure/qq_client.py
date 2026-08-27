import json
import logging
import threading
import time
from collections.abc import Iterator
from contextlib import AbstractContextManager

import requests
import websocket

TOKEN_URL = 'https://bots.qq.com/app/getAppAccessToken'
API_BASE_URL = 'https://api.sgroup.qq.com'
PUBLIC_MESSAGES_INTENT = 1 << 25

# Stay below both the character and UTF-8 byte limits used by QQ text messages.
MAX_LENGTH = 1800
MAX_BYTES = 3800
SEPARATOR = '\n\n---\n\n'
CONTINUATION = '（续上）\n\n'


class QQBotError(RuntimeError):
    pass


class QQGatewayReconnect(QQBotError):
    pass


def _fits_qq_message(text: str) -> bool:
    return len(text) <= MAX_LENGTH and len(text.encode('utf-8')) <= MAX_BYTES


def _split_paragraph(paragraph: str) -> list[str]:
    """Split one digest item while reserving room for the continuation marker."""
    max_chars = MAX_LENGTH - len(CONTINUATION)
    max_bytes = MAX_BYTES - len(CONTINUATION.encode('utf-8'))
    parts = []
    current_chars = []
    current_bytes = 0

    for char in paragraph:
        char_bytes = len(char.encode('utf-8'))
        if current_chars and (len(current_chars) >= max_chars or current_bytes + char_bytes > max_bytes):
            parts.append(''.join(current_chars))
            current_chars = []
            current_bytes = 0

        current_chars.append(char)
        current_bytes += char_bytes

    if current_chars or not parts:
        parts.append(''.join(current_chars))
    return parts


def split_qq_message(text: str) -> list[str]:
    """Split a long message without breaking digest items whenever possible."""
    if _fits_qq_message(text):
        return [text]

    chunks = []
    current = ''

    for paragraph in text.split(SEPARATOR):
        for part in _split_paragraph(paragraph):
            candidate = part if not current else current + SEPARATOR + part
            if _fits_qq_message(candidate):
                current = candidate
                continue

            if current:
                chunks.append(current)
            current = CONTINUATION + part

    if current:
        chunks.append(current)

    return chunks


class QQGateway(AbstractContextManager):
    """Minimal QQ Bot Gateway connection used to keep the bot online and receive C2C events."""

    def __init__(self, url: str, authorization: str, shard_count: int = 1):
        self.url = url
        self.authorization = authorization
        self.shard_count = shard_count
        self.sequence = None
        self.session_id = None
        self._socket = None
        self._heartbeat_interval = 30.0
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread = None
        self._send_lock = threading.Lock()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def _receive_payload(self) -> dict:
        if self._socket is None:
            raise QQBotError('QQ Gateway is not connected.')

        raw_payload = self._socket.recv()
        if not raw_payload:
            raise QQGatewayReconnect('QQ Gateway connection was closed.')

        try:
            return json.loads(raw_payload)
        except json.JSONDecodeError as e:
            raise QQBotError(f'Invalid payload received from QQ Gateway: {raw_payload!r}') from e

    def _send_payload(self, payload: dict):
        if self._socket is None:
            raise QQBotError('QQ Gateway is not connected.')

        with self._send_lock:
            self._socket.send(json.dumps(payload))

    def _send_heartbeat(self):
        self._send_payload({'op': 1, 'd': self.sequence})

    def _heartbeat_loop(self):
        while not self._heartbeat_stop.wait(self._heartbeat_interval):
            try:
                self._send_heartbeat()
            except Exception as e:
                logging.warning(f'Failed to send QQ Gateway heartbeat: {e}')
                return

    def connect(self):
        logging.info('Connecting to QQ Bot Gateway...')
        self._socket = websocket.create_connection(self.url, timeout=20, enable_multithread=True)

        hello = self._receive_payload()
        if hello.get('op') != 10:
            self.close()
            raise QQBotError(f'Expected QQ Gateway HELLO, received: {hello}')

        heartbeat_ms = hello.get('d', {}).get('heartbeat_interval', 30000)
        self._heartbeat_interval = max(float(heartbeat_ms) / 1000, 1.0)
        self._send_payload({
            'op': 2,
            'd': {
                'token': self.authorization,
                'intents': PUBLIC_MESSAGES_INTENT,
                'shard': [0, self.shard_count],
            }
        })

        while True:
            payload = self._receive_payload()
            opcode = payload.get('op')
            if opcode == 0:
                self.sequence = payload.get('s', self.sequence)
                if payload.get('t') == 'READY':
                    self.session_id = payload.get('d', {}).get('session_id')
                    break
            elif opcode == 1:
                self._send_heartbeat()
            elif opcode == 7:
                self.close()
                raise QQGatewayReconnect('QQ Gateway requested reconnection before READY.')
            elif opcode == 9:
                self.close()
                raise QQBotError(f'QQ Gateway rejected the session: {payload}')

        self._socket.settimeout(max(self._heartbeat_interval * 2.5, 20))
        self._heartbeat_stop.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name='qq-gateway-heartbeat',
            daemon=True,
        )
        self._heartbeat_thread.start()
        logging.info('QQ Bot Gateway connection is ready.')

    def iter_events(self) -> Iterator[dict]:
        while True:
            payload = self._receive_payload()
            opcode = payload.get('op')

            if opcode == 0:
                self.sequence = payload.get('s', self.sequence)
                yield payload
            elif opcode == 1:
                self._send_heartbeat()
            elif opcode == 7:
                raise QQGatewayReconnect('QQ Gateway requested reconnection.')
            elif opcode == 9:
                raise QQBotError(f'QQ Gateway session became invalid: {payload}')

    def close(self):
        self._heartbeat_stop.set()

        if self._socket is not None:
            try:
                self._socket.close()
            finally:
                self._socket = None

        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=2)
        self._heartbeat_thread = None


class QQBotClient:
    """Small QQ Bot client containing only the authentication, Gateway and C2C APIs we use."""

    def __init__(self, app_id: str, app_secret: str, timeout: int = 20):
        self.app_id = app_id
        self.app_secret = app_secret
        self.timeout = timeout
        self._access_token = None
        self._token_expires_at = 0.0
        self._session = requests.Session()

    def _request_access_token(self, force_refresh: bool = False) -> str:
        if not force_refresh and self._access_token and time.time() < self._token_expires_at:
            return self._access_token

        response = self._session.post(
            TOKEN_URL,
            json={
                'appId': self.app_id,
                'clientSecret': self.app_secret,
            },
            timeout=self.timeout,
        )
        self._raise_for_status(response, 'get QQ Bot access token')

        data = response.json()
        if not data.get('access_token') or not data.get('expires_in'):
            raise QQBotError(f'Invalid QQ Bot access token response: {data}')

        self._access_token = data['access_token']
        # Refresh one minute early to avoid using a token while it expires in flight.
        self._token_expires_at = time.time() + max(int(data['expires_in']) - 60, 0)
        return self._access_token

    def _authorization(self, force_refresh: bool = False) -> str:
        return f'QQBot {self._request_access_token(force_refresh=force_refresh)}'

    def _headers(self, force_refresh: bool = False) -> dict:
        return {
            'Authorization': self._authorization(force_refresh=force_refresh),
            'X-Union-Appid': self.app_id,
            'Content-Type': 'application/json',
        }

    @staticmethod
    def _raise_for_status(response: requests.Response, action: str):
        if response.ok:
            return

        trace_id = response.headers.get('X-Tps-trace-Id', '')
        raise QQBotError(
            f'Failed to {action}: HTTP {response.status_code}, '
            f'response={response.text}, trace_id={trace_id}'
        )

    def _api_request(self, method: str, path: str, **kwargs) -> dict:
        response = self._session.request(
            method,
            API_BASE_URL + path,
            headers=self._headers(),
            timeout=self.timeout,
            **kwargs,
        )

        if response.status_code == 401:
            response = self._session.request(
                method,
                API_BASE_URL + path,
                headers=self._headers(force_refresh=True),
                timeout=self.timeout,
                **kwargs,
            )

        self._raise_for_status(response, f'call QQ Bot API {path}')
        if not response.content:
            return {}
        return response.json()

    def connect_gateway(self) -> QQGateway:
        gateway_data = self._api_request('GET', '/gateway/bot')
        gateway_url = gateway_data.get('url')
        if not gateway_url:
            raise QQBotError(f'QQ Bot Gateway URL is missing: {gateway_data}')

        shard_count = int(gateway_data.get('shards', 1))
        if shard_count > 1:
            logging.warning(
                f'QQ Bot Gateway returned {shard_count} shards; '
                'the minimal client will connect to shard 0 only.'
            )

        return QQGateway(
            url=gateway_url,
            authorization=self._authorization(),
            shard_count=shard_count,
        )

    def send_c2c_message(self, openid: str, text: str, msg_id: str | None = None) -> bool:
        if not text:
            logging.warning('No text to send to QQ.')
            return False

        for index, message_part in enumerate(split_qq_message(text), start=1):
            payload = {
                'msg_type': 2,  # Markdown
                'markdown': {
                    'content': message_part,
                },
                'msg_seq': index if msg_id else 1,
            }
            if msg_id:
                payload['msg_id'] = msg_id

            self._api_request(
                'POST',
                f'/v2/users/{openid}/messages',
                json=payload,
            )
            logging.info(f'QQ message part {index} sent successfully.')

            if len(text) > MAX_LENGTH:
                time.sleep(1)

        return True
