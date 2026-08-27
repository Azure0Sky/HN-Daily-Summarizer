## Configuration

**1. Install Dependencies**

This project uses `uv` for dependency management.

```bash
uv sync --frozen --no-dev
```


**2. Environment Variables**

Create a `.env` file based on the provided example:

```Bash
cp .env.example .env
```

Edit the `.env` file to include your specific credentials.

### QQ Bot push setup

Configure `QQ_BOT_APP_ID` and `QQ_BOT_APP_SECRET`, then start the minimal QQ Bot listener:

```bash
uv run main.py qq-bot
```

Send a private message to the Bot from the QQ account that should receive the daily digest. The first user's OpenID is written to `qq_bot_user_openid.json`; later users cannot overwrite it. To change the recipient, stop the listener and delete that file before registering again.

For GitHub Actions, copy the recorded `openid` value into the following repository secrets:

```text
QQ_BOT_APP_ID
QQ_BOT_APP_SECRET
QQ_BOT_USER_OPENID
```

`QQ_BOT_USER_OPENID` takes precedence over the local file. The local file is ignored by Git and is intended only for initial registration or persistent server deployments.

## Running the Application

**1. API Server (Background via systemd)**

Create the service file at `/etc/systemd/system/hn-api.service`:

```TOML
[Unit]
Description=HN Daily Summarizer FastAPI Server
After=network.target

[Service]
Type=simple
User=your_linux_user
WorkingDirectory=/path/to/HN-Daily-Summarizer
ExecStart=/path/to/HN-Daily-Summarizer/.venv/bin/python main.py api
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**2. Telegram Bot (Background via systemd)**

Create the service file at `/etc/systemd/system/hn-agent.service`:

```TOML
[Unit]
Description=HN Daily Summarizer Telegram Agent
After=network.target

[Service]
Type=simple
User=your_linux_user
WorkingDirectory=/path/to/HN-Daily-Summarizer
ExecStart=/path/to/HN-Daily-Summarizer/.venv/bin/python main.py bot
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**3. Enable and Start Services**

Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now hn-api
sudo systemctl enable --now hn-agent

# Check status
sudo systemctl status hn-api hn-agent
```

**4. Chroma Database (Background via systemd)**

Create the service file at `/etc/systemd/system/chromadb.service`:

```TOML
[Unit]
Description=ChromaDB Vector Database Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/path/to/HN-Daily-Summarizer
Environment=PATH=/path/to/HN-Daily-Summarizer/.env
ExecStart=/path/to/HN-Daily-Summarizer/.venv/bin/uv run chroma run --path ./chroma_data --host 127.0.0.1 --port ${CHROMA_SERVER_PORT}

Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

**5. Caddy Server**

[Install Caddy](https://caddyserver.com/docs/install) and configure it to reverse proxy to the FastAPI server. Create a Caddyfile:

```Caddyfile
yourdomain.com {
    reverse_proxy localhost:{FastAPI_Port}
}
```

Enable and start Caddy:

```bash
sudo systemctl enable caddy
sudo systemctl restart caddy
```
