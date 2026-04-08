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

Edit the .env file to include your specific credentials.

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
