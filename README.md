Forex Telegram Bot - Render-ready
Files:
- main.py : bot code (set env vars TG_BOT_TOKEN, TG_CHAT_ID, DATA_API_KEY on Render)
- requirements.txt : Python deps
How to use:
1) Upload this repository to GitHub.
2) On Render, create a new Web Service or Background Worker using this repo.
3) Build command: pip install -r requirements.txt
4) Start command: python main.py
5) Set environment variables: TG_BOT_TOKEN, TG_CHAT_ID, DATA_API_KEY
Security: If your token leaked, revoke it via BotFather and create a new one.
