README - Forex Telegram Bot (Professional)

Files:
- bot.py : main bot code (H1, professional signals)
- requirements.txt : Python dependencies

Quick steps to run on Railway (short):
1. Create GitHub repo and push these files or upload via GitHub web UI.
2. Go to https://railway.app, sign in with GitHub.
3. New Project -> Deploy from GitHub -> select your repo.
4. In Railway Project -> Variables, set:
   - TG_BOT_TOKEN  (your Telegram bot token)
   - TG_CHAT_ID    (chat id or @groupname)
   - DATA_API_KEY  (TwelveData API key)
5. Set start command to: python bot.py
6. Deploy and check Logs.

Notes:
- Recommended: use Railway environment variables instead of hardcoding secrets.
- After deploying, monitor logs for errors and ensure bot is added to the target Telegram group as admin.

Security warning:
- If you ever exposed your bot token publicly, revoke it via BotFather and create a new one.
