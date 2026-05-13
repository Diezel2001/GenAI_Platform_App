# bot/bot.py

import httpx
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    MessageHandler,
    CommandHandler,
    ContextTypes,
    filters,
)

TOKEN = "8787163223:AAHh28HknJeRqcoMevE8qTkzF_6gLZNsvl0"
API_URL = "http://localhost:8000/agent/"  # 👈 your router endpoint


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Connected to agent 🤖")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)  # 👈 session_id
    text = update.message.text

    await update.message.chat.send_action("typing")

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            res = await client.post(API_URL, json={
                "message": text,
                "k": 5,
                "session_id": user_id
            })

        data = res.json()

        # 👇 choose what to show
        reply = data.get("results", "No response")

        # Optional: debug info
        # reply += f"\n\n[route: {data.get('route')}]"

    except Exception as e:
        reply = f"Error: {str(e)}"

    await update.message.reply_text(reply)


def main():
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Bot running...")
    app.run_polling()


if __name__ == "__main__":
    main()