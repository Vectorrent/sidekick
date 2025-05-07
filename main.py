import os
from sidekick.discord import bot
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the Discord token from environment variables
TOKEN = os.getenv('DISCORD_TOKEN')

if __name__ == "__main__":
    bot.run(TOKEN)