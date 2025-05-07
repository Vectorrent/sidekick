import os
import sys
from sidekick.discord import bot
from dotenv import load_dotenv

def main():
    # Print welcome banner
    print("=" * 50)
    print("Sidekick Discord Bot with SmolLM Integration")
    print("=" * 50)
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Get the Discord token from environment variables
    TOKEN = os.getenv('DISCORD_TOKEN')
    
    if not TOKEN:
        print("Error: DISCORD_TOKEN not found in environment variables")
        print("Please create a .env file with your Discord token:")
        print("DISCORD_TOKEN=your_token_here")
        sys.exit(1)
    
    print("Starting bot...")
    print("Available commands:")
    print("  !ping - Check if the bot is responsive")
    print("  !chat <message> - Chat with the AI")
    print("  !clear - Clear conversation history")
    print("  !system <prompt> - Set a custom system prompt")
    print("The bot will also respond to @mentions")
    print("=" * 50)
    
    # Run the bot
    bot.run(TOKEN)

if __name__ == "__main__":
    main()