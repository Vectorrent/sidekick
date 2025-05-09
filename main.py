import os
import sys
import asyncio
from sidekick.discord import bot, ENABLE_LOGGING
from dotenv import load_dotenv

async def run_bot():
    """Run the Discord bot with proper command syncing"""
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
    print("  /system <prompt> - Set a custom system prompt")
    print("  /good_bot [comment] - Give positive feedback on the last response")
    print("  /bad_bot [comment] - Give negative feedback on the last response")
    print("  /engagement - [Owner only] Check engagement level for debugging")
    print("  /logging - [Owner only] Toggle prompt logging on/off")
    print("  /rl_metrics - [Owner only] View reinforcement learning metrics")
    print()
    print("Conversation Engagement Features:")
    print("  - The bot has a small random chance to respond to any message")
    print("  - After engagement, the bot becomes more likely to respond")
    print("  - Engagement naturally decays over time (~6 minutes)")
    print("  - Each channel has its own independent engagement level")
    print("  - The bot will always respond to direct @mentions and !chat commands")
    print()
    print("Text Generation Parameters:")
    print("  - Each channel can have custom generation settings")
    print("  - Parameters like temperature, top_p, top_k can be adjusted")
    print("  - Changes persist until the bot is restarted")
    print("  - Use !config to view and adjust parameters")
    print()
    print("Debugging Features:")
    print("  - Prompt logging to terminal for debugging AI responses")
    print("  - Toggle logging on/off with the /logging command (owner-only)")
    print("  - All prompts and system messages visible in terminal when enabled")
    print("=" * 50)
    
    # Enable logging by default in development mode
    if os.getenv('DEVELOPMENT') == 'true':
        global ENABLE_LOGGING
        ENABLE_LOGGING = True
        print("Development mode: Prompt logging enabled by default")
    
    # Run the bot with the provided token
    await bot.start(TOKEN)

def main():
    """Main entry point that properly handles asyncio"""
    try:
        # Run the bot using asyncio
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        print("Bot shutting down...")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()