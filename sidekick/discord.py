import os
import discord
from discord.ext import commands
# from dotenv import load_dotenv

# # Load environment variables from .env file
# load_dotenv()

# # Get the Discord token from environment variables
# TOKEN = os.getenv('DISCORD_TOKEN')

# Set up intents (required in Discord.py v2.0+)
intents = discord.Intents.default()
intents.message_content = True  # Necessary to read message content

# Create bot instance with command prefix
bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    """Event triggered when the bot successfully connects to Discord"""
    print(f'{bot.user.name} has connected to Discord!')
    print(f'Bot is connected to {len(bot.guilds)} guild(s)')

# This event triggers on every message sent that the bot can see
@bot.event
async def on_message(message):
    # Don't respond to our own messages to prevent infinite loops
    if message.author == bot.user:
        return
        
    # Example: Check for specific words or phrases
    if 'hello' in message.content.lower():
        await message.channel.send(f'Hello, {message.author.display_name}!')
        
    if 'good bot' in message.content.lower():
        await message.channel.send('Thank you! 😊')
        
    # Example: Respond to mentions
    if bot.user.mentioned_in(message):
        await message.channel.send(f'You mentioned me, {message.author.display_name}?')
    
    # IMPORTANT: Process commands if you're also using command system
    # Without this, your commands won't work when using on_message
    await bot.process_commands(message)

@bot.command(name='ping')
async def ping(ctx):
    """Simple command that responds with 'Pong!' to check if bot is responsive"""
    await ctx.send('Pong!')

# Run the bot
if __name__ == '__main__':
    bot.run(TOKEN)