import os
import discord
import asyncio
from discord.ext import commands
from collections import defaultdict
from sidekick.chat import load_model, generate_response

# Set up intents (required in Discord.py v2.0+)
intents = discord.Intents.default()
intents.message_content = True  # Necessary to read message content

# Create bot instance with command prefix
bot = commands.Bot(command_prefix='!', intents=intents)

# Dictionary to store conversation history for each channel
# Format: {channel_id: [{"role": "...", "content": "..."}]}
conversation_histories = defaultdict(list)

# Maximum conversation history length
MAX_HISTORY_LENGTH = 10

# System prompt for the AI
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant named Samn. You are polite, knowledgeable, and concise."

@bot.event
async def on_ready():
    """Event triggered when the bot successfully connects to Discord"""
    print(f'{bot.user.name} has connected to Discord!')
    print(f'Bot is connected to {len(bot.guilds)} guild(s)')
    
    # Pre-load the model on startup
    await asyncio.to_thread(load_model)

def get_conversation_history(channel_id):
    """Get conversation history for a specific channel"""
    history = conversation_histories[channel_id]
    
    # Initialize with system prompt if empty
    if not history:
        history.append({"role": "system", "content": DEFAULT_SYSTEM_PROMPT})
        conversation_histories[channel_id] = history
    
    return history

def add_to_conversation(channel_id, role, content):
    """Add a message to the conversation history for a channel"""
    history = get_conversation_history(channel_id)
    history.append({"role": role, "content": content})
    
    # Trim history if it gets too long
    if len(history) > MAX_HISTORY_LENGTH + 1:  # +1 for the system message
        # Keep system message and recent messages
        history = [history[0]] + history[-(MAX_HISTORY_LENGTH):]
        conversation_histories[channel_id] = history

# This event triggers on every message sent that the bot can see
@bot.event
async def on_message(message):
    # Don't respond to our own messages to prevent infinite loops
    if message.author == bot.user:
        return
        
    # Respond to mentions
    if bot.user.mentioned_in(message):
        # Remove mention from the message to get clean content
        content = message.content
        for mention in message.mentions:
            content = content.replace(f'<@{mention.id}>', '').replace(f'<@!{mention.id}>', '')
        content = content.strip()
        
        if content:  # Only respond if there's actual content after removing mentions
            # Send typing indicator
            async with message.channel.typing():
                # Add user message to conversation
                add_to_conversation(message.channel.id, "user", f"{message.author.display_name}: {content}")
                
                # Get conversation history for this channel
                history = get_conversation_history(message.channel.id)
                
                # Generate response using the model (run in thread pool to not block the event loop)
                response = await asyncio.to_thread(
                    generate_response, 
                    history,
                    max_new_tokens=256,
                    temperature=0.7
                )
                
                # Add bot response to conversation history
                add_to_conversation(message.channel.id, "assistant", response)
                
                # Send the response
                await message.channel.send(response)
    
    # IMPORTANT: Process commands if you're also using command system
    # Without this, your commands won't work when using on_message
    await bot.process_commands(message)

@bot.command(name='ping')
async def ping(ctx):
    """Simple command that responds with 'Pong!' to check if bot is responsive"""
    await ctx.send('Pong!')

@bot.command(name='chat')
async def chat(ctx, *, message: str = None):
    """Chat with the AI model"""
    if not message:
        await ctx.send("Please provide a message to chat with the AI.")
        return
    
    # Send typing indicator
    async with ctx.typing():
        # Add user message to conversation
        add_to_conversation(ctx.channel.id, "user", f"{ctx.author.display_name}: {message}")
        
        # Get conversation history for this channel
        history = get_conversation_history(ctx.channel.id)
        
        # Generate response using the model (run in thread pool to not block the event loop)
        response = await asyncio.to_thread(
            generate_response, 
            history,
            max_new_tokens=256,
            temperature=0.7
        )
        
        # Add bot response to conversation history
        add_to_conversation(ctx.channel.id, "assistant", response)
        
        # Send the response
        await ctx.send(response)

@bot.command(name='clear')
async def clear_history(ctx):
    """Clear the conversation history for the current channel"""
    if ctx.channel.id in conversation_histories:
        # Reset history but keep the system prompt
        system_prompt = DEFAULT_SYSTEM_PROMPT
        for msg in conversation_histories[ctx.channel.id]:
            if msg["role"] == "system":
                system_prompt = msg["content"]
                break
        
        conversation_histories[ctx.channel.id] = [{"role": "system", "content": system_prompt}]
        await ctx.send("Conversation history cleared!")
    else:
        await ctx.send("No conversation history to clear.")

@bot.command(name='system')
async def set_system_prompt(ctx, *, prompt: str = None):
    """Set a custom system prompt for the AI in the current channel"""
    if not prompt:
        await ctx.send("Please provide a system prompt.")
        return
    
    history = get_conversation_history(ctx.channel.id)
    
    # Find and update system message if it exists
    system_updated = False
    for i, msg in enumerate(history):
        if msg["role"] == "system":
            history[i] = {"role": "system", "content": prompt}
            system_updated = True
            break
    
    # Add system message if it doesn't exist
    if not system_updated:
        history.insert(0, {"role": "system", "content": prompt})
    
    await ctx.send(f"System prompt updated to: '{prompt}'")

# Run the bot
if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    TOKEN = os.getenv('DISCORD_TOKEN')
    bot.run(TOKEN)