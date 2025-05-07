import os
import discord
import asyncio
import random
import time
from discord.ext import commands, tasks
from collections import defaultdict
from sidekick.chat import load_model, generate_response

# Set up intents (required in Discord.py v2.0+)
intents = discord.Intents.default()
intents.message_content = True  # Necessary to read message content

# Create bot instance with command prefix
class SidekickBot(commands.Bot):
    async def setup_hook(self):
        """A hook that is called when the bot is starting to sync commands and do setup"""
        print("Setting up commands...")
        # List existing commands (for debug purposes only)
        for command in self.commands:
            print(f"Command available: {command.name}")
            
        # Sync commands with Discord
        print("Syncing commands with Discord...")
        try:
            # sync() will automatically register all slash commands
            await self.tree.sync()
            print("Commands synced successfully!")
        except Exception as e:
            print(f"Error syncing commands: {e}")
            print("Commands will still be available as text commands with ! prefix")

# Initialize the bot with our custom class
bot = SidekickBot(command_prefix='!', intents=intents)

# Dictionary to store conversation history for each channel
# Format: {channel_id: [{"role": "...", "content": "..."}]}
conversation_histories = defaultdict(list)

# Dictionary to track conversation engagement levels for each channel
# Format: {channel_id: {"level": float, "last_update": timestamp}}
conversation_engagement = {}

# Configuration for random responses
RESPONSE_CONFIG = {
    # Base chance (percentage) to respond to any message (0.5%)
    "BASE_CHANCE": 0.5,
    
    # Maximum chance (percentage) to respond when fully engaged (25%)
    "MAX_CHANCE": 80.0,
    
    # How much to boost engagement when bot is mentioned or used in a command
    "DIRECT_ENGAGEMENT_BOOST": 1.0,  # 100% engagement
    
    # How much to boost engagement when the bot decides to respond to a message
    "RESPONSE_ENGAGEMENT_BOOST": 0.7,  # 70% boost
    
    # How much engagement decays per second
    "DECAY_RATE": 0.0028,  # Decays to ~0 in about 6 minutes
    
    # Threshold below which engagement is considered zero
    "MIN_THRESHOLD": 0.01
}

# Discord message size limit (slightly below the actual 2000 for safety)
DISCORD_MESSAGE_LIMIT = 1900

# Maximum conversation history length
MAX_HISTORY_LENGTH = 10

# Logging settings
ENABLE_LOGGING = False  # Enable prompt logging by default for debugging
LOG_LEVEL = 1  # 0 = minimal, 1 = basic, 2 = detailed

# System prompts for the AI
DEFAULT_SYSTEM_PROMPT = "You are named Maya Chen, and you are an AI researcher. You are polite, knowledgeable, and concise."
DM_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT

# Default generation parameters for LLM responses
# These can be overridden per channel
DEFAULT_GENERATION_PARAMS = {
    "max_new_tokens": 512,     # Maximum length of generated text
    "temperature": 0.45,        # Controls randomness (0.0-1.0) - higher = more random
    # "top_p": 0.9,            # Nucleus sampling - keep tokens with cumulative probability >= top_p
    # "top_k": 50,             # Keep only the top k tokens - 0 means no filtering
    # "min_p": 0.02,             # Minimum token probability, which will be scaled by the probability of the most likely token.
    "repetition_penalty": 1.1, # Penalty for repeating tokens (1.0 = no penalty)
    "do_sample": True,         # Whether to use sampling vs greedy decoding
    # "num_beams": 1,          # Number of beams for beam search (1 = no beam search)
    "no_repeat_ngram_size": 9  # Size of n-grams to prevent repetition (0 = no filtering)
}

# Store generation parameters per channel
# Format: {channel_id: {param_name: value, ...}}
generation_params = defaultdict(lambda: DEFAULT_GENERATION_PARAMS.copy())

@bot.event
async def on_ready():
    """Event triggered when the bot successfully connects to Discord"""
    print(f'{bot.user.name} has connected to Discord!')
    print(f'Bot is connected to {len(bot.guilds)} guild(s)')
    
    # Pre-load the model on startup
    await asyncio.to_thread(load_model)
    
    # Start the engagement decay task
    engagement_decay_task.start()
    
    # Print registered commands
    print("Registered commands:")
    for command in bot.commands:
        print(f"  !{command.name} - {command.help}")

def get_conversation_history(channel_id, is_dm=False):
    """Get conversation history for a specific channel
    
    Args:
        channel_id: The Discord channel ID
        is_dm: Whether this is a direct message channel
    """
    history = conversation_histories[channel_id]
    
    # Initialize with system prompt if empty
    if not history:
        # Use DM-specific system prompt for direct message channels
        system_prompt = DM_SYSTEM_PROMPT if is_dm else DEFAULT_SYSTEM_PROMPT
        history.append({"role": "system", "content": system_prompt})
        conversation_histories[channel_id] = history
    
    return history

def add_to_conversation(channel_id, role, content, is_dm=False):
    """Add a message to the conversation history for a channel"""
    history = get_conversation_history(channel_id, is_dm=is_dm)
    history.append({"role": role, "content": content})
    
    # Trim history if it gets too long
    if len(history) > MAX_HISTORY_LENGTH + 1:  # +1 for the system message
        # Keep system message and recent messages
        history = [history[0]] + history[-(MAX_HISTORY_LENGTH):]
        conversation_histories[channel_id] = history

def get_generation_params(channel_id):
    """Get generation parameters for a specific channel"""
    # If channel has custom params, return those. Otherwise return defaults.
    return generation_params[channel_id]

def get_engagement_level(channel_id):
    """Get the current engagement level for a channel (0.0 to 1.0)"""
    if channel_id not in conversation_engagement:
        return 0.0
    
    # Get channel engagement data
    engagement_data = conversation_engagement[channel_id]
    
    # Calculate decay since last update
    current_time = time.time()
    time_elapsed = current_time - engagement_data["last_update"]
    
    # Apply decay based on time elapsed
    engagement_level = engagement_data["level"]
    engagement_level *= (1.0 - RESPONSE_CONFIG["DECAY_RATE"] * time_elapsed)
    
    # Enforce threshold
    if engagement_level < RESPONSE_CONFIG["MIN_THRESHOLD"]:
        engagement_level = 0.0
    
    # Update the stored engagement level
    conversation_engagement[channel_id] = {
        "level": engagement_level,
        "last_update": current_time
    }
    
    return engagement_level

def boost_engagement(channel_id, boost_amount):
    """Boost the engagement level for a channel"""
    # Get current engagement level first (this handles initialization and decay)
    current_level = get_engagement_level(channel_id)
    
    # Boost engagement, capped at 1.0
    new_level = min(1.0, current_level + boost_amount)
    
    # Update engagement data
    conversation_engagement[channel_id] = {
        "level": new_level,
        "last_update": time.time()
    }

def should_respond_to_message(message):
    """Determine if the bot should respond to a message based on engagement level and channel type"""
    channel_id = message.channel.id
    
    # Check if this is a DM/private channel (DMChannel instance)
    is_dm_channel = isinstance(message.channel, discord.DMChannel)
    
    # Always respond in DM channels
    if is_dm_channel:
        return True
    
    # For non-DM channels, use engagement-based probability
    engagement_level = get_engagement_level(channel_id)
    
    # Calculate response chance based on engagement level
    base_chance = RESPONSE_CONFIG["BASE_CHANCE"] / 100.0  # Convert to probability
    max_chance = RESPONSE_CONFIG["MAX_CHANCE"] / 100.0    # Convert to probability
    
    # Linear interpolation between base and max chance based on engagement
    response_chance = base_chance + (max_chance - base_chance) * engagement_level
    
    # Random check if we should respond
    return random.random() < response_chance

@tasks.loop(seconds=30)
async def engagement_decay_task():
    """Background task to periodically clean up engagement data for inactive channels"""
    channels_to_remove = []
    
    # Current time
    current_time = time.time()
    
    # Check each channel's engagement
    for channel_id, data in conversation_engagement.items():
        # Calculate how long since last update
        time_elapsed = current_time - data["last_update"]
        
        # Calculate new engagement level after decay
        new_level = data["level"] * (1.0 - RESPONSE_CONFIG["DECAY_RATE"] * time_elapsed)
        
        if new_level < RESPONSE_CONFIG["MIN_THRESHOLD"]:
            # If engagement is below threshold, mark for removal
            channels_to_remove.append(channel_id)
        else:
            # Otherwise, update the engagement level
            conversation_engagement[channel_id] = {
                "level": new_level,
                "last_update": current_time
            }
    
    # Remove inactive channels
    for channel_id in channels_to_remove:
        conversation_engagement.pop(channel_id, None)

@engagement_decay_task.before_loop
async def before_decay_task():
    """Wait for the bot to be ready before starting the engagement decay task"""
    await bot.wait_until_ready()

def log_prompt(history, level=LOG_LEVEL, channel_name="Unknown", author_name="Unknown"):
    """
    Log conversation history to the terminal
    
    Args:
        history (list): List of message dicts with 'role' and 'content' keys
        level (int): Log level (0=minimal, 1=basic, 2=detailed)
        channel_name (str): Name of the Discord channel
        author_name (str): Name of the message author
    """
    if not ENABLE_LOGGING:
        return
    
    print("\n" + "="*50)
    print(f"PROMPT LOG - Channel: {channel_name} - Author: {author_name}")
    print("="*50)
    
    if level == 0:
        # Minimal logging - just show the latest user message and system prompt
        system_prompt = None
        latest_user_message = None
        
        for msg in history:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            if msg["role"] == "user":
                latest_user_message = msg["content"]
        
        if system_prompt:
            print(f"SYSTEM: {system_prompt}")
        if latest_user_message:
            print(f"USER: {latest_user_message}")
    
    elif level == 1:
        # Basic logging - show all messages
        for i, msg in enumerate(history):
            role = msg["role"].upper()
            content = msg["content"]
            # Truncate very long messages
            if len(content) > 500:
                content = content[:500] + "... [truncated]"
            print(f"{i}. {role}: {content}")
    
    elif level == 2:
        # Detailed logging - show all messages and full history
        print("FULL CONVERSATION HISTORY:")
        for i, msg in enumerate(history):
            role = msg["role"].upper()
            content = msg["content"]
            print(f"{i}. {role}: {content}")
        
        # Also print raw history for debugging
        print("\nRAW HISTORY:")
        import json
        print(json.dumps(history, indent=2))
    
    print("="*50 + "\n")

def split_long_message(message, max_length=DISCORD_MESSAGE_LIMIT):
    """
    Split a long message into smaller chunks while respecting line boundaries when possible.
    
    Args:
        message (str): The message to split
        max_length (int): Maximum length for each chunk
        
    Returns:
        list: List of message chunks
    """
    # If message is already short enough, just return it
    if len(message) <= max_length:
        return [message]
    
    chunks = []
    current_chunk = ""
    lines = message.split('\n')
    
    for line in lines:
        # If a single line is longer than max_length, we need to split it
        if len(line) > max_length:
            # If we already have content in the current chunk, add it to chunks and reset
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            
            # Split long line by max_length
            for i in range(0, len(line), max_length):
                chunks.append(line[i:i + max_length])
            
        # If adding this line would make the chunk too long, start a new chunk
        elif len(current_chunk) + len(line) + 1 > max_length:  # +1 for the newline
            chunks.append(current_chunk)
            current_chunk = line
        
        # Otherwise add to the current chunk
        else:
            if current_chunk:
                current_chunk += '\n' + line
            else:
                current_chunk = line
    
    # Add any remaining chunk
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

# This event triggers on every message sent that the bot can see
@bot.event
async def on_message(message):
    # Don't respond to our own messages to prevent infinite loops
    if message.author == bot.user:
        return
        
    # IMPORTANT: Process commands first to ensure they work properly
    # This must be at the beginning of on_message
    await bot.process_commands(message)
    
    # Check if this is a DM/private channel (DMChannel instance)
    is_dm_channel = isinstance(message.channel, discord.DMChannel)
    
    # Flag to track if this is a direct mention
    is_direct_mention = bot.user.mentioned_in(message)
    should_respond = False
    content = message.content
    
    if is_direct_mention:
        # This is a direct mention - we should definitely respond
        should_respond = True
        
        # Remove mention from the message to get clean content
        for mention in message.mentions:
            content = content.replace(f'<@{mention.id}>', '').replace(f'<@!{mention.id}>', '')
        content = content.strip()
        
        # Boost engagement level significantly
        boost_engagement(message.channel.id, RESPONSE_CONFIG["DIRECT_ENGAGEMENT_BOOST"])
    elif is_dm_channel:
        # This is a DM channel - always respond
        should_respond = True
        
        # In DMs, always maintain full engagement
        boost_engagement(message.channel.id, RESPONSE_CONFIG["DIRECT_ENGAGEMENT_BOOST"])
    else:
        # Not a direct mention or DM - check if we should randomly respond
        should_respond = should_respond_to_message(message)
        
        if should_respond:
            # If we're going to respond, boost engagement
            boost_engagement(message.channel.id, RESPONSE_CONFIG["RESPONSE_ENGAGEMENT_BOOST"])
    
    # Generate and send a response if appropriate
    if should_respond and content:
        # Send typing indicator
        async with message.channel.typing():
            # Add user message to conversation
            add_to_conversation(message.channel.id, "user", f"{message.author.display_name}: {content}", is_dm=is_dm_channel)
            
            # Get conversation history for this channel
            history = get_conversation_history(message.channel.id, is_dm=is_dm_channel)
            
            # Log prompt to terminal if logging is enabled
            channel_name = message.channel.name if hasattr(message.channel, 'name') else "DM"
            log_prompt(history, channel_name=channel_name, author_name=message.author.display_name)
            
            # Get generation parameters for this channel
            params = get_generation_params(message.channel.id)
            
            # Generate response using the model (run in thread pool to not block the event loop)
            response = await asyncio.to_thread(
                generate_response, 
                history,
                **params
            )
            
            # Add bot response to conversation history
            add_to_conversation(message.channel.id, "assistant", response, is_dm=is_dm_channel)
            
            # Split the response if it's too long for Discord
            message_chunks = split_long_message(response)
            
            # Send each chunk as a separate message
            for chunk in message_chunks:
                await message.channel.send(chunk)
    
    # NOTE: We're already processing commands at the beginning of this function
    # This was moved to ensure commands are processed before any AI response logic

@bot.hybrid_command(
    name='ping',
    description='Check if the bot is responsive'
)
async def ping(ctx):
    """Simple command that responds with 'Pong!' to check if bot is responsive"""
    # Deliberately not using ephemeral=True here, as ping is often used to check if 
    # the bot is visible to everyone in the channel
    await ctx.send('Pong!')

@bot.hybrid_command(
    name='chat',
    description='Chat with the AI model'
)
async def chat(ctx, *, message: str = None):
    """Chat with the AI model"""
    if not message:
        await ctx.send("Please provide a message to chat with the AI.")
        return
    
    # Check if this is a DM channel
    is_dm_channel = isinstance(ctx.channel, discord.DMChannel)
    
    # Boost engagement level significantly - direct command is similar to a mention
    boost_engagement(ctx.channel.id, RESPONSE_CONFIG["DIRECT_ENGAGEMENT_BOOST"])
    
    # Send typing indicator
    async with ctx.typing():
        # Add user message to conversation
        add_to_conversation(ctx.channel.id, "user", f"{ctx.author.display_name}: {message}", is_dm=is_dm_channel)
        
        # Get conversation history for this channel
        history = get_conversation_history(ctx.channel.id, is_dm=is_dm_channel)
        
        # Log prompt to terminal if logging is enabled
        channel_name = ctx.channel.name if hasattr(ctx.channel, 'name') else "DM"
        log_prompt(history, channel_name=channel_name, author_name=ctx.author.display_name)
        
        # Get generation parameters for this channel
        params = get_generation_params(ctx.channel.id)
        
        # Generate response using the model (run in thread pool to not block the event loop)
        response = await asyncio.to_thread(
            generate_response, 
            history,
            **params
        )
        
        # Add bot response to conversation history
        add_to_conversation(ctx.channel.id, "assistant", response, is_dm=is_dm_channel)
        
        # Split the response if it's too long for Discord
        message_chunks = split_long_message(response)
            
        # Send each chunk as a separate message
        for chunk in message_chunks:
            await ctx.send(chunk)

@bot.hybrid_command(
    name='clear',
    description='Clear the conversation history for the current channel'
)
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
        
        # Reset engagement for this channel (clearing history suggests ending the conversation)
        if ctx.channel.id in conversation_engagement:
            conversation_engagement.pop(ctx.channel.id)
        
        # Send ephemeral confirmation message
        await ctx.send("Conversation history cleared!", ephemeral=True)
    else:
        await ctx.send("No conversation history to clear.", ephemeral=True)

@bot.hybrid_command(
    name='system',
    description='Set a custom system prompt for the AI in the current channel'
)
async def set_system_prompt(ctx, *, prompt: str = None):
    """Set a custom system prompt for the AI in the current channel"""
    if not prompt:
        await ctx.send("Please provide a system prompt.", ephemeral=True)
        return
    
    # Boost engagement slightly - modifying system prompt shows interest
    boost_engagement(ctx.channel.id, 0.3)
    
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
    
    # Send ephemeral confirmation message
    await ctx.send(f"System prompt updated to: '{prompt}'", ephemeral=True)

@bot.hybrid_command(
    name='config',
    description='Configure text generation parameters for this channel'
)
async def config_generation(ctx, param=None, value=None):
    """Configure generation parameters for this channel"""
    channel_id = ctx.channel.id
    
    # Get the current parameters for this channel
    params = get_generation_params(channel_id)
    
    # If no param specified, show current config
    if param is None:
        param_list = '\n'.join([f"- **{k}**: `{v}`" for k, v in params.items()])
        await ctx.send(
            f"Current generation parameters for this channel:\n{param_list}\n\n"
            f"Use `!config <parameter> <value>` to change a parameter.",
            ephemeral=True
        )
        return
    
    # If param is 'reset', reset to defaults
    if param.lower() == 'reset':
        generation_params[channel_id] = DEFAULT_GENERATION_PARAMS.copy()
        await ctx.send("Generation parameters reset to defaults.", ephemeral=True)
        return
    
    # Check if the parameter exists
    if param not in DEFAULT_GENERATION_PARAMS:
        valid_params = '`, `'.join(DEFAULT_GENERATION_PARAMS.keys())
        await ctx.send(
            f"Unknown parameter: `{param}`\nValid parameters: `{valid_params}`", 
            ephemeral=True
        )
        return
    
    # If no value specified, show current value
    if value is None:
        await ctx.send(f"Current value of `{param}`: `{params[param]}`", ephemeral=True)
        return
    
    # Try to convert value to the appropriate type
    current_value = params[param]
    try:
        if isinstance(current_value, bool):
            if value.lower() in ('true', 'yes', '1', 'on'):
                new_value = True
            elif value.lower() in ('false', 'no', '0', 'off'):
                new_value = False
            else:
                raise ValueError("Boolean value must be true/false, yes/no, 1/0, or on/off")
        elif isinstance(current_value, int):
            new_value = int(value)
        elif isinstance(current_value, float):
            new_value = float(value)
        else:
            new_value = value
    except ValueError as e:
        await ctx.send(f"Invalid value for `{param}`: {str(e)}", ephemeral=True)
        return
    
    # Update the parameter
    params[param] = new_value
    generation_params[channel_id] = params
    
    await ctx.send(f"Updated `{param}` to `{new_value}`", ephemeral=True)

@bot.hybrid_command(
    name='logging',
    description='Toggle prompt logging to terminal (owner only)'
)
@commands.is_owner()  # Restrict this command to the bot owner
async def toggle_logging(ctx, setting=None, level=None):
    """Toggle prompt logging to terminal (owner only)"""
    global ENABLE_LOGGING, LOG_LEVEL
    
    # If no setting provided, show current status
    if setting is None:
        await ctx.send(
            f"Logging is currently **{'enabled' if ENABLE_LOGGING else 'disabled'}**\n"
            f"Log level: **{LOG_LEVEL}**\n\n"
            f"Use `!logging on/off` to toggle logging\n"
            f"Use `!logging level 0/1/2` to set log level",
            ephemeral=True  # Make response visible only to the command invoker
        )
        return
    
    # Handle setting the log level
    if setting.lower() == 'level':
        if level is None:
            await ctx.send(
                f"Current log level: **{LOG_LEVEL}**\n"
                f"0 = minimal, 1 = basic, 2 = detailed",
                ephemeral=True
            )
            return
        
        try:
            level_int = int(level)
            if 0 <= level_int <= 2:
                LOG_LEVEL = level_int
                await ctx.send(f"Log level set to **{LOG_LEVEL}**", ephemeral=True)
            else:
                await ctx.send("Log level must be 0, 1, or 2", ephemeral=True)
        except ValueError:
            await ctx.send("Log level must be a number (0, 1, or 2)", ephemeral=True)
        return
    
    # Handle toggling logging on/off
    if setting.lower() in ['on', 'true', 'enable', 'yes', '1']:
        ENABLE_LOGGING = True
        await ctx.send("Prompt logging **enabled**", ephemeral=True)
    elif setting.lower() in ['off', 'false', 'disable', 'no', '0']:
        ENABLE_LOGGING = False
        await ctx.send("Prompt logging **disabled**", ephemeral=True)
    else:
        await ctx.send("Invalid setting. Use `on` or `off`", ephemeral=True)

@bot.hybrid_command(
    name='engagement',
    description='Check the current engagement level for this channel (owner only)'
)
@commands.is_owner()  # Restrict this command to the bot owner
async def check_engagement(ctx):
    """Check the current engagement level for this channel (debug command)"""
    channel_id = ctx.channel.id
    
    # Get current engagement
    engagement_level = get_engagement_level(channel_id)
    
    # Calculate current response chance
    base_chance = RESPONSE_CONFIG["BASE_CHANCE"] / 100.0
    max_chance = RESPONSE_CONFIG["MAX_CHANCE"] / 100.0
    response_chance = base_chance + (max_chance - base_chance) * engagement_level
    
    # Format as percentages
    engagement_percent = engagement_level * 100
    response_percent = response_chance * 100
    
    await ctx.send(
        f"Debug info for this channel:\n"
        f"- Engagement level: {engagement_percent:.2f}%\n"
        f"- Current response chance: {response_percent:.2f}%\n"
        f"- Base response chance: {RESPONSE_CONFIG['BASE_CHANCE']}%\n"
        f"- Max response chance: {RESPONSE_CONFIG['MAX_CHANCE']}%",
        ephemeral=True  # Make response visible only to the command invoker
    )

# Run the bot
if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    TOKEN = os.getenv('DISCORD_TOKEN')
    bot.run(TOKEN)