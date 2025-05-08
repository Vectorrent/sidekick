import os
import discord
import asyncio
import random
import time
from discord.ext import commands, tasks
from collections import defaultdict
from sidekick.chat import load_model, generate_response
from sidekick.rl_pipeline_ppo import (
    add_feedback, 
    process_feedback_queue, 
    get_metrics, 
    initialize_rl_pipeline
)
from sidekick.feedback_slash_commands import setup_feedback_slash_commands
from sidekick.help_command import setup_help_command

# Set up intents (required in Discord.py v2.0+)
intents = discord.Intents.default()
intents.message_content = True  # Necessary to read message content

# Create bot instance with native Discord application commands
class SidekickBot(commands.Bot):
    def __init__(self):
        # Initialize with empty command prefix and disable the default help command
        super().__init__(
            command_prefix=commands.when_mentioned, 
            intents=intents,
            help_command=None  # Disable the default help command
        )
    
    async def setup_hook(self):
        """A hook that is called when the bot is starting to sync commands and do setup"""
        print("Setting up application commands...")
        
        # Setup slash commands before syncing
        setup_feedback_slash_commands(self)
        setup_help_command(self)
        
        # Sync commands with Discord
        print("Syncing commands with Discord...")
        try:
            # sync() will register all slash commands globally
            await self.tree.sync()
            # Check if commands were registered
            commands = await self.tree.fetch_commands()
            print(f"Registered {len(commands)} application commands:")
            for cmd in commands:
                print(f"  /{cmd.name} - {cmd.description}")
            print("Application commands synced successfully!")
        except Exception as e:
            print(f"Error syncing commands: {e}")

# Initialize the bot with our custom class
bot = SidekickBot()

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
    
    # Initialize the RL pipeline
    await initialize_rl_pipeline()
    
    # Start the engagement decay task
    engagement_decay_task.start()
    
    # Start the RL feedback processing task
    rl_feedback_task.start()
    
    # We don't need to print commands here anymore - they're printed in setup_hook
    print("Bot is ready to use! Slash commands are available.")

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

@tasks.loop(seconds=60)
async def rl_feedback_task():
    """Background task to process RL feedback queue"""
    try:
        # Process all pending feedback
        processed = await process_feedback_queue()
        if processed > 0:
            print(f"Processed {processed} feedback items for RL training")
    except Exception as e:
        print(f"Error in RL feedback processing task: {e}")

@rl_feedback_task.before_loop
async def before_rl_feedback_task():
    """Wait for the bot to be ready before starting the RL feedback task"""
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
            
            # Add to RL feedback queue (automatically will be processed by task)
            # Pass an empty string as user_feedback to ensure it uses the automatic rewarder
            add_feedback(
                conversation=history.copy(), 
                response=response,
                user_feedback="",  # Empty string so compute_reward knows to use automatic rewards
                channel_id=str(message.channel.id)
            )
            
            # Split the response if it's too long for Discord
            message_chunks = split_long_message(response)
            
            # Send each chunk as a separate message
            for chunk in message_chunks:
                await message.channel.send(chunk)
    
    # NOTE: We're already processing commands at the beginning of this function
    # This was moved to ensure commands are processed before any AI response logic

# Convert to pure slash command
@bot.tree.command(
    name='ping',
    description='Check if the bot is responsive'
)
async def ping(interaction: discord.Interaction):
    """Simple command that responds with 'Pong!' to check if bot is responsive"""
    # Deliberately not using ephemeral=True here, as ping is often used to check if 
    # the bot is visible to everyone in the channel
    await interaction.response.send_message('Pong!')

@bot.tree.command(
    name='chat',
    description='Chat with the AI model'
)
async def chat(interaction: discord.Interaction, message: str):
    """Chat with the AI model"""
    # Check if this is a DM channel
    is_dm_channel = isinstance(interaction.channel, discord.DMChannel)
    
    # Boost engagement level significantly - direct command is similar to a mention
    boost_engagement(interaction.channel_id, RESPONSE_CONFIG["DIRECT_ENGAGEMENT_BOOST"])
    
    # Defer the response since generation might take a while
    await interaction.response.defer(thinking=True)
    
    # Add user message to conversation
    add_to_conversation(
        interaction.channel_id, 
        "user", 
        f"{interaction.user.display_name}: {message}", 
        is_dm=is_dm_channel
    )
    
    # Get conversation history for this channel
    history = get_conversation_history(interaction.channel_id, is_dm=is_dm_channel)
    
    # Log prompt to terminal if logging is enabled
    channel_name = interaction.channel.name if hasattr(interaction.channel, 'name') else "DM"
    log_prompt(history, channel_name=channel_name, author_name=interaction.user.display_name)
    
    # Get generation parameters for this channel
    params = get_generation_params(interaction.channel_id)
    
    # Generate response using the model (run in thread pool to not block the event loop)
    response = await asyncio.to_thread(
        generate_response, 
        history,
        **params
    )
    
    # Add bot response to conversation history
    add_to_conversation(interaction.channel_id, "assistant", response, is_dm=is_dm_channel)
    
    # Add to RL feedback queue (automatically will be processed by task)
    # Pass an empty string as user_feedback to ensure it uses the automatic rewarder
    add_feedback(
        conversation=history.copy(), 
        response=response,
        user_feedback="",  # Empty string so compute_reward knows to use automatic rewards
        channel_id=str(interaction.channel_id)
    )
    
    # Split the response if it's too long for Discord
    message_chunks = split_long_message(response)
    
    # Send the first chunk as the response to the interaction
    await interaction.followup.send(message_chunks[0])
    
    # Send any additional chunks as follow-up messages
    if len(message_chunks) > 1:
        for chunk in message_chunks[1:]:
            await interaction.channel.send(chunk)

@bot.tree.command(
    name='clear',
    description='Clear the conversation history for the current channel'
)
async def clear_history(interaction: discord.Interaction):
    """Clear the conversation history for the current channel"""
    if interaction.channel_id in conversation_histories:
        # Reset history but keep the system prompt
        system_prompt = DEFAULT_SYSTEM_PROMPT
        for msg in conversation_histories[interaction.channel_id]:
            if msg["role"] == "system":
                system_prompt = msg["content"]
                break
        
        conversation_histories[interaction.channel_id] = [{"role": "system", "content": system_prompt}]
        
        # Reset engagement for this channel (clearing history suggests ending the conversation)
        if interaction.channel_id in conversation_engagement:
            conversation_engagement.pop(interaction.channel_id)
        
        # Send ephemeral confirmation message
        await interaction.response.send_message("Conversation history cleared!", ephemeral=True)
    else:
        await interaction.response.send_message("No conversation history to clear.", ephemeral=True)

@bot.tree.command(
    name='system',
    description='Set a custom system prompt for the AI in the current channel'
)
async def set_system_prompt(interaction: discord.Interaction, prompt: str):
    """Set a custom system prompt for the AI in the current channel"""    
    # Boost engagement slightly - modifying system prompt shows interest
    boost_engagement(interaction.channel_id, 0.3)
    
    history = get_conversation_history(interaction.channel_id)
    
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
    await interaction.response.send_message(f"System prompt updated to: '{prompt}'", ephemeral=True)

@bot.tree.command(
    name='config',
    description='View current generation parameters for this channel'
)
async def config_view(interaction: discord.Interaction):
    """View current generation parameters for this channel"""
    channel_id = interaction.channel_id
    
    # Get the current parameters for this channel
    params = get_generation_params(channel_id)
    
    # Format params for display
    param_list = '\n'.join([f"- **{k}**: `{v}`" for k, v in params.items()])
    
    await interaction.response.send_message(
        f"Current generation parameters for this channel:\n{param_list}\n\n"
        f"Use `/config_set` or `/config_reset` to modify parameters.",
        ephemeral=True
    )

@bot.tree.command(
    name='config_reset',
    description='Reset generation parameters to default values'
)
async def config_reset(interaction: discord.Interaction):
    """Reset generation parameters to default values"""
    channel_id = interaction.channel_id
    
    # Reset to defaults
    generation_params[channel_id] = DEFAULT_GENERATION_PARAMS.copy()
    
    await interaction.response.send_message(
        "Generation parameters reset to defaults.",
        ephemeral=True
    )

@bot.tree.command(
    name='config_set',
    description='Set a specific generation parameter'
)
async def config_set(
    interaction: discord.Interaction, 
    parameter: str,
    value: str
):
    """Set a specific generation parameter"""
    channel_id = interaction.channel_id
    
    # Get the current parameters for this channel
    params = get_generation_params(channel_id)
    
    # Check if the parameter exists
    if parameter not in DEFAULT_GENERATION_PARAMS:
        valid_params = '`, `'.join(DEFAULT_GENERATION_PARAMS.keys())
        await interaction.response.send_message(
            f"Unknown parameter: `{parameter}`\nValid parameters: `{valid_params}`", 
            ephemeral=True
        )
        return
    
    # Try to convert value to the appropriate type
    current_value = params[parameter]
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
        await interaction.response.send_message(
            f"Invalid value for `{parameter}`: {str(e)}", 
            ephemeral=True
        )
        return
    
    # Update the parameter
    params[parameter] = new_value
    generation_params[channel_id] = params
    
    await interaction.response.send_message(
        f"Updated `{parameter}` to `{new_value}`", 
        ephemeral=True
    )

@bot.tree.command(
    name='logging',
    description='View current logging status (owner only)'
)
@commands.is_owner()  # Restrict this command to the bot owner
async def logging_status(interaction: discord.Interaction):
    """View current logging status (owner only)"""
    global ENABLE_LOGGING, LOG_LEVEL
    
    await interaction.response.send_message(
        f"Logging is currently **{'enabled' if ENABLE_LOGGING else 'disabled'}**\n"
        f"Log level: **{LOG_LEVEL}**\n\n"
        f"Use `/logging_toggle` to enable/disable logging\n"
        f"Use `/logging_level` to set log level",
        ephemeral=True  # Make response visible only to the command invoker
    )

@bot.tree.command(
    name='logging_toggle',
    description='Toggle prompt logging on/off (owner only)'
)
@commands.is_owner()  # Restrict this command to the bot owner
async def logging_toggle(interaction: discord.Interaction, enable: bool):
    """Toggle prompt logging on/off (owner only)"""
    global ENABLE_LOGGING
    
    ENABLE_LOGGING = enable
    
    await interaction.response.send_message(
        f"Prompt logging **{'enabled' if ENABLE_LOGGING else 'disabled'}**",
        ephemeral=True
    )

@bot.tree.command(
    name='logging_level',
    description='Set logging detail level (owner only)'
)
@commands.is_owner()  # Restrict this command to the bot owner
async def logging_level(
    interaction: discord.Interaction, 
    level: int
):
    """Set logging detail level (owner only)"""
    global LOG_LEVEL
    
    if 0 <= level <= 2:
        LOG_LEVEL = level
        await interaction.response.send_message(
            f"Log level set to **{LOG_LEVEL}**\n"
            f"0 = minimal, 1 = basic, 2 = detailed",
            ephemeral=True
        )
    else:
        await interaction.response.send_message(
            "Log level must be 0, 1, or 2", 
            ephemeral=True
        )

@bot.tree.command(
    name='engagement',
    description='Check the current engagement level for this channel (owner only)'
)
@commands.is_owner()  # Restrict this command to the bot owner
async def check_engagement(interaction: discord.Interaction):
    """Check the current engagement level for this channel (debug command)"""
    channel_id = interaction.channel_id
    
    # Get current engagement
    engagement_level = get_engagement_level(channel_id)
    
    # Calculate current response chance
    base_chance = RESPONSE_CONFIG["BASE_CHANCE"] / 100.0
    max_chance = RESPONSE_CONFIG["MAX_CHANCE"] / 100.0
    response_chance = base_chance + (max_chance - base_chance) * engagement_level
    
    # Format as percentages
    engagement_percent = engagement_level * 100
    response_percent = response_chance * 100
    
    await interaction.response.send_message(
        f"Debug info for this channel:\n"
        f"- Engagement level: {engagement_percent:.2f}%\n"
        f"- Current response chance: {response_percent:.2f}%\n"
        f"- Base response chance: {RESPONSE_CONFIG['BASE_CHANCE']}%\n"
        f"- Max response chance: {RESPONSE_CONFIG['MAX_CHANCE']}%",
        ephemeral=True  # Make response visible only to the command invoker
    )

# Remove hybrid commands that have been converted to slash commands
@bot.event
async def on_command_error(ctx, error):
    """
    This event is triggered when a command raises an error.
    We'll use it to handle command not found errors gracefully.
    """
    if isinstance(error, commands.CommandNotFound):
        # Silently ignore command not found errors (legacy prefix commands)
        # We're using slash commands now
        pass
    else:
        # Print other errors for debugging
        print(f"Command error: {error}")

# Run the bot
if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    TOKEN = os.getenv('DISCORD_TOKEN')
    bot.run(TOKEN)