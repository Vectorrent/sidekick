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
bot = commands.Bot(command_prefix='!', intents=intents)

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

# Maximum conversation history length
MAX_HISTORY_LENGTH = 10

# System prompt for the AI
DEFAULT_SYSTEM_PROMPT = "You are an assistant named Samn. You are polite, knowledgeable, and concise."

# Default generation parameters for LLM responses
# These can be overridden per channel
DEFAULT_GENERATION_PARAMS = {
    "max_new_tokens": 512,     # Maximum length of generated text
    "temperature": 0.7,        # Controls randomness (0.0-1.0) - higher = more random
    # "top_p": 0.9,            # Nucleus sampling - keep tokens with cumulative probability >= top_p
    # "top_k": 50,             # Keep only the top k tokens - 0 means no filtering
    "min_p": 0.02,             # Minimum token probability, which will be scaled by the probability of the most likely token.
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
    """Determine if the bot should respond to a message based on engagement level"""
    channel_id = message.channel.id
    
    # Get current engagement level for this channel
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

# This event triggers on every message sent that the bot can see
@bot.event
async def on_message(message):
    # Don't respond to our own messages to prevent infinite loops
    if message.author == bot.user:
        return
    
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
    else:
        # Not a direct mention - check if we should randomly respond
        should_respond = should_respond_to_message(message)
        
        if should_respond:
            # If we're going to respond, boost engagement
            boost_engagement(message.channel.id, RESPONSE_CONFIG["RESPONSE_ENGAGEMENT_BOOST"])
    
    # Generate and send a response if appropriate
    if should_respond and content:
        # Send typing indicator
        async with message.channel.typing():
            # Add user message to conversation
            add_to_conversation(message.channel.id, "user", f"{message.author.display_name}: {content}")
            
            # Get conversation history for this channel
            history = get_conversation_history(message.channel.id)
            
            # Get generation parameters for this channel
            params = get_generation_params(message.channel.id)
            
            # Generate response using the model (run in thread pool to not block the event loop)
            response = await asyncio.to_thread(
                generate_response, 
                history,
                **params
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
    
    # Boost engagement level significantly - direct command is similar to a mention
    boost_engagement(ctx.channel.id, RESPONSE_CONFIG["DIRECT_ENGAGEMENT_BOOST"])
    
    # Send typing indicator
    async with ctx.typing():
        # Add user message to conversation
        add_to_conversation(ctx.channel.id, "user", f"{ctx.author.display_name}: {message}")
        
        # Get conversation history for this channel
        history = get_conversation_history(ctx.channel.id)
        
        # Get generation parameters for this channel
        params = get_generation_params(ctx.channel.id)
        
        # Generate response using the model (run in thread pool to not block the event loop)
        response = await asyncio.to_thread(
            generate_response, 
            history,
            **params
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
        
        # Reset engagement for this channel (clearing history suggests ending the conversation)
        if ctx.channel.id in conversation_engagement:
            conversation_engagement.pop(ctx.channel.id)
        
        await ctx.send("Conversation history cleared!")
    else:
        await ctx.send("No conversation history to clear.")

@bot.command(name='system')
async def set_system_prompt(ctx, *, prompt: str = None):
    """Set a custom system prompt for the AI in the current channel"""
    if not prompt:
        await ctx.send("Please provide a system prompt.")
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
    
    await ctx.send(f"System prompt updated to: '{prompt}'")

@bot.command(name='config')
async def config_generation(ctx, param=None, value=None):
    """Configure generation parameters for this channel"""
    channel_id = ctx.channel.id
    
    # Get the current parameters for this channel
    params = get_generation_params(channel_id)
    
    # If no param specified, show current config
    if param is None:
        param_list = '\n'.join([f"- **{k}**: `{v}`" for k, v in params.items()])
        await ctx.send(f"Current generation parameters for this channel:\n{param_list}\n\n"
                      f"Use `!config <parameter> <value>` to change a parameter.")
        return
    
    # If param is 'reset', reset to defaults
    if param.lower() == 'reset':
        generation_params[channel_id] = DEFAULT_GENERATION_PARAMS.copy()
        await ctx.send("Generation parameters reset to defaults.")
        return
    
    # Check if the parameter exists
    if param not in DEFAULT_GENERATION_PARAMS:
        valid_params = '`, `'.join(DEFAULT_GENERATION_PARAMS.keys())
        await ctx.send(f"Unknown parameter: `{param}`\nValid parameters: `{valid_params}`")
        return
    
    # If no value specified, show current value
    if value is None:
        await ctx.send(f"Current value of `{param}`: `{params[param]}`")
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
        await ctx.send(f"Invalid value for `{param}`: {str(e)}")
        return
    
    # Update the parameter
    params[param] = new_value
    generation_params[channel_id] = params
    
    await ctx.send(f"Updated `{param}` to `{new_value}`")

@bot.command(name='engagement')
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
    
    await ctx.send(f"Debug info for this channel:\n"
                  f"- Engagement level: {engagement_percent:.2f}%\n"
                  f"- Current response chance: {response_percent:.2f}%\n"
                  f"- Base response chance: {RESPONSE_CONFIG['BASE_CHANCE']}%\n"
                  f"- Max response chance: {RESPONSE_CONFIG['MAX_CHANCE']}%")

# Run the bot
if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    TOKEN = os.getenv('DISCORD_TOKEN')
    bot.run(TOKEN)