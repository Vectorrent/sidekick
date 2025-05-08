# Sidekick

A Discord bot with SmolLM2-135M-Instruct integration and reinforcement learning capabilities.

## Overview

Sidekick is a small experimental project that integrates a Discord bot with a local LLM (SmolLM2-135M-Instruct). The project consists of:

1. A Discord bot implementation that responds to messages and commands
2. A standalone chat interface for interacting with the SmolLM2 model
3. An integrated system that allows the Discord bot to use the LLM for conversations
4. Reinforcement learning with LoRA/PEFT for continuous model improvement

## Environment Setup

The project requires the following dependencies:
- python-dotenv (for managing environment variables)
- discord.py (for Discord bot functionality)
- torch (for machine learning)
- transformers (for HuggingFace model integration)
- peft (for Parameter-Efficient Fine-Tuning)
- trl (for reinforcement learning)

## Environment Variables

The Discord bot requires a `DISCORD_TOKEN` environment variable to be set in a `.env` file at the root of the project.

Example `.env` file:
```
DISCORD_TOKEN=your_token_here
```

## Running the Application

To run the Discord bot with LLM integration:
```bash
python main.py
```

To run the standalone chat interface (without Discord):
```bash
python -m sidekick.chat
```

## Discord Bot Features

- Command prefix: `!`
- Built-in commands:
  - `!ping`: Responds with "Pong!" to check if the bot is responsive
  - `!chat <message>`: Chat with the AI model
  - `!clear`: Clear conversation history for the current channel
  - `!system <prompt>`: Set a custom system prompt for the AI in the current channel
  - `!config`: View and modify text generation parameters
  - `!feedback <rating> <comment>`: Provide feedback on the last AI response
  - `!rl_metrics`: [Owner only] View reinforcement learning metrics
  - `!engagement`: [Owner only] Debug command to check current engagement level
- Automatic responses:
  - Responds to @mentions with AI-generated responses
  - Random chance to respond to any message, even without being mentioned
  - Increased response chance for channels where the bot is already engaged in conversation

## Reinforcement Learning Integration

The bot includes a custom RL implementation with LoRA parameter-efficient fine-tuning:

- Uses the PEFT library for LoRA adapters
- Supports online single-step RL training from user feedback
- Feedback can be provided through:
  - Explicit ratings via the `!feedback` command
  - Comment sentiment analysis from users
- Training metrics can be viewed with the `!rl_metrics` command
- LoRA adapters are automatically saved after every 100 training steps
- Every bot response is added to a feedback queue for potential training

## Conversation Engagement System

The bot includes a natural conversation engagement system:

- Has a small random chance to respond to any message
- After responding or being engaged, becomes more likely to respond to subsequent messages in the same channel
- Engagement level affects response probability
- Engagement level naturally decays over time
- Each channel has its own independent engagement tracking