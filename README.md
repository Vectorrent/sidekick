# Sidekick

A Discord bot powered by SmolLM2-135M-Instruct with reinforcement learning capabilities.

## Features

- **Discord Bot Integration**: Responds to commands and messages with AI-generated text
- **Local LLM**: Uses SmolLM2-135M-Instruct model for text generation
- **Reinforcement Learning**: Improves responses over time using user feedback
- **Standalone Chat Interface**: Interact with the model without Discord

## Quick Start

### Requirements

- Python 3.8+
- Discord Bot Token (set in `.env` file)

### Installation

```bash
# Clone this repository
git clone https://github.com/yourusername/sidekick.git
cd sidekick

# Install dependencies
pip install -e .
```

### Usage

1. Create a `.env` file with your Discord token:
   ```
   DISCORD_TOKEN=your_token_here
   ```

2. Run the Discord bot:
   ```bash
   python main.py
   ```

3. Or use the standalone chat interface:
   ```bash
   python -m sidekick.chat
   ```

## Discord Commands

| Command | Description |
|---------|-------------|
| `/system <prompt>` | Set custom system prompt |
| `/good_bot [comment]` | Rate last response positively |
| `/bad_bot [comment]` | Rate last response negatively |

## Core Components

- **Engagement System**: Bot becomes more likely to respond in active channels
- **Reinforcement Learning**: Uses PPO and LoRA for continuous improvement
- **Text Generation Config**: Customizable per-channel generation parameters

## Technical Details

- Uses Hugging Face Transformers for model loading and inference
- Parameter-Efficient Fine-Tuning (PEFT) with LoRA adapters
- Proximal Policy Optimization (PPO) via TRL library
- Automatic GPU detection and utilization