"""
Custom help command for the Sidekick Discord bot using slash commands
"""
import discord

def setup_help_command(bot):
    """
    Create a custom help command using Discord's slash commands
    
    Args:
        bot: The Discord bot instance
    """
    @bot.tree.command(
        name='help',
        description='Shows available commands and how to use them'
    )
    async def help_command(interaction: discord.Interaction):
        """Show help information about the bot and its commands"""
        
        embed = discord.Embed(
            title="Sidekick Bot Help",
            description="Here are the available commands:",
            color=0x3498db
        )
        
        # Add command sections
        embed.add_field(
            name="Basic Commands",
            value=(
                "• `/system <prompt>` - Set custom system prompt\n"
            ),
            inline=False
        )
        
        embed.add_field(
            name="Feedback Commands",
            value=(
                "• `/good_bot [comment]` - Rate the last response positively\n"
                "• `/bad_bot [comment]` - Rate the last response negatively\n"
            ),
            inline=False
        )
        
        embed.add_field(
            name="Owner Commands",
            value=(
                "• `/engagement` - Check channel engagement level\n"
                "• `/logging` - Toggle prompt logging on/off\n"
                "• `/metrics` - View reinforcement learning metrics\n"
            ),
            inline=False
        )
        
        embed.add_field(
            name="How It Works",
            value=(
                "The bot uses a small language model (SmolLM2) with reinforcement learning. "
                "Every time you use `/good_bot` or `/bad_bot`, you help train the model to "
                "generate better responses. The bot will also occasionally respond to messages "
                "without being directly mentioned."
            ),
            inline=False
        )
        
        await interaction.response.send_message(embed=embed, ephemeral=True)
    
    return help_command