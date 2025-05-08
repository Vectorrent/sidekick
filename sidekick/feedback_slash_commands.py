"""
Discord slash commands for RL feedback collection via good_bot and bad_bot commands
"""
import discord
from discord.ext import commands
from typing import Optional

from sidekick.rl_pipeline import add_feedback, get_metrics

def setup_feedback_slash_commands(bot):
    """
    Set up good_bot/bad_bot commands for the Discord bot
    
    Args:
        bot: The Discord bot instance
    """
    @bot.tree.command(
        name='good_bot',
        description='Give positive feedback to the bot\'s last response'
    )
    async def good_bot(interaction: discord.Interaction, comment: Optional[str] = None):
        """
        Provide positive feedback on the bot's most recent response
        
        Args:
            comment: Optional feedback comment
        """
        channel_id = interaction.channel_id
        
        # Verify we have conversation history
        from sidekick.discord import conversation_histories
        
        if channel_id not in conversation_histories or len(conversation_histories[channel_id]) < 2:
            await interaction.response.send_message(
                "No recent conversation found to rate.", 
                ephemeral=True
            )
            return
        
        # Get the most recent messages (focusing on the last assistant response)
        history = conversation_histories[channel_id]
        
        # Find the most recent assistant message
        assistant_idx = None
        for i in range(len(history) - 1, -1, -1):
            if history[i]["role"] == "assistant":
                assistant_idx = i
                break
        
        if assistant_idx is None:
            await interaction.response.send_message(
                "No assistant responses found to rate.", 
                ephemeral=True
            )
            return
        
        # Get the conversation leading up to this response
        previous_convo = history[:assistant_idx]
        assistant_response = history[assistant_idx]["content"]
        
        # Submit positive feedback (rating=1)
        success = add_feedback(
            conversation=previous_convo,
            response=assistant_response,
            user_feedback=comment,
            binary_rating=1,  # Positive rating
            channel_id=str(channel_id)
        )
        
        if success:
            if comment:
                confirmation_msg = f"Thanks for the positive feedback! Comment: '{comment}'"
            else:
                confirmation_msg = "Thanks for the positive feedback!"
                
            await interaction.response.send_message(
                confirmation_msg, 
                ephemeral=True
            )
        else:
            await interaction.response.send_message(
                "Error processing feedback. Please try again.", 
                ephemeral=True
            )

    @bot.tree.command(
        name='bad_bot',
        description='Give negative feedback to the bot\'s last response'
    )
    async def bad_bot(interaction: discord.Interaction, comment: Optional[str] = None):
        """
        Provide negative feedback on the bot's most recent response
        
        Args:
            comment: Optional feedback comment
        """
        channel_id = interaction.channel_id
        
        # Verify we have conversation history
        from sidekick.discord import conversation_histories
        
        if channel_id not in conversation_histories or len(conversation_histories[channel_id]) < 2:
            await interaction.response.send_message(
                "No recent conversation found to rate.", 
                ephemeral=True
            )
            return
        
        # Get the most recent messages (focusing on the last assistant response)
        history = conversation_histories[channel_id]
        
        # Find the most recent assistant message
        assistant_idx = None
        for i in range(len(history) - 1, -1, -1):
            if history[i]["role"] == "assistant":
                assistant_idx = i
                break
        
        if assistant_idx is None:
            await interaction.response.send_message(
                "No assistant responses found to rate.", 
                ephemeral=True
            )
            return
        
        # Get the conversation leading up to this response
        previous_convo = history[:assistant_idx]
        assistant_response = history[assistant_idx]["content"]
        
        # Submit negative feedback (rating=0)
        success = add_feedback(
            conversation=previous_convo,
            response=assistant_response,
            user_feedback=comment,
            binary_rating=0,  # Negative rating
            channel_id=str(channel_id)
        )
        
        if success:
            if comment:
                confirmation_msg = f"Thanks for the feedback. I'll try to do better. Comment: '{comment}'"
            else:
                confirmation_msg = "Thanks for the feedback. I'll try to do better."
                
            await interaction.response.send_message(
                confirmation_msg, 
                ephemeral=True
            )
        else:
            await interaction.response.send_message(
                "Error processing feedback. Please try again.", 
                ephemeral=True
            )

    @bot.tree.command(
        name='rl_metrics',
        description='View current RL training metrics (owner only)'
    )
    @commands.is_owner()  # Restrict to bot owner
    async def show_rl_metrics(interaction: discord.Interaction):
        """Show current reinforcement learning metrics (owner only)"""
        metrics = get_metrics()
        
        # Format the metrics for display
        embed = discord.Embed(
            title="Reinforcement Learning Metrics",
            description="Current training statistics",
            color=0x00ff00
        )
        
        # Add general stats
        embed.add_field(
            name="Training Stats",
            value=(
                f"**Total Feedback**: {metrics['total_feedback']}\n"
                f"**Training Steps**: {metrics['total_training_steps']}\n"
                f"**Avg Reward**: {metrics['avg_reward']:.4f}\n"
            ),
            inline=False
        )
        
        # Add feedback breakdown
        embed.add_field(
            name="Feedback Breakdown",
            value=(
                f"**Positive**: {metrics['positive_feedback']} "
                f"({100 * metrics['positive_feedback'] / max(1, metrics['total_feedback']):.1f}%)\n"
                f"**Neutral**: {metrics['neutral_feedback']} "
                f"({100 * metrics['neutral_feedback'] / max(1, metrics['total_feedback']):.1f}%)\n"
                f"**Negative**: {metrics['negative_feedback']} "
                f"({100 * metrics['negative_feedback'] / max(1, metrics['total_feedback']):.1f}%)\n"
            ),
            inline=False
        )
        
        # Loss if available
        if metrics['recent_losses']:
            avg_loss = sum(metrics['recent_losses']) / len(metrics['recent_losses'])
            embed.add_field(
                name="Recent Training",
                value=f"**Avg Loss**: {avg_loss:.4f}\n",
                inline=False
            )
        
        await interaction.response.send_message(embed=embed, ephemeral=True)

    # Return the commands
    return good_bot, bad_bot, show_rl_metrics