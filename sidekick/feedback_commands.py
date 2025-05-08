"""
Discord commands for RL feedback collection and management
"""
import discord
from discord.ext import commands
from typing import Optional

from sidekick.rl_pipeline import add_feedback, get_metrics

def setup_feedback_commands(bot):
    """
    Set up feedback-related commands for the Discord bot
    
    Args:
        bot: The Discord bot instance
    """
    @bot.hybrid_command(
        name='feedback',
        description='Provide feedback on the previous AI response'
    )
    async def provide_feedback(ctx, rating: int = None, *, comment: str = None):
        """
        Provide feedback on the bot's most recent response in this channel
        
        Args:
            rating: Optional integer rating (1 = positive, 0 = negative)
            comment: Optional feedback comment
        """
        channel_id = ctx.channel.id
        
        # Verify we have conversation history
        from sidekick.discord import conversation_histories
        
        if channel_id not in conversation_histories or len(conversation_histories[channel_id]) < 2:
            await ctx.send("No recent conversation found to rate.", ephemeral=True)
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
            await ctx.send("No assistant responses found to rate.", ephemeral=True)
            return
        
        # Get the conversation leading up to this response
        previous_convo = history[:assistant_idx]
        assistant_response = history[assistant_idx]["content"]
        
        # Add the feedback
        if rating is None and comment is None:
            await ctx.send("Please provide a rating (1 for good, 0 for bad) and/or a comment.", ephemeral=True)
            return
        
        # Convert rating to integer if needed
        if rating is not None:
            rating = int(rating)
            if rating not in [0, 1]:
                await ctx.send("Rating must be 0 (negative) or 1 (positive).", ephemeral=True)
                return
        
        # Submit feedback
        success = add_feedback(
            conversation=previous_convo,
            response=assistant_response,
            user_feedback=comment,
            binary_rating=rating,
            channel_id=str(channel_id)
        )
        
        if success:
            # Send confirmation
            if rating is not None and comment is not None:
                confirmation_msg = f"Thank you for your {'positive' if rating == 1 else 'negative'} feedback and comment!"
            elif rating is not None:
                confirmation_msg = f"Thank you for your {'positive' if rating == 1 else 'negative'} rating!"
            else:
                confirmation_msg = "Thank you for your feedback comment!"
                
            await ctx.send(confirmation_msg, ephemeral=True)
        else:
            await ctx.send("Error processing feedback. Please try again.", ephemeral=True)

    @bot.hybrid_command(
        name='rl_metrics',
        description='View current RL training metrics'
    )
    @commands.is_owner()  # Restrict to bot owner
    async def show_rl_metrics(ctx):
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
        
        await ctx.send(embed=embed, ephemeral=True)

    # Return the commands
    return provide_feedback, show_rl_metrics