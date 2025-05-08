"""
Discord slash commands for RL feedback collection via good_bot and bad_bot commands
"""
import discord
from discord.ext import commands
from typing import Optional

from sidekick.rl_pipeline_ppo import add_feedback, get_metrics

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
            description="Current PPO training statistics",
            color=0x00ff00
        )
        
        # Add general stats
        embed.add_field(
            name="Feedback Stats",
            value=(
                f"**Total Feedback**: {metrics['total_feedback']}\n"
                f"**Positive Feedback**: {metrics['positive_feedback']}\n"
                f"**Negative Feedback**: {metrics['negative_feedback']}\n"
                f"**Neutral Feedback**: {metrics['neutral_feedback']}\n"
            ),
            inline=False
        )
        
        # Add PPO stats if available
        if 'ppo_stats' in metrics:
            ppo = metrics['ppo_stats']
            embed.add_field(
                name="PPO Training Stats",
                value=(
                    f"**Total Steps**: {ppo['total_steps']}\n"
                    f"**Avg Reward**: {ppo['avg_reward']:.4f}\n"
                    f"**Avg Policy Loss**: {ppo['avg_policy_loss']:.6f if ppo['avg_policy_loss'] is not None else 'N/A'}\n"
                    f"**Avg Value Loss**: {ppo['avg_value_loss']:.6f if ppo['avg_value_loss'] is not None else 'N/A'}\n"
                    f"**Avg KL Divergence**: {ppo['avg_kl_div']:.6f if ppo['avg_kl_div'] is not None else 'N/A'}\n"
                ),
                inline=False
            )
        
        await interaction.response.send_message(embed=embed, ephemeral=True)

    # Return the commands
    return good_bot, bad_bot, show_rl_metrics