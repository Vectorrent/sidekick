"""
Online RL training pipeline for Sidekick
Handles feedback collection and single-step RL training
"""
import asyncio
import time
import json
import os
from typing import Dict, List, Any, Optional, Tuple, Union

from sidekick.rl import (
    load_model_for_rl,
    compute_reward,
    train_step,
    save_model
)

# Global variables
rl_metrics = {
    "total_feedback": 0,
    "positive_feedback": 0,
    "negative_feedback": 0,
    "neutral_feedback": 0,
    "total_training_steps": 0,
    "avg_reward": 0.0,
    "recent_rewards": [],
    "recent_losses": [],
}

feedback_queue = []
SAVE_INTERVAL = 100  # Save after this many training steps
MAX_RECENT_METRICS = 50  # Number of recent metrics to track

class FeedbackRecord:
    """
    Class to store feedback and conversation for RL training
    """
    def __init__(
        self,
        conversation: List[Dict[str, str]],
        response: str,
        user_feedback: Optional[str] = None,
        binary_rating: Optional[int] = None,
        channel_id: Optional[str] = None
    ):
        self.conversation = conversation
        self.response = response
        self.user_feedback = user_feedback
        self.binary_rating = binary_rating
        self.channel_id = channel_id
        self.timestamp = time.time()
        self.processed = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "conversation": self.conversation,
            "response": self.response,
            "user_feedback": self.user_feedback,
            "binary_rating": self.binary_rating,
            "channel_id": self.channel_id,
            "timestamp": self.timestamp,
            "processed": self.processed
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeedbackRecord':
        """Create from dictionary"""
        record = cls(
            conversation=data["conversation"],
            response=data["response"],
            user_feedback=data.get("user_feedback"),
            binary_rating=data.get("binary_rating"),
            channel_id=data.get("channel_id")
        )
        record.timestamp = data.get("timestamp", time.time())
        record.processed = data.get("processed", False)
        return record

def add_feedback(
    conversation: List[Dict[str, str]],
    response: str,
    user_feedback: Optional[str] = None,
    binary_rating: Optional[int] = None,
    channel_id: Optional[str] = None
) -> bool:
    """
    Add feedback to the queue for training
    
    Args:
        conversation: The conversation history (list of role/content dictionaries)
        response: The model's response that's being rated
        user_feedback: Optional textual feedback
        binary_rating: Optional explicit rating (1 for positive, 0 for negative)
        channel_id: Optional Discord channel ID for tracking
    
    Returns:
        bool: True if feedback was successfully added
    """
    global feedback_queue
    
    try:
        # Create feedback record
        record = FeedbackRecord(
            conversation=conversation,
            response=response,
            user_feedback=user_feedback,
            binary_rating=binary_rating,
            channel_id=channel_id
        )
        
        # Add to queue
        feedback_queue.append(record)
        return True
    except Exception as e:
        print(f"Error adding feedback: {e}")
        return False

async def process_single_feedback(record: FeedbackRecord) -> Dict[str, Any]:
    """
    Process a single feedback record with RL training
    
    Args:
        record: The feedback record to process
    
    Returns:
        dict: Training metrics
    """
    global rl_metrics
    
    try:
        # Compute reward
        reward = compute_reward(
            response=record.response,
            user_feedback=record.user_feedback,
            binary_rating=record.binary_rating
        )
        
        # Update metrics based on reward
        rl_metrics["total_feedback"] += 1
        if reward > 0.2:
            rl_metrics["positive_feedback"] += 1
        elif reward < -0.2:
            rl_metrics["negative_feedback"] += 1
        else:
            rl_metrics["neutral_feedback"] += 1
        
        # Perform training step
        metrics = await asyncio.to_thread(
            train_step,
            record.conversation,
            record.response,
            reward
        )
        
        # Record metrics
        rl_metrics["total_training_steps"] += 1
        rl_metrics["recent_rewards"].append(reward)
        if "loss" in metrics:
            rl_metrics["recent_losses"].append(metrics["loss"])
        
        # Keep only recent metrics
        if len(rl_metrics["recent_rewards"]) > MAX_RECENT_METRICS:
            rl_metrics["recent_rewards"] = rl_metrics["recent_rewards"][-MAX_RECENT_METRICS:]
        if len(rl_metrics["recent_losses"]) > MAX_RECENT_METRICS:
            rl_metrics["recent_losses"] = rl_metrics["recent_losses"][-MAX_RECENT_METRICS:]
        
        # Calculate average reward
        if rl_metrics["recent_rewards"]:
            rl_metrics["avg_reward"] = sum(rl_metrics["recent_rewards"]) / len(rl_metrics["recent_rewards"])
        
        # Save periodically
        if rl_metrics["total_training_steps"] % SAVE_INTERVAL == 0:
            await asyncio.to_thread(save_model, "./rl_model")
            
            # Also save metrics
            save_metrics()
        
        # Mark as processed
        record.processed = True
        
        return {**metrics, "reward": reward}
    
    except Exception as e:
        print(f"Error processing feedback: {e}")
        return {"error": str(e)}

async def process_feedback_queue() -> int:
    """
    Process all pending feedback in the queue
    
    Returns:
        int: Number of feedback items processed
    """
    global feedback_queue
    
    processed_count = 0
    
    # Process all items in queue
    for record in list(feedback_queue):
        if not record.processed:
            await process_single_feedback(record)
            processed_count += 1
    
    # Remove processed items
    feedback_queue = [record for record in feedback_queue if not record.processed]
    
    return processed_count

def save_metrics(filename: str = "rl_metrics.json") -> bool:
    """
    Save training metrics to file
    
    Args:
        filename: The file to save metrics to
    
    Returns:
        bool: True if saved successfully
    """
    try:
        # Create metrics with timestamp
        save_data = {
            **rl_metrics,
            "timestamp": time.time()
        }
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        return True
    except Exception as e:
        print(f"Error saving metrics: {e}")
        return False

def load_metrics(filename: str = "rl_metrics.json") -> bool:
    """
    Load training metrics from file
    
    Args:
        filename: The file to load metrics from
    
    Returns:
        bool: True if loaded successfully
    """
    global rl_metrics
    
    try:
        if not os.path.exists(filename):
            return False
        
        with open(filename, 'r') as f:
            loaded_metrics = json.load(f)
        
        # Update metrics
        for key, value in loaded_metrics.items():
            if key in rl_metrics:
                rl_metrics[key] = value
        
        return True
    except Exception as e:
        print(f"Error loading metrics: {e}")
        return False

def get_metrics() -> Dict[str, Any]:
    """
    Get current RL training metrics
    
    Returns:
        dict: Training metrics
    """
    return rl_metrics.copy()

async def initialize_rl_pipeline():
    """
    Initialize the RL pipeline
    - Load model with LoRA
    - Load existing metrics if available
    """
    # Load model in background thread
    await asyncio.to_thread(load_model_for_rl)
    
    # Load metrics
    load_metrics()
    
    print("RL pipeline initialized")