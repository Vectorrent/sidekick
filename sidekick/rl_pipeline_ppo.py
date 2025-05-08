"""
Online RL training pipeline for Sidekick using PPO and TRL
Handles feedback collection and training with PPO
"""
import asyncio
import time
import json
import os
from typing import Dict, List, Any, Optional, Tuple, Union
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

from sidekick.ppo_trainer import (
    initialize_ppo_trainer,
    train_step,
    save_model,
    get_stats,
    save_stats_history,
    stats_history
)

# Global variables
model = None
tokenizer = None
device = None
feedback_queue = []
feedback_stats = {
    "total_feedback": 0,
    "positive_feedback": 0,
    "negative_feedback": 0,
    "neutral_feedback": 0
}

# Save interval
SAVE_INTERVAL = 100

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

def compute_reward(response: str, user_feedback: Optional[str] = None, binary_rating: Optional[int] = None) -> float:
    """
    Compute reward for RL training
    
    Args:
        response (str): The model's response to rate
        user_feedback (str, optional): Optional feedback text from user
        binary_rating (int, optional): Optional binary rating (1 for positive, 0 for negative)
        
    Returns:
        float: Reward value between -1 and 1
    """
    # Simple implementation: use explicit binary rating if provided
    if binary_rating is not None:
        return 1.0 if binary_rating > 0 else -1.0
    
    # If user feedback available, we can use simple heuristics 
    if user_feedback:
        # Extremely simple sentiment analysis - just a placeholder
        positive_words = ["good", "great", "excellent", "helpful", "thanks", "thank", "nice", "perfect"]
        negative_words = ["bad", "wrong", "incorrect", "not helpful", "terrible", "useless"]
        
        feedback_lower = user_feedback.lower()
        
        # Count positive and negative indicators
        positive_count = sum(1 for word in positive_words if word in feedback_lower)
        negative_count = sum(1 for word in negative_words if word in feedback_lower)
        
        # Compute a simple score
        total = positive_count + negative_count
        if total == 0:
            return 0.0
        return (positive_count - negative_count) / total
    
    # Default: neutral reward
    return 0.0

async def load_model_for_ppo(model_name="HuggingFaceTB/SmolLM2-135M-Instruct", adapter_path="./rl_model"):
    """
    Load model with LoRA for PPO training
    
    Args:
        model_name (str): HuggingFace model name/path
        adapter_path (str): Path to save/load LoRA adapters
    """
    global model, tokenizer, device
    
    try:
        print(f"Loading {model_name} model for PPO...")
        
        # Set device (more flexible device selection)
        if torch.cuda.is_available():
            # Try to find the GPU with the most available memory
            if torch.cuda.device_count() > 1:
                max_free_memory = 0
                best_device_idx = 0
                for i in range(torch.cuda.device_count()):
                    # This is a heuristic approach since PyTorch doesn't directly expose free memory info
                    # We allocate a small tensor and see if it succeeds
                    try:
                        with torch.cuda.device(i):
                            torch.zeros(1, device=f"cuda:{i}")
                            # If we get here, the device is usable
                            free_memory = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
                            if free_memory > max_free_memory:
                                max_free_memory = free_memory
                                best_device_idx = i
                    except Exception:
                        # Skip this device if there's an error
                        continue
                device = f"cuda:{best_device_idx}"
            else:
                device = "cuda:0"
            print(f"Using CUDA device: {device}")
        else:
            device = "cpu"
            print("No CUDA device found, using CPU")
        
        # Load tokenizer with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"Loading tokenizer (attempt {attempt+1}/{max_retries})...")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                print("Tokenizer loaded successfully")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Error loading tokenizer: {e}, retrying...")
                    await asyncio.sleep(1)
                else:
                    print(f"Failed to load tokenizer after {max_retries} attempts: {e}")
                    return None, None, None
        
        # Load base model with retry logic
        for attempt in range(max_retries):
            try:
                print(f"Loading base model (attempt {attempt+1}/{max_retries})...")
                base_model = AutoModelForCausalLM.from_pretrained(model_name)
                print("Base model loaded successfully")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Error loading base model: {e}, retrying...")
                    await asyncio.sleep(1)
                else:
                    print(f"Failed to load base model after {max_retries} attempts: {e}")
                    return None, None, None
        
        # Use simple fixed target modules to ensure compatibility
        # This avoids issues with unsupported module types
        target_modules = ["q_proj", "k_proj"]
        print(f"Using fixed target modules: {target_modules}")
        
        # Define LoRA configuration with simple target modules
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,  # Rank of the update matrices
            lora_alpha=32,  # Alpha scaling factor
            lora_dropout=0.1,  # Dropout probability for LoRA layers
            target_modules=target_modules,  # Use simple fixed modules for compatibility
        )
        
        # Check if adapters exist and load them if they do
        adapter_exists = os.path.exists(adapter_path) and os.path.exists(os.path.join(adapter_path, "adapter_config.json"))
        
        # Apply PEFT model with error handling
        try:
            if adapter_exists:
                try:
                    print(f"Loading existing PEFT adapters from {adapter_path}...")
                    model = PeftModel.from_pretrained(base_model, adapter_path)
                    model.to(device)
                    print("Existing PEFT adapters loaded successfully!")
                except Exception as adapter_e:
                    import traceback
                    print(f"Error loading existing adapters: {adapter_e}")
                    print(traceback.format_exc())
                    print("Creating new PEFT model instead...")
                    model = get_peft_model(base_model, peft_config)
                    model.to(device)
            else:
                # Create new PEFT model
                print("No existing adapters found. Creating new PEFT model...")
                model = get_peft_model(base_model, peft_config)
                model.to(device)
        except Exception as peft_e:
            import traceback
            print(f"Critical error applying PEFT to model: {peft_e}")
            print(traceback.format_exc())
            print("Will continue with base model without PEFT adapters")
            model = base_model.to(device)
        
        # Initialize PPO trainer with robust error handling
        ppo_trainer_initialized = False
        try:
            print("Initializing PPO trainer with minimal configuration...")
            trainer = await asyncio.to_thread(
                initialize_ppo_trainer,
                model,
                tokenizer,
                learning_rate=5e-5,
                enable_kl_penalty=False,  # Disable KL penalty to simplify
                kl_penalty_factor=None
            )
            if trainer is None:
                print("WARNING: PPO trainer initialization returned None")
            else:
                print("PPO trainer successfully initialized!")
                ppo_trainer_initialized = True
        except Exception as e:
            import traceback
            print(f"Detailed error initializing PPO trainer:")
            print(traceback.format_exc())
            print("Will continue without PPO training capabilities")
        
        if ppo_trainer_initialized:
            print(f"Model loaded successfully on {device} with LoRA adapters and PPO trainer!")
        else:
            print(f"Model loaded on {device}, but PPO trainer could not be initialized")
        
        return model, tokenizer, device
    
    except Exception as e:
        import traceback
        print(f"Critical error in load_model_for_ppo: {e}")
        print(traceback.format_exc())
        return None, None, None

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
    global feedback_queue, feedback_stats
    
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
        
        # Update stats 
        feedback_stats["total_feedback"] += 1
        
        if binary_rating is not None:
            if binary_rating > 0:
                feedback_stats["positive_feedback"] += 1
            elif binary_rating == 0:
                feedback_stats["negative_feedback"] += 1
        
        return True
    except Exception as e:
        print(f"Error adding feedback: {e}")
        return False

async def process_single_feedback(record: FeedbackRecord) -> Dict[str, Any]:
    """
    Process a single feedback record with PPO
    
    Args:
        record: The feedback record to process
    
    Returns:
        dict: Training metrics
    """
    global model, tokenizer
    
    try:
        # Compute reward
        reward = compute_reward(
            response=record.response,
            user_feedback=record.user_feedback,
            binary_rating=record.binary_rating
        )
        
        # Update feedback stats
        feedback_stats["total_feedback"] += 1
        if reward > 0.2:
            feedback_stats["positive_feedback"] += 1
        elif reward < -0.2:
            feedback_stats["negative_feedback"] += 1
        else:
            feedback_stats["neutral_feedback"] += 1
        
        try:
            # Run PPO training step
            metrics = await asyncio.to_thread(
                train_step,
                record.conversation,
                record.response,
                reward
            )
            
            if "error" in metrics:
                print(f"PPO training step failed: {metrics['error']}")
                # Still mark as processed even if PPO training fails
                record.processed = True
                return {"reward": reward, "error": metrics["error"], "ppo_training": False}
        except Exception as train_error:
            import traceback
            print(f"PPO training step exception: {train_error}")
            print(traceback.format_exc())
            # Still mark as processed even if PPO training fails
            record.processed = True
            return {"reward": reward, "error": str(train_error), "ppo_training": False}
        
        # Mark as processed
        record.processed = True
        
        return {**metrics, "reward": reward, "ppo_training": True}
    
    except Exception as e:
        import traceback
        print(f"Error processing feedback with PPO: {e}")
        print(traceback.format_exc())
        
        # Still mark as processed to avoid getting stuck in the queue
        if record:
            record.processed = True
            
        return {"error": str(e), "ppo_training": False}

async def process_feedback_queue() -> int:
    """
    Process all pending feedback in the queue
    
    Returns:
        int: Number of feedback items processed
    """
    global feedback_queue, model
    
    processed_count = 0
    
    # Skip processing if model isn't loaded
    if model is None:
        print("Skipping feedback processing - model not initialized")
        return 0
    
    try:
        # Verify current model state before processing
        if not isinstance(model, torch.nn.Module):
            print("Warning: model is not a torch Module - cannot process feedback")
            return 0

        # Process all items in queue with error handling per item
        queue_copy = list(feedback_queue)
        for record in queue_copy:
            if not record.processed:
                try:
                    await process_single_feedback(record)
                    processed_count += 1
                except Exception as e:
                    import traceback
                    print(f"Error processing feedback for a record: {e}")
                    print(traceback.format_exc())
                    # Mark as processed to avoid retrying forever
                    record.processed = True
        
        # Remove processed items
        feedback_queue = [record for record in feedback_queue if not record.processed]
        
        # Periodically save stats
        if processed_count > 0:
            try:
                # Save feedback stats
                save_feedback_stats()
                
                # Also save training stats if available 
                from sidekick.ppo_trainer import stats_history
                if stats_history and len(stats_history) > 0:
                    try:
                        save_stats_history()
                    except Exception as stats_e:
                        print(f"Error saving PPO stats history: {stats_e}")
            except Exception as e:
                print(f"Error saving feedback stats: {e}")
        
        return processed_count
        
    except Exception as e:
        import traceback
        print(f"Critical error in process_feedback_queue: {e}")
        print(traceback.format_exc())
        return 0

def save_feedback_stats(filename: str = "feedback_stats.json") -> bool:
    """
    Save feedback statistics to file
    
    Args:
        filename: The file to save stats to
    
    Returns:
        bool: True if saved successfully
    """
    global feedback_stats
    
    try:
        # Create stats with timestamp
        save_data = {
            **feedback_stats,
            "timestamp": time.time()
        }
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        return True
    except Exception as e:
        print(f"Error saving feedback stats: {e}")
        return False

def load_feedback_stats(filename: str = "feedback_stats.json") -> bool:
    """
    Load feedback statistics from file
    
    Args:
        filename: The file to load stats from
    
    Returns:
        bool: True if loaded successfully
    """
    global feedback_stats
    
    try:
        if not os.path.exists(filename):
            return False
        
        with open(filename, 'r') as f:
            loaded_stats = json.load(f)
        
        # Update stats
        for key, value in loaded_stats.items():
            if key in feedback_stats:
                feedback_stats[key] = value
        
        return True
    except Exception as e:
        print(f"Error loading feedback stats: {e}")
        return False

def get_metrics() -> Dict[str, Any]:
    """
    Get current RL training metrics
    
    Returns:
        dict: Training metrics including PPO stats
    """
    global feedback_stats
    
    # Get PPO stats
    ppo_stats = get_stats()
    
    # Combine with feedback stats
    combined_stats = {
        **feedback_stats,
        "ppo_stats": ppo_stats
    }
    
    return combined_stats

async def initialize_rl_pipeline():
    """
    Initialize the PPO-based RL pipeline
    """
    # Load model with LoRA and initialize PPO trainer
    await load_model_for_ppo()
    
    # Load feedback stats
    load_feedback_stats()
    
    print("PPO-based RL pipeline initialized")