"""
Reinforcement Learning module for Sidekick using PEFT/LoRA for online single-step training
"""
import os
import torch
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any, Tuple, Optional

# Global variables
model = None
tokenizer = None
device = None
peft_config = None
optimizer = None
reward_model = None

def load_model_for_rl(model_name="HuggingFaceTB/SmolLM2-135M-Instruct", adapter_path="./rl_model"):
    """
    Load model with LoRA configuration for fine-tuning
    
    Args:
        model_name (str): HuggingFace model name/path
        adapter_path (str): Path to save/load LoRA adapters
        
    Returns:
        tuple: (tokenizer, model, device) or None if loading failed
    """
    global model, tokenizer, device, peft_config, optimizer
    
    # Only load if not already loaded
    if model is not None and tokenizer is not None and peft_config is not None:
        return tokenizer, model, device, peft_config, optimizer
    
    try:
        print(f"Loading {model_name} model for RL fine-tuning...")
        print("This may take a moment depending on your hardware...")
        
        # Set device
        device = "cuda:1" if torch.cuda.is_available() else "cpu"
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Define LoRA configuration
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,  # Rank of the update matrices
            lora_alpha=32,  # Alpha scaling factor
            lora_dropout=0.1,  # Dropout probability for LoRA layers
            target_modules=["q_proj", "v_proj"],  # Target attention matrices
        )
        
        # Check if adapters exist and load them if they do
        adapter_exists = os.path.exists(adapter_path) and os.path.exists(os.path.join(adapter_path, "adapter_config.json"))
        
        if adapter_exists:
            # Try to load existing adapters
            try:
                print(f"Loading existing PEFT adapters from {adapter_path}...")
                model = PeftModel.from_pretrained(base_model, adapter_path)
                model.to(device)
                
                # Put model in training mode
                model.train()
                model.enable_adapters_training()
                
                print("Existing PEFT adapters loaded successfully!")
            except Exception as adapter_e:
                print(f"Error loading existing adapters: {adapter_e}")
                print("Creating new PEFT model...")
                model = get_peft_model(base_model, peft_config).to(device)
        else:
            # Create new PEFT model
            print("No existing adapters found. Creating new PEFT model...")
            model = get_peft_model(base_model, peft_config).to(device)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        
        print(f"Model loaded successfully on {device} with LoRA adapters!")
        return tokenizer, model, device, peft_config, optimizer
    except Exception as e:
        print(f"Error loading model for RL: {e}")
        return None, None, None, None, None

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
    # In a full implementation, you would use a proper reward model here
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

def train_step(
    messages: List[Dict[str, str]], 
    response: str, 
    reward_value: float
) -> Dict[str, float]:
    """
    Perform a single RL training step using the given conversation and reward
    
    Args:
        messages (list): List of message dicts with 'role' and 'content' keys
        response (str): The generated response to train on
        reward_value (float): The reward value for this response (-1 to 1)
        
    Returns:
        dict: Training metrics
    """
    global model, tokenizer, device, optimizer
    
    if model is None or tokenizer is None:
        tokenizer, model, device, peft_config, optimizer = load_model_for_rl()
        if model is None:
            return {"error": "Failed to load model for RL training"}
    
    model.train()
    
    try:
        # Format conversation using chat template (up to but not including the response)
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Format the full sequence including response
        full_sequence = input_text + response
        
        # Encode the sequences
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
        target_ids = tokenizer.encode(full_sequence, return_tensors="pt").to(device)
        
        # Create labels tensor (only train on the assistant's response part)
        labels = torch.full_like(input_ids, -100)  # -100 is the ignore index
        labels_part = target_ids[:, input_ids.shape[1]-1:]
        labels = torch.cat([labels[:, :-1], labels_part], dim=1)
        
        # Forward pass
        outputs = model(input_ids=target_ids, labels=labels)
        loss = outputs.loss
        
        # Scale loss by reward
        scaled_loss = loss * (-reward_value)  # Negative reward means we want to maximize loss
        
        # Backward and optimize
        optimizer.zero_grad()
        scaled_loss.backward()
        optimizer.step()
        
        # Return metrics
        return {
            "loss": loss.item(),
            "scaled_loss": scaled_loss.item(),
            "reward": reward_value
        }
        
    except Exception as e:
        print(f"Error in RL training step: {e}")
        return {"error": str(e)}

def save_model(save_path: str = "./rl_model"):
    """
    Save the LoRA adapters
    
    Args:
        save_path (str): Path to save the model adapters
    """
    global model
    
    if model is not None:
        try:
            # Create directory if it doesn't exist
            os.makedirs(save_path, exist_ok=True)
            
            # Save only the LoRA adapters
            model.save_pretrained(save_path)
            print(f"LoRA adapters saved to {save_path}")
        except Exception as e:
            print(f"Error saving model: {e}")
    else:
        print("No model loaded to save")

def load_adapters(adapters_path: str):
    """
    Load saved LoRA adapters
    
    Args:
        adapters_path (str): Path to the saved adapters
    """
    global model, tokenizer, device
    
    if model is not None:
        try:
            model = PeftModel.from_pretrained(model, adapters_path)
            print(f"LoRA adapters loaded from {adapters_path}")
        except Exception as e:
            print(f"Error loading adapters: {e}")
    else:
        print("Base model must be loaded before adapters")