#!/usr/bin/env python3
"""
Interactive chat script for SmolLM2-135M-Instruct model using Transformers
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import glob

# Global variables to hold the model and tokenizer
model = None
tokenizer = None
device = None
is_peft_model = False

def load_model(model_name="HuggingFaceTB/SmolLM2-135M-Instruct", adapter_path="./model"):
    """
    Load the LLM model and tokenizer with PEFT adapters if available
    
    Args:
        model_name (str): HuggingFace model name/path
        adapter_path (str): Path to PEFT adapter weights
        
    Returns:
        tuple: (tokenizer, model, device) or None if loading failed
    """
    global model, tokenizer, device, is_peft_model
    
    # Only load if not already loaded
    if model is not None and tokenizer is not None:
        return tokenizer, model, device
    
    try:
        print(f"Loading {model_name} model...")
        print("This may take a moment depending on your hardware...")
        
        # Set device
        device = "cuda:1" if torch.cuda.is_available() else "cpu"
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Check if adapters exist
        adapter_exists = os.path.exists(adapter_path) and os.path.exists(os.path.join(adapter_path, "adapter_config.json"))
        
        if adapter_exists:
            # Load model with PEFT adapters
            try:
                print(f"Loading PEFT adapters from {adapter_path}...")
                model = PeftModel.from_pretrained(base_model, adapter_path)
                is_peft_model = True
                print("PEFT adapters loaded successfully!")
            except Exception as adapter_e:
                print(f"Error loading adapters: {adapter_e}")
                print("Falling back to base model")
                model = base_model
                is_peft_model = False
        else:
            # No adapters found, use base model
            model = base_model
            is_peft_model = False
            
        # Move model to device
        model = model.to(device)
        
        print(f"Model loaded successfully on {device}!")
        if is_peft_model:
            print("Using PEFT-adapted model for inference")
        else:
            print("Using base model for inference (no PEFT adapters found)")
            
        return tokenizer, model, device
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None

def generate_response(messages, **generation_kwargs):
    """
    Generate a response from the model based on the conversation history
    
    Args:
        messages (list): List of message dicts with 'role' and 'content' keys
        **generation_kwargs: Keyword arguments passed to model.generate()
            Common parameters include:
            - max_new_tokens (int): Maximum number of tokens to generate
            - min_new_tokens (int): Minimum number of tokens to generate
            - temperature (float): Temperature for sampling
            - top_k (int): Number of highest probability tokens to keep for sampling
            - top_p (float): Keep tokens with cumulative probability >= top_p
            - repetition_penalty (float): Penalty for repeating tokens
            - no_repeat_ngram_size (int): Size of n-grams to prevent repetition of
            - do_sample (bool): Whether to use sampling or greedy decoding
            - num_beams (int): Number of beams for beam search
            - num_return_sequences (int): Number of sequences to return
            
    Returns:
        str: The generated response
    """
    global model, tokenizer, device
    
    # Ensure model is loaded
    if model is None or tokenizer is None:
        tokenizer, model, device = load_model()
        if model is None:
            return "Error: Failed to load the AI model."
    
    try:
        # Set default generation parameters if not provided
        default_params = {
            'max_new_tokens': 256,
            'temperature': 0.15,
            'do_sample': True,
            'pad_token_id': tokenizer.eos_token_id
        }
        
        # Update defaults with provided parameters
        generate_params = default_params.copy()
        generate_params.update(generation_kwargs)
        
        # Format the conversation using chat template
        input_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Encode the formatted conversation
        inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
        
        # Generate a response with all the specified parameters
        outputs = model.generate(
            inputs,
            **generate_params
        )
        
        # Decode the response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Extract only the new assistant response
        assistant_response = extract_assistant_response(full_response, input_text)
        
        # Clean up response
        clean_response = assistant_response.replace("<|im_end|>", "").strip()
        
        return clean_response
    
    except Exception as e:
        return f"Error generating response: {str(e)}"

def extract_assistant_response(full_response, input_prompt):
    """
    Extract only the new assistant response from the model output.
    This is a simple implementation and may need to be adjusted based on 
    the specific output format of the model.
    """
    # Remove the input prompt from the beginning
    if full_response.startswith(input_prompt):
        new_content = full_response[len(input_prompt):]
    else:
        new_content = full_response
    
    # Clean up any special tokens or formatting
    # This is a simplified version and may need adjustment
    new_content = new_content.replace("<|endoftext|>", "").strip()
    
    return new_content

def main():
    """
    Run the interactive chat application from the command line
    """
    # Load the model
    tokenizer, model, device = load_model()
    if model is None:
        return
    
    # Initialize conversation history
    messages = []
    
    # Optional: Add a system message to set the assistant's behavior
    system_message = input("Enter a system message (or press Enter to skip): ")
    if system_message:
        messages.append({"role": "system", "content": system_message})
    
    print("\nChat with SmolLM2-135M-Instruct (type 'exit', 'quit', or 'q' to end the conversation)")
    print("=" * 50)
    
    # Start the conversation loop
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        # Check for exit commands
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Ending conversation. Goodbye!")
            break
        
        # Add user message to conversation history
        messages.append({"role": "user", "content": user_input})
        
        # Generate response
        print("\nAssistant is thinking...")
        response = generate_response(messages)
        
        print(f"\nAssistant: {response}")
        
        # Add assistant response to conversation history
        messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()