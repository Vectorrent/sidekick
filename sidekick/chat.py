#!/usr/bin/env python3
"""
Interactive chat script for SmolLM2-135M-Instruct model using Transformers
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Global variables to hold the model and tokenizer
model = None
tokenizer = None
device = None

def load_model(model_name="HuggingFaceTB/SmolLM2-135M-Instruct"):
    """
    Load the LLM model and tokenizer
    
    Args:
        model_name (str): HuggingFace model name/path
        
    Returns:
        tuple: (tokenizer, model, device) or None if loading failed
    """
    global model, tokenizer, device
    
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
        
        # For single GPU or CPU
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        
        print(f"Model loaded successfully on {device}!")
        return tokenizer, model, device
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None

def generate_response(messages, max_new_tokens=256, temperature=0.7):
    """
    Generate a response from the model based on the conversation history
    
    Args:
        messages (list): List of message dicts with 'role' and 'content' keys
        max_new_tokens (int): Maximum number of tokens to generate
        temperature (float): Temperature parameter for generation
        
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
        # Format the conversation using chat template
        input_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Encode the formatted conversation
        inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
        
        # Generate a response
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
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