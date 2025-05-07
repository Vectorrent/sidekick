#!/usr/bin/env python3
"""
Interactive chat script for SmolLM2-135M-Instruct model using Transformers
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    # Print welcome message
    print("Loading SmolLM2-135M-Instruct model...")
    print("This may take a moment depending on your hardware...")
    
    # Configure model loading parameters
    model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    
    # Load tokenizer and model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # For multiple GPUs, uncomment the line below instead
        # model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        
        # For single GPU or CPU
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        
        print(f"Model loaded successfully on {device}!")
    except Exception as e:
        print(f"Error loading model: {e}")
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
        
        # Format the conversation using chat template
        input_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Encode the formatted conversation
        inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
        
        # Generate a response
        print("\nAssistant is thinking...")
        outputs = model.generate(
            inputs,
            max_new_tokens=256,  # Adjust based on your needs
            temperature=0.7,     # Lower for more deterministic outputs
            # top_p=0.9,           # Nucleus sampling parameter
            do_sample=True,      # Enable sampling
            pad_token_id=tokenizer.eos_token_id  # Ensure proper padding
        )
        
        # Decode the response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Extract only the new assistant response
        # This may require adjustment based on the model's output format
        assistant_response = extract_assistant_response(full_response, input_text)
        
        clean_response = assistant_response.replace("<|im_end|>", "").strip()

        print(f"\nAssistant: {clean_response}")
        
        # Add assistant response to conversation history
        messages.append({"role": "assistant", "content": assistant_response})

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

if __name__ == "__main__":
    main()