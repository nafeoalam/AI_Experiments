import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def run_qwq_model_quantized(prompt):
    """
    Run the quantized QwQ-32B-AWQ model with the given prompt.
    This version requires less VRAM than the full precision model.
    
    Args:
        prompt (str): The user prompt to send to the model
        
    Returns:
        str: The model's response
    """
    model_name = "Qwen/QwQ-32B-AWQ"  # Using the AWQ quantized version
    
    print(f"Loading quantized model {model_name}...")
    
    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use half precision
        device_map="auto",          # Automatically determine the best device mapping
        trust_remote_code=True      # Required for some models with custom code
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Format the prompt as a chat message
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # Apply the chat template to format the messages
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize the input
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    print("Generating response...")
    
    # Generate the response
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1024,  # Adjust based on how long you want the response to be
        temperature=0.7,      # Controls randomness (lower = more deterministic)
        top_p=0.9,            # Nucleus sampling parameter
        do_sample=True        # Use sampling instead of greedy decoding
    )
    
    # Extract only the newly generated tokens
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    # Decode the generated tokens to text
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return response

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the quantized QwQ-32B model")
    parser.add_argument("--prompt", type=str, default="How many r's are in the word 'strawberry'?",
                        help="The prompt to send to the model")
    
    args = parser.parse_args()
    
    response = run_qwq_model_quantized(args.prompt)
    print("\nModel Response:")
    print(response) 