from openai import OpenAI
import os
import argparse

def run_qwq_model_api(prompt):
    """
    Run the QwQ-32B model via the DashScope API.
    This method doesn't require any local GPU resources.
    
    Args:
        prompt (str): The user prompt to send to the model
        
    Returns:
        str: The model's response
    """
    # Check if the API key is set
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("DASHSCOPE_API_KEY environment variable is not set. Please set it with your API key.")
    
    # Initialize OpenAI client
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    
    print("Sending request to DashScope API...")
    
    # Create a chat completion
    completion = client.chat.completions.create(
        model="qwq-32b",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    # Extract the response
    response = completion.choices[0].message.content
    
    return response

def run_qwq_model_api_with_reasoning(prompt):
    """
    Run the QwQ-32B model via the DashScope API with reasoning content.
    This method shows both the reasoning process and the final answer.
    
    Args:
        prompt (str): The user prompt to send to the model
        
    Returns:
        tuple: (reasoning_content, content) - The model's reasoning and final response
    """
    # Check if the API key is set
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("DASHSCOPE_API_KEY environment variable is not set. Please set it with your API key.")
    
    # Initialize OpenAI client
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    
    print("Sending request to DashScope API with reasoning...")
    
    reasoning_content = ""
    content = ""
    is_answering = False
    
    # Create a streaming chat completion to get reasoning content
    completion = client.chat.completions.create(
        model="qwq-32b",
        messages=[
            {"role": "user", "content": prompt}
        ],
        stream=True
    )
    
    print("\n" + "=" * 20 + " Reasoning Content " + "=" * 20 + "\n")
    
    # Process the streaming response
    for chunk in completion:
        # If chunk.choices is empty, print usage
        if not chunk.choices:
            continue
        else:
            delta = chunk.choices[0].delta
            # Print reasoning content
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                print(delta.reasoning_content, end='', flush=True)
                reasoning_content += delta.reasoning_content
            else:
                if delta.content != "" and is_answering is False:
                    print("\n" + "=" * 20 + " Final Answer " + "=" * 20 + "\n")
                    is_answering = True
                # Print content
                print(delta.content, end='', flush=True)
                content += delta.content
    
    return reasoning_content, content

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the QwQ-32B model via DashScope API")
    parser.add_argument("--prompt", type=str, default="How many r's are in the word 'strawberry'?",
                        help="The prompt to send to the model")
    parser.add_argument("--show-reasoning", action="store_true",
                        help="Show the model's reasoning process")
    
    args = parser.parse_args()
    
    if args.show_reasoning:
        reasoning, response = run_qwq_model_api_with_reasoning(args.prompt)
        # The response is already printed during streaming
    else:
        response = run_qwq_model_api(args.prompt)
        print("\nModel Response:")
        print(response) 