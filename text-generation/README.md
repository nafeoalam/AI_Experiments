# Running QwQ-32B Model

This repository contains a simple script to run the [QwQ-32B model](https://huggingface.co/Qwen/QwQ-32B) from Hugging Face.

## Requirements

- Python 3.8+
- CUDA-compatible GPU with at least 40GB VRAM (for full precision)
  - For lower VRAM requirements, consider using the quantized versions:
    - [QwQ-32B-GGUF](https://huggingface.co/Qwen/QwQ-32B-GGUF) (for llama.cpp)
    - [QwQ-32B-AWQ](https://huggingface.co/Qwen/QwQ-32B-AWQ) (4-bit quantized)

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the script with a prompt:

```bash
python run_qwq_32b.py --prompt "Your prompt here"
```

Example:

```bash
python run_qwq_32b.py --prompt "How many r's are in the word 'strawberry'?"
```

## Alternative Methods

### Using Quantized Models

For systems with limited VRAM, you can use quantized versions of the model:

1. GGUF format (for use with llama.cpp):
   - Download from [QwQ-32B-GGUF](https://huggingface.co/Qwen/QwQ-32B-GGUF)
   - Run with llama.cpp

2. AWQ format (4-bit quantized):
   - Use [QwQ-32B-AWQ](https://huggingface.co/Qwen/QwQ-32B-AWQ) with transformers

### Using Hugging Face Inference API

If you don't have sufficient hardware, you can use the Hugging Face Inference API:

```python
import requests

API_URL = "https://api-inference.huggingface.co/models/Qwen/QwQ-32B"
headers = {"Authorization": "Bearer YOUR_API_KEY"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

output = query({
    "inputs": "How many r's are in the word 'strawberry'?",
})
print(output)
```

### Using DashScope API (Alibaba Cloud)

You can also use the DashScope API as shown in the official documentation:

```python
from openai import OpenAI
import os

# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

completion = client.chat.completions.create(
    model="qwq-32b",
    messages=[
        {"role": "user", "content": "How many r's are in the word 'strawberry'?"}
    ]
)

print(completion.choices[0].message.content)
```

## Notes

- The QwQ-32B model requires a significant amount of VRAM (40GB+) to run at full precision.
- For systems with less VRAM, consider using the quantized versions or cloud APIs.
- The model requires transformers version 4.37.0 or higher. 