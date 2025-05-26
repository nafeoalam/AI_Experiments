# AI/ML Proof of Concepts Repository

This repository contains various AI and Machine Learning experiments and proof-of-concepts (POCs) demonstrating different techniques and applications.

## 🧪 Experiments Overview

### 1. RFP Finder (`rfp-finder/`)
A web application that analyzes documents (PDF, Word, Excel, Text) to extract relevant information based on specified keywords using Azure OpenAI.

**Features:**
- Document upload and processing (multiple formats)
- AI-powered keyword extraction using GPT-4o
- Chunk-based processing for large documents
- Flask web interface

**Tech Stack:** Flask, Azure OpenAI, PyPDF2, python-docx, openpyxl

### 2. Text-to-Image Generation (`text-to-image/`)
Experiments with Stable Diffusion XL for generating images from text prompts.

**Features:**
- Uses Stable Diffusion XL Base 1.0 model
- Optimized for Apple Silicon (MPS)
- Simple prompt-to-image generation

**Tech Stack:** PyTorch, Diffusers, Stable Diffusion XL

### 3. Web Search Agent (`web_search_agent/`)
An intelligent agent that searches the web, analyzes results, and provides comprehensive answers using Azure OpenAI and Google Custom Search API.

**Features:**
- Google Custom Search API integration
- Web scraping capabilities (NYC City Record)
- AI-powered query generation and result analysis
- RFP/RFX opportunity detection

**Tech Stack:** Azure OpenAI, Google Custom Search API, BeautifulSoup, Requests

### 4. RAG Chat with Documents (`rag-intro-chat-with-docs/`)
A Retrieval-Augmented Generation (RAG) system that allows users to chat with documents using vector embeddings and similarity search.

**Features:**
- Document ingestion and vectorization
- ChromaDB for persistent vector storage
- Interactive chat interface
- Semantic search over documents

**Tech Stack:** ChromaDB, Vector embeddings, Flask/Streamlit

### 5. Text Generation (`text-generation/`)
Scripts for running and experimenting with large language models, specifically the QwQ-32B model from Qwen.

**Features:**
- Local model execution
- Quantized model support for lower VRAM
- API integration options (HuggingFace, DashScope)
- Shell scripts for easy model deployment

**Tech Stack:** Transformers, PyTorch, HuggingFace, Qwen/QwQ-32B

## 🚀 Getting Started

Each experiment directory contains its own README with specific setup instructions. General requirements:

### Prerequisites
- Python 3.8+ (3.12+ recommended)
- Git
- Virtual environment tool (venv, conda, etc.)

### Common Dependencies
Most experiments use:
- OpenAI/Azure OpenAI APIs
- PyTorch (for ML models)
- Flask (for web interfaces)
- Various document processing libraries

### Environment Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd AI_ML_POCS
   ```

2. Navigate to the specific experiment directory
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Configure API keys and environment variables as specified in each experiment's README

## 📁 Repository Structure

# Introduction 
TODO: Give a short introduction of your project. Let this section explain the objectives or the motivation behind this project. 

# Getting Started
TODO: Guide users through getting your code up and running on their own system. In this section you can talk about:
1.	Installation process
2.	Software dependencies
3.	Latest releases
4.	API references

# Build and Test
TODO: Describe and show how to build your code and run the tests. 

# Contribute
TODO: Explain how other users and developers can contribute to make your code better. 

If you want to learn more about creating good readme files then refer the following [guidelines](https://docs.microsoft.com/en-us/azure/devops/repos/git/create-a-readme?view=azure-devops). You can also seek inspiration from the below readme files:
- [ASP.NET Core](https://github.com/aspnet/Home)
- [Visual Studio Code](https://github.com/Microsoft/vscode)
- [Chakra Core](https://github.com/Microsoft/ChakraCore)