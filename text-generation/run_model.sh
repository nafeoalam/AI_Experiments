#!/bin/bash

# Script to run QwQ-32B model with different options

# Display help message
show_help() {
    echo "Usage: ./run_model.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help                 Show this help message"
    echo "  -p, --prompt TEXT          Prompt to send to the model (default: 'How many r's are in the word \"strawberry\"?')"
    echo "  -m, --mode MODE            Mode to run the model in"
    echo "                             Modes:"
    echo "                               full - Run full precision model (requires 40GB+ VRAM)"
    echo "                               quantized - Run quantized model (requires 12GB+ VRAM)"
    echo "                               api - Run via DashScope API (requires DASHSCOPE_API_KEY)"
    echo "  -r, --reasoning            Show reasoning process (only works with api mode)"
    echo ""
    echo "Examples:"
    echo "  ./run_model.sh -p \"What is the capital of France?\""
    echo "  ./run_model.sh -m full -p \"Explain quantum computing\""
    echo "  ./run_model.sh -m api -r -p \"Solve this math problem: 3x + 5 = 14\""
    echo ""
}

# Default values
PROMPT="How many r's are in the word \"strawberry\"?"
SHOW_REASONING=false

# Determine default mode based on API key availability
if [ -n "$DASHSCOPE_API_KEY" ]; then
    # If API key is available, use API mode
    MODE="api"
else
    # If no API key, default to quantized mode which requires less resources
    MODE="quantized"
    echo "No DASHSCOPE_API_KEY found. Defaulting to quantized mode."
    echo "For API mode, set your API key with: export DASHSCOPE_API_KEY=your_api_key_here"
fi

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            exit 0
            ;;
        -p|--prompt)
            PROMPT="$2"
            shift 2
            ;;
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -r|--reasoning)
            SHOW_REASONING=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check if DASHSCOPE_API_KEY is set when using API mode
if [[ "$MODE" == "api" && -z "$DASHSCOPE_API_KEY" ]]; then
    echo "Error: API mode selected but DASHSCOPE_API_KEY environment variable is not set."
    echo "Please set it with your API key:"
    echo "  export DASHSCOPE_API_KEY=your_api_key_here"
    echo ""
    echo "Switching to quantized mode instead..."
    MODE="quantized"
fi

# Run the model based on the selected mode
case "$MODE" in
    full)
        echo "Running full precision model (requires 40GB+ VRAM)..."
        python run_qwq_32b.py --prompt "$PROMPT"
        ;;
    quantized)
        echo "Running quantized model (requires 12GB+ VRAM)..."
        python run_qwq_32b_quantized.py --prompt "$PROMPT"
        ;;
    api)
        echo "Running via DashScope API..."
        if $SHOW_REASONING; then
            python run_qwq_32b_api.py --prompt "$PROMPT" --show-reasoning
        else
            python run_qwq_32b_api.py --prompt "$PROMPT"
        fi
        ;;
    *)
        echo "Invalid mode: $MODE"
        show_help
        exit 1
        ;;
esac 