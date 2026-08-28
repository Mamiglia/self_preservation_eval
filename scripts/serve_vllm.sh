#!/bin/bash

HF_HOME="/media/pinas/huggingface_cache"
MODEL_NAME="Qwen/Qwen3-8B"
PORT=${PORT:-8000}
EXTRA_ARGS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --moe)
            ENABLE_MOE=true
            shift
            ;;
        *)
            # Assume it's the model name if it doesn't start with --
            if [[ ! $1 == --* ]]; then
                MODEL_NAME="$1"
            else 
                EXTRA_ARGS+=("$1")
                if [[ $# -gt 1 && ! $2 == --* ]]; then
                    EXTRA_ARGS+=("$2")
                    shift
                fi
            fi
            shift
            ;;
    esac
done

if [[ "$MODEL_NAME" == *"mistral"* ]]; then
    EXTRA_ARGS+=(--tokenizer_mode mistral --config_format mistral --load_format mistral)
    if [[ "$MODEL_NAME" == *"Reasoning"* ]]; then
        EXTRA_ARGS+=(--reasoning-parser mistral)
    fi
fi

if [[ $MODEL_NAME == *"qwen"* ]]; then
    if [[ "$MODEL_NAME" == *"Thinking"* ]]; then
        EXTRA_ARGS+=(--reasoning-parser qwen)
    fi
fi

if [[ "$MODEL_NAME" == *"openai"* ]]; then
    EXTRA_ARGS+=(--async-scheduling --no-enable-prefix-caching)
fi

VLLM_USE_DEEP_GEMM=0
if [[ "$MODEL_NAME" == *"FP8"* ]]; then
    VLLM_USE_DEEP_GEMM=1
fi


podman run --security-opt=label=disable \
  --device nvidia.com/gpu=all -p $PORT:$PORT \
  -v $HF_HOME:/home/user/.cache/huggingface \
  -v $(pwd)/assets:/assets \
  -e HF_HOME="/home/user/.cache/huggingface" \
  -e VLLM_USE_DEEP_GEMM=$VLLM_USE_DEEP_GEMM \
  nvcr.io/nvidia/vllm:25.11-py3 \
  vllm serve "$MODEL_NAME" --port $PORT --enable-chunked-prefill --enable-log-requests "${EXTRA_ARGS[@]}"
