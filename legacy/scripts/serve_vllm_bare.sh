#!/usr/bin/env bash
set -e

MODEL_NAME="Qwen/Qwen3-30B-A3B-Instruct-2507"
ENABLE_MOE=false

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
            fi
            shift
            ;;
    esac
done

# Set MoE environment variable if enabled
if [ "$ENABLE_MOE" = true ]; then
    export VLLM_USE_FLASHINFER_MOE_FP16=1
    echo "MoE mode enabled"
fi

# Function to cleanup vLLM on exit
cleanup() {
    if [ -n "$VLLM_PID" ] && kill -0 $VLLM_PID 2>/dev/null; then
        echo "Stopping vLLM (PID $VLLM_PID)..."
        kill $VLLM_PID
        wait $VLLM_PID 2>/dev/null || true
        echo "vLLM stopped."
    fi
}

# Set trap to cleanup on exit (success, error, or interrupt)
trap cleanup EXIT INT TERM

# Build vLLM command
VLLM_CMD="vllm serve $MODEL_NAME --tensor-parallel-size 2 --port 8000 --max-model-len 8192"

# Add MoE flag if enabled
if [ "$ENABLE_MOE" = true ]; then
    VLLM_CMD="$VLLM_CMD --enable-expert-parallel"
fi

# 1. Start vLLM in background
echo "Starting vLLM with command: $VLLM_CMD"
$VLLM_CMD > vllm.log 2>&1 &

VLLM_PID=$!
echo "Started vLLM with PID $VLLM_PID"

# 2. Wait until vLLM is ready (with timeout)
echo -n "Waiting for vLLM to be ready..."
MAX_WAIT=2400  # 40 minutes timeout
WAITED=0
until curl -s http://localhost:8000/health > /dev/null; do
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo " Timeout!"
        echo "vLLM failed to start within $MAX_WAIT seconds. Check vllm.log for details."
        exit 1
    fi
    #echo -n "."
    tail -n 20 vllm.log
    sleep 2
    WAITED=$((WAITED + 2))
done
echo " Ready!"

# 3. Run your Python script
bash scripts/eval.sh --model vllm/"$MODEL_NAME"
# 4. Cleanup happens automatically via trap
echo "Evaluation complete."