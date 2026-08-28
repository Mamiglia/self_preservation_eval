#!/bin/bash

kill_vllm() {
    # Kill existing vLLM servers
    kill -9 $(pgrep -f "vllm serve") 2>/dev/null
    sleep 10
}

wait_vllm() {
    # Wait for server to be ready (check health endpoint)
    echo "Waiting for vLLM server to be ready..."
    local timeout=$((2 * 60 * 60))  # 2 hours in seconds
    local elapsed=0
    until curl -s http://localhost:8000/health > /dev/null 2>&1; do
        if [ $elapsed -ge $timeout ]; then
            echo "Timeout: Server did not become ready within $timeout seconds."
            kill_vllm
            return 1
        fi
        sleep 30s
        elapsed=$((elapsed + 30))
    done
    echo "Server ready!"
}