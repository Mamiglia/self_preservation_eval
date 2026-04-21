bash script/kappa.sh meta-llama/Llama-3.1-8B-Instruct > vllm.log 2>&1 &

bash script/eval.sh --model vllm/meta-llama/Llama-3.1-8B-Instruct --seed 42 --max-connections 32
bash script/eval.sh --model vllm/meta-llama/Llama-3.1-8B-Instruct --seed 1789 --max-connections 32
bash script/eval.sh --model vllm/meta-llama/Llama-3.1-8B-Instruct --seed 1455 --max-connections 32
bash script/eval.sh --model vllm/meta-llama/Llama-3.1-8B-Instruct --seed 1300 --max-connections 32
bash script/eval.sh --model vllm/meta-llama/Llama-3.1-8B-Instruct --seed 1948 --max-connections 32

bash script/eval.sh --model vllm/meta-llama/Llama-3.1-8B-Instruct --seed 10 --max-connections 32
bash script/eval.sh --model vllm/meta-llama/Llama-3.1-8B-Instruct --seed 20 --max-connections 32
bash script/eval.sh --model vllm/meta-llama/Llama-3.1-8B-Instruct --seed 30 --max-connections 32
bash script/eval.sh --model vllm/meta-llama/Llama-3.1-8B-Instruct --seed 40 --max-connections 32
bash script/eval.sh --model vllm/meta-llama/Llama-3.1-8B-Instruct --seed 50 --max-connections 32

bash script/eval.sh --model vllm/meta-llama/Llama-3.1-8B-Instruct --seed 42 --extra --max-connections 32
bash script/eval.sh --model vllm/meta-llama/Llama-3.1-8B-Instruct --seed 1789 --extra --max-connections 32
bash script/eval.sh --model vllm/meta-llama/Llama-3.1-8B-Instruct --seed 1455 --extra --max-connections 32
bash script/eval.sh --model vllm/meta-llama/Llama-3.1-8B-Instruct --seed 1300 --extra --max-connections 32
bash script/eval.sh --model vllm/meta-llama/Llama-3.1-8B-Instruct --seed 1948 --extra --max-connections 32

# Kill existing vLLM servers
kill -9 $(pgrep -f "vllm serve") 2>/dev/null
sleep 10

# Start vLLM server
bash script/kappa.sh hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4 &

# Wait for server to be ready (check health endpoint)
echo "Waiting for vLLM server to be ready..."
until curl -s http://localhost:8000/health > /dev/null 2>&1; do
    sleep 10
done
echo "Server ready!"

# Run evaluation
bash script/eval.sh --model vllm/hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4 --seed 42 --max-connections 32
bash script/eval.sh --model vllm/hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4 --seed 1789 --max-connections 32
bash script/eval.sh --model vllm/hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4 --seed 1455 --max-connections 32
bash script/eval.sh --model vllm/hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4 --seed 1300 --max-connections 32
bash script/eval.sh --model vllm/hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4 --seed 1948 --max-connections 32


# Kill existing vLLM servers
kill -9 $(pgrep -f "vllm serve") 2>/dev/null
