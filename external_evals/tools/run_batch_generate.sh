#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <path_to_models_directory> [candidate_port] [gpu_memory_utilization] [tensor_parallel_size]"
    echo "Example: $0 /path/to/my/checkpoints 8001 0.85 1"
    exit 1
fi

MODELS_DIR=$(realpath "$1")
CANDIDATE_PORT=${2:-8001}
GPU_MEM_UTIL=${3:-0.85}
TP_SIZE=${4:-1}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TOOLS_DIR="$ROOT_DIR/external_evals/tools"

# Activate the correct conda environment for vLLM
source /root/miniconda3/etc/profile.d/conda.sh
conda activate learnarena

# Function to wait for the vLLM server to be ready
wait_for_server() {
    local port=$1
    local max_retries=120 # 10 minutes max wait (120 * 5s)
    local retry=0
    
    echo "Waiting for candidate vLLM server on port $port to be ready..."
    while [ $retry -lt $max_retries ]; do
        if curl -s "http://localhost:$port/v1/models" > /dev/null; then
            echo "Server on port $port is ready!"
            return 0
        fi
        sleep 5
        retry=$((retry+1))
    done
    
    echo "Timeout waiting for server on port $port!"
    return 1
}

# Iterate through subdirectories (checkpoints) in the provided folder
for MODEL_PATH in "$MODELS_DIR"/*; do
    if [ ! -d "$MODEL_PATH" ]; then
        continue
    fi
    
    MODEL_ID=$(basename "$MODEL_PATH")
    echo "=========================================================="
    echo "[INFO] GENERATION PHASE: Starting for model $MODEL_ID"
    echo "=========================================================="
    
    # 1. Start candidate vLLM server
    echo "[INFO] Spinning up vLLM server for $MODEL_ID on port $CANDIDATE_PORT (TP=$TP_SIZE)..."
    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --served-model-name "$MODEL_ID" \
        --port "$CANDIDATE_PORT" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --tensor-parallel-size "$TP_SIZE" \
        --trust-remote-code \
        > "$TOOLS_DIR/vllm_${MODEL_ID}.log" 2>&1 &
        
    VLLM_PID=$!
    echo "[INFO] vLLM server PID: $VLLM_PID. Logs at: $TOOLS_DIR/vllm_${MODEL_ID}.log"
    
    # 2. Wait for server to be ready
    if ! wait_for_server "$CANDIDATE_PORT"; then
        echo "[ERROR] Failed to start vLLM for $MODEL_ID. Moving to next model."
        kill -9 $VLLM_PID 2>/dev/null || true
        continue
    fi
    
    # 3. Write temporary config.yaml (only candidate is needed for generation)
    TMP_CONFIG="$TOOLS_DIR/config_${MODEL_ID}.yaml"
    cat << YAML > "$TMP_CONFIG"
models:
  - id: "${MODEL_ID}"
    api_name: "${MODEL_ID}"
    api_base: "http://localhost:${CANDIDATE_PORT}/v1"
    api_key: "dummy"

benchmarks:
  livebench: true
  truthfulqa: true
  ifeval: true
  alpacaeval: true
YAML

    # 4. Run the generation script
    echo "[INFO] Running data generation for $MODEL_ID..."
    if ! python "$TOOLS_DIR/run_evals.py" "$TMP_CONFIG" --phase generate; then
        echo "[ERROR] Generation failed for $MODEL_ID."
    else
        echo "[INFO] Successfully generated outputs for $MODEL_ID."
    fi
    
    # 5. Tear down the server
    echo "[INFO] Stopping vLLM server (PID: $VLLM_PID) to release VRAM..."
    kill $VLLM_PID 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true
    echo "[INFO] Server stopped."
    
    # Optional: Clean up tmp config
    rm -f "$TMP_CONFIG"
    
    echo "=========================================================="
    echo "[INFO] Finished generating for model: $MODEL_ID"
    echo "=========================================================="
    sleep 5 # Small buffer to ensure port/VRAM is fully released
done

echo "[SUCCESS] All models in $MODELS_DIR have been generated."
