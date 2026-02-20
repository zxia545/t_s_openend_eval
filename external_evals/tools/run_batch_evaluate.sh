#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <path_to_models_directory> [judge_port]"
    echo "Example: $0 /path/to/my/checkpoints 8000"
    exit 1
fi

MODELS_DIR=$(realpath "$1")
JUDGE_PORT=${2:-8000}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TOOLS_DIR="$ROOT_DIR/external_evals/tools"

# Activate the correct conda environment
source /root/miniconda3/etc/profile.d/conda.sh
conda activate learnarena

# Set judge model name directly since we are using OpenRouter
JUDGE_MODEL_NAME="stepfun/step-3.5-flash:free"
echo "[INFO] Using judge model: $JUDGE_MODEL_NAME via OpenRouter"

# Iterate through subdirectories (checkpoints) in the provided folder
for MODEL_PATH in "$MODELS_DIR"/*; do
    if [ ! -d "$MODEL_PATH" ]; then
        continue
    fi
    
    MODEL_ID=$(basename "$MODEL_PATH")
    echo "=========================================================="
    echo "[INFO] EVALUATION PHASE: Starting for model $MODEL_ID"
    echo "=========================================================="
    
    # Write temporary config.yaml (candidate port/api is not queried during evaluate)
    TMP_CONFIG="$TOOLS_DIR/config_${MODEL_ID}.yaml"
    cat << YAML > "$TMP_CONFIG"
models:
  - id: "${MODEL_ID}"
    api_name: "${MODEL_ID}"
    api_base: "http://localhost:8001/v1"
    api_key: "dummy"

judge:
  api_base: "https://openrouter.ai/api/v1"
  api_key: "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"
  model_name: "${JUDGE_MODEL_NAME}"

benchmarks:
  livebench: true
  truthfulqa: true
  ifeval: true
  alpacaeval: true
YAML

    # Run the evaluation script
    echo "[INFO] Running data evaluation against judge for $MODEL_ID..."
    if ! python "$TOOLS_DIR/run_evals.py" "$TMP_CONFIG" --phase evaluate; then
        echo "[ERROR] Evaluation failed for $MODEL_ID."
    else
        echo "[INFO] Successfully evaluated $MODEL_ID."
    fi
    
    # Optional: Clean up tmp config
    rm -f "$TMP_CONFIG"
    
    echo "=========================================================="
    echo "[INFO] Finished evaluating model: $MODEL_ID"
    echo "=========================================================="
done

echo "[SUCCESS] All models in $MODELS_DIR have been evaluated."
