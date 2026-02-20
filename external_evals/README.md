# Local End-to-End Evaluation Framework

A unified, complete environment to run LiveBench, TruthfulQA, IFEval, and AlpacaEval 2 on locally hosted models (e.g., using vLLM). 

## Prerequisites

1. Models must be running locally and accessible via an OpenAI-compatible API (e.g. `vllm serve`).
2. A single judge model (e.g., Qwen2.5-72B-Instruct) should be hosted locally to act as the evaluator for LiveBench and AlpacaEval 2.
3. Install the environment:
   ```bash
   pip install -r external_evals/tools/requirements.txt
   ```

## Configuration

Configure your candidates and the judge in `external_evals/tools/config.yaml`.
Example:
```yaml
models:
  - id: "llama-3-70b-instruct"
    api_name: "meta-llama/Llama-3-70b-chat-hf"
    api_base: "http://localhost:8001/v1"
    api_key: "dummy"

judge:
  api_base: "http://localhost:8000/v1"
  api_key: "dummy"
  model_name: "Qwen/Qwen2.5-72B-Instruct"

benchmarks:
  livebench: true
  truthfulqa: true
  ifeval: true
  alpacaeval: true
```

## Running the Pipeline

You can run the evaluations in two phases, or all at once. The orchestrator handles pulling datasets, generating outputs, and scoring them.

**To run both phases (Generate + Evaluate):**
```bash
python external_evals/tools/run_evals.py external_evals/tools/config.yaml --phase all
```

**To run only the Generation phase:**
```bash
python external_evals/tools/run_evals.py external_evals/tools/config.yaml --phase generate
```

**To run only the Evaluation phase:**
```bash
python external_evals/tools/run_evals.py external_evals/tools/config.yaml --phase evaluate
```

## Automated Batch Runner

If you have a folder full of model checkpoints (e.g. `checkpoints/step-1000`, `checkpoints/step-2000`) and want to evaluate all of them sequentially, you can use the two-phase automated batch scripts. 

This approach is highly recommended because it ensures your candidate models and the large judge model do not compete for GPU memory.

### Phase 1: Generate Answers (Candidate Models)

First, run the generation script pointing to your models directory. The script will automatically loop through each model, host it, generate responses, and tear it down before moving to the next.

```bash
cd external_evals
./tools/run_batch_generate.sh /path/to/my/checkpoints 8001 0.85 1
```
*Arguments:* `<models_dir> [port=8001] [gpu_mem_util=0.85] [tensor_parallel_size=1]*

### Phase 2: Evaluate Answers (Judge Model)

After all generation is complete, host your large judge model independently.

```bash
# In a separate terminal, start your judge model:
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-72B-Instruct --port 8000 --tensor-parallel-size 4
```

Then, run the evaluation batch script. It will loop through your generated outputs and send them to the judge for scoring. No candidate models will be hosted in this phase!

```bash
cd external_evals
./tools/run_batch_evaluate.sh /path/to/my/checkpoints 8000
```
*Arguments:* `<models_dir> [judge_port=8000]*

## Results

Results are automatically structured into the `results/` directory by model ID and benchmark:

```text
results/
  <model_id>/
    alpacaeval2/
    ifeval/
    livebench/
    truthfulqa/
```
