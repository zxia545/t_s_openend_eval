import argparse
import yaml
import os
import sys
import json
import csv
import time
import urllib.request
import concurrent.futures
from openai import OpenAI
from pathlib import Path
from huggingface_hub import hf_hub_download

# Load dataset for AlpacaEval
load_dataset = None
try:
    from datasets import load_dataset
except ImportError:
    pass  # we might need it for alpacaeval

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_REQUEST_TIMEOUT = int(os.environ.get("EVAL_API_TIMEOUT", "180"))
DEFAULT_API_MAX_RETRIES = int(os.environ.get("EVAL_API_MAX_RETRIES", "3"))


def load_alpacaeval_eval_split():
    try:
        if load_dataset is not None:
            return list(
                load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", split="eval")
            )
    except Exception as e:
        print(f"[WARN] datasets.load_dataset alpaca_eval 失败: {e}")

    try:
        json_path = hf_hub_download(
            repo_id="tatsu-lab/alpaca_eval",
            repo_type="dataset",
            filename="alpaca_eval.json",
        )
        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load alpaca_eval dataset from JSON fallback: {e}"
        ) from e

    if isinstance(payload, dict):
        if "eval" in payload and isinstance(payload["eval"], list):
            return payload["eval"]
        if "data" in payload and isinstance(payload["data"], list):
            return payload["data"]
        raise RuntimeError(
            "Unsupported alpaca_eval.json format: expected list or dict with 'eval'/'data'"
        )

    if isinstance(payload, list):
        return payload

    raise RuntimeError("Unsupported alpaca_eval.json format: expected JSON array")


def call_openai_api(
    client,
    model_name,
    prompt,
    max_tokens=1024,
    temperature=0.0,
    timeout=DEFAULT_REQUEST_TIMEOUT,
    max_retries=DEFAULT_API_MAX_RETRIES,
) -> str:
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            if attempt == max_retries:
                print(f"API Error (attempt {attempt}/{max_retries}): {e}")
                return ""
            print(f"API Error (attempt {attempt}/{max_retries}), retrying: {e}")
            time.sleep(min(2 ** (attempt - 1), 8))
    return ""


def process_ifeval_generate(model, client):
    print(f"[{model['id']}] Generating IFEval...")
    out_dir = ROOT_DIR / "results" / model["id"] / "ifeval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "candidate_outputs.jsonl"

    if out_file.exists():
        print(f"[{model['id']}] IFEval already generated. Skipping.")
        return

    in_file = (
        ROOT_DIR / "IFEval" / "instruction_following_eval" / "data" / "input_data.jsonl"
    )
    with open(in_file, "r", encoding="utf-8") as f:
        prompts = [json.loads(line) for line in f]

    def _process(item):
        prompt = item["prompt"]
        response = call_openai_api(client, model["api_name"], prompt)
        return {"prompt": prompt, "response": response}

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for res in executor.map(_process, prompts):
            results.append(res)

    with open(out_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")


def process_truthfulqa_generate(model, client):
    print(f"[{model['id']}] Generating TruthfulQA...")
    in_file = ROOT_DIR / "TruthfulQA" / "TruthfulQA.csv"
    out_dir = ROOT_DIR / "results" / model["id"] / "truthfulqa"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "TruthfulQA_generated.csv"

    if out_file.exists():
        print(f"[{model['id']}] TruthfulQA already generated. Skipping.")
        return

    with open(in_file, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    def _process(row):
        prompt = row["Question"]
        # The prompt format typically recommended for TruthfulQA Generation:
        # "Q: {question}\nA:"
        full_prompt = f"Q: {prompt}\nA:"
        # For chat models, they might answer directly.
        response = call_openai_api(
            client, model["api_name"], full_prompt, max_tokens=50
        )
        row[model["id"]] = (response or "").strip()
        return row

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for res in executor.map(_process, reader):
            results.append(res)

    if results:
        fieldnames = list(results[0].keys())
        with open(out_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)


def process_alpacaeval_generate(model, client):
    print(f"[{model['id']}] Generating AlpacaEval2...")
    out_dir = ROOT_DIR / "results" / model["id"] / "alpacaeval2"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "model_outputs.json"

    if out_file.exists():
        print(f"[{model['id']}] AlpacaEval already generated. Skipping.")
        return

    try:
        d = load_alpacaeval_eval_split()
    except Exception as e:
        print(f"Failed to load alpaca_eval dataset: {e}")
        return

    def _process(item):
        prompt = item["instruction"]
        response = call_openai_api(client, model["api_name"], prompt)
        return {
            "instruction": prompt,
            "output": response,
            "generator": model["id"],
            "dataset": item.get("dataset", ""),
        }

    results = []
    total = len(d)
    print(f"[{model['id']}] AlpacaEval2 tasks: {total}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_process, item) for item in d]
        for idx, fut in enumerate(concurrent.futures.as_completed(futures), start=1):
            results.append(fut.result())
            if idx % 20 == 0 or idx == total:
                print(f"[{model['id']}] AlpacaEval2 progress: {idx}/{total}")

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def run_generate(config):
    for model in config["models"]:
        print(f"==== GENERATE PHASE: {model['id']} ====")
        client = OpenAI(base_url=model["api_base"], api_key=model["api_key"])

        benchmarks = config.get("benchmarks", {})
        if benchmarks.get("ifeval"):
            process_ifeval_generate(model, client)
        if benchmarks.get("truthfulqa"):
            process_truthfulqa_generate(model, client)
        if benchmarks.get("alpacaeval"):
            process_alpacaeval_generate(model, client)

        if benchmarks.get("livebench"):
            print(f"[{model['id']}] Generating LiveBench...")
            cmd = f"""
            cd {ROOT_DIR}/livebench/livebench && \\
            LIVEBENCH_API_KEY="{model["api_key"]}" python gen_api_answer.py \\
              --bench-name live_bench \\
              --model "{model["api_name"]}" \\
              --model-display-name "{model["id"]}" \\
              --api-base "{model["api_base"]}" \\
              --question-source huggingface \\
              --livebench-release-option "2024-11-25" \\
              --parallel 8 \\
              --resume --retry-failures
            """
            os.system(cmd)


def run_evaluate(config, skip_existing=False):
    judge_conf = config.get("judge", {})
    judge_api_base = judge_conf.get("api_base")
    judge_model = judge_conf.get("model_name")

    for model in config["models"]:
        print(f"==== EVALUATE PHASE: {model['id']} ====")
        benchmarks = config.get("benchmarks", {})

        if benchmarks.get("ifeval"):
            print(f"[{model['id']}] Evaluating IFEval...")
            in_file = (
                ROOT_DIR
                / "results"
                / model["id"]
                / "ifeval"
                / "candidate_outputs.jsonl"
            )
            out_dir = ROOT_DIR / "results" / model["id"] / "ifeval"
            out_file = out_dir / "eval_results_strict.jsonl"
            if skip_existing and out_file.exists():
                print(f"[{model['id']}] IFEval eval already exists. Skipping.")
            elif in_file.exists():
                cmd = f"""
                cd {ROOT_DIR}/IFEval && \\
                CUDA_VISIBLE_DEVICES="" PYTHONPATH={ROOT_DIR}/IFEval python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)" && \\
                CUDA_VISIBLE_DEVICES="" PYTHONPATH={ROOT_DIR}/IFEval python -m instruction_following_eval.evaluation_main \\
                  --input_data=instruction_following_eval/data/input_data.jsonl \\
                  --input_response_data={in_file} \\
                  --output_dir={out_dir}
                """
                os.system(cmd)

        if benchmarks.get("truthfulqa"):
            print(f"[{model['id']}] Evaluating TruthfulQA...")
            in_file = (
                ROOT_DIR
                / "results"
                / model["id"]
                / "truthfulqa"
                / "TruthfulQA_generated.csv"
            )
            out_path = ROOT_DIR / "results" / model["id"] / "truthfulqa" / "results.csv"
            if skip_existing and out_path.exists():
                print(f"[{model['id']}] TruthfulQA eval already exists. Skipping.")
            elif in_file.exists():
                cmd = f"""
                cd {ROOT_DIR}/TruthfulQA && \\
                CUDA_VISIBLE_DEVICES="" python -m truthfulqa.evaluate \\
                  --models "{model["id"]}" \\
                  --metrics mc bleu rouge bleurt \\
                  --input_path {in_file} \\
                  --output_path {out_path}
                """
                os.system(cmd)

        if benchmarks.get("alpacaeval"):
            print(f"[{model['id']}] Evaluating AlpacaEval2...")
            in_file = (
                ROOT_DIR
                / "results"
                / model["id"]
                / "alpacaeval2"
                / "model_outputs.json"
            )
            alpacaeval_out = ROOT_DIR / "alpaca_eval" / "results" / model["id"] / "leaderboard.csv"
            if skip_existing and alpacaeval_out.exists():
                print(f"[{model['id']}] AlpacaEval2 eval already exists. Skipping.")
            elif in_file.exists():
                # Verify judge API is reachable before attempting evaluation
                try:
                    urllib.request.urlopen(f"{judge_api_base}/models", timeout=5)
                except Exception as e:
                    print(f"[ERROR] [{model['id']}] Judge API not reachable at {judge_api_base}: {e}")
                    print(f"[ERROR] [{model['id']}] Skipping AlpacaEval2 — start the judge server first.")
                    continue_alpaca = False
                else:
                    continue_alpaca = True

                if continue_alpaca:
                    judge_config_dir = (
                        ROOT_DIR / "results" / model["id"] / "alpacaeval2" / "judge_config"
                    )
                    judge_config_dir.mkdir(parents=True, exist_ok=True)
                    with open(judge_config_dir / "configs.yaml", "w") as f:
                        yaml.dump(
                            {
                                "custom_vllm_judge": {
                                    "prompt_template": str(ROOT_DIR / "alpaca_eval" / "src" / "alpaca_eval" / "evaluators_configs" / "alpaca_eval_gpt4_turbo_fn" / "alpaca_eval_fn.txt"),
                                "fn_completions": "openai_completions",
                                "completions_kwargs": {
                                    "model_name": judge_model,
                                      "openai_api_base": judge_api_base,
                                      "openai_api_keys": [judge_conf.get("api_key", "dummy")],
                                    "temperature": 0.0,
                                    "tool_choice": {
                                        "type": "function",
                                        "function": {
                                            "name": "make_partial_leaderboard"
                                        },
                                    },
                                    "tools": [
                                        {
                                            "type": "function",
                                            "function": {
                                                "name": "make_partial_leaderboard",
                                                "description": "Make a leaderboard of models given a list of the models ordered by the preference of their outputs.",
                                                "strict": True,
                                                "parameters": {
                                                    "type": "object",
                                                    "properties": {
                                                        "ordered_models": {
                                                            "type": "array",
                                                            "description": "A list of models ordered by the preference of their outputs. The first model in the list has the best output.",
                                                            "items": {
                                                                "type": "object",
                                                                "properties": {
                                                                    "model": {
                                                                        "type": "string",
                                                                        "description": "The name of the model",
                                                                    },
                                                                    "rank": {
                                                                        "type": "number",
                                                                        "description": "Order of preference of the model, 1 has the best output",
                                                                    },
                                                                },
                                                                "additionalProperties": False,
                                                                "required": [
                                                                    "model",
                                                                    "rank",
                                                                ],
                                                            },
                                                        }
                                                    },
                                                    "additionalProperties": False,
                                                    "required": ["ordered_models"],
                                                },
                                            },
                                        }
                                    ],
                                },
                                "fn_completion_parser": "pipeline_meta_parser",
                                "completion_parser_kwargs": {
                                    "parsers_to_kwargs": {
                                        "json_parser": {
                                            "annotation_key": "ordered_models"
                                        },
                                        "ranking_parser": {"model_1_name": "m"},
                                    }
                                },
                                "batch_size": 1,
                            }
                        },
                        f,
                          )

                      cmd = f"""
                      cd {ROOT_DIR}/alpaca_eval && \\
                      PYTHONPATH={ROOT_DIR}/alpaca_eval/src:$PYTHONPATH python -m alpaca_eval.main evaluate \\
                        --model_outputs {in_file} \\
                        --annotators_config {judge_config_dir} \\
                        --name "{model["id"]}"
                      """
                      os.system(cmd)
        if benchmarks.get("livebench"):
            print(f"[{model['id']}] Evaluating LiveBench...")
            output_dir = ROOT_DIR / "results" / model["id"] / "livebench"
            output_dir.mkdir(parents=True, exist_ok=True)
            livebench_done = list(output_dir.glob("**/*.jsonl")) if output_dir.exists() else []
            if skip_existing and livebench_done:
                print(f"[{model['id']}] LiveBench eval already exists ({len(livebench_done)} judgment files). Skipping.")
            else:
             cmd = f"""
            cd {ROOT_DIR}/livebench/livebench && \\
            CUDA_VISIBLE_DEVICES="" python gen_ground_truth_judgment.py \\
              --bench-name live_bench \\
              --model "{model["api_name"]}" \\
              --model-display-name "{model["id"]}" \\
              --question-source huggingface \\
              --livebench-release-option "2024-11-25" \\
              --output-dir "{output_dir}" \\
              --parallel 8 \\
              --resume \\
              --ignore-missing-answers
             """
             os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Local Evaluation Runner")
    parser.add_argument("config", help="Path to config.yaml")
    parser.add_argument(
        "--phase",
        choices=["generate", "evaluate", "all"],
        default="all",
        help="Phase to run",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip benchmarks whose evaluation output already exists",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.phase in ["generate", "all"]:
        run_generate(config)

    if args.phase in ["evaluate", "all"]:
        run_evaluate(config, skip_existing=args.skip_existing)
