import argparse
import yaml
import os
import sys
import json
import csv
import concurrent.futures
from openai import OpenAI
from pathlib import Path

# Load dataset for AlpacaEval
try:
    from datasets import load_dataset
except ImportError:
    pass  # we might need it for alpacaeval

ROOT_DIR = Path(__file__).resolve().parent.parent


def call_openai_api(client, model_name, prompt, max_tokens=1024, temperature=0.0):
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API Error: {e}")
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
        row[model["id"]] = response.strip()
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
        from datasets import load_dataset

        d = load_dataset(
            "tatsu-lab/alpaca_eval", "alpaca_eval", split="eval", trust_remote_code=True
        )
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
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for res in executor.map(_process, d):
            results.append(res)

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


def run_evaluate(config):
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
            if in_file.exists():
                cmd = f"""
                cd {ROOT_DIR}/IFEval && \\
                PYTHONPATH={ROOT_DIR}/IFEval python -m instruction_following_eval.evaluation_main \\
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
            if in_file.exists():
                cmd = f"""
                cd {ROOT_DIR}/TruthfulQA && \\
                python -m truthfulqa.evaluate \\
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
            if in_file.exists():
                config_yaml = (
                    ROOT_DIR
                    / "results"
                    / model["id"]
                    / "alpacaeval2"
                    / "judge_config.yaml"
                )
                config_yaml.parent.mkdir(parents=True, exist_ok=True)
                with open(config_yaml, "w") as f:
                    yaml.dump(
                        {
                            judge_model: [
                                {
                                    "base_url": judge_api_base,
                                    "api_key": judge_conf.get("api_key", "dummy"),
                                }
                            ]
                        },
                        f,
                    )

                os.environ["OPENAI_CLIENT_CONFIG_PATH"] = str(config_yaml)

                judge_config_dir = (
                    ROOT_DIR / "results" / model["id"] / "alpacaeval2" / "judge_config"
                )
                judge_config_dir.mkdir(parents=True, exist_ok=True)
                with open(judge_config_dir / "configs.yaml", "w") as f:
                    yaml.dump(
                        {
                            "custom_vllm_judge": {
                                "prompt_template": "alpaca_eval_gpt4_turbo_fn/alpaca_eval_fn.txt",
                                "fn_completions": "openai_completions",
                                "completions_kwargs": {
                                    "model_name": judge_model,
                                    "max_tokens": 100,
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
                alpaca_eval evaluate \\
                  --model_outputs {in_file} \\
                  --annotators_config {judge_config_dir} \\
                  --name "{model["id"]}"
                """
                os.system(cmd)

        if benchmarks.get("livebench"):
            print(f"[{model['id']}] Evaluating LiveBench...")
            output_dir = ROOT_DIR / "results" / model["id"] / "livebench"
            output_dir.mkdir(parents=True, exist_ok=True)
            cmd = f"""
            cd {ROOT_DIR}/livebench/livebench && \\
            python gen_ground_truth_judgment.py \\
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
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.phase in ["generate", "all"]:
        run_generate(config)

    if args.phase in ["evaluate", "all"]:
        run_evaluate(config)
