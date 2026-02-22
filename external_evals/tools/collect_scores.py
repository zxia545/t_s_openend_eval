#!/usr/bin/env python3
"""
Collect and aggregate evaluation scores for all models.

Usage:
    python collect_scores.py [--output-dir results/summaries]

Output:
    - results/summaries/<model_id>_scores.jsonl
    - Each line contains: {"benchmark": "...", "metric": "...", "value": ...}
"""

import json
import csv
import argparse
import math
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent


def collect_ifeval_scores(model_id: str) -> list[dict]:
    """Collect IFEval scores from eval_results files."""
    scores = []
    results_dir = ROOT_DIR / "results" / model_id / "ifeval"

    for result_type in ["strict", "loose"]:
        result_file = results_dir / f"eval_results_{result_type}.jsonl"
        if not result_file.exists():
            continue

        with open(result_file) as f:
            results = [json.loads(line) for line in f]

        total = len(results)
        correct = sum(1 for r in results if r.get("follow_all_instructions", False))
        accuracy = correct / total if total > 0 else 0

        scores.append(
            {
                "benchmark": "ifeval",
                "metric": f"accuracy_{result_type}",
                "value": round(accuracy, 4),
                "count": total,
                "correct": correct,
            }
        )

    return scores


def collect_truthfulqa_scores(model_id: str) -> list[dict]:
    """Collect TruthfulQA scores from results.csv."""
    scores = []
    results_file = ROOT_DIR / "results" / model_id / "truthfulqa" / "results.csv"

    if not results_file.exists():
        return scores

    with open(results_file) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return scores

    metrics = ["bleu acc", "rouge1 acc", "rouge2 acc", "rougeL acc"]

    for metric in metrics:
        col_name = f"{model_id} {metric}"
        if col_name not in rows[0]:
            continue

        values = [float(row[col_name]) for row in rows if row.get(col_name)]
        if values:
            avg_value = sum(values) / len(values)
            scores.append(
                {
                    "benchmark": "truthfulqa",
                    "metric": metric.replace(" ", "_"),
                    "value": round(avg_value, 4),
                    "count": len(values),
                }
            )

    if scores:
        avg_truthful = sum(s["value"] for s in scores) / len(scores)
        scores.append(
            {
                "benchmark": "truthfulqa",
                "metric": "overall_truthful",
                "value": round(avg_truthful, 4),
                "count": len(rows),
            }
        )

    return scores


def collect_alpacaeval_scores(model_id: str) -> list[dict]:
    scores = []

    # Check both possible locations
    leaderboard_locations = [
        ROOT_DIR / "results" / model_id / "alpacaeval2" / "leaderboard.csv",
        ROOT_DIR / "alpaca_eval" / "results" / model_id / "leaderboard.csv",
    ]

    leaderboard_file = None
    for loc in leaderboard_locations:
        if loc.exists():
            leaderboard_file = loc
            break

    if not leaderboard_file:
        return _collect_alpacaeval_scores_from_annotations(model_id)

    with open(leaderboard_file) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return _collect_alpacaeval_scores_from_annotations(model_id)

    row = _select_alpacaeval_row(rows, model_id)

    metric_aliases = {
        "win_rate": ["win_rate", "winrate"],
        "length_controlled_winrate": [
            "length_controlled_winrate",
            "length_controlled_win_rate",
        ],
        "standard_error": ["standard_error", "std_error", "stderr"],
        "avg_length": ["avg_length", "avg_len"],
        "n_total": ["n_total", "count", "num_examples", "n"],
        "lc_standard_error": ["lc_standard_error", "length_controlled_standard_error"],
        "discrete_win_rate": ["discrete_win_rate"],
    }

    for metric_name, candidate_cols in metric_aliases.items():
        value = _extract_numeric_metric(row, candidate_cols)
        if value is None:
            continue
        scores.append(
            {
                "benchmark": "alpacaeval2",
                "metric": metric_name,
                "value": round(value, 4),
            }
        )

    if not scores:
        return _collect_alpacaeval_scores_from_annotations(model_id)

    return scores


def _normalize_text(value: Any) -> str:
    return str(value).strip().lower()


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _select_alpacaeval_row(rows: list[dict[str, Any]], model_id: str) -> dict[str, Any]:
    if len(rows) == 1:
        return rows[0]

    model_norm = _normalize_text(model_id)
    name_keys = ["", "name", "model", "generator", "model_id"]

    for row in rows:
        for key in name_keys:
            if key in row and row[key] and _normalize_text(row[key]) == model_norm:
                return row

    return rows[0]


def _extract_numeric_metric(row: dict[str, Any], candidates: list[str]) -> float | None:
    row_lut = {_normalize_text(k): v for k, v in row.items()}
    for key in candidates:
        if _normalize_text(key) in row_lut:
            value = _safe_float(row_lut[_normalize_text(key)])
            if value is not None:
                return value
    return None


def _collect_alpacaeval_scores_from_annotations(model_id: str) -> list[dict]:
    scores = []

    shared_alpacaeval2_dir = ROOT_DIR.parent / "alpacaeval2"

    candidate_files = [
        ROOT_DIR / "results" / model_id / "alpacaeval2" / "annotations.json",
        ROOT_DIR
        / "results"
        / model_id
        / "alpacaeval2"
        / "judge_config"
        / "annotations_seed0_configs.json",
        ROOT_DIR / "alpaca_eval" / "results" / model_id / "annotations.json",
        shared_alpacaeval2_dir / "judge_config" / "annotations_seed0_configs.json",
        shared_alpacaeval2_dir / "annotations.json",
    ]

    annotations_file = None
    for path in candidate_files:
        if path.exists():
            if path.is_relative_to(shared_alpacaeval2_dir):
                shared_outputs = shared_alpacaeval2_dir / "model_outputs.json"
                if shared_outputs.exists():
                    try:
                        with open(shared_outputs) as f:
                            outputs = json.load(f)
                    except (json.JSONDecodeError, OSError):
                        continue

                    generators = {
                        str(row.get("generator", "")).strip()
                        for row in outputs
                        if isinstance(row, dict) and row.get("generator")
                    }
                    if generators and model_id not in generators:
                        continue
            annotations_file = path
            break

    if not annotations_file:
        return scores

    try:
        with open(annotations_file) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return scores

    if not isinstance(data, list):
        return scores

    preferences = []
    output_2_lengths = []
    for item in data:
        if not isinstance(item, dict):
            continue

        pref = _safe_float(item.get("preference"))
        if pref is None:
            continue
        if pref == 0:
            pref = 1.5
        if pref < 1 or pref > 2:
            continue
        preferences.append(pref)

        output_2 = item.get("output_2")
        if isinstance(output_2, str):
            output_2_lengths.append(len(output_2))

    n_total = len(preferences)
    if n_total == 0:
        return scores

    probs = [p - 1 for p in preferences]
    mean_prob = sum(probs) / n_total
    if n_total > 1:
        variance = sum((x - mean_prob) ** 2 for x in probs) / (n_total - 1)
        standard_error = math.sqrt(variance / n_total) * 100
    else:
        standard_error = 0.0

    n_wins = sum(1 for x in probs if x > 0.5)
    n_wins_base = sum(1 for x in probs if x < 0.5)
    n_draws = n_total - n_wins - n_wins_base
    win_rate = mean_prob * 100

    scores.extend(
        [
            {
                "benchmark": "alpacaeval2",
                "metric": "win_rate",
                "value": round(win_rate, 4),
            },
            {
                "benchmark": "alpacaeval2",
                "metric": "standard_error",
                "value": round(standard_error, 4),
            },
            {
                "benchmark": "alpacaeval2",
                "metric": "n_total",
                "value": float(n_total),
            },
            {
                "benchmark": "alpacaeval2",
                "metric": "n_wins",
                "value": float(n_wins),
            },
            {
                "benchmark": "alpacaeval2",
                "metric": "n_wins_base",
                "value": float(n_wins_base),
            },
            {
                "benchmark": "alpacaeval2",
                "metric": "n_draws",
                "value": float(n_draws),
            },
        ]
    )

    if output_2_lengths:
        scores.append(
            {
                "benchmark": "alpacaeval2",
                "metric": "avg_length",
                "value": round(sum(output_2_lengths) / len(output_2_lengths), 4),
            }
        )

    return scores


def collect_livebench_scores(model_id: str) -> list[dict]:
    """Collect LiveBench scores from judgment files."""
    scores = []

    livebench_dir = ROOT_DIR / "results" / model_id / "livebench"

    if not livebench_dir.exists():
        return scores

    category_scores = {}
    all_scores = []

    for category_dir in livebench_dir.iterdir():
        if not category_dir.is_dir():
            continue
        category = category_dir.name

        for task_dir in category_dir.iterdir():
            if not task_dir.is_dir():
                continue
            task = task_dir.name

            judgment_file = task_dir / "judgment.jsonl"

            if not judgment_file.exists():
                continue

            task_scores = []
            with open(judgment_file) as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if "score" in data and data["score"] is not None:
                            task_scores.append(float(data["score"]))
                    except (json.JSONDecodeError, ValueError):
                        pass

            if task_scores:
                avg_score = sum(task_scores) / len(task_scores)
                all_scores.append(avg_score)

                scores.append(
                    {
                        "benchmark": "livebench",
                        "metric": f"{category}_{task}",
                        "value": round(avg_score, 4),
                        "count": len(task_scores),
                    }
                )

                if category not in category_scores:
                    category_scores[category] = []
                category_scores[category].append(avg_score)

    for category, cat_scores in category_scores.items():
        scores.append(
            {
                "benchmark": "livebench",
                "metric": f"category_{category}",
                "value": round(sum(cat_scores) / len(cat_scores), 4),
                "count": len(cat_scores),
            }
        )

    if all_scores:
        scores.append(
            {
                "benchmark": "livebench",
                "metric": "overall_average",
                "value": round(sum(all_scores) / len(all_scores), 4),
                "count": len(all_scores),
            }
        )

    return scores

    model_name_lower = model_id.lower()
    category_scores = {}
    all_scores = []

    for category_dir in livebench_dir.iterdir():
        if not category_dir.is_dir():
            continue
        category = category_dir.name

        for task_dir in category_dir.iterdir():
            if not task_dir.is_dir():
                continue
            task = task_dir.name

            if use_unified:
                judgment_file = task_dir / "judgment.jsonl"
            else:
                judgment_file = (
                    task_dir / "model_judgment" / f"{model_name_lower}_judgment.jsonl"
                )
                if not judgment_file.exists():
                    judgment_file = (
                        task_dir / "model_judgment" / "ground_truth_judgment.jsonl"
                    )

            if not judgment_file.exists():
                continue

            task_scores = []
            with open(judgment_file) as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if "score" in data and data["score"] is not None:
                            score_val = data["score"]
                            if data.get("model", "").lower() != model_name_lower:
                                continue
                            task_scores.append(float(score_val))
                    except (json.JSONDecodeError, ValueError):
                        pass

            if task_scores:
                avg_score = sum(task_scores) / len(task_scores)
                all_scores.append(avg_score)

                scores.append(
                    {
                        "benchmark": "livebench",
                        "metric": f"{category}_{task}",
                        "value": round(avg_score, 4),
                        "count": len(task_scores),
                    }
                )

                if category not in category_scores:
                    category_scores[category] = []
                category_scores[category].append(avg_score)

    for category, cat_scores in category_scores.items():
        scores.append(
            {
                "benchmark": "livebench",
                "metric": f"category_{category}",
                "value": round(sum(cat_scores) / len(cat_scores), 4),
                "count": len(cat_scores),
            }
        )

    if all_scores:
        scores.append(
            {
                "benchmark": "livebench",
                "metric": "overall_average",
                "value": round(sum(all_scores) / len(all_scores), 4),
                "count": len(all_scores),
            }
        )

    return scores

    category_scores = {}
    all_scores = []

    for category_dir in livebench_dir.iterdir():
        if not category_dir.is_dir():
            continue
        category = category_dir.name

        for task_dir in category_dir.iterdir():
            if not task_dir.is_dir():
                continue
            task = task_dir.name

            if use_unified:
                judgment_file = task_dir / "judgment.jsonl"
            else:
                judgment_file = (
                    task_dir / "model_judgment" / "ground_truth_judgment.jsonl"
                )

            if not judgment_file.exists():
                continue

            task_scores = []
            with open(judgment_file) as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if "score" in data and data["score"] is not None:
                            task_scores.append(float(data["score"]))
                    except (json.JSONDecodeError, ValueError):
                        pass

            if task_scores:
                avg_score = sum(task_scores) / len(task_scores)
                all_scores.append(avg_score)

                scores.append(
                    {
                        "benchmark": "livebench",
                        "metric": f"{category}_{task}",
                        "value": round(avg_score, 4),
                        "count": len(task_scores),
                    }
                )

                if category not in category_scores:
                    category_scores[category] = []
                category_scores[category].append(avg_score)

    for category, cat_scores in category_scores.items():
        scores.append(
            {
                "benchmark": "livebench",
                "metric": f"category_{category}",
                "value": round(sum(cat_scores) / len(cat_scores), 4),
                "count": len(cat_scores),
            }
        )

    if all_scores:
        scores.append(
            {
                "benchmark": "livebench",
                "metric": "overall_average",
                "value": round(sum(all_scores) / len(all_scores), 4),
                "count": len(all_scores),
            }
        )

    return scores


def collect_all_scores(model_id: str) -> dict[str, list[dict]]:
    """Collect all scores for a model."""
    return {
        "ifeval": collect_ifeval_scores(model_id),
        "truthfulqa": collect_truthfulqa_scores(model_id),
        "alpacaeval2": collect_alpacaeval_scores(model_id),
        "livebench": collect_livebench_scores(model_id),
    }


def get_available_models() -> list[str]:
    """Get list of models with results."""
    models = set()
    results_dir = ROOT_DIR / "results"

    if results_dir.exists():
        for model_dir in results_dir.iterdir():
            if model_dir.is_dir() and model_dir.name not in [
                "summaries",
                "judge_openai_config.yaml",
            ]:
                models.add(model_dir.name)

    return sorted(models)


def main():
    parser = argparse.ArgumentParser(description="Collect evaluation scores")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT_DIR / "results" / "summaries"),
        help="Output directory for score files",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific model to collect (default: all models)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["jsonl", "json"],
        default="jsonl",
        help="Output format",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.model:
        models = [args.model]
    else:
        models = get_available_models()

    if not models:
        print("No models found with results.")
        return

    print(f"Collecting scores for {len(models)} model(s): {', '.join(models)}")

    for model_id in models:
        print(f"\n=== {model_id} ===")

        all_scores = collect_all_scores(model_id)

        output_file = output_dir / f"{model_id}_scores.{args.format}"

        if args.format == "jsonl":
            with open(output_file, "w") as f:
                for benchmark, scores in all_scores.items():
                    for score in scores:
                        record = {"model": model_id, "benchmark": benchmark, **score}
                        f.write(json.dumps(record) + "\n")
        else:
            with open(output_file, "w") as f:
                json.dump({"model": model_id, "scores": all_scores}, f, indent=2)

        print(f"  Saved to: {output_file}")

        for benchmark, scores in all_scores.items():
            if scores:
                print(f"  {benchmark}: {len(scores)} metrics")
                for s in scores:
                    if "overall" in s.get("metric", ""):
                        print(f"    - {s['metric']}: {s['value']}")

    print(f"\nDone! All scores saved to {output_dir}/")


if __name__ == "__main__":
    main()
