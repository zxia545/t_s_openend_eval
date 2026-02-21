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
    """Collect AlpacaEval2 scores from leaderboard.csv."""
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
        return scores

    with open(leaderboard_file) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return scores

    row = rows[0]

    for col in [
        "win_rate",
        "length_controlled_winrate",
        "standard_error",
        "avg_length",
        "n_total",
    ]:
        if col in row and row[col]:
            try:
                value = float(row[col])
                scores.append(
                    {
                        "benchmark": "alpacaeval2",
                        "metric": col,
                        "value": round(value, 4),
                    }
                )
            except (ValueError, TypeError):
                pass

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
