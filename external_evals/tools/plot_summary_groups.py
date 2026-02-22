#!/usr/bin/env python3
"""Plot summary score curves for each training group.

Expected input files live under results/summaries and look like:
- model--Qwen-Qwen2.5-3B-Instruct_scores.jsonl
- reordered_teacher_jsonl_group_1_checkpoint-781_scores.jsonl
- reordered_teacher_jsonl_group_1_scores.jsonl
"""

import argparse
import json
import re
from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parent.parent

CHECKPOINT_RE = re.compile(
    r"^reordered_teacher_jsonl_group_(\d+)_checkpoint-(\d+)_scores\.jsonl$"
)
FINAL_RE = re.compile(r"^reordered_teacher_jsonl_group_(\d+)_scores\.jsonl$")


class RunPoint(NamedTuple):
    model_name: str
    file_path: Path
    x_label: str


def _read_metrics(file_path: Path) -> dict[tuple[str, str], float]:
    metrics: dict[tuple[str, str], float] = {}
    with file_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            benchmark = str(row.get("benchmark", "")).strip()
            metric = str(row.get("metric", "")).strip()
            value = row.get("value")
            if not benchmark or not metric or value is None:
                continue
            try:
                metrics[(benchmark, metric)] = float(value)
            except (TypeError, ValueError):
                continue
    return metrics


def discover_group_runs(summary_dir: Path) -> dict[int, list[RunPoint]]:
    summary_dir = Path(summary_dir)
    files = list(summary_dir.glob("*_scores.jsonl"))

    original_files = [p for p in files if p.name.startswith("model--")]
    original_file = original_files[0] if original_files else None

    checkpoints: dict[int, list[tuple[int, Path]]] = {}
    finals: dict[int, Path] = {}

    for p in files:
        ckpt_match = CHECKPOINT_RE.match(p.name)
        if ckpt_match:
            group_id = int(ckpt_match.group(1))
            step = int(ckpt_match.group(2))
            checkpoints.setdefault(group_id, []).append((step, p))
            continue

        final_match = FINAL_RE.match(p.name)
        if final_match:
            group_id = int(final_match.group(1))
            finals[group_id] = p

    all_groups = sorted(set(checkpoints.keys()) | set(finals.keys()))
    result: dict[int, list[RunPoint]] = {}

    for group_id in all_groups:
        run_points: list[RunPoint] = []

        if original_file is not None:
            run_points.append(
                RunPoint(
                    model_name=original_file.stem.replace("_scores", ""),
                    file_path=original_file,
                    x_label="Original",
                )
            )

        sorted_ckpts = sorted(checkpoints.get(group_id, []), key=lambda x: x[0])
        for i, (_step, ckpt_file) in enumerate(sorted_ckpts, start=1):
            run_points.append(
                RunPoint(
                    model_name=ckpt_file.stem.replace("_scores", ""),
                    file_path=ckpt_file,
                    x_label=f"checkpoint {i}",
                )
            )

        if group_id in finals:
            final_file = finals[group_id]
            run_points.append(
                RunPoint(
                    model_name=final_file.stem.replace("_scores", ""),
                    file_path=final_file,
                    x_label="Final",
                )
            )

        if run_points:
            result[group_id] = run_points

    return result


def _plot_metric_lines(
    run_points: list[RunPoint],
    lines: list[tuple[str, tuple[str, str]]],
    title: str,
    out_path: Path,
    dpi: int,
) -> bool:
    x_labels = [p.x_label for p in run_points]
    x_values = list(range(len(run_points)))

    metrics_by_run = [_read_metrics(p.file_path) for p in run_points]

    fig, ax = plt.subplots(figsize=(10, 5))
    plotted_any = False

    for legend_name, metric_key in lines:
        y_values = [m.get(metric_key) for m in metrics_by_run]
        if all(v is None for v in y_values):
            continue
        ax.plot(x_values, y_values, marker="o", linewidth=2, label=legend_name)
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return False

    ax.set_title(title)
    ax.set_xlabel("Model Stage")
    ax.set_ylabel("Score")
    ax.set_xticks(x_values)
    ax.set_xticklabels(x_labels, rotation=20, ha="right")
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend()
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return True


def generate_all_plots(
    summary_dir: Path, output_dir: Path, dpi: int = 140
) -> list[Path]:
    groups = discover_group_runs(Path(summary_dir))
    output_dir = Path(output_dir)
    written: list[Path] = []

    for group_id, run_points in groups.items():
        group_out = output_dir / f"group_{group_id}"

        if _plot_metric_lines(
            run_points=run_points,
            lines=[
                ("ifeval strict acc", ("ifeval", "accuracy_strict")),
                ("ifeval loose acc", ("ifeval", "accuracy_loose")),
            ],
            title=f"Group {group_id} - IFEval",
            out_path=group_out / "ifeval.png",
            dpi=dpi,
        ):
            written.append(group_out / "ifeval.png")

        if _plot_metric_lines(
            run_points=run_points,
            lines=[("overall_truthful", ("truthfulqa", "overall_truthful"))],
            title=f"Group {group_id} - TruthfulQA",
            out_path=group_out / "truthfulqa.png",
            dpi=dpi,
        ):
            written.append(group_out / "truthfulqa.png")

        if _plot_metric_lines(
            run_points=run_points,
            lines=[
                ("win_rate", ("alpacaeval2", "win_rate")),
                (
                    "length_controlled_winrate",
                    ("alpacaeval2", "length_controlled_winrate"),
                ),
            ],
            title=f"Group {group_id} - AlpacaEval2",
            out_path=group_out / "alpacaeval2.png",
            dpi=dpi,
        ):
            written.append(group_out / "alpacaeval2.png")

        if _plot_metric_lines(
            run_points=run_points,
            lines=[
                ("category_coding", ("livebench", "category_coding")),
                ("category_data_analysis", ("livebench", "category_data_analysis")),
                (
                    "category_instruction_following",
                    ("livebench", "category_instruction_following"),
                ),
                ("category_language", ("livebench", "category_language")),
                ("category_math", ("livebench", "category_math")),
                ("category_reasoning", ("livebench", "category_reasoning")),
                ("overall_average", ("livebench", "overall_average")),
            ],
            title=f"Group {group_id} - LiveBench",
            out_path=group_out / "livebench.png",
            dpi=dpi,
        ):
            written.append(group_out / "livebench.png")

    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot evaluation summary curves by group."
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=ROOT_DIR / "results" / "summaries",
        help="Directory with *_scores.jsonl files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "results" / "plots",
        help="Directory to write plot PNGs",
    )
    parser.add_argument("--dpi", type=int, default=140, help="PNG output DPI")
    args = parser.parse_args()

    written = generate_all_plots(args.summary_dir, args.output_dir, dpi=args.dpi)
    if written:
        print(f"Generated {len(written)} plot files in {args.output_dir}")
    else:
        print("No plots generated (no matching files/metrics found).")


if __name__ == "__main__":
    main()
