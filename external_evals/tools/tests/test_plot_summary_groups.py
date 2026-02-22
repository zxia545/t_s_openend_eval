import importlib.util
import json
from pathlib import Path


def _load_plot_module():
    repo_root = Path(__file__).resolve().parents[3]
    target = repo_root / "external_evals" / "tools" / "plot_summary_groups.py"
    spec = importlib.util.spec_from_file_location("plot_summary_groups_mod", target)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _score_rows(model_name: str, shift: float):
    return [
        {
            "model": model_name,
            "benchmark": "ifeval",
            "metric": "accuracy_strict",
            "value": round(0.50 + shift, 4),
        },
        {
            "model": model_name,
            "benchmark": "ifeval",
            "metric": "accuracy_loose",
            "value": round(0.60 + shift, 4),
        },
        {
            "model": model_name,
            "benchmark": "truthfulqa",
            "metric": "overall_truthful",
            "value": round(0.40 + shift, 4),
        },
        {
            "model": model_name,
            "benchmark": "alpacaeval2",
            "metric": "win_rate",
            "value": round(10.0 + shift * 10, 4),
        },
        {
            "model": model_name,
            "benchmark": "alpacaeval2",
            "metric": "length_controlled_winrate",
            "value": round(12.0 + shift * 10, 4),
        },
        {
            "model": model_name,
            "benchmark": "livebench",
            "metric": "category_coding",
            "value": round(0.20 + shift, 4),
        },
        {
            "model": model_name,
            "benchmark": "livebench",
            "metric": "category_data_analysis",
            "value": round(0.30 + shift, 4),
        },
        {
            "model": model_name,
            "benchmark": "livebench",
            "metric": "category_instruction_following",
            "value": round(0.40 + shift, 4),
        },
        {
            "model": model_name,
            "benchmark": "livebench",
            "metric": "category_language",
            "value": round(0.10 + shift, 4),
        },
        {
            "model": model_name,
            "benchmark": "livebench",
            "metric": "category_math",
            "value": round(0.22 + shift, 4),
        },
        {
            "model": model_name,
            "benchmark": "livebench",
            "metric": "category_reasoning",
            "value": round(0.18 + shift, 4),
        },
        {
            "model": model_name,
            "benchmark": "livebench",
            "metric": "overall_average",
            "value": round(0.24 + shift, 4),
        },
    ]


def _write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_discover_group_runs_orders_original_checkpoints_and_final(tmp_path):
    mod = _load_plot_module()

    base_name = "model--Qwen-Qwen2.5-3B-Instruct"
    _write_jsonl(
        tmp_path / f"{base_name}_scores.jsonl",
        _score_rows(base_name, 0.00),
    )

    ckpt_100 = "reordered_teacher_jsonl_group_1_checkpoint-100"
    ckpt_200 = "reordered_teacher_jsonl_group_1_checkpoint-200"
    final_name = "reordered_teacher_jsonl_group_1"

    _write_jsonl(tmp_path / f"{ckpt_200}_scores.jsonl", _score_rows(ckpt_200, 0.02))
    _write_jsonl(tmp_path / f"{ckpt_100}_scores.jsonl", _score_rows(ckpt_100, 0.01))
    _write_jsonl(tmp_path / f"{final_name}_scores.jsonl", _score_rows(final_name, 0.03))

    groups = mod.discover_group_runs(tmp_path)
    runs = groups[1]

    assert [run.x_label for run in runs] == [
        "Original",
        "checkpoint 1",
        "checkpoint 2",
        "Final",
    ]


def test_generate_all_plots_writes_expected_files(tmp_path):
    mod = _load_plot_module()

    base_name = "model--Qwen-Qwen2.5-3B-Instruct"
    _write_jsonl(
        tmp_path / f"{base_name}_scores.jsonl",
        _score_rows(base_name, 0.00),
    )

    ckpt_100 = "reordered_teacher_jsonl_group_1_checkpoint-100"
    ckpt_200 = "reordered_teacher_jsonl_group_1_checkpoint-200"
    final_name = "reordered_teacher_jsonl_group_1"

    _write_jsonl(tmp_path / f"{ckpt_100}_scores.jsonl", _score_rows(ckpt_100, 0.01))
    _write_jsonl(tmp_path / f"{ckpt_200}_scores.jsonl", _score_rows(ckpt_200, 0.02))
    _write_jsonl(tmp_path / f"{final_name}_scores.jsonl", _score_rows(final_name, 0.03))

    output_dir = tmp_path / "plots"
    written = mod.generate_all_plots(tmp_path, output_dir, dpi=80)

    expected = {
        output_dir / "group_1" / "ifeval.png",
        output_dir / "group_1" / "truthfulqa.png",
        output_dir / "group_1" / "alpacaeval2.png",
        output_dir / "group_1" / "livebench.png",
    }

    assert expected.issubset(set(written))
    for path in expected:
        assert path.exists()
        assert path.stat().st_size > 0
