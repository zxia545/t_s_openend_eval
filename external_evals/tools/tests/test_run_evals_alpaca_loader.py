import importlib.util
import json
from pathlib import Path


def _load_run_evals_module():
    repo_root = Path(__file__).resolve().parents[3]
    target = repo_root / "external_evals" / "tools" / "run_evals.py"
    spec = importlib.util.spec_from_file_location("run_evals_mod", target)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_load_alpacaeval_eval_split_fallbacks_to_json(monkeypatch, tmp_path):
    mod = _load_run_evals_module()

    data = [
        {"instruction": "inst-1", "dataset": "d1"},
        {"instruction": "inst-2", "dataset": "d2"},
    ]
    json_file = tmp_path / "alpaca_eval.json"
    json_file.write_text(json.dumps(data), encoding="utf-8")

    def fail_load_dataset(*args, **kwargs):
        raise RuntimeError(
            "Dataset scripts are no longer supported, but found alpac_eval.py"
        )

    monkeypatch.setattr(mod, "load_dataset", fail_load_dataset, raising=False)
    monkeypatch.setattr(mod, "hf_hub_download", lambda **kwargs: str(json_file))

    loaded = mod.load_alpacaeval_eval_split()
    assert isinstance(loaded, list)
    assert len(loaded) == 2
    assert loaded[0]["instruction"] == "inst-1"


def test_call_openai_api_retries_then_succeeds():
    mod = _load_run_evals_module()

    class _RespMsg:
        content = "ok"

    class _RespChoice:
        message = _RespMsg()

    class _Resp:
        choices = [_RespChoice()]

    class _Completions:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            if self.calls < 3:
                raise RuntimeError("transient")
            return _Resp()

    class _Chat:
        def __init__(self):
            self.completions = _Completions()

    class _Client:
        def __init__(self):
            self.chat = _Chat()

    client = _Client()
    out = mod.call_openai_api(
        client,
        model_name="dummy",
        prompt="hi",
        timeout=1,
        max_retries=3,
    )
    assert out == "ok"
    assert client.chat.completions.calls == 3
