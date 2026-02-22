import importlib.util
from types import SimpleNamespace
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


def test_run_shell_command_raises_on_failure():
    mod = _load_run_evals_module()

    try:
        mod.run_shell_command("exit 7", "unit-test")
        raised = False
    except RuntimeError as exc:
        raised = True
        assert "unit-test" in str(exc)
        assert "7" in str(exc)

    assert raised


def test_run_shell_command_succeeds_on_zero_exit():
    mod = _load_run_evals_module()
    mod.run_shell_command("true", "unit-test")


def test_run_evaluate_alpacaeval_uses_vendored_module_env(monkeypatch, tmp_path):
    mod = _load_run_evals_module()

    root_dir = tmp_path / "external_evals"
    (root_dir / "results" / "m1" / "alpacaeval2").mkdir(parents=True)
    (root_dir / "results" / "m1" / "alpacaeval2" / "model_outputs.json").write_text(
        "[]", encoding="utf-8"
    )
    (root_dir / "alpaca_eval" / "src").mkdir(parents=True)

    monkeypatch.setattr(mod, "ROOT_DIR", root_dir)

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(mod.urllib.request, "urlopen", lambda *args, **kwargs: _Resp())

    calls = []

    def _fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)

    config = {
        "models": [
            {"id": "m1", "api_name": "m1", "api_base": "http://x", "api_key": "k"}
        ],
        "judge": {"api_base": "http://judge", "api_key": "jk", "model_name": "jmodel"},
        "benchmarks": {"alpacaeval": True},
    }

    mod.run_evaluate(config)

    assert calls
    _args, kwargs = calls[-1]
    assert kwargs.get("cwd") == root_dir / "alpaca_eval"
    assert kwargs.get("shell") is False
    assert (
        kwargs.get("env", {})
        .get("PYTHONPATH", "")
        .startswith(str(root_dir / "alpaca_eval" / "src"))
    )
    cmd = kwargs.get("args")
    assert isinstance(cmd, list)
    assert cmd[:4] == ["python", "-m", "alpaca_eval.main", "evaluate"]
