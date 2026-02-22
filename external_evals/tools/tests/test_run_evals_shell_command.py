import importlib.util
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
