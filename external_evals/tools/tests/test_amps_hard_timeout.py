import importlib.util
import os
import time
from pathlib import Path
import sys


def _load_amps_utils_module():
    repo_root = Path(__file__).resolve().parents[3]
    livebench_pkg_root = repo_root / "external_evals" / "livebench"
    sys.path.insert(0, str(livebench_pkg_root))
    target = (
        repo_root
        / "external_evals"
        / "livebench"
        / "livebench"
        / "process_results"
        / "math"
        / "AMPS_Hard"
        / "utils.py"
    )
    spec = importlib.util.spec_from_file_location("amps_hard_utils_mod", target)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_run_with_timeout_returns_value():
    mod = _load_amps_utils_module()

    out = mod.run_with_timeout(lambda x: x + 1, args=(1,), timeout=1)
    assert out == 2


def test_run_with_timeout_times_out():
    mod = _load_amps_utils_module()

    def _slow():
        time.sleep(2)
        return 1

    try:
        mod.run_with_timeout(_slow, timeout=0.2)
        raised = False
    except TimeoutError:
        raised = True

    assert raised


def test_run_with_timeout_handles_worker_crash_without_hanging():
    mod = _load_amps_utils_module()

    def _crash_now():
        os._exit(11)

    start = time.time()
    try:
        mod.run_with_timeout(_crash_now, timeout=1)
        raised = False
    except RuntimeError as exc:
        raised = True
        assert "without result" in str(exc)

    elapsed = time.time() - start
    assert raised
    assert elapsed < 3
