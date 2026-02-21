import importlib.util
from pathlib import Path
import sys


def _load_livebench_common_module():
    repo_root = Path(__file__).resolve().parents[3]
    livebench_pkg_root = repo_root / "external_evals" / "livebench"
    sys.path.insert(0, str(livebench_pkg_root))
    target = repo_root / "external_evals" / "livebench" / "livebench" / "common.py"
    spec = importlib.util.spec_from_file_location("livebench_common_mod", target)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_livebench_adds_list_feature_compat_alias(monkeypatch):
    mod = _load_livebench_common_module()
    from datasets.features import features as ds_features

    original = ds_features._FEATURE_TYPES.copy()
    monkeypatch.setattr(ds_features, "_FEATURE_TYPES", original, raising=False)
    original.pop("List", None)

    mod._ensure_datasets_list_feature_compat()

    assert "List" in ds_features._FEATURE_TYPES
    assert ds_features._FEATURE_TYPES["List"] is ds_features.Sequence
