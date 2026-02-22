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
    from datasets import Features, Sequence, Value
    from datasets.features import features as ds_features

    monkeypatch.setattr(mod, "_LIST_FEATURE_COMPAT_PATCHED", False)
    original_generate_from_dict = ds_features.generate_from_dict

    mod._ensure_datasets_list_feature_compat()

    assert ds_features.generate_from_dict is not original_generate_from_dict

    parsed = ds_features.generate_from_dict(
        {
            "kwargs": {
                "_type": "List",
                "feature": {
                    "num_bullets": {"_type": "Value", "dtype": "int64"},
                    "first_word": {"_type": "Value", "dtype": "string"},
                    "forbidden_words": {
                        "_type": "Sequence",
                        "feature": {"_type": "Value", "dtype": "string"},
                        "length": -1,
                    },
                },
            }
        }
    )

    assert isinstance(parsed, dict)
    assert isinstance(parsed["kwargs"], list)
    assert len(parsed["kwargs"]) == 1
    assert set(parsed["kwargs"][0].keys()) == {
        "num_bullets",
        "first_word",
        "forbidden_words",
    }

    source = Features(parsed)
    target = Features(
        {
            "kwargs": [
                {
                    "num_bullets": Value("int64"),
                    "first_word": Value("string"),
                    "forbidden_words": Sequence(Value("string")),
                }
            ]
        }
    )
    source.reorder_fields_as(target)


def test_livebench_list_compat_patch_is_idempotent(monkeypatch):
    mod = _load_livebench_common_module()
    from datasets.features import features as ds_features

    monkeypatch.setattr(mod, "_LIST_FEATURE_COMPAT_PATCHED", False)
    mod._ensure_datasets_list_feature_compat()
    patched_once = ds_features.generate_from_dict
    mod._ensure_datasets_list_feature_compat()

    assert ds_features.generate_from_dict is patched_once
