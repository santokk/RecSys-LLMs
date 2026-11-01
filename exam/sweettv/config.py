"""Configuration loading. Paths in config.json are relative to the repository root."""
import json
import os
from pathlib import Path

EXAM_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXAM_DIR.parent
RESULTS_DIR = EXAM_DIR / "results"


def load_config(path=None):
    with open(path or EXAM_DIR / "config.json", encoding="utf-8") as f:
        return json.load(f)


def _resolve(env_name, configured):
    value = os.environ.get(env_name) or configured
    return (REPO_ROOT / value).resolve()


def data_dir(cfg):
    return _resolve("SWEETTV_DATA_DIR", cfg["paths"]["data_dir"])


def artifacts_dir(cfg):
    return _resolve("SWEETTV_ARTIFACT_DIR", cfg["paths"]["artifacts_dir"])


def source_path(cfg, key):
    return data_dir(cfg) / cfg["files"][key]
