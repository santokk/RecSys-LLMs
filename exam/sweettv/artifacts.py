"""External artifact stages guarded by a manifest of sources, config and code."""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from . import csvio
from .config import EXAM_DIR, artifacts_dir, data_dir

_source_hashes = {}


class StaleArtifactError(RuntimeError):
    pass


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes(cfg):
    """Hash every source once per process and enforce the pinned checksums."""
    if not _source_hashes:
        for name, expected in cfg["expected_sha256"].items():
            found = sha256_file(data_dir(cfg) / name)
            if found != expected:
                raise StaleArtifactError(f"{name} does not match its pinned checksum")
            _source_hashes[name] = found
    return dict(_source_hashes)


def config_hash(cfg):
    settings = {key: value for key, value in cfg.items() if key != "paths"}
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


def code_hash(root=EXAM_DIR):
    digest = hashlib.sha256()
    for path in sorted([*root.glob("sweettv/*.py"), *root.glob("scripts/0[1-35-8]_*.py"), *root.glob("scripts/1[0-4]_*.py")]):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def fingerprint(cfg):
    return {"config_sha256": config_hash(cfg), "code_sha256": code_hash(), "sources": source_hashes(cfg)}


def stage_dir(cfg, stage):
    return artifacts_dir(cfg) / stage


def save_stage(cfg, stage, arrays, texts=None, depends_on=()):
    """Write all files, then the manifest, then swap the stage into place."""
    target = stage_dir(cfg, stage)
    temporary = target.with_name(f"{stage}.partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    for name, members in arrays.items():
        np.savez(temporary / f"{name}.npz", **members)
    for name, text in (texts or {}).items():
        (temporary / name).write_text(text, encoding="utf-8")
    manifest = {
        **fingerprint(cfg),
        "stage": stage,
        "parsed_sources": sorted(csvio.parsed_sources),
        "depends_on": {name: sha256_file(stage_dir(cfg, name) / "manifest.json") for name in depends_on},
        "files": {p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in sorted(temporary.iterdir())},
    }
    (temporary / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    if target.exists():
        shutil.rmtree(target)
    temporary.rename(target)
    return manifest


def load_manifest(cfg, stage):
    """Return a stage manifest only if it still matches sources, config and code."""
    directory = stage_dir(cfg, stage)
    path = directory / "manifest.json"
    if not path.exists():
        raise StaleArtifactError(f"{stage}: no manifest; run its script first")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    current = fingerprint(cfg)
    for key, value in current.items():
        if manifest.get(key) != value:
            raise StaleArtifactError(f"{stage}: {key} changed since this artifact was built")
    for name in manifest["depends_on"]:
        if sha256_file(stage_dir(cfg, name) / "manifest.json") != manifest["depends_on"][name]:
            raise StaleArtifactError(f"{stage}: dependency {name} was rebuilt")
    for name, entry in manifest["files"].items():
        if sha256_file(directory / name) != entry["sha256"]:
            raise StaleArtifactError(f"{stage}: {name} was altered")
    return manifest


def load_arrays(cfg, stage, name):
    load_manifest(cfg, stage)
    with np.load(stage_dir(cfg, stage) / f"{name}.npz", allow_pickle=False) as members:
        return {key: members[key] for key in members.files}


def load_text(cfg, stage, name):
    return Path(stage_dir(cfg, stage) / name).read_text(encoding="utf-8")


def write_result(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
