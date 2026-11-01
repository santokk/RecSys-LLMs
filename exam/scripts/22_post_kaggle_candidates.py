"""POST-KAGGLE: write the two bounded candidates from frozen weeks 11-30 models.

Fits no PPR or popularity model. The test pool comes from the existing stage, so
the test schedule is not reopened. Existing candidate files are never touched.
"""
import os

for name in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[name] = "1"

import hashlib
import json
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from post_kaggle.repeat import PROTECTED_CANDIDATES, positive_rows, read_lock, tables_match, variant_lists
from sweettv import csvio
from sweettv.artifacts import load_arrays, load_manifest, sha256_file, source_hashes, stage_dir, write_result
from sweettv.baseline import rank_items
from sweettv.config import EXAM_DIR, REPO_ROOT, RESULTS_DIR, load_config
from sweettv.csvio import read_cohort
from sweettv.hybrid import hybrid_lists
from sweettv.pipeline import build_interactions
from sweettv.ppr import score_candidates
from sweettv.runner import fallback_usage
from sweettv.split import TRAIN, watched_tables
from sweettv.submission import render_csv, validate_csv
from sweettv.timeutil import WallClock


def experiment_code_hash():
    digest = hashlib.sha256()
    for path in sorted([*EXAM_DIR.glob("post_kaggle/*.py"), *EXAM_DIR.glob("scripts/2[0-2]_*.py")]):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def write_candidate(folder, name, text):
    if name in PROTECTED_CANDIDATES:
        raise SystemExit(f"{name} must not be overwritten")
    (folder / name).write_bytes(text.encode("ascii"))


def main():
    started = time.perf_counter()
    cfg = load_config()
    lock = read_lock(RESULTS_DIR)
    decision = json.loads((RESULTS_DIR / "post_kaggle_validation.json").read_text(encoding="utf-8"))
    if decision["lock_sha256"] != lock["lock_sha256"]:
        raise SystemExit("validation result belongs to another lock")
    k = cfg["top_k"]
    source_hashes(cfg)
    template_users = read_cohort(cfg)

    pool = load_arrays(cfg, "test_pool", "pool")["candidates"]
    popularity = load_arrays(cfg, "refit", "popularity")
    frozen = load_arrays(cfg, "refit", "ppr")
    cohort = frozen["cohort"]
    users = cohort.tolist()
    pool_ids = set(pool.tolist())
    position = np.searchsorted(popularity["items"], pool).clip(max=len(popularity["items"]) - 1)
    order = rank_items(pool, np.where(popularity["items"][position] == pool, popularity["users_watching"][position], 0))
    ppr_scores, ppr_valid = score_candidates(frozen["item_ids"], frozen["probabilities"], pool)

    folder = stage_dir(cfg, "kaggle_candidates")
    protected_before = {name: sha256_file(folder / name) for name in PROTECTED_CANDIDATES}
    if protected_before != lock["protected_candidate_sha256"]:
        raise SystemExit("protected candidates differ from the lock")

    outputs = {}
    settings = lock["candidates"]["hybrid"]
    params = {"ppr_weight": 0.8, "rank_constant": 10}
    lists, _, counts = hybrid_lists(users, pool, ppr_scores, ppr_valid, order, params["ppr_weight"], params["rank_constant"], k)
    outputs["hybrid"] = {"file": settings["file"], "lists": lists, "counts": counts, "configuration": {**settings, **params}}

    refit_detail = {}
    if decision["acceptance"]["accepted"]:
        variant = next(v for v in lock["repeat_family"] if v["name"] == decision["selected_variant"])
        clock = WallClock(cfg["timezone"])
        start = clock.view_ts(cfg["final"]["train"]["start"])
        end = clock.view_ts(cfg["final"]["train"]["end"])
        threshold, fraction = cfg["watched"]["primary_seconds"], cfg["watched"]["airing_fraction"]
        airings, interactions, _, user_ids, _ = build_interactions(cfg, clock, [(start, end), (end, end)])
        rows = positive_rows(airings, interactions, TRAIN, threshold, fraction)
        verified = load_arrays(cfg, "refit", "train_table")
        if not tables_match(rows, verified) or len(watched_tables(airings, interactions, threshold, fraction)["validation"]["user"]):
            raise SystemExit("rebuilt per-airing rows do not match the verified refit table")
        if user_ids != (stage_dir(cfg, "refit") / "users.txt").read_text(encoding="utf-8").split():
            raise SystemExit("user index differs from the verified refit")
        lists, _, counts = variant_lists(variant, rows, cohort, pool, order, end, k)
        outputs["repeat"] = {"file": lock["candidates"]["repeat"]["file"], "lists": lists, "counts": counts,
                             "configuration": {"variant": variant, "prediction_start_utc": end}}
        refit_detail = {"per_airing_rows_rebuilt": int(len(rows["user"])), "equal_to_verified_refit_table": True,
                        "latest_airing_start_utc": int(rows["airing_start"].max()), "prediction_start_utc": end,
                        "users_with_any_repeat_candidate": int((counts > 0).sum())}

    entries = {}
    for key, output in outputs.items():
        ordered = [output["lists"][user] for user in users]
        text = render_csv(template_users, ordered, pool_ids, k)
        validate_csv(text, template_users, pool_ids, k)
        write_candidate(folder, output["file"], text)
        entries[key] = {"file": output["file"], "configuration": output["configuration"], "rows": len(ordered),
                        "bytes": len(text.encode("ascii")), "sha256": hashlib.sha256(text.encode("ascii")).hexdigest(),
                        "distinct_ids": len({item for row in ordered for item in row}),
                        "fallback": fallback_usage(output["counts"], k)}
    if not decision["acceptance"]["accepted"]:
        entries["repeat"] = {"file": None, "not_generated": "the selected variant failed the locked acceptance rule"}

    git = lambda *args: subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    summary = {
        "post_kaggle_feedback": True,
        "label": "POST-KAGGLE candidates: produced after PPR and popularity Kaggle scores were observed",
        "lock_sha256": lock["lock_sha256"],
        "candidates": entries,
        "candidate_pool_size": int(len(pool)),
        "repeat_refit": refit_detail,
        "parsed_sources": sorted(csvio.parsed_sources),
        "test_schedule_reopened": csvio.test_schedule_name(cfg) in csvio.parsed_sources,
        "protected_candidates_unchanged": {name: sha256_file(folder / name) == sha for name, sha in protected_before.items()},
        "artifact_manifest_sha256": {stage: sha256_file(stage_dir(cfg, stage) / "manifest.json") for stage in ("refit", "test_pool")},
        "test_viewing_behaviour": "no test-period viewing data exists in the supplied files, and none was used",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "code_version": {"git_head": git("rev-parse", "HEAD"), "uncommitted_changes": bool(git("status", "--porcelain")),
                         "experiment_code_sha256": experiment_code_hash()},
        "execution": {"seconds": round(time.perf_counter() - started, 1),
                      "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20), 1)},
    }
    load_manifest(cfg, "kaggle_candidates")
    (folder / "post_kaggle_manifest.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    write_result(RESULTS_DIR / "post_kaggle_candidates.json", summary)
    print(json.dumps({key: {k2: entry.get(k2) for k2 in ("file", "bytes", "sha256")} for key, entry in entries.items()}, indent=1))


if __name__ == "__main__":
    main()
