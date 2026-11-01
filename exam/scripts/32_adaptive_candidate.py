"""EXPLORATORY, POST-KAGGLE: write one candidate only if a variant qualified.

Uses frozen weeks 11-30 models and the verified test pool. Fits nothing and does
not reopen the test schedule. Writes to its own folder, never `kaggle_candidates`.
"""
import os

for name in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[name] = "1"

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from adaptive.fusion import (CANDIDATE_FILE, CANDIDATE_FOLDER, active_days, adaptive_lists, assign_weights, group_names,
                             programme_counts, read_lock)
from post_kaggle.repeat import positive_rows, tables_match
from sweettv import csvio
from sweettv.artifacts import load_arrays, stage_dir, write_result
from sweettv.baseline import rank_items
from sweettv.config import RESULTS_DIR, load_config
from sweettv.csvio import read_cohort
from sweettv.pipeline import build_interactions
from sweettv.ppr import score_candidates
from sweettv.runner import fallback_usage
from sweettv.split import TRAIN
from sweettv.submission import render_csv, validate_csv
from sweettv.timeutil import WallClock

CONSTANT = 10


def main():
    cfg = load_config()
    lock = read_lock(RESULTS_DIR)
    decision = json.loads((RESULTS_DIR / "adaptive_validation.json").read_text(encoding="utf-8"))
    if decision["lock_sha256"] != lock["lock_sha256"]:
        raise SystemExit("validation result belongs to another lock")
    if decision["selected_variant"] is None:
        print("no variant qualified; no CSV generated")
        return
    variant = next(v for v in lock["variants"] if v["name"] == decision["selected_variant"])
    k = cfg["top_k"]
    template_users = read_cohort(cfg)
    pool = load_arrays(cfg, "test_pool", "pool")["candidates"]
    popularity = load_arrays(cfg, "refit", "popularity")
    frozen = load_arrays(cfg, "refit", "ppr")
    train = load_arrays(cfg, "refit", "train_table")
    cohort = frozen["cohort"]
    users = cohort.tolist()
    pool_ids = set(pool.tolist())
    position = np.searchsorted(popularity["items"], pool).clip(max=len(popularity["items"]) - 1)
    order = rank_items(pool, np.where(popularity["items"][position] == pool, popularity["users_watching"][position], 0))
    scores, valid = score_candidates(frozen["item_ids"], frozen["probabilities"], pool)

    if variant["label"] == "A":
        values = programme_counts(train, cohort)
    else:
        clock = WallClock(cfg["timezone"])
        start, end = clock.view_ts(cfg["final"]["train"]["start"]), clock.view_ts(cfg["final"]["train"]["end"])
        airings, interactions, _, _, _ = build_interactions(cfg, clock, [(start, end), (end, end)])
        rows = positive_rows(airings, interactions, TRAIN, cfg["watched"]["primary_seconds"], cfg["watched"]["airing_fraction"])
        if not tables_match(rows, train):
            raise SystemExit("rebuilt rows do not match the verified refit table")
        values = active_days(rows, cohort, cfg["timezone"])
    weights = assign_weights(values, variant["lower"], variant["upper"], variant["weights"])
    lists, _, counts = adaptive_lists(users, pool, scores, valid, order, weights, CONSTANT, k)

    text = render_csv(template_users, [lists[user] for user in users], pool_ids, k)
    validate_csv(text, template_users, pool_ids, k)
    folder = stage_dir(cfg, CANDIDATE_FOLDER)
    folder.mkdir(exist_ok=True)
    (folder / CANDIDATE_FILE).write_bytes(text.encode("ascii"))
    groups = group_names(values, variant["lower"], variant["upper"])
    summary = {
        "post_kaggle_feedback": True,
        "exploratory": True,
        "label": "EXPLORATORY POST-KAGGLE candidate, produced after four Kaggle score pairs were observed",
        "lock_sha256": lock["lock_sha256"],
        "variant": variant,
        "file": f"{CANDIDATE_FOLDER}/{CANDIDATE_FILE}",
        "rows": len(users),
        "bytes": len(text.encode("ascii")),
        "sha256": hashlib.sha256(text.encode("ascii")).hexdigest(),
        "candidate_pool_size": int(len(pool)),
        "history_groups": {group: int((groups == group).sum()) for group in ("cold", "low", "middle", "high")},
        "fallback": fallback_usage(counts, k),
        "distinct_ids": len({item for user in users for item in lists[user]}),
        "parsed_sources": sorted(csvio.parsed_sources),
        "test_viewing_behaviour": "no test-period viewing data exists in the supplied files, and none was used",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (folder / "adaptive_manifest.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    write_result(RESULTS_DIR / "adaptive_candidate.json", summary)
    print(summary["file"], summary["bytes"], summary["sha256"])


if __name__ == "__main__":
    main()
