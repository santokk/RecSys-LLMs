"""Write validated Kaggle candidate files outside Git from frozen models.

Fits nothing. Popularity counts and PPR probabilities come from the refit stage;
the test pool only restricts which ids may appear.
"""
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from sweettv.artifacts import (code_hash, config_hash, load_arrays, load_manifest, load_text, save_stage, sha256_file,
                               source_hashes, stage_dir, write_result)
from sweettv.baseline import lists_from_scores, rank_items, recommend
from sweettv.config import REPO_ROOT, RESULTS_DIR, load_config
from sweettv.csvio import read_cohort, test_schedule_name
from sweettv.hybrid import hybrid_lists
from sweettv.ppr import score_candidates
from sweettv.runlog import RunTimer
from sweettv.runner import fallback_usage, hybrid_config_id, hybrid_grid
from sweettv.submission import render_csv, validate_csv

STAGE = "kaggle_candidates"


def git(*args):
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()


def main():
    timer = RunTimer("14_make_candidates")
    cfg = load_config()
    k = cfg["top_k"]
    template_users = read_cohort(cfg)
    if len(template_users) != cfg["final"]["expected_cohort_rows"]:
        raise SystemExit("the submission template changed")
    pool = load_arrays(cfg, "test_pool", "pool")["candidates"]
    popularity = load_arrays(cfg, "refit", "popularity")
    frozen = load_arrays(cfg, "refit", "ppr")
    cohort = frozen["cohort"]
    users = cohort.tolist()
    pool_ids = set(pool.tolist())

    position = np.searchsorted(popularity["items"], pool).clip(max=len(popularity["items"]) - 1)
    scores = np.where(popularity["items"][position] == pool, popularity["users_watching"][position], 0)
    order = rank_items(pool, scores)
    fallback = order.tolist()
    ppr_scores, ppr_valid = score_candidates(frozen["item_ids"], frozen["probabilities"], pool)

    selection = json.loads((RESULTS_DIR / "hybrid_validation.json").read_text(encoding="utf-8"))
    strategies = {"popularity": (recommend(users, {}, fallback, k), np.zeros(len(users), dtype=np.int64)),}
    ppr_lists, _, ppr_counts = lists_from_scores(users, pool, ppr_scores, ppr_valid, fallback, k)
    strategies["ppr"] = (ppr_lists, ppr_counts)
    configuration = {"popularity": {"definition": "distinct weeks 11-30 users with a watched airing"},
                     "ppr": {"run": cfg["hybrid"]["ppr_run"]}}
    if selection["selected_is_hybrid"]:
        params = next(p for p in hybrid_grid(cfg) if hybrid_config_id(p) == selection["selected_strategy"])
        lists, _, counts = hybrid_lists(users, pool, ppr_scores, ppr_valid, order, params["ppr_weight"], params["rank_constant"], k)
        strategies["hybrid"] = (lists, counts)
        configuration["hybrid"] = {"run": selection["selected_strategy"], **params, "ppr_run": cfg["hybrid"]["ppr_run"]}

    texts, entries = {}, {}
    for name, (lists, counts) in strategies.items():
        ordered = [lists[user] for user in users]
        text = render_csv(template_users, ordered, pool_ids, k)
        validate_csv(text, template_users, pool_ids, k)
        filename = f"{name}_submission.csv"
        texts[filename] = text
        entries[name] = {
            "file": filename,
            "configuration": configuration[name],
            "rows": len(ordered),
            "bytes": len(text.encode("utf-8")),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "distinct_ids": len({item for row in ordered for item in row}),
            "fallback": {"full_fallback_users": len(users), "partial_fallback_users": 0} if name == "popularity" else fallback_usage(counts, k),
        }

    refit_manifest = load_manifest(cfg, "refit")
    summary = {
        "validation_selected_strategy": selection["selected_strategy"],
        "designated_candidate_after_validation": "hybrid" if selection["selected_is_hybrid"] else "ppr",
        "candidates": entries,
        "candidate_pool_policy": json.loads((RESULTS_DIR / "test_pool.json").read_text(encoding="utf-8"))["policy"],
        "candidate_pool_size": int(len(pool)),
        "row_count": len(template_users),
        "source_sha256": source_hashes(cfg),
        "artifact_manifest_sha256": {stage: sha256_file(stage_dir(cfg, stage) / "manifest.json") for stage in ("refit", "test_pool")},
        "test_schedule_parsed_by_refit": test_schedule_name(cfg) in refit_manifest["parsed_sources"],
        "test_viewing_behaviour": "no test-period viewing data exists in the supplied files, and none was used",
        "test_schedule_use": "candidate availability and title/category lookup only",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "code_version": {"git_head": git("rev-parse", "HEAD"), "uncommitted_changes": bool(git("status", "--porcelain")),
                         "code_sha256": code_hash(), "config_sha256": config_hash(cfg)},
    }
    save_stage(cfg, STAGE, {}, {**texts, "candidate_manifest.json": json.dumps(summary, indent=1, ensure_ascii=False)},
               depends_on=("refit", "test_pool"))
    write_result(RESULTS_DIR / "candidate_summary.json", summary)
    timer.finish()


if __name__ == "__main__":
    main()
