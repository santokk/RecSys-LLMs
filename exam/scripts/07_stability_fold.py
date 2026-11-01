"""Robustness diagnostic for the two selected configurations only.

Rebuilds interactions from the raw log through the normal pipeline with the
boundary moved to week 28, so views are clipped there rather than approximated.
Nothing here feeds selection.
"""
import os

for name in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[name] = "1"

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from sweettv import csvio
from sweettv.artifacts import source_hashes, write_result
from sweettv.baseline import recommend
from sweettv.config import RESULTS_DIR, load_config
from sweettv.evaluation import evaluate_lists, validation_targets
from sweettv.runlog import RunTimer
from sweettv.runner import fallback_usage, mf_checkpoints, popularity_order, ppr_lists, stability_split


def summary(report):
    keep = {key: report[key] for key in ("primary_top5_preference", "secondary_broad_watched_diagnostic", "concentration")}
    if "paired_difference_vs_popularity" in report:
        keep["paired_difference_vs_popularity"] = report["paired_difference_vs_popularity"]
    return keep


def main():
    timer = RunTimer("07_stability_fold")
    cfg = load_config()
    source_hashes(cfg)
    k = cfg["top_k"]
    fold = stability_split(cfg)
    train, validation, pool, cohort = fold["train"], fold["validation"], fold["pool"], fold["cohort"]
    users = cohort.tolist()

    target, broad = validation_targets(validation, cohort, pool, k)
    order, _ = popularity_order(train, pool)
    fallback = order.tolist()
    baseline = recommend(users, {}, fallback, k)
    baseline_report, baseline_ap = evaluate_lists(baseline, users, target, broad, len(pool), cfg)

    grid = json.loads((RESULTS_DIR / "model_grid.json").read_text(encoding="utf-8"))
    chosen = {record["run_id"]: record for record in grid["runs"] if record["run_id"] in grid["selected"].values()}
    results = {"popularity": {"run_id": "popularity", **summary(baseline_report)}}
    selected_mf = chosen[grid["selected"]["mf"]]
    params = {key: selected_mf["params"][key] for key in ("factors", "confidence_alpha", "regularization")}
    iteration = selected_mf["params"]["iterations"]
    for _, lists, _, counts, _ in mf_checkpoints(train, params, cfg, cohort, pool, fallback, [iteration]):
        report, _ = evaluate_lists(lists, users, target, broad, len(pool), cfg, baseline_ap)
        results["mf"] = {"run_id": selected_mf["run_id"], "fallback": fallback_usage(counts, k), **summary(report)}
    selected_ppr = chosen[grid["selected"]["ppr"]]
    lists, _, counts, details = ppr_lists(train, selected_ppr["params"], cfg, cohort, pool, fallback)
    report, _ = evaluate_lists(lists, users, target, broad, len(pool), cfg, baseline_ap)
    results["ppr"] = {"run_id": selected_ppr["run_id"], "fallback": fallback_usage(counts, k), "graph": details, **summary(report)}

    write_result(RESULTS_DIR / "stability_fold.json", {
        "purpose": "robustness diagnostic, never a tuning target",
        "train": cfg["stability"]["train"],
        "validation": cfg["stability"]["validation"],
        "parsed_sources": sorted(csvio.parsed_sources),
        "fold_size": {
            "candidate_pool": int(len(pool)),
            "train_user_programme_pairs": int(len(train["user"])),
            "validation_user_programme_pairs": int(len(validation["user"])),
            "validation_user_airing_rows_all_thresholds": fold["validation_rows"],
            "cohort_with_nonempty_target": sum(1 for user in users if target[user]),
            "cohort_cold_no_train_positive": int((~np.isin(cohort, train["user"])).sum()),
        },
        "results": results,
    })
    timer.finish()


if __name__ == "__main__":
    main()
