"""Evaluate the four locked hybrids, select by the locked rule, then run the fold."""
import os

for name in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[name] = "1"

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from sweettv import csvio
from sweettv.artifacts import load_arrays, save_stage, write_result
from sweettv.baseline import recommend
from sweettv.config import RESULTS_DIR, load_config
from sweettv.evaluation import evaluate_lists, paired_difference, validation_targets
from sweettv.hybrid import hybrid_lists
from sweettv.ppr import score_candidates
from sweettv.runlog import RunTimer
from sweettv.runner import (fallback_usage, hybrid_config_id, hybrid_grid, popularity_order, ppr_grid,
                            ppr_config_id, ppr_lists, ppr_probabilities, stability_split)
from sweettv.split import side
from sweettv.submission import validate_row

PRIMARY = ("primary_top5_preference", "map_at_5_all_users")


def primary_of(record):
    return record[PRIMARY[0]][PRIMARY[1]]


def compare_split(cfg, train, validation, pool, cohort, hybrids):
    """Popularity, PPR and the named hybrids on one split, sharing all inputs."""
    k = cfg["top_k"]
    users = cohort.tolist()
    pool_ids = set(pool.tolist())
    params = next(p for p in ppr_grid(cfg) if ppr_config_id(p) == cfg["hybrid"]["ppr_run"])
    order, _ = popularity_order(train, pool)
    fallback = order.tolist()
    target, broad = validation_targets(validation, cohort, pool, k)

    baseline = recommend(users, {}, fallback, k)
    popularity_report, popularity_ap = evaluate_lists(baseline, users, target, broad, len(pool), cfg)
    item_ids, probabilities, details = ppr_probabilities(train, params, cfg, cohort)
    scores, valid = score_candidates(item_ids, probabilities, pool)
    lists, _, counts, _ = ppr_lists(train, params, cfg, cohort, pool, fallback)
    ppr_report, ppr_ap = evaluate_lists(lists, users, target, broad, len(pool), cfg, popularity_ap)
    records = {
        "popularity": {"run_id": "popularity", **popularity_report},
        "ppr": {"run_id": cfg["hybrid"]["ppr_run"], "fallback": fallback_usage(counts, k), "graph": details, **ppr_report},
    }
    arrays = {}
    for run in hybrids:
        begin = time.perf_counter()
        lists, list_scores, counts = hybrid_lists(users, pool, scores, valid, order, run["ppr_weight"], run["rank_constant"], k)
        for user in users:
            validate_row(lists[user], pool_ids, k)
        report, ap = evaluate_lists(lists, users, target, broad, len(pool), cfg, popularity_ap)
        run_id = hybrid_config_id(run)
        records[run_id] = {"run_id": run_id, "params": run, **report,
                           "paired_difference_vs_ppr": paired_difference(ap, ppr_ap, cfg),
                           "fallback": fallback_usage(counts, k),
                           "execution": {"fuse_and_score_seconds": round(time.perf_counter() - begin, 2)}}
        arrays[run_id] = {"lists": np.array([lists[user] for user in users], dtype=np.int64), "ap": ap}
    return records, arrays


def main():
    timer = RunTimer("11_evaluate_hybrid")
    cfg = load_config()
    lock = json.loads((RESULTS_DIR / "hybrid_lock.json").read_text(encoding="utf-8"))
    if lock["grid"] != cfg["hybrid"]:
        raise SystemExit("config.json hybrid grid differs from the lock")

    tables = load_arrays(cfg, "split", f"tables_{cfg['watched']['primary_seconds']}")
    pool_arrays = load_arrays(cfg, "split", "pool")
    grid = hybrid_grid(cfg)
    records, arrays = compare_split(cfg, side(tables, "train"), side(tables, "validation"),
                                    pool_arrays["candidates"], pool_arrays["cohort"], grid)

    best_hybrid = None
    for run in grid:
        record = records[hybrid_config_id(run)]
        if best_hybrid is None or primary_of(record) > primary_of(best_hybrid):
            best_hybrid = record
    tolerance = cfg["hybrid"]["equal_score_tolerance"]
    hybrid_wins = primary_of(best_hybrid) > primary_of(records["ppr"]) + tolerance
    selected = best_hybrid["run_id"] if hybrid_wins else cfg["hybrid"]["ppr_run"]

    fold = stability_split(cfg)
    fold_records, _ = compare_split(cfg, fold["train"], fold["validation"], fold["pool"], fold["cohort"], [best_hybrid["params"]])

    save_stage(cfg, "hybrid", arrays, depends_on=("split", "models"))
    write_result(RESULTS_DIR / "hybrid_validation.json", {
        "selection_rule": lock["selection_rule"],
        "grid_sha256": lock["grid_sha256"],
        "selected_strategy": selected,
        "selected_is_hybrid": bool(hybrid_wins),
        "best_hybrid": best_hybrid["run_id"],
        "best_hybrid_minus_ppr_primary": primary_of(best_hybrid) - primary_of(records["ppr"]),
        "main_validation": records,
        "stability_fold": {
            "purpose": "diagnostic for the best hybrid; cannot override selection",
            "parsed_sources": sorted(csvio.parsed_sources),
            "results": fold_records,
        },
    })
    print("selected", selected)
    timer.finish()


if __name__ == "__main__":
    main()
