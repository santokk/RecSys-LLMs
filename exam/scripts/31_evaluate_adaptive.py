"""EXPLORATORY, POST-KAGGLE: evaluate the two locked variants on both folds.

Kaggle scores are read nowhere here. Frozen PPR settings are recomputed only from
each fold's training table and must reproduce the stored fixed-fusion result.
"""
import os

for name in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[name] = "1"

import json
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from adaptive.fusion import (active_days, adaptive_lists, assign_weights, group_names, programme_counts, read_lock,
                             selection_optimism)
from post_kaggle.repeat import positive_rows, tables_match
from sweettv import csvio
from sweettv.artifacts import load_arrays, save_stage, write_result
from sweettv.baseline import recommend
from sweettv.config import RESULTS_DIR, load_config
from sweettv.csvio import read_cohort
from sweettv.evaluation import evaluate_lists, paired_difference, validation_targets
from sweettv.metrics import average_precision_at_k, bootstrap_interval, per_user
from sweettv.pipeline import build_interactions
from sweettv.ppr import score_candidates
from sweettv.runner import fallback_usage, popularity_order, ppr_config_id, ppr_grid, ppr_probabilities
from sweettv.split import TRAIN, VALIDATION, candidate_pool, side, watched_tables
from sweettv.submission import validate_row
from sweettv.timeutil import WallClock

CONSTANT = 10


def primary(report):
    return report["primary_top5_preference"]["map_at_5_all_users"]


def history(variant, train, rows, cohort, cfg):
    if variant["label"] == "A":
        return programme_counts(train, cohort)
    return active_days(rows, cohort, cfg["timezone"])


def group_report(variant, values, ap, fixed_ap, popularity_ap, broad_ap, fixed_broad_ap, cfg):
    groups = group_names(values, variant["lower"], variant["upper"])
    report = {}
    for group in ("cold", "low", "middle", "high"):
        mask = groups == group
        entry = {"users": int(mask.sum())}
        if mask.any():
            difference = ap[mask] - fixed_ap[mask]
            entry.update({
                "primary_map_variant": float(ap[mask].mean()),
                "primary_map_fixed": float(fixed_ap[mask].mean()),
                "primary_map_popularity": float(popularity_ap[mask].mean()),
                "broad_map_variant": float(broad_ap[mask].mean()),
                "broad_map_fixed": float(fixed_broad_ap[mask].mean()),
                "paired_vs_fixed_mean": float(difference.mean()),
                "paired_vs_fixed_95ci": bootstrap_interval(difference, cfg["bootstrap_resamples"], cfg["seed"]),
                "users_better": int((difference > 0).sum()),
                "users_worse": int((difference < 0).sum()),
            })
        report[group] = entry
    return report


def evaluate_fold(lock, cfg, train, validation, rows, pool, cohort, stored_fixed=None):
    k = cfg["top_k"]
    users = cohort.tolist()
    pool_ids = set(pool.tolist())
    params = next(p for p in ppr_grid(cfg) if ppr_config_id(p) == cfg["hybrid"]["ppr_run"])
    order, _ = popularity_order(train, pool)
    target, broad = validation_targets(validation, cohort, pool, k)
    item_ids, probabilities, _ = ppr_probabilities(train, params, cfg, cohort)
    scores, valid = score_candidates(item_ids, probabilities, pool)

    popularity_lists = recommend(users, {}, order.tolist(), k)
    popularity_report, popularity_ap = evaluate_lists(popularity_lists, users, target, broad, len(pool), cfg)
    fixed_lists, _, fixed_counts = adaptive_lists(users, pool, scores, valid, order, np.full(len(users), 0.8), CONSTANT, k)
    fixed_report, fixed_ap = evaluate_lists(fixed_lists, users, target, broad, len(pool), cfg, popularity_ap)
    fixed_broad_ap = per_user(average_precision_at_k, fixed_lists, broad, users, k)
    if stored_fixed is not None and np.array([fixed_lists[user] for user in users]).tolist() != stored_fixed.tolist():
        raise SystemExit("all-0.8 adaptive code does not reproduce the stored fixed fusion lists")

    records, differences, arrays = {}, {}, {}
    for variant in lock["variants"]:
        started = time.perf_counter()
        values = history(variant, train, rows, cohort, cfg)
        weights = assign_weights(values, variant["lower"], variant["upper"], variant["weights"])
        lists, list_scores, counts = adaptive_lists(users, pool, scores, valid, order, weights, CONSTANT, k)
        for user in users:
            validate_row(lists[user], pool_ids, k)
        report, ap = evaluate_lists(lists, users, target, broad, len(pool), cfg, popularity_ap)
        broad_ap = per_user(average_precision_at_k, lists, broad, users, k)
        name = variant["name"]
        differences[name] = ap - fixed_ap
        records[name] = {
            "label": variant["label"], **report,
            "paired_difference_vs_fixed": paired_difference(ap, fixed_ap, cfg),
            "broad_paired_difference_vs_fixed": paired_difference(broad_ap, fixed_broad_ap, cfg),
            "by_history_group": group_report(variant, values, ap, fixed_ap, popularity_ap, broad_ap, fixed_broad_ap, cfg),
            "users_with_a_changed_list": sum(lists[user] != fixed_lists[user] for user in users),
            "fallback": fallback_usage(counts, k),
            "execution": {"seconds": round(time.perf_counter() - started, 2)},
        }
        arrays[name] = {"lists": np.array([lists[user] for user in users], dtype=np.int64), "list_scores": list_scores,
                        "weights": weights, "history": values, "ap": ap}
    return {"fixed": fixed_report, "fixed_fallback": fallback_usage(fixed_counts, k), "popularity": popularity_report,
            "variants": records}, differences, arrays


def main():
    started = time.perf_counter()
    cfg = load_config()
    lock = read_lock(RESULTS_DIR)
    threshold, fraction = cfg["watched"]["primary_seconds"], cfg["watched"]["airing_fraction"]
    clock = WallClock(cfg["timezone"])

    airings = load_arrays(cfg, "preprocess", "airings")
    interactions = load_arrays(cfg, "preprocess", "interactions")
    tables = load_arrays(cfg, "split", f"tables_{threshold}")
    pool_arrays = load_arrays(cfg, "split", "pool")
    train, validation = side(tables, "train"), side(tables, "validation")
    rows = positive_rows(airings, interactions, TRAIN, threshold, fraction)
    if not tables_match(rows, train):
        raise SystemExit("per-airing rows do not aggregate to the main training table")
    stored_fixed = load_arrays(cfg, "hybrid", "hybrid_w0.8_k10")["lists"]
    main_result, main_differences, main_arrays = evaluate_fold(lock, cfg, train, validation, rows, pool_arrays["candidates"],
                                                               pool_arrays["cohort"], stored_fixed)

    windows = [(clock.view_ts(cfg["stability"][part]["start"]), clock.view_ts(cfg["stability"][part]["end"]))
               for part in ("train", "validation")]
    fold_airings, fold_interactions, coverage, user_ids, _ = build_interactions(cfg, clock, windows)
    fold_tables = watched_tables(fold_airings, fold_interactions, threshold, fraction)
    fold_rows = positive_rows(fold_airings, fold_interactions, TRAIN, threshold, fraction)
    if not tables_match(fold_rows, fold_tables["train"]):
        raise SystemExit("per-airing rows do not aggregate to the fold training table")
    position = {user: index for index, user in enumerate(user_ids)}
    fold_cohort = np.array([position[user] for user in read_cohort(cfg)], dtype=np.int32)
    fold_pool = candidate_pool(fold_airings, windows[VALIDATION], coverage["train_channels"])
    fold_result, _, fold_arrays = evaluate_fold(lock, cfg, fold_tables["train"], fold_tables["validation"], fold_rows, fold_pool, fold_cohort)

    earlier = json.loads((RESULTS_DIR / "hybrid_validation.json").read_text(encoding="utf-8"))
    if primary(main_result["fixed"]) != primary(earlier["main_validation"]["hybrid_w0.8_k10"]):
        raise SystemExit("fixed fusion differs from the verified main value")
    if primary(fold_result["fixed"]) != primary(earlier["stability_fold"]["results"]["hybrid_w0.8_k10"]):
        raise SystemExit("fixed fusion differs from the verified fold value")

    decisions = {}
    for variant in lock["variants"]:
        name = variant["name"]
        beats_main = primary(main_result["variants"][name]) > primary(main_result["fixed"])
        beats_fold = primary(fold_result["variants"][name]) > primary(fold_result["fixed"])
        decisions[name] = {"beats_fixed_on_main": bool(beats_main), "beats_fixed_on_fold": bool(beats_fold), "qualifies": bool(beats_main and beats_fold)}
    selected = None
    for variant in lock["variants"]:
        name = variant["name"]
        if decisions[name]["qualifies"] and (selected is None or primary(main_result["variants"][name]) > primary(main_result["variants"][selected])):
            selected = name

    names = [variant["name"] for variant in lock["variants"]]
    arrays = {f"main_{key}": value for key, value in main_arrays.items()}
    arrays.update({f"fold_{key}": value for key, value in fold_arrays.items()})
    save_stage(cfg, "adaptive", arrays, depends_on=("split", "hybrid", "refit"))
    write_result(RESULTS_DIR / "adaptive_validation.json", {
        "post_kaggle_feedback": True,
        "exploratory": True,
        "label": "EXPLORATORY POST-KAGGLE result, produced after four Kaggle score pairs were observed; not pre-Kaggle evidence",
        "target_note": lock["target_note"],
        "lock_sha256": lock["lock_sha256"],
        "kaggle_scores_used_in_acceptance": False,
        "parsed_sources": sorted(csvio.parsed_sources),
        "main_split": main_result,
        "stability_fold": fold_result,
        "decisions": decisions,
        "selected_variant": selected,
        "selection_optimism": {
            "method": "paired user bootstrap on main-split differences from fixed fusion: pick the better variant in each "
                      "resample, then measure its resample mean minus its original mean",
            "main_map_estimate": selection_optimism(main_differences[names[0]], main_differences[names[1]],
                                                    cfg["bootstrap_resamples"], cfg["seed"]),
            "resamples": cfg["bootstrap_resamples"],
            "seed": cfg["seed"],
            "not_captured": "the hypothesis and thresholds were chosen after Kaggle feedback, and the fold's training weeks "
                            "include most of the main validation weeks, so the folds are not independent",
        },
        "execution": {"seconds": round(time.perf_counter() - started, 1),
                      "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20), 1)},
    })
    print("decisions", decisions, "selected", selected)


if __name__ == "__main__":
    main()
