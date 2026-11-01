"""POST-KAGGLE: evaluate the fixed repeat family and apply the locked acceptance rule.

Kaggle scores are read nowhere in this script. Selection and acceptance use the
main validation split and the stability fold only.
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

from post_kaggle.repeat import positive_rows, read_lock, tables_match, variant_lists
from sweettv import csvio
from sweettv.artifacts import load_arrays, save_stage, write_result
from sweettv.baseline import recommend
from sweettv.config import RESULTS_DIR, load_config
from sweettv.csvio import read_cohort
from sweettv.evaluation import evaluate_lists, paired_difference, validation_targets
from sweettv.pipeline import build_interactions
from sweettv.runner import fallback_usage, popularity_order
from sweettv.split import TRAIN, VALIDATION, candidate_pool, side, watched_tables
from sweettv.submission import validate_row
from sweettv.timeutil import WallClock


def primary(report):
    return report["primary_top5_preference"]["map_at_5_all_users"]


def peak_mib():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20), 1)


def run_variant(variant, rows, split, order, baseline_ap, cfg):
    k = cfg["top_k"]
    users = split["cohort"].tolist()
    started = time.perf_counter()
    lists, list_scores, counts = variant_lists(variant, rows, split["cohort"], split["pool"], order, split["prediction_start"], k)
    pool_ids = set(split["pool"].tolist())
    for user in users:
        validate_row(lists[user], pool_ids, k)
    report, ap = evaluate_lists(lists, users, split["target"], split["broad"], len(split["pool"]), cfg, baseline_ap)
    record = {"name": variant["name"], "definition": variant, **report, "fallback": fallback_usage(counts, k),
              "users_with_any_repeat_candidate": int((counts > 0).sum()),
              "execution": {"seconds": round(time.perf_counter() - started, 2)}}
    return record, {"lists": np.array([lists[user] for user in users], dtype=np.int64), "list_scores": list_scores, "ap": ap}


def main():
    started = time.perf_counter()
    cfg = load_config()
    lock = read_lock(RESULTS_DIR)
    k = cfg["top_k"]
    threshold, fraction = cfg["watched"]["primary_seconds"], cfg["watched"]["airing_fraction"]
    clock = WallClock(cfg["timezone"])

    airings = load_arrays(cfg, "preprocess", "airings")
    interactions = load_arrays(cfg, "preprocess", "interactions")
    tables = load_arrays(cfg, "split", f"tables_{threshold}")
    pool_arrays = load_arrays(cfg, "split", "pool")
    train, validation = side(tables, "train"), side(tables, "validation")
    pool, cohort = pool_arrays["candidates"], pool_arrays["cohort"]
    main_rows = positive_rows(airings, interactions, TRAIN, threshold, fraction)
    if not tables_match(main_rows, train):
        raise SystemExit("per-airing rows do not aggregate to the verified main training table")
    order, _ = popularity_order(train, pool)
    if order.tolist() != load_arrays(cfg, "popularity", "popularity")["fallback_order"].tolist():
        raise SystemExit("popularity order differs from the verified stage")

    users = cohort.tolist()
    target, broad = validation_targets(validation, cohort, pool, k)
    main_split = {"cohort": cohort, "pool": pool, "target": target, "broad": broad,
                  "prediction_start": clock.view_ts(cfg["windows"]["validation"]["start"])}
    popularity_report, popularity_ap = evaluate_lists(recommend(users, {}, order.tolist(), k), users, target, broad, len(pool), cfg)
    ppr_ap = load_arrays(cfg, "models", cfg["hybrid"]["ppr_run"])["ap"]

    records, arrays = [], {}
    for variant in lock["repeat_family"]:
        record, stored = run_variant(variant, main_rows, main_split, order, popularity_ap, cfg)
        record["paired_difference_vs_ppr"] = paired_difference(stored["ap"], ppr_ap, cfg)
        records.append(record)
        arrays[variant["name"]] = stored

    selected = None
    for record in records:
        if selected is None or primary(record) > primary(selected):
            selected = record
    variant = next(v for v in lock["repeat_family"] if v["name"] == selected["name"])

    windows = [(clock.view_ts(cfg["stability"][part]["start"]), clock.view_ts(cfg["stability"][part]["end"]))
               for part in ("train", "validation")]
    fold_airings, fold_interactions, coverage, user_ids, _ = build_interactions(cfg, clock, windows)
    fold_tables = watched_tables(fold_airings, fold_interactions, threshold, fraction)
    fold_rows = positive_rows(fold_airings, fold_interactions, TRAIN, threshold, fraction)
    if not tables_match(fold_rows, fold_tables["train"]):
        raise SystemExit("per-airing rows do not aggregate to the fold training table")
    fold_pool = candidate_pool(fold_airings, windows[VALIDATION], coverage["train_channels"])
    position = {user: index for index, user in enumerate(user_ids)}
    fold_cohort = np.array([position[user] for user in read_cohort(cfg)], dtype=np.int32)
    fold_order, _ = popularity_order(fold_tables["train"], fold_pool)
    fold_users = fold_cohort.tolist()
    fold_target, fold_broad = validation_targets(fold_tables["validation"], fold_cohort, fold_pool, k)
    fold_split = {"cohort": fold_cohort, "pool": fold_pool, "target": fold_target, "broad": fold_broad,
                  "prediction_start": windows[VALIDATION][0]}
    fold_popularity, fold_popularity_ap = evaluate_lists(recommend(fold_users, {}, fold_order.tolist(), k), fold_users,
                                                         fold_target, fold_broad, len(fold_pool), cfg)
    fold_record, fold_stored = run_variant(variant, fold_rows, fold_split, fold_order, fold_popularity_ap, cfg)
    arrays["fold_selected"] = fold_stored

    earlier_fold = json.loads((RESULTS_DIR / "stability_fold.json").read_text(encoding="utf-8"))["results"]
    if primary(fold_popularity) != primary(earlier_fold["popularity"]):
        raise SystemExit("fold popularity differs from the verified Step 3 value")

    beats_main = primary(selected) > primary(popularity_report)
    beats_fold = primary(fold_record) > primary(fold_popularity)
    accepted = bool(beats_main and beats_fold)
    save_stage(cfg, "post_kaggle", arrays, depends_on=("split", "models", "refit"))
    write_result(RESULTS_DIR / "post_kaggle_validation.json", {
        "post_kaggle_feedback": True,
        "label": "POST-KAGGLE result: produced after PPR and popularity Kaggle scores were observed; not pre-Kaggle evidence",
        "lock_sha256": lock["lock_sha256"],
        "kaggle_scores_used_in_selection_or_acceptance": False,
        "parsed_sources": sorted(csvio.parsed_sources),
        "main_validation_references": {
            "popularity": primary(popularity_report),
            "ppr": primary(json.loads((RESULTS_DIR / "hybrid_validation.json").read_text(encoding="utf-8"))["main_validation"]["ppr"]),
            "hybrid_w0.8_k10_pre_kaggle": primary(json.loads((RESULTS_DIR / "hybrid_validation.json").read_text(encoding="utf-8"))["main_validation"]["hybrid_w0.8_k10"]),
        },
        "repeat_family_main_validation": records,
        "selected_variant": selected["name"],
        "stability_fold": {
            "variant": selected["name"],
            "popularity": primary(fold_popularity),
            "ppr": primary(earlier_fold["ppr"]),
            "selected_variant_record": fold_record,
        },
        "acceptance": {
            "main_map_strictly_above_popularity": bool(beats_main),
            "fold_map_strictly_above_popularity": bool(beats_fold),
            "accepted": accepted,
        },
        "execution": {"seconds": round(time.perf_counter() - started, 1), "peak_rss_mib": peak_mib()},
    })
    print("selected", selected["name"], "accepted", accepted)


if __name__ == "__main__":
    main()
