"""Apply the watched rule, build user-programme tables and the candidate pool."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from sweettv.artifacts import load_arrays, load_text, save_stage, write_result
from sweettv.config import RESULTS_DIR, load_config
from sweettv.csvio import read_cohort
from sweettv.runlog import RunTimer
from sweettv.split import TRAIN, VALIDATION, candidate_pool, watched_tables, window_bounds
from sweettv.timeutil import WallClock


def pair_keys(table):
    return table["user"].astype(np.int64) << 32 | table["item"]


def relevance_audit(train, validation, pool, population, train_view_users):
    """Measured counts for one user population at one threshold."""
    eligible = np.isin(validation["item"], pool)
    members = np.isin(validation["user"], population)
    positive_users = np.unique(validation["user"][members])
    relevant_users, relevant_sizes = np.unique(validation["user"][members & eligible], return_counts=True)
    trained = np.unique(train["user"])
    relevant_keys = pair_keys(validation)[members & eligible]
    item_trained = np.isin(validation["item"][members & eligible], train["item"])
    return {
        "users": int(len(population)),
        "without_any_train_view": int((~np.isin(population, train_view_users)).sum()),
        "cold_no_train_positive": int((~np.isin(population, trained)).sum()),
        "without_validation_positive": int(len(population) - len(positive_users)),
        "without_eligible_watched_item": int(len(population) - len(relevant_users)),
        "with_eligible_watched_item": int(len(relevant_users)),
        "validation_positive_pairs": int(members.sum()),
        "eligible_watched_pairs": int(len(relevant_keys)),
        "positive_pairs_outside_pool": int((members & ~eligible).sum()),
        "watched_set_size_median": float(np.median(relevant_sizes)) if len(relevant_sizes) else 0.0,
        "watched_set_size_mean": float(relevant_sizes.mean()) if len(relevant_sizes) else 0.0,
        "watched_pairs_repeating_user_train_item": int(np.isin(relevant_keys, pair_keys(train)).sum()),
        "watched_pairs_on_cold_items": int((~item_trained).sum()),
    }


def main():
    timer = RunTimer("02_build_split")
    cfg = load_config()
    clock = WallClock(cfg["timezone"])
    windows = window_bounds(cfg, clock)
    airings = load_arrays(cfg, "preprocess", "airings")
    interactions = load_arrays(cfg, "preprocess", "interactions")
    coverage = load_arrays(cfg, "preprocess", "coverage")
    users = load_text(cfg, "preprocess", "users.txt").split()

    position = {user: index for index, user in enumerate(users)}
    cohort = np.array([position[user] for user in read_cohort(cfg)], dtype=np.int32)
    everyone = np.arange(len(users), dtype=np.int32)
    fraction = cfg["watched"]["airing_fraction"]
    thresholds = sorted({cfg["watched"]["primary_seconds"], *cfg["watched"]["sensitivity_seconds"]})

    pool = candidate_pool(airings, windows[VALIDATION], coverage["train_channels"])
    pool_any_channel = candidate_pool(airings, windows[VALIDATION], np.unique(airings["channel"]))
    train_window_ids = candidate_pool(airings, windows[TRAIN], np.unique(airings["channel"]))

    tables, by_threshold = {}, {}
    for seconds in thresholds:
        sides = watched_tables(airings, interactions, seconds, fraction)
        entry = by_threshold.setdefault(str(seconds), {})
        for name, table in sides.items():
            entry[name] = {
                "watched_airing_rows": int(table["airings"].sum()),
                "user_programme_pairs": int(len(table["user"])),
                "users": int(len(np.unique(table["user"]))),
                "programmes": int(len(np.unique(table["item"]))),
            }
        tables[f"tables_{seconds}"] = {f"{name}_{key}": value for name, table in sides.items() for key, value in table.items()}
        entry["pool_items_cold"] = int((~np.isin(pool, sides["train"]["item"])).sum())
        entry["cohort_1260"] = relevance_audit(sides["train"], sides["validation"], pool, cohort, coverage["train_users"])
        entry["all_log_users"] = relevance_audit(sides["train"], sides["validation"], pool, everyone, coverage["train_users"])

    audit = {
        "definition": "overlap_capped >= min(threshold_seconds, airing_fraction * airing_seconds)",
        "airing_fraction": fraction,
        "primary_seconds": cfg["watched"]["primary_seconds"],
        "candidate_pool": {
            "rule": "labelled ids with an airing overlapping the validation window on a channel that has a training-window view",
            "size": int(len(pool)),
            "size_without_channel_restriction": int(len(pool_any_channel)),
            "train_view_channels": int(len(coverage["train_channels"])),
            "ids_not_airing_in_training_window": int((~np.isin(pool, train_window_ids)).sum()),
        },
        "by_threshold": by_threshold,
    }
    arrays = {**tables, "pool": {"candidates": pool, "cohort": cohort}}
    save_stage(cfg, "split", arrays, depends_on=("preprocess",))
    write_result(RESULTS_DIR / "split_audit.json", audit)
    timer.finish()


if __name__ == "__main__":
    main()
