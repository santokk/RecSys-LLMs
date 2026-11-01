"""Lock the validation target, the model grids and the two report users.

Runs before any model. Report users are chosen from training statistics alone,
so they cannot be picked for how well a model happens to serve them.
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from sweettv.artifacts import load_arrays, load_text, write_result
from sweettv.config import RESULTS_DIR, load_config
from sweettv.runlog import RunTimer
from sweettv.split import side

PROTOCOL_FILE = "protocol_lock.json"


def grid_hash(cfg):
    return hashlib.sha256(json.dumps(cfg["models"], sort_keys=True).encode()).hexdigest()


def training_statistics(train, user, user_id):
    mine = train["user"] == user
    return {
        "user_id": user_id,
        "distinct_positive_training_programmes": int(mine.sum()),
        "watched_training_airings": int(train["airings"][mine].sum()),
        "watched_training_seconds": int(train["seconds"][mine].sum()),
    }


def main():
    timer = RunTimer("05_lock_protocol")
    cfg = load_config()
    train = side(load_arrays(cfg, "split", f"tables_{cfg['watched']['primary_seconds']}"), "train")
    cohort = load_arrays(cfg, "split", "pool")["cohort"]
    user_ids = load_text(cfg, "preprocess", "users.txt").split()
    counts = np.bincount(train["user"], minlength=len(user_ids))
    active = [user for user in cohort.tolist() if counts[user] > 0]
    light = min(active, key=lambda user: (counts[user], int(user_ids[user])))
    heavy = min(active, key=lambda user: (-counts[user], int(user_ids[user])))
    lock = {
        "primary_target": {
            "start_from": "validation user-programme pairs satisfying overlap >= min(300 seconds, 0.5 * airing duration)",
            "restrict_to": "the common validation candidate pool",
            "rank_by": "summed capped watched seconds, descending",
            "tie_break": "ascending numeric tv_show_id",
            "keep": "at most five distinct ids",
            "empty_target": "AP@5 = 0",
            "basis": "competition wording '5 most preferred' and its five-id submission format",
        },
        "selection_metric": "MAP@5 across all 1,260 users, empty targets scoring zero",
        "selection_tie_break": "first configuration in declared grid order",
        "grids": cfg["models"],
        "grids_sha256": grid_hash(cfg),
        "report_users": {
            "rule": {
                "heavy": "cohort user with the most distinct positive training programmes",
                "light": "cohort user with the fewest, but at least one, distinct positive training programmes",
                "tie_break": "numerically ascending user_id",
            },
            "heavy": training_statistics(train, heavy, user_ids[heavy]),
            "light": training_statistics(train, light, user_ids[light]),
        },
    }
    write_result(RESULTS_DIR / PROTOCOL_FILE, lock)
    print(json.dumps({"grids_sha256": lock["grids_sha256"], "report_users": lock["report_users"]}, indent=1))
    timer.finish()


if __name__ == "__main__":
    main()
