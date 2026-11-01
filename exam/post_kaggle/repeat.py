"""Personal-repeat scores and the immutable experiment lock."""
import hashlib
import json
import math

import numpy as np

from sweettv.baseline import lists_from_scores
from sweettv.hybrid import hybrid_lists
from sweettv.matching import watched_mask

SECONDS_PER_DAY = 86400
LOCK_FILE = "post_kaggle_improvement_lock.json"
PROTECTED_CANDIDATES = ("ppr_submission.csv", "popularity_submission.csv")


def lock_hash(lock):
    """SHA-256 of the canonical lock body, excluding the hash field itself."""
    body = {key: value for key, value in lock.items() if key != "lock_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_lock(results_dir):
    """Load the lock and refuse it if missing or altered."""
    path = results_dir / LOCK_FILE
    if not path.exists():
        raise SystemExit("the experiment lock does not exist; run 20_lock_post_kaggle.py first")
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("lock_sha256") != lock_hash(lock):
        raise SystemExit("the experiment lock was altered")
    if lock.get("post_kaggle_feedback") is not True:
        raise SystemExit("the experiment lock must be labelled post_kaggle_feedback")
    return lock


def positive_rows(airings, interactions, window, seconds, fraction):
    """Watched training rows, one per (user, airing), with the airing's start."""
    durations = (airings["end"] - airings["start"])[interactions["airing"]]
    watched = watched_mask(interactions["overlap_capped"], durations, seconds, fraction)
    mask = watched & (interactions["window"] == window)
    chosen = interactions["airing"][mask]
    return {
        "user": interactions["user"][mask],
        "item": airings["show_id"][chosen],
        "capped": interactions["overlap_capped"][mask],
        "airing_start": airings["start"][chosen],
    }


def repeat_scores(rows, users, pool, half_life_days, prediction_start):
    """Users x pool scores; only programmes the user watched in training are valid.

    With a half-life each row counts capped * exp(-ln(2) * age_days / half_life),
    where age runs from the airing's start to the prediction-window start.
    """
    if np.any(rows["airing_start"] >= prediction_start):
        raise ValueError("a training row starts at or after the prediction window")
    weight = rows["capped"].astype(np.float64)
    if half_life_days is not None:
        age_days = (prediction_start - rows["airing_start"]) / SECONDS_PER_DAY
        weight = weight * np.exp(-math.log(2) * age_days / half_life_days)
    order = np.argsort(users)
    ordered = users[order]
    row = np.searchsorted(ordered, rows["user"]).clip(max=len(users) - 1)
    column = np.searchsorted(pool, rows["item"]).clip(max=len(pool) - 1)
    keep = (ordered[row] == rows["user"]) & (pool[column] == rows["item"])
    scores = np.zeros((len(users), len(pool)))
    np.add.at(scores, (order[row[keep]], column[keep]), weight[keep])
    return scores, scores > 0


def variant_lists(variant, rows, users, pool, order, prediction_start, k):
    """One fixed family member through the shared list-building paths."""
    half_life = variant["half_life_days"]
    scores, valid = repeat_scores(rows, users, pool, half_life, prediction_start)
    if variant["rrf_with_popularity"]:
        settings = variant["rrf_with_popularity"]
        return hybrid_lists(users.tolist(), pool, scores, valid, order, settings["repeat_weight"], settings["rank_constant"], k)
    return lists_from_scores(users.tolist(), pool, scores, valid, order.tolist(), k)


def tables_match(rows, table):
    """True when per-airing rows aggregate exactly to a verified user-item table."""
    totals = {}
    for user, item, capped in zip(rows["user"].tolist(), rows["item"].tolist(), rows["capped"].tolist()):
        totals[(user, item)] = totals.get((user, item), 0) + capped
    return totals == dict(zip(zip(table["user"].tolist(), table["item"].tolist()), table["seconds"].tolist()))
