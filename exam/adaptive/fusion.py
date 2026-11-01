"""Per-user PPR weights from training history, applied to the existing RRF."""
import hashlib
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

from sweettv.baseline import lists_from_scores
from sweettv.hybrid import full_ranks, fuse, popularity_ranks

LOCK_FILE = "adaptive_experiment_lock.json"
CANDIDATE_FILE = "history_adaptive_submission.csv"
CANDIDATE_FOLDER = "adaptive_candidates"


def lock_hash(lock):
    body = {key: value for key, value in lock.items() if key != "lock_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_lock(results_dir):
    path = results_dir / LOCK_FILE
    if not path.exists():
        raise SystemExit("the experiment lock does not exist; run 30_lock_adaptive.py first")
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("lock_sha256") != lock_hash(lock) or lock.get("post_kaggle_feedback") is not True:
        raise SystemExit("the experiment lock is altered or unlabelled")
    return lock


def assign_weights(values, lower, upper, weights):
    """Low, middle or high PPR weight from a history value and two cut points."""
    return np.select([values < lower, values < upper], [weights[0], weights[1]], default=weights[2]).astype(np.float64)


def group_names(values, lower, upper):
    """History group per user, with zero history reported separately as cold."""
    return np.select([values == 0, values < lower, values < upper], ["cold", "low", "middle"], default="high")


def programme_counts(table, users):
    """Distinct positive training programmes per user; `table` has one row per pair."""
    found, counts = np.unique(table["user"], return_counts=True)
    position = np.searchsorted(found, users).clip(max=len(found) - 1)
    return np.where(found[position] == users, counts[position], 0)


def active_days(rows, users, tz_name):
    """Distinct Kyiv calendar dates of the airing starts of positive training rows."""
    zone = ZoneInfo(tz_name)
    starts, inverse = np.unique(rows["airing_start"], return_inverse=True)
    ordinals = np.array([datetime.fromtimestamp(int(t), zone).date().toordinal() for t in starts], dtype=np.int64)
    pairs = np.unique(np.stack([rows["user"].astype(np.int64), ordinals[inverse]], axis=1), axis=0)
    return programme_counts({"user": pairs[:, 0]}, users.astype(np.int64))


def adaptive_lists(users, pool, ppr_scores, ppr_valid, order, weights, constant, k):
    """RRF with one PPR weight per user. Users without a ranking take the fallback."""
    popular = popularity_ranks(pool, order)
    fused = np.zeros_like(ppr_scores)
    seeded = ppr_valid.any(axis=1)
    for row in np.flatnonzero(seeded):
        fused[row] = fuse(full_ranks(pool, ppr_scores[row], ppr_valid[row]), popular, weights[row], constant)
    valid = np.repeat(seeded[:, None], len(pool), axis=1)
    return lists_from_scores(users, pool, fused, valid, order.tolist(), k)


def selection_optimism(first, second, resamples, seed):
    """Bootstrap estimate of the gain from picking the better of two paired differences.

    In each user resample the better variant is picked; optimism is how far its
    resample mean sits above that same variant's mean on the original users.
    """
    picks = np.random.default_rng(seed).integers(0, len(first), size=(resamples, len(first)))
    resampled = np.stack([first[picks].mean(axis=1), second[picks].mean(axis=1)])
    originals = np.array([first.mean(), second.mean()])
    chosen = resampled.argmax(axis=0)
    apparent = resampled.max(axis=0)
    return float((apparent - originals[chosen]).mean())
