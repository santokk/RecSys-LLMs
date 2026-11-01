"""Temporal windows, user-programme tables and candidate construction."""
import numpy as np

from .matching import watched_mask

TRAIN, VALIDATION = 0, 1


def window_bounds(cfg, clock):
    """Training and validation windows as [start, end) UTC pairs."""
    windows = cfg["windows"]
    return [(clock.view_ts(windows[name]["start"]), clock.view_ts(windows[name]["end"]))
            for name in ("train", "validation")]


def user_item_table(user, item, seconds):
    """One row per (user, item): positive airing count and summed seconds."""
    if len(user) == 0:
        return {"user": user, "item": item, "airings": np.zeros(0, dtype=np.int64), "seconds": seconds}
    order = np.lexsort((item, user))
    user, item, seconds = user[order], item[order], seconds[order]
    new_pair = np.r_[True, (user[1:] != user[:-1]) | (item[1:] != item[:-1])]
    first = np.flatnonzero(new_pair)
    return {
        "user": user[first],
        "item": item[first],
        "airings": np.diff(np.r_[first, len(user)]),
        "seconds": np.add.reduceat(seconds, first),
    }


def candidate_pool(airings, window, eligible_channels):
    """Sorted programme ids airing inside `window` on an eligible channel.

    `show_id != 0` is enforced here, so unlabelled airtime can never be a candidate.
    """
    lower, upper = window
    airs = (airings["start"] < upper) & (airings["end"] > lower) & (airings["show_id"] != 0)
    return np.unique(airings["show_id"][airs & np.isin(airings["channel"], eligible_channels)])


def watched_tables(airings, interactions, seconds, fraction):
    """Train and validation user-programme tables under one watched threshold."""
    durations = (airings["end"] - airings["start"])[interactions["airing"]]
    items = airings["show_id"][interactions["airing"]]
    watched = watched_mask(interactions["overlap_capped"], durations, seconds, fraction)
    tables = {}
    for window, name in ((TRAIN, "train"), (VALIDATION, "validation")):
        mask = watched & (interactions["window"] == window)
        tables[name] = user_item_table(interactions["user"][mask], items[mask], interactions["overlap_capped"][mask])
    return tables


def side(tables, name):
    """One window's columns from a stored `tables_*` archive."""
    return {key[len(name) + 1:]: value for key, value in tables.items() if key.startswith(name + "_")}


def broad_watched_sets(table, users, pool):
    """Secondary diagnostic: every watched pool id per user, however briefly."""
    eligible = np.isin(table["item"], pool)
    relevant = {int(u): set() for u in users}
    for u, item in zip(table["user"][eligible].tolist(), table["item"][eligible].tolist()):
        if u in relevant:
            relevant[u].add(item)
    return relevant


def preference_target(table, users, pool, k):
    """Primary selection target: each user's top-k watched pool ids.

    `table` must already satisfy the watched rule. Ids are ranked by summed
    capped seconds, descending, then by ascending numeric id. A user with no
    eligible id has an empty target and scores AP@k = 0.
    """
    eligible = np.isin(table["item"], pool)
    user, item, seconds = table["user"][eligible], table["item"][eligible], table["seconds"][eligible]
    order = np.lexsort((item, -seconds, user))
    target = {int(u): set() for u in users}
    for u, i in zip(user[order].tolist(), item[order].tolist()):
        if u in target and len(target[u]) < k:
            target[u].add(i)
    return target
