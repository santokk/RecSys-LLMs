"""Popularity baseline, deterministic ranking and the shared five-item fallback."""
import numpy as np


def rank_items(item_ids, scores):
    """Item ids by descending score, ties broken by ascending numeric id."""
    item_ids = np.asarray(item_ids, dtype=np.int64)
    return item_ids[np.lexsort((item_ids, -np.asarray(scores, dtype=np.float64)))]


def popularity_scores(table, candidates):
    """Distinct training users with a watched interaction, for each candidate."""
    item, count = np.unique(table["item"], return_counts=True)
    scores = np.zeros(len(candidates), dtype=np.int64)
    position = np.searchsorted(candidates, item)
    found = (position < len(candidates)) & (candidates[position.clip(max=len(candidates) - 1)] == item)
    scores[position[found]] = count[found]
    return scores


def fill_to_k(ranked, fallback, k=5):
    """First k distinct ids of `ranked`, topped up from `fallback` in order."""
    chosen, seen = [], set()
    for source in (ranked, fallback):
        for item in source:
            item = int(item)
            if item not in seen:
                seen.add(item)
                chosen.append(item)
                if len(chosen) == k:
                    return chosen
    raise ValueError(f"cannot fill {k} distinct recommendations")


def recommend(users, personal_ranking, fallback, k=5):
    """Lists for every user; a user missing from `personal_ranking` is cold."""
    return {u: fill_to_k(personal_ranking.get(u, ()), fallback, k) for u in users}


def lists_from_scores(users, candidates, scores, valid, fallback, k=5):
    """Shared path from a users x candidates score matrix to five-item lists.

    Returns the lists, each list's model scores (NaN on fallback slots) and how
    many slots the model filled itself.
    """
    personal, list_scores = {}, np.full((len(users), k), np.nan)
    personal_counts = np.zeros(len(users), dtype=np.int64)
    for row, user in enumerate(users):
        ids, values = candidates[valid[row]], scores[row][valid[row]]
        top = np.lexsort((ids, -values))[:k]
        personal[user] = ids[top].tolist()
        personal_counts[row] = len(top)
        list_scores[row, :len(top)] = values[top]
    return recommend(users, personal, fallback, k), list_scores, personal_counts
