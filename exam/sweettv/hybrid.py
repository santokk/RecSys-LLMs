"""Reciprocal-rank fusion of a frozen PPR ranking and the popularity ranking.

    score(i) = w / (K + rank_ppr(i)) + (1 - w) / (K + rank_popularity(i))

Ranks start at 1 and cover every candidate. A seeded user's candidates that PPR
cannot reach rank after the reachable ones, by ascending id. A user with no
training history has no PPR ranking at all, so fusing would only blend id order
into the list; such a user takes the common popularity fallback instead.
"""
import numpy as np

from .baseline import lists_from_scores


def full_ranks(candidates, scores, valid):
    """Rank 1..n per candidate: valid first by score, then ties and the rest by id."""
    order = np.lexsort((candidates, -np.where(valid, scores, -np.inf)))
    ranks = np.empty(len(candidates), dtype=np.int64)
    ranks[order] = np.arange(1, len(candidates) + 1)
    return ranks


def popularity_ranks(candidates, order):
    """Ranks of `candidates` under the full popularity order."""
    ranks = np.empty(len(candidates), dtype=np.int64)
    ranks[np.searchsorted(candidates, order)] = np.arange(1, len(order) + 1)
    return ranks


def fuse(ppr_rank, popularity_rank, weight, constant):
    return weight / (constant + ppr_rank) + (1 - weight) / (constant + popularity_rank)


def hybrid_scores(candidates, ppr_scores, ppr_valid, order, weight, constant):
    """Users x candidates fused scores, valid only for users with a PPR ranking."""
    popular = popularity_ranks(candidates, order)
    fused = np.zeros_like(ppr_scores)
    seeded = ppr_valid.any(axis=1)
    for row in np.flatnonzero(seeded):
        fused[row] = fuse(full_ranks(candidates, ppr_scores[row], ppr_valid[row]), popular, weight, constant)
    return fused, np.repeat(seeded[:, None], len(candidates), axis=1)


def hybrid_lists(users, candidates, ppr_scores, ppr_valid, order, weight, constant, k):
    """Lists, list scores and personal counts through the shared ranking path."""
    fused, valid = hybrid_scores(candidates, ppr_scores, ppr_valid, order, weight, constant)
    return lists_from_scores(users, candidates, fused, valid, order.tolist(), k)
