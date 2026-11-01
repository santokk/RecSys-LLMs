"""Personalized PageRank on a training-only programme co-view graph.

Nodes are programmes with a training interaction. Edge weight between
programmes i != j is

    W_ij = sum_u r_ui * r_uj / sum_k r_uk

so a heavy user does not dominate the graph by sheer list length. r_ui is the
interaction strength (binary, log seconds or seconds). P = W normalised by row.

A user's personalization vector s is their strengths normalised to sum to one.
The walk follows a link with probability d and restarts at s otherwise; mass on
a zero-degree programme also returns to s, so total probability stays at one:

    p_next = d * (p P + dangling_mass * s) + (1 - d) * s

Dense matrices are cheap here: the graph has only a few thousand nodes.
"""
import numpy as np


def interaction_strength(seconds, weighting, strength_scale):
    seconds = np.asarray(seconds, dtype=np.float64)
    if weighting == "binary":
        return np.ones_like(seconds)
    if weighting == "log_seconds":
        return np.log1p(seconds / strength_scale)
    if weighting == "seconds":
        return seconds / strength_scale
    raise ValueError(f"unknown weighting: {weighting}")


def coview_graph(table, weighting, strength_scale):
    """Return (item_ids, W) with a zero diagonal."""
    user_ids, user = np.unique(table["user"], return_inverse=True)
    item_ids, item = np.unique(table["item"], return_inverse=True)
    strength = interaction_strength(table["seconds"], weighting, strength_scale)
    matrix = np.zeros((len(user_ids), len(item_ids)))
    matrix[user, item] = strength
    scaled = matrix / np.sqrt(matrix.sum(axis=1, keepdims=True))
    weights = scaled.T @ scaled
    np.fill_diagonal(weights, 0.0)
    return item_ids, weights


def transition_matrix(weights):
    """Row-stochastic P and a mask of zero-degree nodes, whose rows stay zero."""
    outgoing = weights.sum(axis=1)
    dangling = outgoing == 0
    transition = np.divide(weights, outgoing[:, None], out=np.zeros_like(weights), where=~dangling[:, None])
    return transition, dangling


def personalization(table, users, item_ids, weighting, strength_scale):
    """Column per user, summing to one. A user with no history stays all zero."""
    seeds = np.zeros((len(item_ids), len(users)))
    column = np.searchsorted(users, table["user"]).clip(max=len(users) - 1)
    known = users[column] == table["user"]
    row = np.searchsorted(item_ids, table["item"])
    seeds[row[known], column[known]] = interaction_strength(table["seconds"], weighting, strength_scale)[known]
    totals = seeds.sum(axis=0, keepdims=True)
    return np.divide(seeds, totals, out=np.zeros_like(seeds), where=totals > 0)


def power_iteration(transition, dangling, seeds, link_probability, tolerance, max_iterations):
    """Iterate until the largest per-user L1 change is below `tolerance`."""
    forward = transition.T
    current = seeds.copy()
    for iteration in range(1, max_iterations + 1):
        dangling_mass = current[dangling].sum(axis=0, keepdims=True)
        updated = link_probability * (forward @ current + seeds * dangling_mass) + (1.0 - link_probability) * seeds
        change = float(np.abs(updated - current).sum(axis=0).max())
        current = updated
        if change < tolerance:
            return current, iteration, change
    raise RuntimeError(f"no convergence after {max_iterations} iterations (change {change})")


def score_candidates(item_ids, probabilities, candidates):
    """Probabilities for users x candidates, valid only where positive."""
    position = np.searchsorted(item_ids, candidates).clip(max=len(item_ids) - 1)
    known = item_ids[position] == candidates
    scores = np.zeros((probabilities.shape[1], len(candidates)))
    scores[:, known] = probabilities[position[known]].T
    return scores, scores > 0
