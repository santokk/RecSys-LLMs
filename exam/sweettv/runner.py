"""Train-only model runs behind one interface.

Every function here takes the training table alone. Validation data can only
reach `evaluate_lists`, which is called after the lists exist.
"""
import itertools

import numpy as np

from . import mf, ppr
from .baseline import lists_from_scores, popularity_scores, rank_items
from .csvio import read_cohort, require_test_schedule_unread
from .pipeline import build_interactions
from .split import VALIDATION, candidate_pool, watched_tables
from .timeutil import WallClock


def popularity_order(train, pool):
    scores = popularity_scores(train, pool)
    return rank_items(pool, scores), scores


def mf_grid(cfg):
    grid = cfg["models"]["mf"]
    return [{"factors": f, "confidence_alpha": a, "regularization": r}
            for f, a, r in itertools.product(grid["factors"], grid["confidence_alpha"], grid["regularization"])]


def ppr_grid(cfg):
    grid = cfg["models"]["ppr"]
    return [{"link_probability": d, "interaction_weighting": w}
            for d, w in itertools.product(grid["link_probability"], grid["interaction_weighting"])]


def hybrid_grid(cfg):
    grid = cfg["hybrid"]
    return [{"ppr_weight": w, "rank_constant": k} for w, k in itertools.product(grid["ppr_weight"], grid["rank_constant"])]


def mf_config_id(params):
    return f"mf_f{params['factors']}_a{params['confidence_alpha']}_r{params['regularization']}"


def ppr_config_id(params):
    return f"ppr_d{params['link_probability']}_{params['interaction_weighting']}"


def hybrid_config_id(params):
    return f"hybrid_w{params['ppr_weight']}_k{params['rank_constant']}"


def mf_checkpoints(train, params, cfg, users, pool, fallback, checkpoints):
    """Yield (iteration, lists, list_scores, personal_counts, losses) at each checkpoint."""
    require_test_schedule_unread(cfg)
    settings = cfg["models"]["mf"]
    losses = []
    states = mf.wals_iterations(train, params["factors"], params["confidence_alpha"], params["regularization"],
                                max(checkpoints), cfg["seed"], settings["init_std"], cfg["models"]["strength_scale_seconds"])
    for iteration, user_ids, item_ids, user_factors, item_factors, loss in states:
        losses.append(loss)
        if iteration in checkpoints:
            scores, valid = mf.score_candidates(user_ids, item_ids, user_factors, item_factors, users, pool)
            yield (iteration, *lists_from_scores(users.tolist(), pool, scores, valid, fallback, cfg["top_k"]), list(losses))


def ppr_probabilities(train, params, cfg, users):
    """Return (item_ids, probabilities with one column per user in `users` order, details)."""
    require_test_schedule_unread(cfg)
    settings = cfg["models"]["ppr"]
    scale = cfg["models"]["strength_scale_seconds"]
    weighting = params["interaction_weighting"]
    item_ids, weights = ppr.coview_graph(train, weighting, scale)
    transition, dangling = ppr.transition_matrix(weights)
    ordered = np.sort(users)
    seeds = ppr.personalization(train, ordered, item_ids, weighting, scale)
    probabilities, iterations, change = ppr.power_iteration(transition, dangling, seeds, params["link_probability"],
                                                            settings["tolerance"], settings["max_iterations"])
    totals = probabilities.sum(axis=0)
    details = {
        "iterations": iterations,
        "final_l1_change": change,
        "graph_nodes": int(len(item_ids)),
        "graph_edges_undirected": int(np.count_nonzero(np.triu(weights))),
        "zero_degree_nodes": int(dangling.sum()),
        "max_probability_total_error": float(np.abs(totals[totals > 0] - 1.0).max()),
        "users_with_seeds": int((seeds.sum(axis=0) > 0).sum()),
    }
    return item_ids, probabilities[:, np.searchsorted(ordered, users)], details


def ppr_lists(train, params, cfg, users, pool, fallback):
    """Return (lists, list_scores, personal_counts, convergence details)."""
    item_ids, probabilities, details = ppr_probabilities(train, params, cfg, users)
    scores, valid = ppr.score_candidates(item_ids, probabilities, pool)
    return (*lists_from_scores(users.tolist(), pool, scores, valid, fallback, cfg["top_k"]), details)


def stability_split(cfg):
    """Rebuild the weeks 11-27 to 28-30 fold through the normal pipeline."""
    clock = WallClock(cfg["timezone"])
    windows = [(clock.view_ts(cfg["stability"][name]["start"]), clock.view_ts(cfg["stability"][name]["end"]))
               for name in ("train", "validation")]
    airings, interactions, coverage, user_ids, _ = build_interactions(cfg, clock, windows)
    sides = watched_tables(airings, interactions, cfg["watched"]["primary_seconds"], cfg["watched"]["airing_fraction"])
    position = {user: index for index, user in enumerate(user_ids)}
    return {
        "train": sides["train"],
        "validation": sides["validation"],
        "pool": candidate_pool(airings, windows[VALIDATION], coverage["train_channels"]),
        "cohort": np.array([position[user] for user in read_cohort(cfg)], dtype=np.int32),
        "validation_rows": int((interactions["window"] == VALIDATION).sum()),
    }


def fallback_usage(personal_counts, k):
    return {"full_fallback_users": int((personal_counts == 0).sum()),
            "partial_fallback_users": int(((personal_counts > 0) & (personal_counts < k)).sum())}
