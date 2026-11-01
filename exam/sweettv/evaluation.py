"""One evaluation path for popularity, MF and PPR."""
import numpy as np

from .metrics import (average_precision_at_k, bootstrap_interval, coverage_and_concentration, per_user,
                      recall_at_k)
from .split import broad_watched_sets, preference_target


def validation_targets(validation, cohort, pool, k):
    """Primary Top-k preference target and the secondary broad watched sets."""
    return preference_target(validation, cohort, pool, k), broad_watched_sets(validation, cohort, pool)


def paired_difference(ap, other_ap, cfg):
    """Per-user AP difference with a fixed-seed percentile bootstrap interval."""
    difference = ap - other_ap
    return {
        "mean": float(difference.mean()),
        "bootstrap_95ci": bootstrap_interval(difference, cfg["bootstrap_resamples"], cfg["seed"]),
        "resamples": cfg["bootstrap_resamples"],
        "seed": cfg["seed"],
        "users_better": int((difference > 0).sum()),
        "users_worse": int((difference < 0).sum()),
    }


def evaluate_lists(recommendations, users, target, broad, pool_size, cfg, baseline_ap=None):
    """Return (report, per-user primary AP)."""
    k = cfg["top_k"]
    nonempty = [user for user in users if target[user]]
    with_broad = [user for user in users if broad[user]]
    ap = per_user(average_precision_at_k, recommendations, target, users, k)
    ap_nonempty = per_user(average_precision_at_k, recommendations, target, nonempty, k)
    recall = per_user(recall_at_k, recommendations, target, nonempty, k)
    broad_ap = per_user(average_precision_at_k, recommendations, broad, users, k)
    broad_recall = per_user(recall_at_k, recommendations, broad, with_broad, k)
    resamples, seed = cfg["bootstrap_resamples"], cfg["seed"]
    report = {
        "primary_top5_preference": {
            "users_evaluated": len(users),
            "users_with_nonempty_target": len(nonempty),
            "map_at_5_all_users": float(ap.mean()),
            "map_at_5_all_users_95ci": bootstrap_interval(ap, resamples, seed),
            "map_at_5_nonempty_users": float(ap_nonempty.mean()),
            "recall_at_5_nonempty_users": float(recall.mean()),
            "hit_rate_all_users": float((ap > 0).mean()),
        },
        "secondary_broad_watched_diagnostic": {
            "not_for_selection": True,
            "map_at_5_all_users": float(broad_ap.mean()),
            "recall_at_5_users_with_broad_set": float(broad_recall.mean()),
        },
        "concentration": coverage_and_concentration(recommendations, users, pool_size),
    }
    if baseline_ap is not None:
        report["paired_difference_vs_popularity"] = paired_difference(ap, baseline_ap, cfg)
    return report, ap
