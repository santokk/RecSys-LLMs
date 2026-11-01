"""AP@k and MAP@k with the min(k, |R|) normalisation from the course slides."""
import numpy as np


def _checked(recommended, k):
    recommended = list(recommended)
    if len(recommended) > k:
        raise ValueError(f"expected at most {k} recommendations, got {len(recommended)}")
    if len(set(recommended)) != len(recommended):
        raise ValueError(f"duplicate recommendation in {recommended}")
    return recommended


def average_precision_at_k(recommended, relevant, k=5):
    """An empty relevant set scores 0."""
    recommended = _checked(recommended, k)
    if not relevant:
        return 0.0
    hits, precision_sum = 0, 0.0
    for rank, item in enumerate(recommended, start=1):
        if item in relevant:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / min(k, len(relevant))


def recall_at_k(recommended, relevant, k=5):
    recommended = _checked(recommended, k)
    if not relevant:
        return 0.0
    return len(set(recommended) & set(relevant)) / len(relevant)


def per_user(metric, recommendations, relevant, users, k=5):
    return np.array([metric(recommendations[u], relevant.get(u, ()), k) for u in users], dtype=np.float64)


def mean_average_precision(recommendations, relevant, users, k=5):
    return float(per_user(average_precision_at_k, recommendations, relevant, users, k).mean())


def bootstrap_interval(values, resamples, seed):
    """Percentile 95% interval of the mean, resampling users."""
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[picks].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def coverage_and_concentration(recommendations, users, pool_size):
    """Catalogue coverage plus slot concentration of the recommended ids."""
    ids, counts = np.unique([i for u in users for i in recommendations[u]], return_counts=True)
    shares = counts / counts.sum()
    ordered = np.sort(np.r_[np.zeros(pool_size - len(ids)), counts.astype(np.float64)])
    position = np.arange(1, pool_size + 1)
    gini = float((2 * position - pool_size - 1).dot(ordered) / (pool_size * ordered.sum()))
    return {
        "distinct_ids_recommended": int(len(ids)),
        "catalogue_coverage": len(ids) / pool_size,
        "largest_slot_share": float(shares.max()),
        "herfindahl_index": float((shares ** 2).sum()),
        "gini_over_pool": gini,
    }
