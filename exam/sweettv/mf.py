"""Weighted ALS for implicit feedback (Hu, Koren and Volinsky, ICDM 2008).

Minimises, over user factors x_u and item factors y_i,

    sum_observed c_ui (1 - x_u.y_i)^2 + sum_unobserved (x_u.y_i)^2
    + regularization * (|x|^2 + |y|^2)

with confidence c_ui = 1 + alpha * log(1 + seconds / strength_scale). Only
programmes with a training interaction get factors, so a cold candidate has no
score at all rather than a learned one.
"""
import numpy as np


def confidence(seconds, alpha, strength_scale):
    return 1.0 + alpha * np.log1p(np.asarray(seconds, dtype=np.float64) / strength_scale)


def _grouped(rows, columns, weights, row_count):
    order = np.lexsort((columns, rows))
    pointer = np.searchsorted(rows[order], np.arange(row_count + 1))
    return pointer, columns[order], weights[order]


def _solve_side(fixed, pointer, columns, weights, regularization):
    """Exact least-squares update of one side, with the other held fixed."""
    size = fixed.shape[1]
    shared = fixed.T @ fixed
    ridge = regularization * np.eye(size)
    solved = np.empty((len(pointer) - 1, size))
    for row in range(len(pointer) - 1):
        low, high = pointer[row], pointer[row + 1]
        observed, weight = fixed[columns[low:high]], weights[low:high]
        system = shared + (observed.T * (weight - 1.0)) @ observed + ridge
        solved[row] = np.linalg.solve((system + system.T) / 2, observed.T @ weight)
    return solved


def objective(user_factors, item_factors, user, item, weights, regularization):
    predicted = np.sum(user_factors[user] * item_factors[item], axis=1)
    everything = np.sum((user_factors.T @ user_factors) * (item_factors.T @ item_factors))
    data = np.sum(weights * (1.0 - predicted) ** 2) + everything - np.sum(predicted ** 2)
    return float(data + regularization * (np.sum(user_factors ** 2) + np.sum(item_factors ** 2)))


def wals_iterations(table, factors, alpha, regularization, iterations, seed, init_std, strength_scale):
    """Yield (iteration, user_ids, item_ids, user_factors, item_factors, loss)."""
    user_ids, user = np.unique(table["user"], return_inverse=True)
    item_ids, item = np.unique(table["item"], return_inverse=True)
    weights = confidence(table["seconds"], alpha, strength_scale)
    by_user = _grouped(user, item, weights, len(user_ids))
    by_item = _grouped(item, user, weights, len(item_ids))
    item_factors = np.random.default_rng(seed).normal(0.0, init_std, (len(item_ids), factors))
    for iteration in range(1, iterations + 1):
        user_factors = _solve_side(item_factors, *by_user, regularization)
        item_factors = _solve_side(user_factors, *by_item, regularization)
        loss = objective(user_factors, item_factors, user, item, weights, regularization)
        yield iteration, user_ids, item_ids, user_factors, item_factors, loss


def fit_wals(table, factors, alpha, regularization, iterations, seed, init_std=0.1, strength_scale=300):
    """Run to `iterations` and return the final state plus the loss history."""
    losses = []
    for state in wals_iterations(table, factors, alpha, regularization, iterations, seed, init_std, strength_scale):
        losses.append(state[-1])
    return {"user_ids": state[1], "item_ids": state[2], "user_factors": state[3], "item_factors": state[4], "losses": losses}


def score_candidates(user_ids, item_ids, user_factors, item_factors, users, candidates):
    """Scores for users x candidates, with a mask of what the model can score."""
    scores = np.zeros((len(users), len(candidates)))
    valid = np.zeros_like(scores, dtype=bool)
    user_row = np.searchsorted(user_ids, users).clip(max=len(user_ids) - 1)
    known_user = user_ids[user_row] == users
    item_column = np.searchsorted(item_ids, candidates).clip(max=len(item_ids) - 1)
    known_item = item_ids[item_column] == candidates
    rows, columns = np.flatnonzero(known_user), np.flatnonzero(known_item)
    scores[np.ix_(rows, columns)] = user_factors[user_row[rows]] @ item_factors[item_column[columns]].T
    valid[np.ix_(rows, columns)] = True
    return scores, valid
