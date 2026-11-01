import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from sweettv import mf, ppr
from sweettv.baseline import lists_from_scores
from sweettv.config import load_config
from sweettv.evaluation import evaluate_lists, validation_targets
from sweettv.runner import fallback_usage, mf_checkpoints, ppr_lists
from sweettv.split import preference_target, side, user_item_table
from sweettv.submission import validate_row


def block_table():
    """Users 0-9 watch items 100-104, users 10-19 watch items 200-204, with holes."""
    user, item, seconds = [], [], []
    for u in range(20):
        base = 100 if u < 10 else 200
        for offset in range(5):
            if (u + offset) % 5:
                user.append(u), item.append(base + offset), seconds.append(600 + 60 * offset)
    return {"user": np.array(user), "item": np.array(item), "seconds": np.array(seconds)}


def config():
    cfg = load_config()
    cfg["models"]["mf"].update({"iteration_checkpoints": [5, 10]})
    return cfg


class WeightedAls(unittest.TestCase):
    params = dict(factors=4, alpha=10, regularization=1, iterations=10, seed=7)

    def test_same_seed_gives_identical_factors(self):
        a, b = mf.fit_wals(block_table(), **self.params), mf.fit_wals(block_table(), **self.params)
        self.assertTrue(np.array_equal(a["user_factors"], b["user_factors"]))
        self.assertTrue(np.array_equal(a["item_factors"], b["item_factors"]))

    def test_different_seed_changes_factors(self):
        a = mf.fit_wals(block_table(), **self.params)
        b = mf.fit_wals(block_table(), **{**self.params, "seed": 8})
        self.assertFalse(np.array_equal(a["item_factors"], b["item_factors"]))

    def test_loss_is_finite_and_never_increases(self):
        losses = np.array(mf.fit_wals(block_table(), **self.params)["losses"])
        self.assertTrue(np.all(np.isfinite(losses)))
        self.assertTrue(np.all(np.diff(losses) <= 1e-8 * losses[0]))

    def test_objective_matches_a_direct_double_sum(self):
        rng = np.random.default_rng(0)
        x, y = rng.normal(size=(3, 2)), rng.normal(size=(4, 2))
        user, item, weight = np.array([0, 1, 2]), np.array([0, 2, 3]), np.array([2.0, 3.0, 4.0])
        predictions = x @ y.T
        direct = (predictions ** 2).sum() + 0.5 * ((x ** 2).sum() + (y ** 2).sum())
        for u, i, c in zip(user, item, weight):
            direct += c * (1 - predictions[u, i]) ** 2 - predictions[u, i] ** 2
        self.assertAlmostEqual(mf.objective(x, y, user, item, weight, 0.5), direct)

    def test_solved_side_is_a_stationary_point_of_the_objective(self):
        table = block_table()
        model = mf.fit_wals(table, factors=3, alpha=10, regularization=2, iterations=3, seed=7, strength_scale=300)
        _, user = np.unique(table["user"], return_inverse=True)
        _, item = np.unique(table["item"], return_inverse=True)
        weights = mf.confidence(table["seconds"], 10, 300)
        x, y = model["user_factors"], model["item_factors"]

        def loss(factors):
            return mf.objective(x, factors, user, item, weights, 2)

        step = 1e-4
        for row, column in ((0, 0), (3, 1), (7, 2)):
            shift = np.zeros_like(y)
            shift[row, column] = step
            self.assertAlmostEqual((loss(y + shift) - loss(y - shift)) / (2 * step), 0.0, places=4)

    def test_positive_block_outranks_unrelated_items(self):
        model = mf.fit_wals(block_table(), **self.params)
        users, candidates = np.array([0, 15]), np.array([100, 101, 102, 103, 104, 200, 201, 202, 203, 204])
        scores, valid = mf.score_candidates(model["user_ids"], model["item_ids"], model["user_factors"], model["item_factors"], users, candidates)
        self.assertTrue(valid.all())
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertGreater(scores[0, :5].min(), scores[0, 5:].max())
        self.assertGreater(scores[1, 5:].min(), scores[1, :5].max())

    def test_unwatched_item_in_block_is_scored_highly(self):
        table = block_table()
        model = mf.fit_wals(table, **self.params)
        watched = set(table["item"][table["user"] == 0].tolist())
        self.assertEqual(sorted(set(range(100, 105)) - watched), [100])
        scores, _ = mf.score_candidates(model["user_ids"], model["item_ids"], model["user_factors"], model["item_factors"], np.array([0]), np.array([100, 200]))
        self.assertGreater(scores[0, 0], scores[0, 1])

    def test_regularization_shrinks_factors(self):
        loose = mf.fit_wals(block_table(), **self.params)
        tight = mf.fit_wals(block_table(), **{**self.params, "regularization": 100})
        self.assertLess(np.abs(tight["item_factors"]).sum(), np.abs(loose["item_factors"]).sum())

    def test_confidence_changes_the_solution(self):
        low = mf.fit_wals(block_table(), **{**self.params, "alpha": 1})
        high = mf.fit_wals(block_table(), **{**self.params, "alpha": 40})
        self.assertGreater(np.abs(low["item_factors"] - high["item_factors"]).max(), 1e-3)

    def test_confidence_uses_interaction_strength(self):
        self.assertAlmostEqual(float(mf.confidence(np.array([0.0]), 20, 300)[0]), 1.0)
        values = mf.confidence(np.array([300.0, 3000.0]), 20, 300)
        self.assertLess(values[0], values[1])

    def test_cold_candidate_and_cold_user_get_no_learned_score(self):
        model = mf.fit_wals(block_table(), **self.params)
        users, candidates = np.array([0, 999]), np.array([100, 555])
        scores, valid = mf.score_candidates(model["user_ids"], model["item_ids"], model["user_factors"], model["item_factors"], users, candidates)
        self.assertEqual(valid.tolist(), [[True, False], [False, False]])
        self.assertEqual(scores[0, 1], 0.0)
        self.assertEqual(scores[1, 0], 0.0)


class Graph(unittest.TestCase):
    def setUp(self):
        self.item_ids, self.weights = ppr.coview_graph(block_table(), "binary", 300)
        self.transition, self.dangling = ppr.transition_matrix(self.weights)

    def test_weights_are_symmetric_with_empty_diagonal(self):
        self.assertTrue(np.allclose(self.weights, self.weights.T))
        self.assertTrue(np.all(np.diag(self.weights) == 0))

    def test_heavy_user_does_not_scale_edge_weight(self):
        table = {"user": np.array([0, 0, 0, 0]), "item": np.array([1, 2, 3, 4]), "seconds": np.array([600] * 4)}
        _, weights = ppr.coview_graph(table, "binary", 300)
        self.assertAlmostEqual(weights[0, 1], 0.25)

    def test_edge_weight_formula_with_strengths(self):
        table = {"user": np.array([0, 0]), "item": np.array([1, 2]), "seconds": np.array([300, 900])}
        _, weights = ppr.coview_graph(table, "seconds", 300)
        self.assertAlmostEqual(weights[0, 1], 1 * 3 / 4)

    def test_transition_rows_sum_to_one(self):
        self.assertTrue(np.allclose(self.transition.sum(axis=1), 1.0))
        self.assertFalse(self.dangling.any())

    def test_zero_degree_row_is_flagged_and_stays_zero(self):
        weights = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        transition, dangling = ppr.transition_matrix(weights)
        self.assertEqual(dangling.tolist(), [False, False, True])
        self.assertEqual(transition[2].tolist(), [0.0, 0.0, 0.0])


class PowerIteration(unittest.TestCase):
    def run_walk(self, weights, seeds, d=0.85):
        transition, dangling = ppr.transition_matrix(weights)
        return ppr.power_iteration(transition, dangling, seeds, d, 1e-12, 1000)

    def test_probability_is_conserved_with_dangling_nodes(self):
        weights = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
        seeds = np.array([[1.0], [0.0], [0.0]])
        result, _, _ = self.run_walk(weights, seeds)
        self.assertAlmostEqual(float(result.sum()), 1.0)
        self.assertTrue(np.all(result >= 0))

    def test_dangling_mass_returns_to_the_seed_only(self):
        weights = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        seeds = np.array([[1.0], [0.0], [0.0]])
        result, _, _ = self.run_walk(weights, seeds)
        self.assertEqual(result[2, 0], 0.0)
        self.assertAlmostEqual(result[0, 0], 1 / (1 + 0.85))

    def test_converges_to_the_exact_solution(self):
        weights = ppr.coview_graph(block_table(), "binary", 300)[1]
        transition, dangling = ppr.transition_matrix(weights)
        seeds = np.zeros((len(weights), 1))
        seeds[0, 0] = 1.0
        result, iterations, change = ppr.power_iteration(transition, dangling, seeds, 0.85, 1e-12, 1000)
        exact = np.linalg.solve(np.eye(len(weights)) - 0.85 * transition.T, 0.15 * seeds)
        self.assertLess(change, 1e-12)
        self.assertLess(iterations, 1000)
        self.assertTrue(np.allclose(result, exact, atol=1e-10))

    def test_non_convergence_raises(self):
        weights = ppr.coview_graph(block_table(), "binary", 300)[1]
        transition, dangling = ppr.transition_matrix(weights)
        seeds = np.zeros((len(weights), 1))
        seeds[0, 0] = 1.0
        with self.assertRaises(RuntimeError):
            ppr.power_iteration(transition, dangling, seeds, 0.99, 1e-15, 2)

    def test_seed_changes_the_ranking(self):
        table = block_table()
        item_ids, weights = ppr.coview_graph(table, "binary", 300)
        transition, dangling = ppr.transition_matrix(weights)
        users = np.array([0, 15])
        seeds = ppr.personalization(table, users, item_ids, "binary", 300)
        result, _, _ = ppr.power_iteration(transition, dangling, seeds, 0.85, 1e-10, 500)
        self.assertLess(item_ids[np.argmax(result[:, 0])], 200)
        self.assertGreaterEqual(item_ids[np.argmax(result[:, 1])], 200)

    def test_personalization_sums_to_one_and_cold_user_is_empty(self):
        table = block_table()
        item_ids = np.unique(table["item"])
        seeds = ppr.personalization(table, np.array([0, 999]), item_ids, "log_seconds", 300)
        self.assertAlmostEqual(float(seeds[:, 0].sum()), 1.0)
        self.assertEqual(float(seeds[:, 1].sum()), 0.0)

    def test_weighting_changes_personalization(self):
        table = block_table()
        item_ids = np.unique(table["item"])
        binary = ppr.personalization(table, np.array([1]), item_ids, "binary", 300)
        seconds = ppr.personalization(table, np.array([1]), item_ids, "seconds", 300)
        self.assertFalse(np.allclose(binary, seconds))

    def test_unknown_weighting_is_refused(self):
        with self.assertRaises(ValueError):
            ppr.interaction_strength([1.0], "linear", 300)


class SharedPolicy(unittest.TestCase):
    def setUp(self):
        self.cfg = config()
        self.table = block_table()
        self.pool = np.array([100, 101, 102, 103, 104, 200, 201, 202, 203, 204, 300])
        self.fallback = [300, 204, 203, 202, 201, 200, 104, 103, 102, 101, 100]
        self.users = np.array([0, 15, 999])

    def mf_result(self):
        params = {"factors": 4, "confidence_alpha": 10, "regularization": 1}
        return list(mf_checkpoints(self.table, params, self.cfg, self.users, self.pool, self.fallback, [10]))[0]

    def ppr_result(self):
        return ppr_lists(self.table, {"link_probability": 0.6, "interaction_weighting": "binary"}, self.cfg, self.users, self.pool, self.fallback)

    def test_both_models_give_five_distinct_pool_ids(self):
        for lists in (self.mf_result()[1], self.ppr_result()[0]):
            for user in self.users.tolist():
                validate_row(lists[user], set(self.pool.tolist()), 5)

    def test_models_never_recommend_outside_the_pool(self):
        pool = np.array([100, 101, 102, 103, 104])
        fallback = pool.tolist()
        params = {"factors": 4, "confidence_alpha": 10, "regularization": 1}
        mf_lists = list(mf_checkpoints(self.table, params, self.cfg, np.array([15]), pool, fallback, [5]))[0][1]
        ppr_result = ppr_lists(self.table, {"link_probability": 0.6, "interaction_weighting": "binary"}, self.cfg, np.array([15]), pool, fallback)[0]
        self.assertLessEqual(set(mf_lists[15]), set(pool.tolist()))
        self.assertLessEqual(set(ppr_result[15]), set(pool.tolist()))

    def test_cold_user_gets_the_common_fallback_from_both(self):
        self.assertEqual(self.mf_result()[1][999], self.fallback[:5])
        self.assertEqual(self.ppr_result()[0][999], self.fallback[:5])

    def test_fallback_usage_is_counted(self):
        self.assertEqual(fallback_usage(self.mf_result()[3], 5), {"full_fallback_users": 1, "partial_fallback_users": 0})
        self.assertEqual(fallback_usage(self.ppr_result()[2], 5), {"full_fallback_users": 1, "partial_fallback_users": 0})

    def test_previously_watched_items_stay_eligible(self):
        watched = set(self.table["item"][self.table["user"] == 0].tolist())
        self.assertTrue(set(self.mf_result()[1][0]) & watched)
        self.assertTrue(set(self.ppr_result()[0][0]) & watched)

    def test_short_list_is_filled_from_fallback(self):
        scores = np.array([[9.0, 8.0, 0.0]])
        valid = np.array([[True, True, False]])
        lists, values, counts = lists_from_scores([1], np.array([10, 20, 30]), scores, valid, [40, 10, 50, 60, 70], 5)
        self.assertEqual(lists[1], [10, 20, 40, 50, 60])
        self.assertEqual(counts.tolist(), [2])
        self.assertTrue(np.isnan(values[0, 2:]).all())

    def test_equal_scores_break_by_numeric_id(self):
        scores = np.array([[5.0, 5.0, 5.0, 5.0, 5.0, 5.0]])
        lists, _, _ = lists_from_scores([1], np.array([900, 30, 100, 2000, 7, 55]), scores, np.ones((1, 6), dtype=bool), [], 5)
        self.assertEqual(lists[1], [7, 30, 55, 100, 900])

    def test_side_exposes_only_the_named_window(self):
        tables = {"train_user": 1, "train_item": 2, "validation_user": 3, "validation_item": 4}
        self.assertEqual(side(tables, "train"), {"user": 1, "item": 2})

    def test_changing_validation_cannot_change_a_fit(self):
        archive = {f"train_{key}": value for key, value in self.table.items()}
        archive["validation_user"], archive["validation_item"] = np.array([0]), np.array([200])
        changed = dict(archive)
        changed["validation_user"], changed["validation_item"] = np.array([1, 2]), np.array([100, 101])
        first = mf.fit_wals(side(archive, "train"), 4, 10, 1, 5, 7)
        second = mf.fit_wals(side(changed, "train"), 4, 10, 1, 5, 7)
        self.assertTrue(np.array_equal(first["item_factors"], second["item_factors"]))

    def test_changing_training_does_change_a_fit(self):
        extra = {key: np.r_[value, value[:1] * 0 + {"user": 0, "item": 200, "seconds": 900}[key]] for key, value in self.table.items()}
        first = mf.fit_wals(self.table, 4, 10, 1, 5, 7)
        second = mf.fit_wals(extra, 4, 10, 1, 5, 7)
        self.assertFalse(np.array_equal(first["user_factors"], second["user_factors"]))

    def test_validation_only_interaction_stays_out_of_the_graph(self):
        item_ids, _ = ppr.coview_graph(self.table, "binary", 300)
        self.assertNotIn(300, item_ids.tolist())

    def test_mf_and_ppr_lists_are_deterministic(self):
        self.assertEqual(self.mf_result()[1], self.mf_result()[1])
        self.assertEqual(self.ppr_result()[0], self.ppr_result()[0])

    def test_ppr_lists_follow_users_in_any_cohort_order(self):
        params = {"link_probability": 0.6, "interaction_weighting": "binary"}
        ordered = ppr_lists(self.table, params, self.cfg, np.array([0, 15]), self.pool, self.fallback)[0]
        shuffled = ppr_lists(self.table, params, self.cfg, np.array([15, 0]), self.pool, self.fallback)[0]
        self.assertEqual(ordered, shuffled)

    def test_ppr_reports_probability_conservation(self):
        details = self.ppr_result()[3]
        self.assertLess(details["max_probability_total_error"], 1e-12)
        self.assertEqual(details["users_with_seeds"], 2)


class PrimaryTarget(unittest.TestCase):
    pool = np.array([1, 2, 3, 4, 5, 6, 7])

    def table(self, rows):
        user, item, seconds = zip(*rows)
        return {"user": np.array(user), "item": np.array(item), "seconds": np.array(seconds)}

    def test_seconds_aggregate_across_airings_before_ranking(self):
        grouped = user_item_table(np.array([1, 1, 1, 1, 1, 1, 1]), np.array([1, 1, 2, 3, 4, 5, 6]),
                                  np.array([300, 300, 500, 400, 350, 340, 330]))
        self.assertEqual(preference_target(grouped, np.array([1]), self.pool, 5)[1], {1, 2, 3, 4, 5})

    def test_equal_seconds_break_by_ascending_id(self):
        rows = [(1, i, 600) for i in (7, 6, 5, 4, 3, 2)]
        self.assertEqual(preference_target(self.table(rows), np.array([1]), self.pool, 5)[1], {2, 3, 4, 5, 6})

    def test_fewer_than_five_items_are_all_kept(self):
        self.assertEqual(preference_target(self.table([(1, 1, 10), (1, 2, 20)]), np.array([1]), self.pool, 5)[1], {1, 2})

    def test_empty_user_stays_with_an_empty_target(self):
        self.assertEqual(preference_target(self.table([(1, 1, 10)]), np.array([1, 2]), self.pool, 5)[2], set())

    def test_pool_restriction_happens_before_truncation(self):
        rows = [(1, 99, 9000), (1, 98, 8000), (1, 1, 10), (1, 2, 20)]
        self.assertEqual(preference_target(self.table(rows), np.array([1]), self.pool, 5)[1], {1, 2})

    def test_truncation_keeps_exactly_five(self):
        rows = [(1, i, 100 * i) for i in range(1, 8)]
        result = preference_target(self.table(rows), np.array([1]), self.pool, 5)[1]
        self.assertEqual(result, {3, 4, 5, 6, 7})

    def test_empty_target_scores_zero_and_stays_in_the_mean(self):
        cfg = config()
        table = self.table([(1, i, 1000 - i) for i in range(1, 6)])
        target, broad = validation_targets(table, np.array([1, 2]), self.pool, 5)
        lists = {1: [1, 2, 3, 4, 5], 2: [1, 2, 3, 4, 5]}
        report, ap = evaluate_lists(lists, [1, 2], target, broad, 7, cfg)
        self.assertEqual(ap.tolist(), [1.0, 0.0])
        self.assertEqual(report["primary_top5_preference"]["map_at_5_all_users"], 0.5)
        self.assertEqual(report["primary_top5_preference"]["map_at_5_nonempty_users"], 1.0)
        self.assertEqual(report["primary_top5_preference"]["users_with_nonempty_target"], 1)

    def test_broad_diagnostic_stays_separate_from_primary(self):
        cfg = config()
        table = self.table([(1, i, 1000 - i) for i in range(1, 8)])
        target, broad = validation_targets(table, np.array([1]), self.pool, 5)
        self.assertEqual(len(target[1]), 5)
        self.assertEqual(len(broad[1]), 7)
        report, _ = evaluate_lists({1: [7, 6, 1, 2, 3]}, [1], target, broad, 7, cfg)
        self.assertLess(report["primary_top5_preference"]["map_at_5_all_users"], report["secondary_broad_watched_diagnostic"]["map_at_5_all_users"])
        self.assertTrue(report["secondary_broad_watched_diagnostic"]["not_for_selection"])

    def test_paired_difference_is_against_the_baseline(self):
        cfg = config()
        table = self.table([(1, 1, 100)])
        target, broad = validation_targets(table, np.array([1]), self.pool, 5)
        report, _ = evaluate_lists({1: [1, 2, 3, 4, 5]}, [1], target, broad, 7, cfg, baseline_ap=np.array([0.25]))
        self.assertEqual(report["paired_difference_vs_popularity"]["mean"], 0.75)


if __name__ == "__main__":
    unittest.main()
