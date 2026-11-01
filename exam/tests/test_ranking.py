import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from sweettv.baseline import fill_to_k, popularity_scores, rank_items, recommend
from sweettv.metrics import (average_precision_at_k, bootstrap_interval, coverage_and_concentration,
                             mean_average_precision, recall_at_k)
from sweettv.split import broad_watched_sets, preference_target
from sweettv.submission import format_row, validate_row


class AveragePrecision(unittest.TestCase):
    def test_slide_method_x(self):
        self.assertAlmostEqual(average_precision_at_k("ABDCE", set("ABC")), (1.0 + 1.0 + 0.75) / 3)
        self.assertEqual(round(average_precision_at_k("ABDCE", set("ABC")), 3), 0.917)

    def test_slide_method_y(self):
        self.assertAlmostEqual(average_precision_at_k("DAEBC", set("ABC")), (0.5 + 0.5 + 0.6) / 3)
        self.assertEqual(round(average_precision_at_k("DAEBC", set("ABC")), 3), 0.533)

    def test_denominator_is_min_of_k_and_relevant(self):
        self.assertAlmostEqual(average_precision_at_k("ABCDE", set("ABCDEFGH")), 1.0)
        self.assertAlmostEqual(average_precision_at_k("XWYZA", {"A"}), 0.2)

    def test_empty_relevant_scores_zero(self):
        self.assertEqual(average_precision_at_k("ABCDE", set()), 0.0)

    def test_no_hit_scores_zero(self):
        self.assertEqual(average_precision_at_k("ABCDE", set("Z")), 0.0)

    def test_map_is_mean_over_users_with_empty_as_zero(self):
        recommendations = {1: list("ABDCE"), 2: list("DAEBC"), 3: list("ABCDE")}
        relevant = {1: set("ABC"), 2: set("ABC")}
        expected = (2.75 / 3 + 1.6 / 3 + 0.0) / 3
        self.assertAlmostEqual(mean_average_precision(recommendations, relevant, [1, 2, 3]), expected)

    def test_recall(self):
        self.assertAlmostEqual(recall_at_k("ABDCE", set("AXYZ")), 0.25)

    def test_duplicates_are_rejected(self):
        for metric in (average_precision_at_k, recall_at_k):
            with self.assertRaises(ValueError):
                metric(["A", "A", "B", "C", "D"], {"A"})

    def test_more_than_k_is_rejected(self):
        with self.assertRaises(ValueError):
            average_precision_at_k("ABCDEF", {"A"})


class Ranking(unittest.TestCase):
    def test_ties_break_by_ascending_numeric_id(self):
        self.assertEqual(rank_items([30, 9, 100], [1, 1, 1]).tolist(), [9, 30, 100])

    def test_numeric_not_lexical_order(self):
        self.assertEqual(rank_items([100, 20], [5, 5]).tolist(), [20, 100])

    def test_score_outranks_id(self):
        self.assertEqual(rank_items([1, 2, 3], [1, 3, 2]).tolist(), [2, 3, 1])

    def test_result_independent_of_input_order(self):
        a = rank_items([3, 1, 2], [7, 7, 9]).tolist()
        b = rank_items([1, 2, 3], [7, 9, 7]).tolist()
        self.assertEqual(a, b)

    def test_popularity_counts_distinct_users_inside_pool(self):
        table = {"user": np.array([1, 1, 2, 3]), "item": np.array([5, 9, 5, 7])}
        scores = popularity_scores(table, np.array([5, 7, 8]))
        self.assertEqual(scores.tolist(), [2, 1, 0])


class Fallback(unittest.TestCase):
    fallback = [10, 20, 30, 40, 50, 60]

    def test_cold_user_gets_fallback_top_five(self):
        self.assertEqual(recommend(["cold"], {}, self.fallback)["cold"], [10, 20, 30, 40, 50])

    def test_short_list_is_filled_without_duplicates(self):
        self.assertEqual(fill_to_k([20, 99], self.fallback), [20, 99, 10, 30, 40])

    def test_duplicate_in_ranked_input_collapses(self):
        self.assertEqual(fill_to_k([7, 7, 8], self.fallback), [7, 8, 10, 20, 30])

    def test_every_user_gets_same_fallback(self):
        lists = recommend(["a", "b"], {}, self.fallback)
        self.assertEqual(lists["a"], lists["b"])

    def test_too_small_pool_raises(self):
        with self.assertRaises(ValueError):
            fill_to_k([], [1, 2, 3])


class SubmissionRow(unittest.TestCase):
    def test_valid_row(self):
        self.assertEqual(format_row("18417531283109304442", [1, 2, 3, 4, 5]), "18417531283109304442,1 2 3 4 5")

    def test_needs_exactly_five(self):
        for ids in ([1, 2, 3, 4], [1, 2, 3, 4, 5, 6]):
            with self.assertRaises(ValueError):
                validate_row(ids)

    def test_needs_distinct_ids(self):
        with self.assertRaises(ValueError):
            validate_row([1, 2, 3, 4, 4])

    def test_rejects_zero_id(self):
        with self.assertRaises(ValueError):
            validate_row([0, 2, 3, 4, 5])

    def test_rejects_id_outside_pool(self):
        with self.assertRaises(ValueError):
            validate_row([1, 2, 3, 4, 5], pool={1, 2, 3, 4})

    def test_user_id_must_stay_a_string(self):
        with self.assertRaises(TypeError):
            format_row(18417531283109304442, [1, 2, 3, 4, 5])

    def test_large_user_id_survives_unchanged(self):
        self.assertTrue(format_row("18446744073709551615", [1, 2, 3, 4, 5]).startswith("18446744073709551615,"))


class RelevanceAndConcentration(unittest.TestCase):
    def test_relevant_sets_respect_pool_and_keep_empty_users(self):
        table = {"user": np.array([1, 1, 2]), "item": np.array([5, 6, 7])}
        result = broad_watched_sets(table, np.array([1, 2, 3]), np.array([5, 7]))
        self.assertEqual(result, {1: {5}, 2: {7}, 3: set()})

    def test_preference_target_uses_seconds_then_id_inside_pool(self):
        table = {"user": np.array([1, 1, 1, 1]), "item": np.array([5, 6, 7, 8]), "seconds": np.array([10, 90, 90, 500])}
        result = preference_target(table, np.array([1, 2]), np.array([5, 6, 7]), 2)
        self.assertEqual(result, {1: {6, 7}, 2: set()})

    def test_identical_lists_are_fully_concentrated_on_five_ids(self):
        recommendations = {u: [1, 2, 3, 4, 5] for u in range(10)}
        result = coverage_and_concentration(recommendations, list(range(10)), 100)
        self.assertEqual(result["distinct_ids_recommended"], 5)
        self.assertAlmostEqual(result["catalogue_coverage"], 0.05)
        self.assertAlmostEqual(result["largest_slot_share"], 0.2)
        self.assertAlmostEqual(result["herfindahl_index"], 0.2)
        self.assertAlmostEqual(result["gini_over_pool"], 0.95)

    def test_bootstrap_is_seeded(self):
        values = np.arange(50, dtype=np.float64)
        self.assertEqual(bootstrap_interval(values, 100, 1), bootstrap_interval(values, 100, 1))


if __name__ == "__main__":
    unittest.main()
