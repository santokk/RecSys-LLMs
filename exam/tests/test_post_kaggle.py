import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from post_kaggle import repeat
from sweettv.split import VALIDATION, TRAIN

DAY = 86400
START = 100 * DAY
POOL = np.array([10, 20, 30, 40, 50, 60])
ORDER = np.array([60, 50, 40, 30, 20, 10])


def rows(*entries):
    user, item, capped, age = zip(*entries)
    return {"user": np.array(user), "item": np.array(item), "capped": np.array(capped),
            "airing_start": START - np.array(age) * DAY}


class Scores(unittest.TestCase):
    def test_undecayed_score_is_summed_seconds(self):
        scores, valid = repeat.repeat_scores(rows((1, 10, 300, 5), (1, 10, 200, 90)), np.array([1]), POOL, None, START)
        self.assertEqual(scores[0, 0], 500)
        self.assertEqual(valid[0].tolist(), [True, False, False, False, False, False])

    def test_one_half_life_halves_the_weight(self):
        scores, _ = repeat.repeat_scores(rows((1, 10, 1000, 28)), np.array([1]), POOL, 28, START)
        self.assertAlmostEqual(scores[0, 0], 500)

    def test_formula_by_hand(self):
        scores, _ = repeat.repeat_scores(rows((1, 10, 600, 7), (1, 10, 400, 56)), np.array([1]), POOL, 56, START)
        expected = 600 * math.exp(-math.log(2) * 7 / 56) + 400 * math.exp(-math.log(2) * 56 / 56)
        self.assertAlmostEqual(scores[0, 0], expected)

    def test_age_runs_from_the_prediction_start(self):
        table = rows((1, 10, 1000, 28))
        early, _ = repeat.repeat_scores(table, np.array([1]), POOL, 28, START)
        later, _ = repeat.repeat_scores(table, np.array([1]), POOL, 28, START + 28 * DAY)
        self.assertAlmostEqual(later[0, 0], early[0, 0] / 2)

    def test_shorter_half_life_favours_recent_viewing(self):
        table = rows((1, 10, 1000, 2), (1, 20, 1200, 60))
        short, _ = repeat.repeat_scores(table, np.array([1]), POOL, 28, START)
        none, _ = repeat.repeat_scores(table, np.array([1]), POOL, None, START)
        self.assertGreater(short[0, 0], short[0, 1])
        self.assertLess(none[0, 0], none[0, 1])

    def test_unseen_programme_gets_no_score(self):
        _, valid = repeat.repeat_scores(rows((1, 10, 500, 1)), np.array([1]), POOL, 56, START)
        self.assertEqual(int(valid.sum()), 1)

    def test_watched_programme_outside_the_pool_is_ignored(self):
        scores, valid = repeat.repeat_scores(rows((1, 99, 500, 1)), np.array([1]), POOL, 56, START)
        self.assertFalse(valid.any())
        self.assertEqual(scores.sum(), 0)

    def test_unsorted_users_keep_their_own_rows(self):
        table = rows((5, 10, 500, 1), (2, 20, 500, 1))
        _, valid = repeat.repeat_scores(table, np.array([5, 2]), POOL, None, START)
        self.assertTrue(valid[0, 0] and valid[1, 1])
        self.assertFalse(valid[0, 1] or valid[1, 0])

    def test_future_row_is_refused(self):
        with self.assertRaises(ValueError):
            repeat.repeat_scores(rows((1, 10, 500, 0)), np.array([1]), POOL, 56, START)

    def test_tables_match_compares_seconds_exactly(self):
        table = rows((1, 10, 500, 1))
        self.assertTrue(repeat.tables_match(table, {"user": np.array([1]), "item": np.array([10]), "seconds": np.array([500])}))
        self.assertFalse(repeat.tables_match(table, {"user": np.array([1]), "item": np.array([10]), "seconds": np.array([501])}))


class Lists(unittest.TestCase):
    plain = {"name": "a", "half_life_days": None, "rrf_with_popularity": None}
    fused = {"name": "d", "half_life_days": 56,
             "rrf_with_popularity": {"repeat_weight": 0.8, "popularity_weight": 0.2, "rank_constant": 10}}

    def lists(self, variant, table, users=(1, 2)):
        return repeat.variant_lists(variant, table, np.array(users), POOL, ORDER, START, 5)

    def test_short_list_is_filled_from_the_fallback(self):
        lists, _, counts = self.lists(self.plain, rows((1, 10, 500, 1)))
        self.assertEqual(lists[1], [10, 60, 50, 40, 30])
        self.assertEqual(counts.tolist(), [1, 0])

    def test_cold_user_gets_the_fallback(self):
        self.assertEqual(self.lists(self.plain, rows((1, 10, 500, 1)))[0][2], ORDER[:5].tolist())
        self.assertEqual(self.lists(self.fused, rows((1, 10, 500, 1)))[0][2], ORDER[:5].tolist())

    def test_ties_break_by_ascending_id(self):
        table = rows(*[(1, item, 500, 1) for item in (50, 30, 10)])
        self.assertEqual(self.lists(self.plain, table)[0][1][:3], [10, 30, 50])

    def test_five_distinct_ids(self):
        table = rows(*[(1, item, 100 * item, 1) for item in POOL.tolist()])
        for variant in (self.plain, self.fused):
            self.assertEqual(len(set(self.lists(variant, table)[0][1])), 5)

    def test_fused_variant_uses_the_declared_weights(self):
        table = rows(*[(1, item, 100 * item, 1) for item in POOL.tolist()])
        _, values, _ = self.lists(self.fused, table, (1,))
        expected = 0.8 / (10 + 1) + (1 - 0.8) / (10 + 1)
        self.assertAlmostEqual(values[0, 0], expected)


    def test_repeat_side_carries_the_larger_weight_when_components_disagree(self):
        table = rows(*[(1, item, 1000 - item, 1) for item in POOL.tolist()])
        lists, values, _ = self.lists(self.fused, table, (1,))
        self.assertEqual(lists[1][0], 10)
        self.assertAlmostEqual(values[0, 0], 0.8 / (10 + 1) + (1 - 0.8) / (10 + 6))


class Lock(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.dir = Path(self.folder.name)
        lock = {"post_kaggle_feedback": True, "experiment": "x"}
        lock["lock_sha256"] = repeat.lock_hash(lock)
        self.lock = lock

    def tearDown(self):
        self.folder.cleanup()

    def write(self, lock):
        (self.dir / repeat.LOCK_FILE).write_text(json.dumps(lock), encoding="utf-8")

    def test_valid_lock_loads(self):
        self.write(self.lock)
        self.assertEqual(repeat.read_lock(self.dir), self.lock)

    def test_missing_lock_is_refused(self):
        with self.assertRaises(SystemExit):
            repeat.read_lock(self.dir)

    def test_altered_lock_is_refused(self):
        self.write({**self.lock, "experiment": "y"})
        with self.assertRaises(SystemExit):
            repeat.read_lock(self.dir)

    def test_lock_must_be_labelled_post_kaggle(self):
        lock = {"post_kaggle_feedback": False}
        lock["lock_sha256"] = repeat.lock_hash(lock)
        self.write(lock)
        with self.assertRaises(SystemExit):
            repeat.read_lock(self.dir)


class PositiveRows(unittest.TestCase):
    def test_keeps_only_watched_rows_of_the_requested_window(self):
        airings = {"start": np.array([0, 1000, 2000]), "end": np.array([3000, 4000, 5000]), "show_id": np.array([7, 8, 9])}
        interactions = {"airing": np.array([0, 1, 2]), "overlap_capped": np.array([600, 100, 600]),
                        "user": np.array([1, 1, 1]), "window": np.array([TRAIN, TRAIN, VALIDATION])}
        result = repeat.positive_rows(airings, interactions, TRAIN, 300, 0.5)
        self.assertEqual(result["item"].tolist(), [7])
        self.assertEqual(result["airing_start"].tolist(), [0])

    def test_protected_files_are_named(self):
        self.assertEqual(set(repeat.PROTECTED_CANDIDATES), {"ppr_submission.csv", "popularity_submission.csv"})


if __name__ == "__main__":
    unittest.main()
