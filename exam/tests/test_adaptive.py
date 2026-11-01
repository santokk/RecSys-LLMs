import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from adaptive import fusion
from sweettv.hybrid import hybrid_lists

POOL = np.array([10, 20, 30, 40, 50, 60])
ORDER = np.array([60, 50, 40, 30, 20, 10])
SCORES = np.array([[.9, .8, .7, .6, .5, .4], [.9, .8, .7, .6, .5, .4], [0, 0, 0, 0, 0, 0]])
VALID = SCORES > 0
KYIV_DAY = 86400


class Weights(unittest.TestCase):
    def test_programme_thresholds_at_each_boundary(self):
        values = np.array([1, 4, 5, 19, 20, 100])
        self.assertEqual(fusion.assign_weights(values, 5, 20, [.4, .6, .8]).tolist(), [.4, .4, .6, .6, .8, .8])

    def test_day_thresholds_at_each_boundary(self):
        values = np.array([1, 2, 3, 13, 14])
        self.assertEqual(fusion.assign_weights(values, 3, 14, [.4, .6, .8]).tolist(), [.4, .4, .6, .6, .8])

    def test_groups_report_cold_separately(self):
        self.assertEqual(fusion.group_names(np.array([0, 1, 5, 20]), 5, 20).tolist(), ["cold", "low", "middle", "high"])


class History(unittest.TestCase):
    def test_programme_count_is_per_user_and_zero_when_absent(self):
        table = {"user": np.array([1, 1, 3]), "item": np.array([7, 8, 7])}
        self.assertEqual(fusion.programme_counts(table, np.array([3, 1, 2])).tolist(), [1, 2, 0])

    def rows(self, *entries):
        user, start = zip(*entries)
        return {"user": np.array(user), "airing_start": np.array(start)}

    def test_same_day_counts_once(self):
        noon = 1584000000
        rows = self.rows((1, noon), (1, noon + 3600), (1, noon + KYIV_DAY))
        self.assertEqual(fusion.active_days(rows, np.array([1, 2]), "Europe/Kyiv").tolist(), [2, 0])

    def test_day_uses_the_kyiv_calendar_not_utc(self):
        # 22:30 UTC on 2020-03-09 is already 00:30 on 2020-03-10 in Kyiv; 21:30 UTC is still the 9th.
        before, after = 1583789400, 1583793000
        rows = self.rows((1, before), (1, after))
        self.assertEqual(fusion.active_days(rows, np.array([1]), "Europe/Kyiv").tolist(), [2])

    def test_users_do_not_share_days(self):
        rows = self.rows((1, 1584000000), (2, 1584000000))
        self.assertEqual(fusion.active_days(rows, np.array([2, 1]), "Europe/Kyiv").tolist(), [1, 1])


class Lists(unittest.TestCase):
    def test_all_equal_weights_reproduce_fixed_fusion(self):
        users = [1, 2, 3]
        adaptive = fusion.adaptive_lists(users, POOL, SCORES, VALID, ORDER, np.full(3, 0.8), 10, 5)[0]
        fixed = hybrid_lists(users, POOL, SCORES, VALID, ORDER, 0.8, 10, 5)[0]
        self.assertEqual(adaptive, fixed)

    def test_each_user_uses_their_own_weight(self):
        lists, values, _ = fusion.adaptive_lists([1, 2, 3], POOL, SCORES, VALID, ORDER, np.array([.99, .01, .5]), 10, 5)
        self.assertEqual(lists[1][0], 10)
        self.assertEqual(lists[2][0], 60)
        self.assertAlmostEqual(values[0, 0], .99 / 11 + (1 - .99) / 16)

    def test_cold_user_receives_popularity_directly(self):
        lists, _, counts = fusion.adaptive_lists([1, 2, 3], POOL, SCORES, VALID, ORDER, np.array([.8, .8, .4]), 10, 5)
        self.assertEqual(lists[3], ORDER[:5].tolist())
        self.assertEqual(counts.tolist(), [5, 5, 0])

    def test_lists_hold_five_distinct_pool_ids_deterministically(self):
        first = fusion.adaptive_lists([1, 2, 3], POOL, SCORES, VALID, ORDER, np.array([.4, .6, .8]), 10, 5)[0]
        second = fusion.adaptive_lists([1, 2, 3], POOL, SCORES, VALID, ORDER, np.array([.4, .6, .8]), 10, 5)[0]
        self.assertEqual(first, second)
        self.assertTrue(all(len(set(ids)) == 5 and set(ids) <= set(POOL.tolist()) for ids in first.values()))


class Optimism(unittest.TestCase):
    def test_identical_variants_have_no_optimism_beyond_monte_carlo_noise(self):
        values = np.random.default_rng(0).normal(size=200)
        self.assertLess(abs(fusion.selection_optimism(values, values, 200, 1)), 0.02)

    def test_independent_noise_gives_positive_optimism(self):
        rng = np.random.default_rng(0)
        self.assertGreater(fusion.selection_optimism(rng.normal(size=500), rng.normal(size=500), 300, 1), 0)

    def test_seeded(self):
        values = np.arange(50.0)
        self.assertEqual(fusion.selection_optimism(values, values[::-1], 50, 1), fusion.selection_optimism(values, values[::-1], 50, 1))


class Lock(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.dir = Path(self.folder.name)

    def tearDown(self):
        self.folder.cleanup()

    def write(self, lock):
        (self.dir / fusion.LOCK_FILE).write_text(json.dumps(lock), encoding="utf-8")

    def sealed(self, **fields):
        lock = dict(fields)
        lock["lock_sha256"] = fusion.lock_hash(lock)
        return lock

    def test_valid_lock_loads(self):
        lock = self.sealed(post_kaggle_feedback=True)
        self.write(lock)
        self.assertEqual(fusion.read_lock(self.dir), lock)

    def test_missing_lock_is_refused(self):
        with self.assertRaises(SystemExit):
            fusion.read_lock(self.dir)

    def test_altered_lock_is_refused(self):
        self.write({**self.sealed(post_kaggle_feedback=True), "extra": 1})
        with self.assertRaises(SystemExit):
            fusion.read_lock(self.dir)

    def test_unlabelled_lock_is_refused(self):
        self.write(self.sealed(post_kaggle_feedback=False))
        with self.assertRaises(SystemExit):
            fusion.read_lock(self.dir)


if __name__ == "__main__":
    unittest.main()
