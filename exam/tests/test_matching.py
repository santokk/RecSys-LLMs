import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from sweettv.matching import AiringIndex, aggregate_fragments, clip_to_windows, match_segments, watched_mask
from sweettv.schedule import resolve_overlaps
from sweettv.split import candidate_pool, user_item_table


def arr(*values):
    return np.array(values, dtype=np.int64)


def aggregate(user, airing, window, start, stop, lengths, view=None):
    return aggregate_fragments(arr(*user), arr(*airing), arr(*window), arr(*start), arr(*stop),
                               arr(*(view or [stop[i] - start[i] for i in range(len(start))])), arr(*lengths))


class IntervalOverlap(unittest.TestCase):
    def setUp(self):
        self.channel, self.start, self.end = arr(1, 1, 2), arr(100, 200, 100), arr(200, 300, 200)
        self.show = arr(11, 22, 33)
        self.index = AiringIndex(self.channel, self.start, self.end)

    def pairs(self, channel, start, stop):
        segment, airing = self.index.overlaps(arr(channel), arr(start), arr(stop))
        return airing.tolist()

    def test_inside_one_airing(self):
        self.assertEqual(self.pairs(1, 120, 150), [0])

    def test_touching_ends_are_not_overlap(self):
        self.assertEqual(self.pairs(1, 50, 100), [])
        self.assertEqual(self.pairs(1, 300, 350), [])

    def test_one_second_overlap_counts(self):
        self.assertEqual(self.pairs(1, 99, 101), [0])
        self.assertEqual(self.pairs(1, 299, 310), [1])

    def test_view_stopping_at_boundary_hits_only_first(self):
        self.assertEqual(self.pairs(1, 150, 200), [0])
        self.assertEqual(self.pairs(1, 200, 250), [1])

    def test_view_spanning_both_airings(self):
        self.assertEqual(self.pairs(1, 150, 250), [0, 1])

    def test_channels_do_not_mix(self):
        self.assertEqual(self.pairs(2, 150, 250), [2])
        self.assertEqual(self.pairs(3, 150, 250), [])

    def test_overlap_seconds(self):
        match = match_segments(self.index, self.start, self.end, self.show, arr(1), arr(150), arr(250))
        self.assertEqual((match["overlap_stop"] - match["overlap_start"]).tolist(), [50, 50])

    def test_rejects_overlapping_timeline(self):
        with self.assertRaises(ValueError):
            AiringIndex(arr(1, 1), arr(100, 150), arr(200, 250))


class ZeroIdExclusion(unittest.TestCase):
    def test_unlabelled_airing_matches_but_is_not_an_item(self):
        start, end, show = arr(100, 200), arr(200, 300), arr(0, 7)
        index = AiringIndex(arr(1, 1), start, end)
        match = match_segments(index, start, end, show, arr(1), arr(150), arr(250))
        self.assertEqual(match["is_item"].tolist(), [False, True])

    def test_zero_never_enters_candidate_pool(self):
        airings = {"channel": arr(1, 1), "start": arr(0, 100), "end": arr(100, 200), "show_id": arr(0, 9)}
        self.assertEqual(candidate_pool(airings, (0, 300), arr(1)).tolist(), [9])


class TemporalClipping(unittest.TestCase):
    windows = [(0, 100), (100, 200)]

    def test_view_is_cut_at_the_boundary(self):
        view, window, start, stop = clip_to_windows(arr(90), arr(120), self.windows)
        self.assertEqual(window.tolist(), [0, 1])
        self.assertEqual(list(zip(start.tolist(), stop.tolist())), [(90, 100), (100, 120)])
        self.assertEqual(view.tolist(), [0, 0])

    def test_view_ending_on_boundary_stays_in_train(self):
        _, window, _, _ = clip_to_windows(arr(90), arr(100), self.windows)
        self.assertEqual(window.tolist(), [0])

    def test_view_past_validation_end_is_cut(self):
        _, window, _, stop = clip_to_windows(arr(190), arr(500), self.windows)
        self.assertEqual((window.tolist(), stop.tolist()), ([1], [200]))

    def test_view_after_all_windows_is_dropped(self):
        view, _, _, _ = clip_to_windows(arr(250), arr(300), self.windows)
        self.assertEqual(len(view), 0)

    def test_airing_spanning_boundary_keeps_windows_apart(self):
        result = aggregate([0, 0], [0, 0], [0, 1], [90, 100], [100, 120], [1000])
        self.assertEqual(result["window"].tolist(), [0, 1])
        self.assertEqual(result["overlap_capped"].tolist(), [10, 20])

    def test_candidate_must_air_inside_window(self):
        airings = {"channel": arr(1, 1, 1), "start": arr(0, 90, 200), "end": arr(50, 110, 300), "show_id": arr(1, 2, 3)}
        self.assertEqual(candidate_pool(airings, (100, 200), arr(1)).tolist(), [2])

    def test_candidate_channel_must_be_eligible(self):
        airings = {"channel": arr(1, 2), "start": arr(0, 0), "end": arr(50, 50), "show_id": arr(1, 2)}
        self.assertEqual(candidate_pool(airings, (0, 50), arr(2)).tolist(), [2])


class FragmentAggregation(unittest.TestCase):
    def test_fragments_sum_before_threshold(self):
        result = aggregate([5, 5], [0, 0], [0, 0], [0, 200], [200, 400], [1800])
        self.assertEqual(result["overlap_capped"].tolist(), [400])
        self.assertEqual(result["fragments"].tolist(), [2])
        self.assertTrue(watched_mask(result["overlap_capped"], arr(1800), 300, 0.5)[0])
        self.assertFalse(watched_mask(arr(200), arr(1800), 300, 0.5)[0])

    def test_sum_is_capped_at_airing_duration(self):
        result = aggregate([1, 1], [0, 0], [0, 0], [0, 0], [100, 100], [100])
        self.assertEqual(result["overlap_sum"].tolist(), [200])
        self.assertEqual(result["overlap_capped"].tolist(), [100])

    def test_union_removes_concurrent_seconds(self):
        result = aggregate([1, 1], [0, 0], [0, 0], [0, 50], [100, 150], [1000])
        self.assertEqual(result["overlap_sum"].tolist(), [200])
        self.assertEqual(result["overlap_union"].tolist(), [150])

    def test_union_with_separate_fragments(self):
        result = aggregate([1, 1, 1], [0, 0, 0], [0, 0, 0], [0, 10, 500], [100, 20, 600], [1000])
        self.assertEqual(result["overlap_union"].tolist(), [200])

    def test_users_and_airings_stay_separate(self):
        result = aggregate([1, 2, 1], [0, 0, 1], [0, 0, 0], [0, 0, 0], [10, 20, 30], [100, 100])
        self.assertEqual(sorted(result["overlap_capped"].tolist()), [10, 20, 30])

    def test_longest_view_is_kept(self):
        result = aggregate([1, 1], [0, 0], [0, 0], [0, 10], [5, 20], [100], view=[9000, 40])
        self.assertEqual(result["longest_view"].tolist(), [9000])


class WatchedRule(unittest.TestCase):
    def test_primary_threshold_on_long_airing(self):
        mask = watched_mask(arr(299, 300), arr(2700, 2700), 300, 0.5)
        self.assertEqual(mask.tolist(), [False, True])

    def test_half_airing_governs_short_airing(self):
        mask = watched_mask(arr(59, 60), arr(120, 120), 300, 0.5)
        self.assertEqual(mask.tolist(), [False, True])

    def test_sensitivity_thresholds(self):
        self.assertTrue(watched_mask(arr(60), arr(2700), 60, 0.5)[0])
        self.assertFalse(watched_mask(arr(599), arr(2700), 600, 0.5)[0])

    def test_user_item_table_merges_airings(self):
        table = user_item_table(arr(1, 1, 2), arr(7, 7, 7), arr(100, 200, 50))
        self.assertEqual(table["airings"].tolist(), [2, 1])
        self.assertEqual(table["seconds"].tolist(), [300, 50])


class OverlapResolution(unittest.TestCase):
    def test_later_airing_truncates_earlier(self):
        airings, stats = resolve_overlaps(arr(1, 1), arr(0, 100), arr(200, 100), arr(5, 6))
        self.assertEqual(airings["end"].tolist(), [100, 200])
        self.assertEqual(stats["overlaps_truncated"], 1)
        self.assertEqual(stats["overlap_seconds_removed"], 100)
        self.assertEqual(stats["overlaps_between_two_labelled_ids"], 1)

    def test_zero_duration_row_dropped_and_counted(self):
        airings, stats = resolve_overlaps(arr(1, 1), arr(0, 100), arr(0, 50), arr(0, 6))
        self.assertEqual(airings["show_id"].tolist(), [6])
        self.assertEqual(stats["zero_duration_dropped"], 1)

    def test_same_start_keeps_longest_then_larger_id(self):
        airings, stats = resolve_overlaps(arr(1, 1, 1), arr(0, 0, 0), arr(50, 100, 100), arr(9, 3, 4))
        self.assertEqual(airings["show_id"].tolist(), [4])
        self.assertEqual(stats["same_start_dropped"], 2)
        self.assertEqual(stats["same_start_with_different_ids"], 2)

    def test_result_is_stable_under_input_order(self):
        first, _ = resolve_overlaps(arr(1, 1, 2), arr(0, 100, 0), arr(200, 100, 50), arr(5, 6, 7))
        second, _ = resolve_overlaps(arr(2, 1, 1), arr(0, 100, 0), arr(50, 100, 200), arr(7, 6, 5))
        for key in first:
            self.assertEqual(first[key].tolist(), second[key].tolist())

    def test_channels_are_independent(self):
        airings, stats = resolve_overlaps(arr(1, 2), arr(0, 10), arr(100, 100), arr(5, 6))
        self.assertEqual(stats["overlaps_truncated"], 0)
        self.assertEqual(airings["end"].tolist(), [100, 110])

    def test_contained_airing_keeps_the_timeline_disjoint(self):
        airings, _ = resolve_overlaps(arr(1, 1, 1), arr(0, 10, 20), arr(1000, 5, 5), arr(1, 2, 3))
        self.assertTrue(np.all(airings["end"][:-1] <= airings["start"][1:]))


if __name__ == "__main__":
    unittest.main()
