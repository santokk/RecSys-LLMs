import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from sweettv import csvio
from sweettv.config import load_config
from sweettv.evaluation import paired_difference
from sweettv.hybrid import full_ranks, fuse, hybrid_lists, hybrid_scores, popularity_ranks
from sweettv.pipeline import build_interactions
from sweettv.runner import hybrid_config_id, hybrid_grid, mf_checkpoints, ppr_probabilities
from sweettv.split import user_item_table, watched_tables
from sweettv.submission import render_csv, validate_csv

CANDIDATES = np.array([10, 20, 30, 40, 50, 60])


class Ranks(unittest.TestCase):
    def test_ranks_start_at_one_and_cover_everyone(self):
        ranks = full_ranks(CANDIDATES, np.array([.1, .9, .5, .2, .3, .4]), np.ones(6, dtype=bool))
        self.assertEqual(ranks.tolist(), [6, 1, 2, 5, 4, 3])

    def test_equal_scores_break_by_ascending_id(self):
        ranks = full_ranks(CANDIDATES, np.full(6, 0.5), np.ones(6, dtype=bool))
        self.assertEqual(ranks.tolist(), [1, 2, 3, 4, 5, 6])

    def test_unreachable_candidates_rank_last_by_id(self):
        valid = np.array([False, True, False, True, False, False])
        ranks = full_ranks(CANDIDATES, np.array([0, .2, 0, .9, 0, 0]), valid)
        self.assertEqual(ranks.tolist(), [3, 2, 4, 1, 5, 6])

    def test_invalid_candidate_never_outranks_a_valid_one_whatever_its_score(self):
        valid = np.array([False, True, True, True, True, True])
        ranks = full_ranks(CANDIDATES, np.array([9.0, .1, .2, .3, .4, .5]), valid)
        self.assertEqual(ranks[0], 6)

    def test_popularity_ranks_follow_the_order(self):
        order = np.array([30, 10, 60, 20, 40, 50])
        self.assertEqual(popularity_ranks(CANDIDATES, order).tolist(), [2, 4, 1, 5, 6, 3])

    def test_fusion_formula_by_hand(self):
        self.assertAlmostEqual(float(fuse(1, 3, 0.8, 10)), 0.8 / 11 + 0.2 / 13)
        self.assertAlmostEqual(float(fuse(2, 1, 0.6, 60)), 0.6 / 62 + 0.4 / 61)


class Fusion(unittest.TestCase):
    order = np.array([50, 40, 30, 20, 10, 60])
    scores = np.array([[.9, .8, .7, .6, .5, .4], [0, 0, 0, 0, 0, 0]])
    valid = np.array([[True] * 6, [False] * 6])

    def lists(self, weight=0.8, constant=10, users=(1, 2)):
        return hybrid_lists(list(users), CANDIDATES, self.scores, self.valid, self.order, weight, constant, 5)

    def test_cold_user_takes_the_common_fallback(self):
        lists, _, counts = self.lists()
        self.assertEqual(lists[2], self.order[:5].tolist())
        self.assertEqual(counts.tolist(), [5, 0])

    def test_lists_hold_five_distinct_pool_ids(self):
        lists, _, _ = self.lists()
        self.assertEqual(len(set(lists[1])), 5)
        self.assertLessEqual(set(lists[1]), set(CANDIDATES.tolist()))

    def test_weight_moves_the_list_towards_a_component(self):
        ppr_led = self.lists(weight=0.99)[0][1]
        popularity_led = self.lists(weight=0.01)[0][1]
        self.assertEqual(ppr_led[0], 10)
        self.assertEqual(popularity_led[0], 50)

    def test_equal_fused_scores_break_by_ascending_id(self):
        # Opposite rankings at w = 0.5 make the pairs (10, 60), (20, 50), (30, 40) exact ties.
        order = np.array([60, 50, 40, 30, 20, 10])
        scores = np.array([[.6, .5, .4, .3, .2, .1]])
        lists, _, _ = hybrid_lists([1], CANDIDATES, scores, np.ones((1, 6), dtype=bool), order, 0.5, 10, 5)
        self.assertEqual(lists[1], [10, 60, 20, 50, 30])

    def test_deterministic(self):
        self.assertEqual(self.lists()[0], self.lists()[0])

    def test_hand_computed_top_item(self):
        fused, _ = hybrid_scores(CANDIDATES, self.scores, self.valid, self.order, 0.6, 10)
        self.assertAlmostEqual(fused[0, 0], 0.6 / 11 + 0.4 / 15)

    def test_grid_is_exactly_the_four_declared_runs(self):
        ids = [hybrid_config_id(p) for p in hybrid_grid(load_config())]
        self.assertEqual(ids, ["hybrid_w0.6_k10", "hybrid_w0.6_k60", "hybrid_w0.8_k10", "hybrid_w0.8_k60"])

    def test_paired_difference_against_any_component(self):
        report = paired_difference(np.array([.5, .5]), np.array([.25, .75]), load_config())
        self.assertEqual(report["mean"], 0.0)
        self.assertEqual((report["users_better"], report["users_worse"]), (1, 1))


class TestScheduleBoundary(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()
        self.name = self.cfg["files"]["schedule_test"]
        self.saved = set(csvio.parsed_sources)
        csvio.parsed_sources.clear()
        self.folder = tempfile.TemporaryDirectory()
        header = ",".join(csvio.SCHEDULE_HEADER)
        row = "7,Ch,27.07.2020 06:00:00,1800,'Show',123,Инфо,g,,,2000,dir,actor"
        zero = "7,Ch,27.07.2020 06:30:00,1800,'Other',0,Другое,,,,,,"
        Path(self.folder.name, self.name).write_text("\n".join([header, row, zero]) + "\n", encoding="utf-8")
        self.previous = os.environ.get("SWEETTV_DATA_DIR")
        os.environ["SWEETTV_DATA_DIR"] = self.folder.name

    def tearDown(self):
        csvio.parsed_sources.clear()
        csvio.parsed_sources.update(self.saved)
        os.environ.pop("SWEETTV_DATA_DIR", None)
        if self.previous is not None:
            os.environ["SWEETTV_DATA_DIR"] = self.previous
        self.folder.cleanup()

    def clean_refit(self):
        return {"stage": "refit", "parsed_sources": ["dataset11-30.csv"]}

    def test_refused_without_a_refit_manifest(self):
        with self.assertRaises(PermissionError):
            list(csvio.iter_test_availability(self.cfg, {}))

    def test_refused_for_another_stage(self):
        with self.assertRaises(PermissionError):
            list(csvio.iter_test_availability(self.cfg, {"stage": "split", "parsed_sources": []}))

    def test_refused_if_the_refit_parsed_the_test_schedule(self):
        with self.assertRaises(PermissionError):
            list(csvio.iter_test_availability(self.cfg, {"stage": "refit", "parsed_sources": [self.name]}))

    def test_yields_only_labelled_rows_and_availability_fields(self):
        rows = list(csvio.iter_test_availability(self.cfg, self.clean_refit()))
        self.assertEqual(len(rows), 1)
        self.assertEqual(set(rows[0]), {"channel", "start_text", "duration", "show_id", "title", "category"})
        self.assertEqual(rows[0]["show_id"], 123)

    def test_any_fit_is_refused_after_the_file_was_parsed(self):
        csvio.require_test_schedule_unread(self.cfg)
        list(csvio.iter_test_availability(self.cfg, self.clean_refit()))
        with self.assertRaises(PermissionError):
            csvio.require_test_schedule_unread(self.cfg)

    def test_ppr_fit_is_refused_after_the_file_was_parsed(self):
        list(csvio.iter_test_availability(self.cfg, self.clean_refit()))
        table = {"user": np.array([0, 0]), "item": np.array([1, 2]), "seconds": np.array([600, 600])}
        params = {"link_probability": 0.3, "interaction_weighting": "seconds"}
        with self.assertRaises(PermissionError):
            ppr_probabilities(table, params, self.cfg, np.array([0]))

    def test_interaction_build_is_refused_after_the_file_was_parsed(self):
        list(csvio.iter_test_availability(self.cfg, self.clean_refit()))
        with self.assertRaises(PermissionError):
            build_interactions(self.cfg, None, [(0, 1), (1, 2)])

    def test_mf_fit_is_refused_after_the_file_was_parsed(self):
        list(csvio.iter_test_availability(self.cfg, self.clean_refit()))
        table = {"user": np.array([0, 0]), "item": np.array([1, 2]), "seconds": np.array([600, 600])}
        params = {"factors": 2, "confidence_alpha": 2, "regularization": 1}
        with self.assertRaises(PermissionError):
            next(mf_checkpoints(table, params, self.cfg, np.array([0]), np.array([1, 2]), [1, 2], [1]))

    def test_fit_schedule_accessor_still_refuses_the_test_role(self):
        with self.assertRaises(PermissionError):
            list(csvio.iter_fit_schedule(self.cfg, "schedule_test"))


class FullRefitTables(unittest.TestCase):
    def test_empty_second_window_gives_an_empty_table(self):
        table = user_item_table(np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
        self.assertEqual(len(table["user"]), 0)
        self.assertEqual(len(table["seconds"]), 0)

    def test_watched_tables_with_no_hold_out(self):
        airings = {"start": np.array([0]), "end": np.array([3000]), "show_id": np.array([9])}
        interactions = {"airing": np.array([0]), "overlap_capped": np.array([600]), "user": np.array([1]), "window": np.array([0])}
        tables = watched_tables(airings, interactions, 300, 0.5)
        self.assertEqual(tables["train"]["item"].tolist(), [9])
        self.assertEqual(len(tables["validation"]["user"]), 0)


class CandidateFile(unittest.TestCase):
    users = ["18417531283109304442", "3398461054087191302"]
    lists = [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]]
    pool = {1, 2, 3, 4, 5, 6}

    def text(self):
        return render_csv(self.users, self.lists, self.pool)

    def test_exact_bytes(self):
        self.assertEqual(self.text(), "user_id,tv_show_id\n18417531283109304442,1 2 3 4 5\n3398461054087191302,5 4 3 2 1\n")

    def test_valid_file_passes(self):
        validate_csv(self.text(), self.users, self.pool)

    def test_rendering_is_deterministic(self):
        self.assertEqual(self.text().encode(), self.text().encode())

    def test_row_count_mismatch_is_refused_when_rendering(self):
        with self.assertRaises(ValueError):
            render_csv(self.users, self.lists[:1], self.pool)

    def bad(self, text):
        with self.assertRaises(ValueError):
            validate_csv(text, self.users, self.pool)

    def test_missing_trailing_newline(self):
        self.bad(self.text()[:-1])

    def test_extra_blank_line(self):
        self.bad(self.text() + "\n")

    def test_windows_line_ending(self):
        self.bad(self.text().replace("\n", "\r\n"))

    def test_wrong_header(self):
        self.bad(self.text().replace("user_id,tv_show_id", "user,tv_show_id"))

    def test_wrong_user_order(self):
        with self.assertRaises(ValueError):
            validate_csv(self.text(), self.users[::-1], self.pool)

    def test_duplicate_id(self):
        self.bad(self.text().replace("1 2 3 4 5", "1 2 3 4 4"))

    def test_zero_id(self):
        self.bad(self.text().replace("1 2 3 4 5", "0 2 3 4 5"))

    def test_four_ids(self):
        self.bad(self.text().replace("1 2 3 4 5", "1 2 3 4"))

    def test_scientific_notation(self):
        self.bad(self.text().replace("1 2 3 4 5", "1e3 2 3 4 5"))

    def test_nan_cell(self):
        self.bad(self.text().replace("1 2 3 4 5", "nan 2 3 4 5"))

    def test_id_outside_pool(self):
        self.bad(self.text().replace("1 2 3 4 5", "1 2 3 4 99"))

    def test_extra_index_column(self):
        self.bad(self.text().replace("18417531283109304442,", "0,18417531283109304442,"))


if __name__ == "__main__":
    unittest.main()
