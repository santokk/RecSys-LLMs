import calendar
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sweettv.timeutil import WallClock


def utc(*fields):
    return calendar.timegm((*fields,))


class SpringTransition(unittest.TestCase):
    def setUp(self):
        self.clock = WallClock("Europe/Kyiv")

    def test_offset_changes_from_two_to_three_hours(self):
        self.assertEqual(self.clock.view_ts("2020-03-28 12:00:00"), utc(2020, 3, 28, 10, 0, 0))
        self.assertEqual(self.clock.view_ts("2020-03-29 02:59:59"), utc(2020, 3, 29, 0, 59, 59))
        self.assertEqual(self.clock.view_ts("2020-03-29 04:00:00"), utc(2020, 3, 29, 1, 0, 0))
        self.assertEqual(self.clock.view_ts("2020-03-30 12:00:00"), utc(2020, 3, 30, 9, 0, 0))

    def test_jump_takes_one_second(self):
        before = self.clock.view_ts("2020-03-29 02:59:59")
        after = self.clock.view_ts("2020-03-29 04:00:00")
        self.assertEqual(after - before, 1)

    def test_view_across_jump_matches_duraton(self):
        # Real shape of the 117 log rows: stop - start on the wall clock is 3600 s too large.
        start = self.clock.view_ts("2020-03-29 02:30:00")
        stop = self.clock.view_ts("2020-03-29 04:30:00")
        self.assertEqual(stop - start, 3600)

    def test_schedule_and_view_formats_agree(self):
        self.assertEqual(self.clock.schedule_ts("29.03.2020 04:00:00"), self.clock.view_ts("2020-03-29 04:00:00"))
        self.assertEqual(self.clock.schedule_ts("09.03.2020 06:00:00"), self.clock.view_ts("2020-03-09 06:00:00"))

    def test_airing_across_jump_ends_at_next_start(self):
        start = self.clock.schedule_ts("29.03.2020 02:00:00")
        next_start = self.clock.schedule_ts("29.03.2020 05:00:00")
        self.assertEqual(start + 7200, next_start)

    def test_skipped_hour_is_counted_not_silent(self):
        value = self.clock.view_ts("2020-03-29 03:30:00")
        self.assertEqual(self.clock.nonexistent, 1)
        self.assertEqual(value, utc(2020, 3, 29, 1, 30, 0))

    def test_malformed_timestamps_raise(self):
        for text in ("2020-03-09", "2020/03/09 00:00:00", "2020-13-09 00:00:00", "2020-03-09 24:00:00"):
            with self.assertRaises(ValueError):
                self.clock.view_ts(text)


if __name__ == "__main__":
    unittest.main()
