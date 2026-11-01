"""Timestamp conversion.

Both the view log and the schedule store naive Europe/Kyiv wall-clock time, so
every timestamp becomes UTC epoch seconds before any arithmetic. Elapsed values
(`duraton`, schedule `duration`) are true seconds and are added in UTC, which
is what removes the one-hour error around the 2020-03-29 03:00 -> 04:00 jump.

A wall time inside the skipped hour cannot exist; it is counted and resolved
with fold=0 (the pre-transition offset). Ambiguous autumn times do the same.
"""
import calendar
from datetime import datetime
from zoneinfo import ZoneInfo


def _iso_day(day):
    if day[4] != "-" or day[7] != "-":
        raise ValueError(f"bad date: {day!r}")
    return int(day[:4]), int(day[5:7]), int(day[8:10])


def _dotted_day(day):
    if day[2] != "." or day[5] != ".":
        raise ValueError(f"bad date: {day!r}")
    return int(day[6:10]), int(day[3:5]), int(day[:2])


class WallClock:
    def __init__(self, tz_name):
        self.zone = ZoneInfo(tz_name)
        self.days = {}
        self.nonexistent = 0
        self.ambiguous = 0

    def view_ts(self, text):
        """`YYYY-MM-DD HH:MM:SS` to UTC epoch seconds."""
        return self._convert(text, _iso_day)

    def schedule_ts(self, text):
        """`DD.MM.YYYY HH:MM:SS` to UTC epoch seconds."""
        return self._convert(text, _dotted_day)

    def _convert(self, text, parse_day):
        if len(text) != 19 or text[10] != " " or text[13] != ":" or text[16] != ":":
            raise ValueError(f"bad timestamp: {text!r}")
        day = text[:10]
        info = self.days.get(day)
        if info is None:
            info = self.days[day] = self._inspect_day(*parse_day(day))
        hour, minute, second = int(text[11:13]), int(text[14:16]), int(text[17:19])
        if hour > 23 or minute > 59 or second > 59:
            raise ValueError(f"bad timestamp: {text!r}")
        midnight, offset, ymd = info
        if offset is None:
            return self._on_transition_day(ymd, hour, minute, second)
        return midnight + hour * 3600 + minute * 60 + second - offset

    def _inspect_day(self, year, month, day):
        first = datetime(year, month, day, 0, 0, 0, tzinfo=self.zone).utcoffset()
        last = datetime(year, month, day, 23, 59, 59, tzinfo=self.zone).utcoffset()
        offset = int(first.total_seconds()) if first == last else None
        return calendar.timegm((year, month, day, 0, 0, 0)), offset, (year, month, day)

    def _on_transition_day(self, ymd, hour, minute, second):
        wall = datetime(*ymd, hour, minute, second, tzinfo=self.zone)
        early, late = wall.replace(fold=0).utcoffset(), wall.replace(fold=1).utcoffset()
        if early < late:
            self.nonexistent += 1
        elif early > late:
            self.ambiguous += 1
        naive = calendar.timegm((*ymd, hour, minute, second))
        return naive - int(early.total_seconds())
