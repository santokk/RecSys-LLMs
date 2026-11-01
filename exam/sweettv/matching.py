"""View-to-airing matching on half-open intervals [start, end)."""
import numpy as np

CHANNEL_KEY = 1 << 33


def clip_to_windows(start, stop, windows):
    """Cut each view at window edges. Returns (view_pos, window, start, stop)."""
    parts = []
    for window, (lower, upper) in enumerate(windows):
        begin, finish = np.maximum(start, lower), np.minimum(stop, upper)
        inside = np.flatnonzero(begin < finish)
        parts.append((inside, np.full(len(inside), window, dtype=np.int8), begin[inside], finish[inside]))
    return tuple(np.concatenate(column) for column in zip(*parts))


class AiringIndex:
    """Interval lookup over a disjoint timeline sorted by (channel, start)."""

    def __init__(self, channel, start, end):
        base = channel.astype(np.int64) * CHANNEL_KEY
        self.start_key = base + start
        self.end_key = base + end
        if np.any(np.diff(self.start_key) <= 0):
            raise ValueError("airings must be sorted and unique by (channel, start)")
        if np.any(end <= start):
            raise ValueError("airings must have positive length")
        if np.any(self.end_key[:-1] > self.start_key[1:]):
            raise ValueError("airings must be disjoint within a channel")

    def overlaps(self, channel, start, stop):
        """All (segment, airing) pairs with positive overlap."""
        base = channel.astype(np.int64) * CHANNEL_KEY
        low = np.searchsorted(self.end_key, base + start, side="right")
        high = np.searchsorted(self.start_key, base + stop, side="left")
        count = high - low
        if np.any(count < 0):
            raise ValueError("timeline is not disjoint")
        segment = np.repeat(np.arange(len(channel)), count)
        first_of_segment = np.repeat(np.cumsum(count) - count, count)
        airing = np.repeat(low, count) + np.arange(len(segment)) - first_of_segment
        return segment, airing


def match_segments(index, airing_start, airing_end, show_id, channel, start, stop):
    """Overlap pairs for view segments, with the item flag.

    All airings are returned so unlabelled airtime can be audited; `is_item` is
    the only route into interactions, and it is false for `tv_show_id = 0`.
    """
    segment, airing = index.overlaps(channel, start, stop)
    return {
        "segment": segment,
        "airing": airing,
        "overlap_start": np.maximum(start[segment], airing_start[airing]),
        "overlap_stop": np.minimum(stop[segment], airing_end[airing]),
        "is_item": show_id[airing] != 0,
    }


def union_seconds(start, stop, first, fragments):
    """Seconds covered by the union of each group's intervals (sorted by start)."""
    total = stop[first] - start[first]
    for group in np.flatnonzero(fragments > 1):
        i = first[group]
        low, high, covered = start[i], stop[i], 0
        for j in range(i + 1, i + fragments[group]):
            if start[j] > high:
                covered += high - low
                low, high = start[j], stop[j]
            elif stop[j] > high:
                high = stop[j]
        total[group] = covered + high - low
    return total


def aggregate_fragments(user, airing, window, overlap_start, overlap_stop, view_seconds, airing_seconds):
    """Sum overlap per (user, airing, window), capped at the airing's duration.

    `overlap_union` keeps the de-duplicated seconds as a diagnostic of how much
    concurrent or repeated rows inflate the capped sum.
    """
    order = np.lexsort((overlap_start, window, airing, user))
    user, airing, window = user[order], airing[order], window[order]
    overlap_start, overlap_stop, view_seconds = overlap_start[order], overlap_stop[order], view_seconds[order]
    new_group = np.r_[True, (user[1:] != user[:-1]) | (airing[1:] != airing[:-1]) | (window[1:] != window[:-1])]
    first = np.flatnonzero(new_group)
    fragments = np.diff(np.r_[first, len(user)])
    overlap_sum = np.add.reduceat(overlap_stop - overlap_start, first)
    return {
        "user": user[first],
        "airing": airing[first],
        "window": window[first],
        "overlap_sum": overlap_sum,
        "overlap_capped": np.minimum(overlap_sum, airing_seconds[airing[first]]),
        "overlap_union": union_seconds(overlap_start, overlap_stop, first, fragments),
        "fragments": fragments,
        "longest_view": np.maximum.reduceat(view_seconds, first),
    }


def watched_mask(overlap, airing_seconds, threshold_seconds, airing_fraction):
    """overlap >= min(threshold_seconds, airing_fraction * airing duration)."""
    return overlap >= np.minimum(threshold_seconds, airing_fraction * airing_seconds)
