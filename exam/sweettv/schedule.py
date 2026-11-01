"""Airing intervals: schedule start plus schedule duration, in UTC."""
from collections import Counter

import numpy as np

from .csvio import FIT_SCHEDULE_ROLES, iter_fit_schedule

MAX_LOGGED_CONFLICTS = 50


def load_schedule(cfg, clock, not_after):
    """Read the fitting schedules, dropping rows that start at or after `not_after`."""
    channel, start, duration, show_id, source = [], [], [], [], []
    dropped = 0
    for index, role in enumerate(FIT_SCHEDULE_ROLES):
        for row in iter_fit_schedule(cfg, role):
            begins = clock.schedule_ts(row[2])
            if begins >= not_after:
                dropped += 1
                continue
            channel.append(int(row[0]))
            start.append(begins)
            duration.append(int(row[3]))
            show_id.append(int(row[5]))
            source.append(index)
    table = {
        "channel": np.array(channel, dtype=np.int32),
        "start": np.array(start, dtype=np.int64),
        "duration": np.array(duration, dtype=np.int64),
        "show_id": np.array(show_id, dtype=np.int64),
        "source": np.array(source, dtype=np.int8),
    }
    return table, dropped


def programme_labels(cfg, ids):
    """Most frequent title and the category of each id, for report lookup only."""
    wanted = set(ids)
    titles = {show_id: Counter() for show_id in wanted}
    categories = {}
    for role in FIT_SCHEDULE_ROLES:
        for row in iter_fit_schedule(cfg, role):
            show_id = int(row[5])
            if show_id in wanted:
                titles[show_id][row[4]] += 1
                categories[show_id] = row[6]
    return {show_id: {"title": min(counts, key=lambda title: (-counts[title], title)), "category": categories[show_id]}
            for show_id, counts in titles.items()}


def resolve_overlaps(channel, start, duration, show_id):
    """Return a per-channel disjoint timeline and a count of every ambiguity.

    Rules, applied in order:
    1. rows with duration <= 0 are dropped;
    2. for rows sharing (channel, start) keep the longest, then the larger id;
    3. an airing still running when the next one starts ends at that start.
    """
    positive = duration > 0
    zero = int((~positive).sum())
    ch, st, du, sid = channel[positive], start[positive], duration[positive], show_id[positive]

    order = np.lexsort((-sid, -du, st, ch))
    ch, st, du, sid = ch[order], st[order], du[order], sid[order]
    repeat = (ch[1:] == ch[:-1]) & (st[1:] == st[:-1])
    same_start_conflicts = int((repeat & (sid[1:] != sid[:-1])).sum())
    keep = np.r_[True, ~repeat]
    ch, st, du, sid = ch[keep], st[keep], du[keep], sid[keep]

    end = st + du
    same_channel = ch[1:] == ch[:-1]
    overlap = same_channel & (st[1:] < end[:-1])
    earlier = np.flatnonzero(overlap)
    removed = end[earlier] - st[earlier + 1]
    first_id, second_id = sid[earlier], sid[earlier + 1]
    stats = {
        "zero_duration_dropped": zero,
        "same_start_dropped": int(repeat.sum()),
        "same_start_with_different_ids": same_start_conflicts,
        "overlaps_truncated": int(len(earlier)),
        "overlap_seconds_removed": int(removed.sum()),
        "overlaps_touching_label": int(((first_id != 0) | (second_id != 0)).sum()),
        "overlaps_between_two_labelled_ids": int(((first_id != 0) & (second_id != 0) & (first_id != second_id)).sum()),
        "overlap_detail": [
            {"channel": int(ch[i]), "earlier_id": int(sid[i]), "later_id": int(sid[i + 1]),
             "later_start_utc": int(st[i + 1]), "seconds_removed": int(end[i] - st[i + 1])}
            for i in earlier[:MAX_LOGGED_CONFLICTS]
        ],
    }
    end[earlier] = st[earlier + 1]
    gap = st[1:] - end[:-1]
    stats["gaps_between_airings"] = int(((gap > 0) & same_channel).sum())
    stats["gap_seconds"] = int(gap[(gap > 0) & same_channel].sum())
    airings = {"channel": ch, "start": st, "end": end, "show_id": sid}
    return airings, stats
