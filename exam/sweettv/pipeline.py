"""Raw-to-interactions build, reused unchanged by the stability fold."""
import numpy as np

from .csvio import iter_view_chunks, require_test_schedule_unread
from .matching import AiringIndex, aggregate_fragments, clip_to_windows, match_segments, watched_mask
from .schedule import load_schedule, resolve_overlaps
from .split import VALIDATION

WINDOW_NAMES = ("train", "validation")
CLASSES = ("channel_not_in_schedule", "schedule_hole", "unlabelled_only", "one_labelled_id", "multiple_labelled_ids")


def build_interactions(cfg, clock, windows):
    """Stream the view log and return airings, interactions, coverage, users and an audit."""
    require_test_schedule_unread(cfg)
    raw, rows_after_validation = load_schedule(cfg, clock, windows[VALIDATION][1])
    airings, resolution = resolve_overlaps(raw["channel"], raw["start"], raw["duration"], raw["show_id"])
    schedule_clock = {"nonexistent": clock.nonexistent, "ambiguous": clock.ambiguous}
    clock.nonexistent = clock.ambiguous = 0

    airing_seconds = airings["end"] - airings["start"]
    index = AiringIndex(airings["channel"], airings["start"], airings["end"])
    scheduled_channels = np.unique(airings["channel"])

    users = {}
    pairs = {key: [] for key in ("user", "airing", "window", "overlap_start", "overlap_stop", "view_seconds")}
    class_rows = np.zeros((2, len(CLASSES)), dtype=np.int64)
    class_seconds = np.zeros((2, len(CLASSES)), dtype=np.int64)
    covered_seconds, labelled_seconds = np.zeros(2, dtype=np.int64), np.zeros(2, dtype=np.int64)
    channels_seen = [set(), set()]
    users_seen = [set(), set()]
    view_rows = stop_mismatch = views_outside = segments_total = 0
    first_view, last_stop = None, None

    for chunk in iter_view_chunks(cfg):
        user = np.array([users.setdefault(row[0], len(users)) for row in chunk], dtype=np.int32)
        channel = np.array([row[1] for row in chunk], dtype=np.int32)
        start = np.array([clock.view_ts(row[2]) for row in chunk], dtype=np.int64)
        duration_sec = np.array([row[4] for row in chunk], dtype=np.int64)
        stop = start + duration_sec
        stop_mismatch += int((np.array([clock.view_ts(row[3]) for row in chunk], dtype=np.int64) != stop).sum())
        view_rows += len(chunk)
        first_view = int(start.min()) if first_view is None else min(first_view, int(start.min()))
        last_stop = int(stop.max()) if last_stop is None else max(last_stop, int(stop.max()))

        view, window, segment_start, segment_stop = clip_to_windows(start, stop, windows)
        views_outside += len(chunk) - len(np.unique(view))
        segment_channel, segment_user = channel[view], user[view]
        segments_total += len(view)
        for w in (0, 1):
            channels_seen[w].update(np.unique(segment_channel[window == w]).tolist())
            users_seen[w].update(np.unique(segment_user[window == w]).tolist())

        match = match_segments(index, airings["start"], airings["end"], airings["show_id"],
                               segment_channel, segment_start, segment_stop)
        segment, airing, is_item = match["segment"], match["airing"], match["is_item"]
        seconds = match["overlap_stop"] - match["overlap_start"]

        any_airing = np.bincount(segment, minlength=len(view))
        distinct = np.unique(segment[is_item].astype(np.int64) << 32 | airings["show_id"][airing[is_item]])
        distinct_ids = np.bincount(distinct >> 32, minlength=len(view))
        on_schedule = np.isin(segment_channel, scheduled_channels)
        code = np.select([~on_schedule, any_airing == 0, distinct_ids == 0, distinct_ids == 1], [0, 1, 2, 3], default=4)
        np.add.at(class_rows, (window, code), 1)
        np.add.at(class_seconds, (window, code), segment_stop - segment_start)
        np.add.at(covered_seconds, window[segment], seconds)
        np.add.at(labelled_seconds, window[segment[is_item]], seconds[is_item])

        pairs["user"].append(segment_user[segment[is_item]])
        pairs["airing"].append(airing[is_item].astype(np.int32))
        pairs["window"].append(window[segment[is_item]])
        pairs["overlap_start"].append(match["overlap_start"][is_item])
        pairs["overlap_stop"].append(match["overlap_stop"][is_item])
        pairs["view_seconds"].append(duration_sec[view[segment[is_item]]])

    columns = {key: np.concatenate(value) for key, value in pairs.items()}
    interactions = aggregate_fragments(columns["user"], columns["airing"], columns["window"],
                                       columns["overlap_start"], columns["overlap_stop"],
                                       columns["view_seconds"], airing_seconds)
    pair_count = len(columns["user"])
    del columns, pairs

    durations = airing_seconds[interactions["airing"]]
    fraction = cfg["watched"]["airing_fraction"]
    primary = watched_mask(interactions["overlap_capped"], durations, cfg["watched"]["primary_seconds"], fraction)
    primary_union = watched_mask(interactions["overlap_union"], durations, cfg["watched"]["primary_seconds"], fraction)
    long_view = interactions["longest_view"] >= cfg["long_view_seconds"]

    audit = {
        "view_rows": view_rows,
        "view_segments_after_window_clipping": segments_total,
        "views_cut_at_window_boundary": segments_total - view_rows + views_outside,
        "views_outside_both_windows": views_outside,
        "users_in_view_log": len(users),
        "first_view_start_utc": first_view,
        "last_view_stop_utc": last_stop,
        "window_bounds_utc": dict(zip(WINDOW_NAMES, windows)),
        "stop_time_disagrees_with_start_plus_duration_after_utc": stop_mismatch,
        "wall_clock_special_cases": {
            "schedule": schedule_clock,
            "views": {"nonexistent": clock.nonexistent, "ambiguous": clock.ambiguous},
        },
        "schedule": {
            "rows_kept_for_fitting": int(len(raw["start"])),
            "rows_starting_at_or_after_validation_end_dropped": rows_after_validation,
            "airings_after_resolution": int(len(airings["start"])),
            "labelled_airings": int((airings["show_id"] != 0).sum()),
            "scheduled_channels": int(len(scheduled_channels)),
            **resolution,
        },
        "views_by_class": {
            name: {
                cls: {"segments": int(class_rows[w, c]), "segment_seconds": int(class_seconds[w, c])}
                for c, cls in enumerate(CLASSES)
            } for w, name in enumerate(WINDOW_NAMES)
        },
        "seconds": {
            name: {
                "viewed": int(class_seconds[w].sum()),
                "covered_by_any_airing": int(covered_seconds[w]),
                "covered_by_labelled_airing": int(labelled_seconds[w]),
            } for w, name in enumerate(WINDOW_NAMES)
        },
        "users_with_views": {name: len(users_seen[w]) for w, name in enumerate(WINDOW_NAMES)},
        "channels_with_views": {name: len(channels_seen[w]) for w, name in enumerate(WINDOW_NAMES)},
        "aggregation": {
            "labelled_overlap_pairs": pair_count,
            "user_airing_window_rows": int(len(interactions["user"])),
            "rows_built_from_multiple_fragments": int((interactions["fragments"] > 1).sum()),
            "rows_where_cap_applied": int((interactions["overlap_sum"] > interactions["overlap_capped"]).sum()),
            "rows_where_union_below_capped_sum": int((interactions["overlap_union"] < interactions["overlap_capped"]).sum()),
            "primary_positives": int(primary.sum()),
            "primary_positives_lost_under_union": int((primary & ~primary_union).sum()),
            "primary_positives_touching_view_at_least_long_seconds": int((primary & long_view).sum()),
            "long_view_seconds": cfg["long_view_seconds"],
        },
    }
    coverage = {
        "train_channels": np.array(sorted(channels_seen[0]), dtype=np.int32),
        "validation_channels": np.array(sorted(channels_seen[1]), dtype=np.int32),
        "train_users": np.array(sorted(users_seen[0]), dtype=np.int32),
        "validation_users": np.array(sorted(users_seen[1]), dtype=np.int32),
    }
    return airings, interactions, coverage, list(users), audit
