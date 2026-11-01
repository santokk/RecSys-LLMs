"""Open the test schedule, after the refit, for availability and lookup only.

The output is a list of programme ids, their airing counts and labels. It holds
nothing about any user. Channel eligibility comes from training views.
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from sweettv.artifacts import load_arrays, load_manifest, save_stage, sha256_file, stage_dir, write_result
from sweettv.config import RESULTS_DIR, load_config
from sweettv.csvio import iter_fit_schedule, iter_test_availability
from sweettv.runlog import RunTimer
from sweettv.split import candidate_pool
from sweettv.timeutil import WallClock


def main():
    timer = RunTimer("13_test_availability")
    cfg = load_config()
    refit_manifest = load_manifest(cfg, "refit")
    clock = WallClock(cfg["timezone"])
    window = (clock.view_ts(cfg["final"]["test_window"]["start"]), clock.view_ts(cfg["final"]["test_window"]["end"]))
    coverage = load_arrays(cfg, "refit", "coverage")
    train_table = load_arrays(cfg, "refit", "train_table")

    channel, start, end, show_id = [], [], [], []
    titles, categories = {}, {}
    for row in iter_test_availability(cfg, refit_manifest):
        begins = clock.schedule_ts(row["start_text"])
        if row["duration"] > 0:
            channel.append(row["channel"]), start.append(begins), end.append(begins + row["duration"]), show_id.append(row["show_id"])
            titles.setdefault(row["show_id"], Counter())[row["title"]] += 1
            categories[row["show_id"]] = row["category"]
    from_test_file = len(channel)

    # A channel's new weekly block can begin after Monday 00:00, so the earlier
    # file still lists its first test-week hours. Those are availability too.
    for row in iter_fit_schedule(cfg, "schedule_validation"):
        identifier, length = int(row[5]), int(row[3])
        if identifier and length > 0:
            begins = clock.schedule_ts(row[2])
            if begins + length > window[0]:
                channel.append(int(row[0])), start.append(begins), end.append(begins + length), show_id.append(identifier)
                titles.setdefault(identifier, Counter())[row[4]] += 1
                categories[identifier] = row[6]

    airings = {key: np.array(value, dtype=np.int64) for key, value in
               (("channel", channel), ("start", start), ("end", end), ("show_id", show_id))}
    pool = candidate_pool(airings, window, coverage["train_channels"])
    any_channel = candidate_pool(airings, window, np.unique(airings["channel"]))
    earlier_rows = {key: value[from_test_file:] for key, value in airings.items()}
    test_rows = {key: value[:from_test_file] for key, value in airings.items()}
    only_earlier = np.setdiff1d(candidate_pool(earlier_rows, window, coverage["train_channels"]),
                                candidate_pool(test_rows, window, coverage["train_channels"]))
    seen = np.isin(pool, train_table["item"])
    in_pool = np.isin(airings["show_id"], pool) & np.isin(airings["channel"], coverage["train_channels"]) \
        & (airings["start"] < window[1]) & (airings["end"] > window[0])
    labels = {str(i): {"title": min(titles[i], key=lambda t: (-titles[i][t], t)), "category": categories[i]} for i in pool.tolist()}

    import json
    save_stage(cfg, "test_pool", {"pool": {"candidates": pool, "seen_in_training": seen}},
               {"labels.json": json.dumps(labels, ensure_ascii=False, sort_keys=True)}, depends_on=("refit",))
    write_result(RESULTS_DIR / "test_pool.json", {
        "policy": "labelled non-zero ids with an airing overlapping weeks 31-42 on a channel with a weeks 11-30 view",
        "test_window": cfg["final"]["test_window"],
        "use_of_test_schedule": "availability and title/category lookup only",
        "pool_size": int(len(pool)),
        "size_without_channel_restriction": int(len(any_channel)),
        "programmes_seen_in_training": int(seen.sum()),
        "programmes_cold_unseen_in_training": int((~seen).sum()),
        "ids_only_in_earlier_file_rows": int(len(only_earlier)),
        "eligible_channels": int(len(coverage["train_channels"])),
        "channels_airing_pool_ids": int(len(np.unique(airings["channel"][in_pool]))),
        "pool_airings": int(in_pool.sum()),
        "pool_airing_seconds_inside_window": int((np.minimum(airings["end"][in_pool], window[1]) - np.maximum(airings["start"][in_pool], window[0])).sum()),
        "labelled_test_rows_read": from_test_file,
        "refit_manifest_sha256_at_open": sha256_file(stage_dir(cfg, "refit") / "manifest.json"),
    })
    timer.finish()


if __name__ == "__main__":
    main()
