"""Refit popularity, the selected PPR and the selected MF on all of weeks 11-30.

There is no hold-out here. Everything is frozen into the `refit` stage before
the test schedule may be opened. MF is a report artifact and feeds no candidate.
"""
import os

for name in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[name] = "1"

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from sweettv import mf
from sweettv.artifacts import save_stage, sha256_file, source_hashes, stage_dir, write_result
from sweettv.config import RESULTS_DIR, load_config
from sweettv.csvio import read_cohort, require_test_schedule_unread
from sweettv.pipeline import build_interactions
from sweettv.runlog import RunTimer
from sweettv.runner import ppr_config_id, ppr_grid, ppr_probabilities
from sweettv.split import watched_tables
from sweettv.timeutil import WallClock


def main():
    timer = RunTimer("12_full_refit")
    cfg = load_config()
    source_hashes(cfg)
    require_test_schedule_unread(cfg)
    clock = WallClock(cfg["timezone"])
    start = clock.view_ts(cfg["final"]["train"]["start"])
    end = clock.view_ts(cfg["final"]["train"]["end"])
    began = time.perf_counter()
    airings, interactions, coverage, user_ids, audit = build_interactions(cfg, clock, [(start, end), (end, end)])
    build_seconds = time.perf_counter() - began
    tables = watched_tables(airings, interactions, cfg["watched"]["primary_seconds"], cfg["watched"]["airing_fraction"])
    train = tables["train"]
    if len(tables["validation"]["user"]):
        raise SystemExit("the full refit must have no hold-out rows")

    position = {user: index for index, user in enumerate(user_ids)}
    cohort = np.array([position[user] for user in read_cohort(cfg)], dtype=np.int32)

    items, counts = np.unique(train["item"], return_counts=True)
    params = next(p for p in ppr_grid(cfg) if ppr_config_id(p) == cfg["hybrid"]["ppr_run"])
    began = time.perf_counter()
    item_ids, probabilities, details = ppr_probabilities(train, params, cfg, cohort)
    ppr_seconds = time.perf_counter() - began

    grid = json.loads((RESULTS_DIR / "model_grid.json").read_text(encoding="utf-8"))
    mf_params = next(r["params"] for r in grid["runs"] if r["run_id"] == cfg["final"]["mf_run"])
    began = time.perf_counter()
    model = mf.fit_wals(train, mf_params["factors"], mf_params["confidence_alpha"], mf_params["regularization"],
                        mf_params["iterations"], cfg["seed"], cfg["models"]["mf"]["init_std"], cfg["models"]["strength_scale_seconds"])
    mf_seconds = time.perf_counter() - began

    arrays = {
        "train_table": train,
        "popularity": {"items": items, "users_watching": counts},
        "ppr": {"item_ids": item_ids, "probabilities": probabilities, "cohort": cohort},
        "mf_report_artifact": {key: model[key] for key in ("user_ids", "item_ids", "user_factors", "item_factors")},
        "coverage": coverage,
    }
    manifest = save_stage(cfg, "refit", arrays, {"users.txt": "\n".join(user_ids) + "\n"})
    write_result(RESULTS_DIR / "full_refit.json", {
        "training_window": cfg["final"]["train"],
        "hold_out": "none",
        "schedules_parsed": manifest["parsed_sources"],
        "view_rows": audit["view_rows"],
        "views_outside_training": audit["views_outside_both_windows"],
        "last_view_stop_utc": audit["last_view_stop_utc"],
        "training_end_utc": end,
        "stop_time_mismatches_after_utc": audit["stop_time_disagrees_with_start_plus_duration_after_utc"],
        "airings": audit["schedule"]["airings_after_resolution"],
        "labelled_airings": audit["schedule"]["labelled_airings"],
        "user_airing_rows": int(len(interactions["user"])),
        "watched_airing_rows": int(train["airings"].sum()),
        "user_programme_pairs": int(len(train["user"])),
        "users_with_positive": int(len(np.unique(train["user"]))),
        "programmes_with_positive": int(len(items)),
        "cohort_cold_no_positive": int((~np.isin(cohort, train["user"])).sum()),
        "channels_with_training_views": int(len(coverage["train_channels"])),
        "ppr": {"run": cfg["hybrid"]["ppr_run"], **details, "fit_seconds": round(ppr_seconds, 2)},
        "mf_report_artifact": {"run": cfg["final"]["mf_run"], "users": int(len(model["user_ids"])), "programmes": int(len(model["item_ids"])),
                               "final_loss": model["losses"][-1], "used_in_candidates": False, "fit_seconds": round(mf_seconds, 2)},
        "popularity_top_5": [[int(i), int(c)] for i, c in sorted(zip(items.tolist(), counts.tolist()), key=lambda p: (-p[1], p[0]))[:5]],
        "interaction_build_seconds": round(build_seconds, 2),
        "refit_manifest_sha256": sha256_file(stage_dir(cfg, "refit") / "manifest.json"),
    })
    timer.finish(artifact_bytes=sum(entry["bytes"] for entry in manifest["files"].values()))


if __name__ == "__main__":
    main()
