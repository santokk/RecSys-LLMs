"""Verify preprocessing, split and baseline artifacts against a second implementation."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import calendar

import numpy as np

import independent as ind
from independent import check
from sweettv.artifacts import load_arrays, load_manifest, load_text, write_result
from sweettv.config import RESULTS_DIR, load_config
from sweettv.runlog import RunTimer

TEST_ROLE = "schedule" + "_test"


def check_manifests(cfg):
    test_name = cfg["files"][TEST_ROLE]
    for stage in ("preprocess", "split", "popularity"):
        manifest = load_manifest(cfg, stage)
        check(f"manifest current: {stage}", True, manifest["parsed_sources"])
        check(f"test schedule never parsed: {stage}", test_name not in manifest["parsed_sources"])
    ind.check_test_boundary_sources(cfg)


def check_schema(cfg):
    airings = load_arrays(cfg, "preprocess", "airings")
    rows = load_arrays(cfg, "preprocess", "interactions")
    pool = load_arrays(cfg, "split", "pool")
    upper = calendar.timegm((2020, 7, 26, 21, 0, 0))  # 2020-07-27 00:00 in Kyiv
    lengths = airings["end"] - airings["start"]
    key = airings["channel"].astype(np.int64) << 33 | airings["start"]
    check("airings: equal column lengths", len({len(v) for v in airings.values()}) == 1, len(key))
    check("airings: sorted unique (channel, start)", bool(np.all(np.diff(key) > 0)))
    check("airings: positive length", bool(np.all(lengths > 0)))
    same = airings["channel"][1:] == airings["channel"][:-1]
    check("airings: disjoint per channel", bool(np.all(airings["end"][:-1][same] <= airings["start"][1:][same])))
    check("airings: none starts at or after validation end", bool(np.all(airings["start"] < upper)))
    check("interactions: equal column lengths", len({len(v) for v in rows.values()}) == 1, len(rows["user"]))
    triple = rows["user"].astype(np.int64) << 33 | rows["airing"].astype(np.int64) << 1 | rows["window"]
    check("interactions: unique (user, airing, window)", len(np.unique(triple)) == len(triple))
    check("interactions: every airing has an id", bool(np.all(airings["show_id"][rows["airing"]] != 0)))
    check("interactions: capped <= airing duration", bool(np.all(rows["overlap_capped"] <= lengths[rows["airing"]])))
    check("interactions: 0 < capped <= sum", bool(np.all((rows["overlap_capped"] > 0) & (rows["overlap_capped"] <= rows["overlap_sum"]))))
    check("interactions: union <= sum and <= airing", bool(np.all((rows["overlap_union"] <= rows["overlap_sum"]) & (rows["overlap_union"] <= lengths[rows["airing"]]))))
    check("interactions: window is 0 or 1", set(np.unique(rows["window"]).tolist()) <= {0, 1})
    for seconds in sorted({cfg["watched"]["primary_seconds"], *cfg["watched"]["sensitivity_seconds"]}):
        tables = load_arrays(cfg, "split", f"tables_{seconds}")
        check(f"tables_{seconds}: no id 0", bool(np.all(tables["train_item"] != 0) and np.all(tables["validation_item"] != 0)))
    check("pool: sorted unique non-zero ids", bool(np.all(np.diff(pool["candidates"]) > 0) and pool["candidates"][0] > 0), len(pool["candidates"]))
    check("cohort: 1,260 distinct users", len(set(pool["cohort"].tolist())) == 1260)


def check_rederivation(cfg):
    lower = ind.view_time(cfg["windows"]["train"]["start"])
    middle = ind.view_time(cfg["windows"]["validation"]["start"])
    upper = ind.view_time(cfg["windows"]["validation"]["end"])
    core = ind.rederive_interactions(cfg, lower, middle, upper)
    check("raw view log: row count", core["rows"] == 5821987, core["rows"])
    check("raw view log: no view stops after week 30", core["latest_stop"] <= upper, core["latest_stop"])
    check("raw view log: no view starts in the test window", core["latest_start"] < upper, core["latest_start"])

    audit = json.loads((RESULTS_DIR / "split_audit.json").read_text(encoding="utf-8"))
    pre = json.loads((RESULTS_DIR / "preprocess_audit.json").read_text(encoding="utf-8"))
    check("rederived user-airing-window rows", len(core["totals"]) == pre["aggregation"]["user_airing_window_rows"], len(core["totals"]))
    capped_total = sum(min(v, core["airing_length"][a]) for (_, a), v in core["totals"].items())
    stored_rows = load_arrays(cfg, "preprocess", "interactions")
    check("rederived capped seconds", capped_total == int(stored_rows["overlap_capped"].sum()), capped_total)

    fraction = cfg["watched"]["airing_fraction"]
    primary = None
    for seconds in (60, 300, 600):
        sides, airing_rows = ind.watched_pairs(core, seconds, fraction)
        if seconds == cfg["watched"]["primary_seconds"]:
            primary = sides
        for window, name in enumerate(("train", "validation")):
            expected = audit["by_threshold"][str(seconds)][name]
            check(f"rederived watched rows {seconds}s {name}", airing_rows[window] == expected["watched_airing_rows"], airing_rows[window])
            check(f"rederived user-programme pairs {seconds}s {name}", len(sides[window]) == expected["user_programme_pairs"], len(sides[window]))

    pool = ind.rederive_pool(core)
    check("rederived candidate pool", pool == load_arrays(cfg, "split", "pool")["candidates"].tolist(), len(pool))
    order, popularity = ind.popularity_order(primary[0], pool)
    stored = load_arrays(cfg, "popularity", "popularity")
    check("rederived popularity scores", [popularity[i] for i in stored["candidates"].tolist()] == stored["scores"].tolist())
    check("rederived popularity order", order == stored["fallback_order"].tolist(), order[:5])
    check("user index matches artifact", core["users"] == {u: i for i, u in enumerate(load_text(cfg, "preprocess", "users.txt").split())})

    cohort = ind.read_cohort_users(cfg, core["users"])
    lists = {user: order[:5] for user in cohort}
    report = json.loads((RESULTS_DIR / "popularity_validation.json").read_text(encoding="utf-8"))
    targets = ind.top5_targets(primary[1], cohort, pool)
    map_all, map_nonempty = ind.map_at_5(lists, targets, cohort)
    primary_report = report["primary_top5_preference"]
    check("rederived primary MAP@5 all users", abs(map_all - primary_report["map_at_5_all_users"]) < 1e-12, map_all)
    check("rederived primary MAP@5 non-empty users", abs(map_nonempty - primary_report["map_at_5_nonempty_users"]) < 1e-12, map_nonempty)
    check("rederived non-empty targets", sum(1 for u in cohort if targets[u]) == primary_report["users_with_nonempty_target"])
    allowed = set(pool)
    broad = {user: set() for user in cohort}
    for (user, item) in primary[1]:
        if item in allowed and user in broad:
            broad[user].add(item)
    broad_map, _ = ind.map_at_5(lists, broad, cohort)
    check("rederived secondary broad MAP@5", abs(broad_map - report["secondary_broad_watched_diagnostic"]["map_at_5_all_users"]) < 1e-12, broad_map)


def main():
    timer = RunTimer("04_verify_artifacts")
    cfg = load_config()
    ind.check_sources(cfg)
    check_manifests(cfg)
    check_schema(cfg)
    check_rederivation(cfg)
    ind.check_repository(cfg)
    failed = [c["check"] for c in ind.checks if not c["passed"]]
    write_result(RESULTS_DIR / "verification.json", {"passed": len(ind.checks) - len(failed), "failed": failed, "checks": ind.checks})
    timer.finish(checks=len(ind.checks), failed=len(failed))
    if failed:
        raise SystemExit(f"verification failed: {failed}")


if __name__ == "__main__":
    main()
