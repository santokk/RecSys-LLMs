"""Verify targets, selected models and metrics from the raw CSVs, fold included.

Training data here comes only from the second implementation's train side. If a
validation or test interaction had entered a model, its lists could not be
reproduced from that side alone.
"""
import json
import os

for name in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[name] = "1"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import independent as ind
from independent import check
from sweettv.artifacts import load_arrays, load_manifest, load_text, write_result
from sweettv.config import RESULTS_DIR, load_config
from sweettv.runlog import RunTimer

TEST_ROLE = "schedule" + "_test"


def main():
    timer = RunTimer("09_verify_models")
    cfg = load_config()
    lock = json.loads((RESULTS_DIR / "protocol_lock.json").read_text(encoding="utf-8"))
    grid = json.loads((RESULTS_DIR / "model_grid.json").read_text(encoding="utf-8"))
    fold = json.loads((RESULTS_DIR / "stability_fold.json").read_text(encoding="utf-8"))
    records = {record["run_id"]: record for record in grid["runs"]}
    selected = {"popularity": "popularity", **grid["selected"]}
    mf_params, ppr_params = records[selected["mf"]]["params"], records[selected["ppr"]]["params"]
    test_name = cfg["files"][TEST_ROLE]

    ind.check_sources(cfg)
    check("protocol: grids match the lock", lock["grids"] == cfg["models"])
    expected_runs = 1 + len(cfg["models"]["ppr"]["link_probability"]) * len(cfg["models"]["ppr"]["interaction_weighting"]) \
        + len(cfg["models"]["mf"]["factors"]) * len(cfg["models"]["mf"]["confidence_alpha"]) \
        * len(cfg["models"]["mf"]["regularization"]) * len(cfg["models"]["mf"]["iteration_checkpoints"])
    check("protocol: every declared run is recorded", len(records) == expected_runs, len(records))
    for method in ("mf", "ppr"):
        ids = [r for r in grid["runs"] if r["method"] == method]
        best = max(r["primary_top5_preference"]["map_at_5_all_users"] for r in ids)
        winner = next(r for r in ids if r["primary_top5_preference"]["map_at_5_all_users"] == best)
        check(f"selection follows primary MAP@5 only: {method}", winner["run_id"] == grid["selected"][method], winner["run_id"])
    for stage in ("preprocess", "split", "popularity", "models"):
        manifest = load_manifest(cfg, stage)
        check(f"manifest current: {stage}", True)
        check(f"test schedule never parsed: {stage}", test_name not in manifest["parsed_sources"])
    check("test schedule never parsed: stability fold", test_name not in fold["parsed_sources"], fold["parsed_sources"])
    ind.check_test_boundary_sources(cfg)

    lower = ind.view_time(cfg["windows"]["train"]["start"])
    middle = ind.view_time(cfg["windows"]["validation"]["start"])
    upper = ind.view_time(cfg["windows"]["validation"]["end"])
    main_split = ind.reproduce(cfg, lower, middle, upper, mf_params, ppr_params)

    stored = load_arrays(cfg, "split", f"tables_{cfg['watched']['primary_seconds']}")
    stored_train = dict(zip(zip(stored["train_user"].tolist(), stored["train_item"].tolist()), stored["train_seconds"].tolist()))
    check("model input equals train-side rederivation exactly", stored_train == dict(main_split["train"]), len(stored_train))
    overlap_keys = set(main_split["train"]) & set(main_split["validation"])
    check("train seconds carry no validation seconds", all(stored_train[key] == main_split["train"][key] for key in overlap_keys), len(overlap_keys))
    airings = load_arrays(cfg, "preprocess", "airings")
    rows = load_arrays(cfg, "preprocess", "interactions")
    train_rows = rows["window"] == 0
    check("every training interaction begins before the boundary", bool(np.all(airings["start"][rows["airing"][train_rows]] < middle)))

    targets_stored = load_arrays(cfg, "models", "targets")
    cohort = targets_stored["cohort"].tolist()
    check("cohort order matches template", cohort == main_split["cohort"])
    rebuilt = {user: {int(i) for i in row if i} for user, row in zip(cohort, targets_stored["target_ids"])}
    check("primary target rederived exactly", rebuilt == main_split["targets"], sum(len(v) for v in rebuilt.values()))
    check("targets hold at most five ids", all(len(v) <= 5 for v in rebuilt.values()))

    for method, run_id in selected.items():
        arrays = load_arrays(cfg, "models", run_id)
        same = sum(arrays["lists"][row].tolist() == main_split["lists"][method][user] for row, user in enumerate(cohort))
        check(f"recommendations rederived: {method}", same == len(cohort), f"{same}/{len(cohort)}")
        pooled = set(main_split["pool"])
        shape = all(len(set(row.tolist())) == 5 and set(row.tolist()) <= pooled for row in arrays["lists"])
        check(f"five distinct pool ids per user: {method}", shape)
        value = records[run_id]["primary_top5_preference"]["map_at_5_all_users"]
        check(f"primary MAP@5 rederived: {method}", abs(main_split["scores"][method][0] - value) < 1e-12, main_split["scores"][method][0])
        nonempty = records[run_id]["primary_top5_preference"]["map_at_5_nonempty_users"]
        check(f"non-empty MAP@5 rederived: {method}", abs(main_split["scores"][method][1] - nonempty) < 1e-12)

    popularity_file = json.loads((RESULTS_DIR / "popularity_validation.json").read_text(encoding="utf-8"))
    check("popularity agrees across scripts 03 and 06",
          popularity_file["primary_top5_preference"]["map_at_5_all_users"] == records["popularity"]["primary_top5_preference"]["map_at_5_all_users"])

    user_ids = load_text(cfg, "preprocess", "users.txt").split()
    counts = {}
    for user, _ in main_split["train"]:
        counts[user] = counts.get(user, 0) + 1
    active = [user for user in cohort if counts.get(user, 0) > 0]
    light = min(active, key=lambda user: (counts[user], int(user_ids[user])))
    heavy = min(active, key=lambda user: (-counts[user], int(user_ids[user])))
    check("heavy report user rederived", user_ids[heavy] == lock["report_users"]["heavy"]["user_id"], counts[heavy])
    check("light report user rederived", user_ids[light] == lock["report_users"]["light"]["user_id"], counts[light])

    fold_lower = ind.view_time(cfg["stability"]["train"]["start"])
    fold_middle = ind.view_time(cfg["stability"]["validation"]["start"])
    fold_upper = ind.view_time(cfg["stability"]["validation"]["end"])
    fold_split = ind.reproduce(cfg, fold_lower, fold_middle, fold_upper, mf_params, ppr_params)
    for method in ("popularity", "mf", "ppr"):
        value = fold["results"][method]["primary_top5_preference"]["map_at_5_all_users"]
        check(f"stability primary MAP@5 rederived: {method}", abs(fold_split["scores"][method][0] - value) < 1e-12, fold_split["scores"][method][0])

    ind.check_repository(cfg)
    failed = [c["check"] for c in ind.checks if not c["passed"]]
    write_result(RESULTS_DIR / "verification_models.json", {"passed": len(ind.checks) - len(failed), "failed": failed, "checks": ind.checks})
    timer.finish(checks=len(ind.checks), failed=len(failed))
    if failed:
        raise SystemExit(f"verification failed: {failed}")


if __name__ == "__main__":
    main()
