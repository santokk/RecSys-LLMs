"""Verify hybrids, the full refit, the test pool and every candidate file.

Everything is rebuilt by the second implementation from the raw CSVs. Final PPR
lists are recomputed for all 1,260 users with an exact linear solve, which
covers more than the required sample.
"""
import hashlib
import json
import os

for name in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[name] = "1"

import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import independent as ind
from independent import check
from sweettv.artifacts import load_arrays, load_manifest, stage_dir, write_result
from sweettv.config import EXAM_DIR, REPO_ROOT, RESULTS_DIR, data_dir, load_config
from sweettv.runlog import RunTimer

ROW = re.compile(r"^(\d+),([1-9]\d*) ([1-9]\d*) ([1-9]\d*) ([1-9]\d*) ([1-9]\d*)$")


def load(name):
    return json.loads((RESULTS_DIR / name).read_text(encoding="utf-8"))


def check_stages(cfg):
    test_name = cfg["files"]["schedule" + "_test"]
    for stage in ("preprocess", "split", "popularity", "models", "hybrid", "refit"):
        manifest = load_manifest(cfg, stage)
        check(f"test schedule never parsed: {stage}", test_name not in manifest["parsed_sources"], manifest["parsed_sources"])
    pool_manifest = load_manifest(cfg, "test_pool")
    check("test schedule parsed only by the availability stage", test_name in pool_manifest["parsed_sources"], pool_manifest["parsed_sources"])
    check("candidate stage is current", bool(load_manifest(cfg, "kaggle_candidates")))
    check("availability stage opened after the refit finished",
          (stage_dir(cfg, "test_pool") / "manifest.json").stat().st_mtime >= (stage_dir(cfg, "refit") / "manifest.json").stat().st_mtime)
    refit_hash = hashlib.sha256((stage_dir(cfg, "refit") / "manifest.json").read_bytes()).hexdigest()
    check("refit unchanged since the test schedule was opened", load("test_pool.json")["refit_manifest_sha256_at_open"] == refit_hash)
    check("hybrid fold parsed no test schedule", test_name not in load("hybrid_validation.json")["stability_fold"]["parsed_sources"])
    ind.check_test_boundary_sources(cfg)


def check_hybrids(cfg):
    lock, report, grid = load("hybrid_lock.json"), load("hybrid_validation.json"), load("model_grid.json")
    records = {record["run_id"]: record for record in grid["runs"]}
    mf_params, ppr_params = records[grid["selected"]["mf"]]["params"], records[grid["selected"]["ppr"]]["params"]
    check("hybrid grid matches its lock", lock["grid"] == cfg["hybrid"], lock["grid_sha256"])
    main = report["main_validation"]
    runs = [key for key in main if key.startswith("hybrid_")]
    check("all four hybrids recorded", len(runs) == 4, runs)
    check("PPR alone equals the Step 3 result",
          main["ppr"]["primary_top5_preference"]["map_at_5_all_users"] == records["ppr_d0.3_seconds"]["primary_top5_preference"]["map_at_5_all_users"])
    value = lambda key: main[key]["primary_top5_preference"]["map_at_5_all_users"]
    best = max(runs, key=value)
    winner = best if value(best) > value("ppr") + cfg["hybrid"]["equal_score_tolerance"] else cfg["hybrid"]["ppr_run"]
    check("selection follows the locked rule", winner == report["selected_strategy"], winner)

    lower = ind.view_time(cfg["windows"]["train"]["start"])
    middle = ind.view_time(cfg["windows"]["validation"]["start"])
    upper = ind.view_time(cfg["windows"]["validation"]["end"])
    split = ind.reproduce(cfg, lower, middle, upper, mf_params, ppr_params)
    cohort = split["cohort"]
    for key in runs:
        params = main[key]["params"]
        lists = ind.fused_lists(split["ppr_items"], split["ppr_known"], split["probabilities"].T, split["pool"], split["order"],
                                params["ppr_weight"], params["rank_constant"], cohort)
        stored = load_arrays(cfg, "hybrid", key)["lists"]
        same = sum(stored[row].tolist() == lists[user] for row, user in enumerate(cohort))
        check(f"hybrid lists rederived: {key}", same == len(cohort), f"{same}/{len(cohort)}")
        score = ind.map_at_5(lists, split["targets"], cohort)[0]
        check(f"hybrid primary MAP@5 rederived: {key}", abs(score - value(key)) < 1e-12, score)

    fold_results = report["stability_fold"]["results"]
    fold = ind.reproduce(cfg, ind.view_time(cfg["stability"]["train"]["start"]), ind.view_time(cfg["stability"]["validation"]["start"]),
                         ind.view_time(cfg["stability"]["validation"]["end"]), mf_params, ppr_params)
    params = main[report["best_hybrid"]]["params"]
    lists = ind.fused_lists(fold["ppr_items"], fold["ppr_known"], fold["probabilities"].T, fold["pool"], fold["order"],
                            params["ppr_weight"], params["rank_constant"], fold["cohort"])
    score = ind.map_at_5(lists, fold["targets"], fold["cohort"])[0]
    check("fold hybrid MAP@5 rederived", abs(score - fold_results[report["best_hybrid"]]["primary_top5_preference"]["map_at_5_all_users"]) < 1e-12, score)
    check("fold PPR MAP@5 rederived", abs(fold["scores"]["ppr"][0] - fold_results["ppr"]["primary_top5_preference"]["map_at_5_all_users"]) < 1e-12)
    check("fold popularity MAP@5 rederived", abs(fold["scores"]["popularity"][0] - fold_results["popularity"]["primary_top5_preference"]["map_at_5_all_users"]) < 1e-12)
    return mf_params, ppr_params


def check_refit(cfg, mf_params, ppr_params):
    start = ind.view_time(cfg["final"]["train"]["start"])
    end = ind.view_time(cfg["final"]["train"]["end"])
    core = ind.rederive_interactions(cfg, start, end, end)
    (train, hold_out), _ = ind.watched_pairs(core, cfg["watched"]["primary_seconds"], cfg["watched"]["airing_fraction"])
    check("refit: every view read", core["rows"] == 5821987, core["rows"])
    check("refit: no view stops after week 30", core["latest_stop"] <= end, core["latest_stop"])
    check("refit: no hold-out rows", len(hold_out) == 0)
    table = load_arrays(cfg, "refit", "train_table")
    stored = dict(zip(zip(table["user"].tolist(), table["item"].tolist()), table["seconds"].tolist()))
    check("refit interactions equal the rederivation exactly", stored == dict(train), len(stored))

    popularity = load_arrays(cfg, "refit", "popularity")
    counts = {}
    for _, item in train:
        counts[item] = counts.get(item, 0) + 1
    check("refit popularity counts rederived", dict(zip(popularity["items"].tolist(), popularity["users_watching"].tolist())) == counts)

    scale = cfg["models"]["strength_scale_seconds"]
    users, items, _, item_factors = ind.als(train, mf_params, cfg["seed"], cfg["models"]["mf"]["init_std"], scale)
    mf_artifact = load_arrays(cfg, "refit", "mf_report_artifact")
    difference = float(np.abs(mf_artifact["item_factors"] - item_factors).max())
    check("refit MF factors rederived", difference < 1e-6 and mf_artifact["item_ids"].tolist() == items, difference)

    cohort = ind.read_cohort_users(cfg, core["users"])
    pool = ind.rederive_test_pool(cfg, core["train_channels"])
    stored_pool = load_arrays(cfg, "test_pool", "pool")["candidates"].tolist()
    check("test pool rederived exactly", pool == stored_pool, len(pool))
    check("test pool has no id 0", 0 not in pool)
    order, _ = ind.popularity_order(train, pool)
    ppr_items, known, probabilities = ind.exact_pagerank(train, cohort, ppr_params, scale)
    lists = {"popularity": {user: order[:5] for user in cohort},
             "ppr": ind.lists_from(ppr_items, known, probabilities.T, pool, order, cohort)}
    report = load("hybrid_validation.json")
    if report["selected_is_hybrid"]:
        params = report["main_validation"][report["selected_strategy"]]["params"]
        lists["hybrid"] = ind.fused_lists(ppr_items, known, probabilities.T, pool, order, params["ppr_weight"], params["rank_constant"], cohort)
    user_ids = {index: user for user, index in core["users"].items()}
    return {name: {user_ids[user]: chosen for user, chosen in value.items()} for name, value in lists.items()}, set(pool)


def check_files(cfg, expected, pool):
    summary = load("candidate_summary.json")
    folder = stage_dir(cfg, "kaggle_candidates")
    template = (data_dir(cfg) / cfg["files"]["submission_template"]).read_text(encoding="utf-8").split("\n")
    template_users = [line.split(",")[0] for line in template[1:] if line]
    check("template read independently", len(template_users) == 1260 and len(set(template_users)) == 1260)
    names = {path.name for path in folder.iterdir()}
    wanted = {f"{name}_submission.csv" for name in expected}
    later = {"ppr_pop_rrf_w08_k10_submission.csv", "repeat_submission.csv"}  # post-Kaggle files, verified by script 23
    check("exactly the required candidate files exist", {n for n in names if n.endswith(".csv")} - later == wanted, sorted(wanted))
    for name, lists in expected.items():
        path = folder / f"{name}_submission.csv"
        raw = path.read_bytes()
        text = raw.decode("ascii")
        check(f"{name}: header and final newline", text.startswith("user_id,tv_show_id\n") and text.endswith("\n") and not text.endswith("\n\n"))
        check(f"{name}: no CR, NaN, exponent or decimal point", not re.search(r"[\reE.]|nan|NaN", text.split("\n", 1)[1]))
        rows = text[:-1].split("\n")[1:]
        check(f"{name}: 1,260 rows", len(rows) == 1260, len(rows))
        parsed = [ROW.match(row) for row in rows]
        check(f"{name}: every row has a user and five positive ids", all(parsed))
        row_users = [match.group(1) for match in parsed]
        check(f"{name}: exact template user order", row_users == template_users)
        check(f"{name}: no missing or duplicate users", len(set(row_users)) == 1260 and set(row_users) == set(template_users))
        ids = [[int(match.group(i)) for i in range(2, 7)] for match in parsed]
        check(f"{name}: five distinct ids per row", all(len(set(row)) == 5 for row in ids))
        check(f"{name}: only test-pool ids", all(set(row) <= pool for row in ids))
        same = sum(row == lists[user] for row, user in zip(ids, row_users))
        check(f"{name}: lists equal the independent recompute", same == 1260, f"{same}/1260")
        check(f"{name}: SHA-256 matches the summary", hashlib.sha256(raw).hexdigest() == summary["candidates"][name]["sha256"], hashlib.sha256(raw).hexdigest())


def check_tracking(cfg):
    git = lambda *args: subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True).stdout
    listed = (git("ls-files") + git("ls-files", "--others", "--exclude-standard", "exam")).split("\n")
    forbidden = re.compile(r"(dataset11-30|export_arh_|_submission\.csv|\.npz$|\.npy$)")
    final = "exam/submission/final_submission.csv"
    check("no raw data or large artifacts tracked or trackable", not [p for p in listed if p != final and forbidden.search(p)])
    designated = EXAM_DIR / "submission" / "final_submission.csv"
    adaptive = stage_dir(cfg, "adaptive_candidates") / "history_adaptive_submission.csv"
    check("designated final file, once present, is the exact adaptive candidate",
          not designated.exists() or designated.read_bytes() == adaptive.read_bytes())


def main():
    timer = RunTimer("15_verify_final")
    cfg = load_config()
    ind.check_sources(cfg)
    check_stages(cfg)
    mf_params, ppr_params = check_hybrids(cfg)
    expected, pool = check_refit(cfg, mf_params, ppr_params)
    check_files(cfg, expected, pool)
    check_tracking(cfg)
    ind.check_repository(cfg)
    failed = [c["check"] for c in ind.checks if not c["passed"]]
    write_result(RESULTS_DIR / "verification_final.json", {"passed": len(ind.checks) - len(failed), "failed": failed, "checks": ind.checks})
    timer.finish(checks=len(ind.checks), failed=len(failed))
    if failed:
        raise SystemExit(f"verification failed: {failed}")


if __name__ == "__main__":
    main()
