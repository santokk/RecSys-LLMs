"""Independently verify adaptive weights, RRF scores, recommendations and the candidate.

Rebuilds from the raw CSVs through `independent.py`. PageRank comes from an exact
solve, history from plain Python sets, and fusion from the plain-Python ranker.
"""
import hashlib
import json
import os

for name in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[name] = "1"

import calendar
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import independent as ind
from adaptive.fusion import CANDIDATE_FILE, CANDIDATE_FOLDER, LOCK_FILE, lock_hash
from independent import check
from sweettv.artifacts import load_arrays, load_manifest, stage_dir, write_result
from sweettv.config import EXAM_DIR, RESULTS_DIR, data_dir, load_config

JUMP_UTC = calendar.timegm((2020, 3, 29, 1, 0, 0))


def load(name):
    return json.loads((RESULTS_DIR / name).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def history(core, seconds, fraction):
    """Programmes and Kyiv dates per user, from positive training rows only."""
    starts = ind.airing_starts(core)
    programmes, days = defaultdict(set), defaultdict(set)
    for (user_window, airing), overlap in core["totals"].items():
        if user_window & 1:
            continue
        length = core["airing_length"][airing]
        if min(overlap, length) >= min(seconds, fraction * length):
            user = user_window >> 1
            offset = 7200 if starts[airing] < JUMP_UTC else 10800
            programmes[user].add(core["airing_id"][airing])
            days[user].add((starts[airing] + offset) // 86400)
    return programmes, days


def weight_of(value, lower, upper):
    return 0.4 if value < lower else 0.6 if value < upper else 0.8


def fused_by_weight(items, known, probabilities, pool, order, weights, cohort):
    """Run the plain ranker once per weight and join the users back together."""
    lists = {user: order[:5] for user in cohort}
    scores = probabilities.T
    for weight in sorted(set(weights.values())):
        rows = [row for row, user in enumerate(known) if weights[user] == weight]
        users = [known[row] for row in rows]
        chosen = ind.fused_lists(items, users, scores[rows], pool, order, weight, 10, users)
        lists.update(chosen)
    return lists


def rrf_by_hand(items, known, probabilities, pool, order, weights, user):
    """Top item and score for one user, from the formula itself."""
    row = known.index(user)
    position = {item: index for index, item in enumerate(items)}
    values = {i: (probabilities[position[i], row] if i in position else 0.0) for i in pool}
    ranked = sorted(pool, key=lambda i: (-(values[i] if values[i] > 1e-15 else -1.0), i))
    ppr_rank = {i: r for r, i in enumerate(ranked, 1)}
    popular_rank = {i: r for r, i in enumerate(order, 1)}
    w = weights[user]
    fused = {i: w / (10 + ppr_rank[i]) + (1 - w) / (10 + popular_rank[i]) for i in pool}
    best = min(pool, key=lambda i: (-fused[i], i))
    return best, fused[best]


def check_fold(cfg, lock, label, lower, middle, upper, ppr_params, prefix, result):
    seconds, fraction = cfg["watched"]["primary_seconds"], cfg["watched"]["airing_fraction"]
    core = ind.rederive_interactions(cfg, lower, middle, upper)
    (train, validation), _ = ind.watched_pairs(core, seconds, fraction)
    pool = ind.rederive_pool(core)
    cohort = ind.read_cohort_users(cfg, core["users"])
    targets = ind.top5_targets(validation, cohort, pool)
    allowed = set(pool)
    broad = {user: set() for user in cohort}
    for user, item in validation:
        if item in allowed and user in broad:
            broad[user].add(item)
    order, _ = ind.popularity_order(train, pool)
    items, known, probabilities = ind.exact_pagerank(train, cohort, ppr_params, cfg["models"]["strength_scale_seconds"])
    programmes, days = history(core, seconds, fraction)
    fixed = ind.fused_lists(items, known, probabilities.T, pool, order, 0.8, 10, cohort)
    fixed_value = ind.map_at_5(fixed, targets, cohort)[0]
    check(f"{label}: fixed fusion MAP@5 rederived", abs(fixed_value - result["fixed"]["primary_top5_preference"]["map_at_5_all_users"]) < 1e-12, fixed_value)
    values = {}
    for variant in lock["variants"]:
        name = variant["name"]
        source = programmes if variant["label"] == "A" else days
        counts = {user: len(source.get(user, ())) for user in cohort}
        weights = {user: weight_of(counts[user], variant["lower"], variant["upper"]) for user in cohort}
        stored = load_arrays(cfg, "adaptive", f"{prefix}_{name}")
        check(f"{label} {name}: history rederived", stored["history"].tolist() == [counts[user] for user in cohort], max(counts.values()))
        check(f"{label} {name}: weights rederived", stored["weights"].tolist() == [weights[user] for user in cohort])
        lists = fused_by_weight(items, known, probabilities, pool, order, weights, cohort)
        same = sum(stored["lists"][row].tolist() == lists[user] for row, user in enumerate(cohort))
        check(f"{label} {name}: recommendations rederived", same == 1260, f"{same}/1260")
        record = result["variants"][name]
        primary = ind.map_at_5(lists, targets, cohort)[0]
        check(f"{label} {name}: primary MAP@5 rederived", abs(primary - record["primary_top5_preference"]["map_at_5_all_users"]) < 1e-12, primary)
        secondary = ind.map_at_5(lists, broad, cohort)[0]
        check(f"{label} {name}: broad MAP@5 rederived", abs(secondary - record["secondary_broad_watched_diagnostic"]["map_at_5_all_users"]) < 1e-12, secondary)
        sparse = [user for user in known if weights[user] < 0.8][:5]
        worst = 0.0
        for user in sparse:
            best, score = rrf_by_hand(items, known, probabilities, pool, order, weights, user)
            row = cohort.index(user)
            worst = max(worst, abs(score - stored["list_scores"][row, 0]))
            check(f"{label} {name}: top item by formula, user {user}", best == int(stored["lists"][row, 0]))
        check(f"{label} {name}: RRF scores by formula", worst < 1e-12, worst)
        values[name] = primary
    return values, fixed_value


def check_candidate(cfg, lock, report, ppr_params):
    variant = next(v for v in lock["variants"] if v["name"] == report["selected_variant"])
    start, end = ind.view_time(cfg["final"]["train"]["start"]), ind.view_time(cfg["final"]["train"]["end"])
    seconds, fraction = cfg["watched"]["primary_seconds"], cfg["watched"]["airing_fraction"]
    core = ind.rederive_interactions(cfg, start, end, end)
    (train, hold_out), _ = ind.watched_pairs(core, seconds, fraction)
    check("final: no view after week 30, no hold-out", core["latest_stop"] <= end and not hold_out, core["latest_stop"])
    table = load_arrays(cfg, "refit", "train_table")
    check("final: history input equals the verified refit", dict(zip(zip(table["user"].tolist(), table["item"].tolist()), table["seconds"].tolist())) == dict(train), len(train))
    pool = ind.rederive_test_pool(cfg, core["train_channels"])
    check("final: test pool rederived", pool == load_arrays(cfg, "test_pool", "pool")["candidates"].tolist(), len(pool))
    order, _ = ind.popularity_order(train, pool)
    cohort = ind.read_cohort_users(cfg, core["users"])
    items, known, probabilities = ind.exact_pagerank(train, cohort, ppr_params, cfg["models"]["strength_scale_seconds"])
    programmes, days = history(core, seconds, fraction)
    source = programmes if variant["label"] == "A" else days
    weights = {user: weight_of(len(source.get(user, ())), variant["lower"], variant["upper"]) for user in cohort}
    lists = fused_by_weight(items, known, probabilities, pool, order, weights, cohort)
    user_ids = {index: user for user, index in core["users"].items()}
    expected = {user_ids[user]: chosen for user, chosen in lists.items()}
    template = [line.split(",")[0] for line in (data_dir(cfg) / cfg["files"]["submission_template"]).read_text(encoding="utf-8").split("\n")[1:] if line]
    check("template has 1,260 unique users", len(set(template)) == 1260)
    summary = load("adaptive_candidate.json")
    path = stage_dir(cfg, CANDIDATE_FOLDER) / CANDIDATE_FILE
    digest, size = ind.check_candidate_file(path, template, set(pool), expected)
    check("candidate SHA-256 and size match the summary", (digest, size) == (summary["sha256"], summary["bytes"]), digest)
    counts = {}
    for user in cohort:
        counts[weights[user]] = counts.get(weights[user], 0) + 1
    fixed_file = stage_dir(cfg, "kaggle_candidates") / "ppr_pop_rrf_w08_k10_submission.csv"
    new_rows, old_rows = path.read_text().split("\n"), fixed_file.read_text().split("\n")
    changed = sum(a != b for a, b in zip(new_rows, old_rows))
    check("candidate differs from fixed fusion only for users with weight below 0.8",
          all(weights[core["users"][a.split(",")[0]]] < 0.8 for a, b in zip(new_rows[1:], old_rows[1:]) if a and a != b), changed)
    return {"rows_changed_from_fixed_file": changed, "weight_counts": {str(k): v for k, v in sorted(counts.items())}}


def main():
    started = time.perf_counter()
    cfg = load_config()
    lock, report = load(LOCK_FILE), load("adaptive_validation.json")
    grid = load("model_grid.json")
    ppr_params = next(r["params"] for r in grid["runs"] if r["run_id"] == grid["selected"]["ppr"])
    ind.check_sources(cfg)
    check("lock hash is intact", lock["lock_sha256"] == lock_hash(lock), lock["lock_sha256"])
    check("results belong to this lock and are labelled", report["lock_sha256"] == lock["lock_sha256"] and report["post_kaggle_feedback"] is True and report["exploratory"] is True)
    check("exactly two locked variants", [v["label"] for v in lock["variants"]] == ["A", "B"])
    check("lock predates results", (RESULTS_DIR / LOCK_FILE).stat().st_mtime <= (RESULTS_DIR / "adaptive_validation.json").stat().st_mtime)
    changed = [name for name, value in lock["pre_existing_result_sha256"].items() if sha(RESULTS_DIR / name) != value]
    check("every earlier result is unchanged", not changed, changed)
    folder = stage_dir(cfg, "kaggle_candidates")
    check("every earlier candidate is unchanged", all(sha(folder / name) == value for name, value in lock["existing_candidate_sha256"].items()), len(lock["existing_candidate_sha256"]))
    check("nothing new in the earlier candidate folder", {p.name for p in folder.glob("*.csv")} == set(lock["existing_candidate_sha256"]))
    test_name = cfg["files"]["schedule" + "_test"]
    check("test schedule not parsed by evaluation", test_name not in report["parsed_sources"], report["parsed_sources"])
    check("test schedule not parsed by the stage", test_name not in load_manifest(cfg, "adaptive")["parsed_sources"])
    new_code = [*EXAM_DIR.glob("adaptive/*.py"), *EXAM_DIR.glob("scripts/3[0-2]_*.py")]
    check("new code never names the test schedule", not [p.name for p in new_code if test_name in p.read_text(encoding="utf-8")])
    check("new code never reads Kaggle scores", not [p.name for p in EXAM_DIR.glob("scripts/3[1-2]_*.py") if "observed_kaggle_scores" in p.read_text(encoding="utf-8")])

    main_values, main_fixed = check_fold(cfg, lock, "main", ind.view_time(cfg["windows"]["train"]["start"]), ind.view_time(cfg["windows"]["validation"]["start"]),
                                         ind.view_time(cfg["windows"]["validation"]["end"]), ppr_params, "main", report["main_split"])
    fold_values, fold_fixed = check_fold(cfg, lock, "fold", ind.view_time(cfg["stability"]["train"]["start"]), ind.view_time(cfg["stability"]["validation"]["start"]),
                                         ind.view_time(cfg["stability"]["validation"]["end"]), ppr_params, "fold", report["stability_fold"])
    qualified = [v["name"] for v in lock["variants"] if main_values[v["name"]] > main_fixed and fold_values[v["name"]] > fold_fixed]
    winner = max(qualified, key=lambda name: main_values[name]) if qualified else None
    check("acceptance and selection rederived", winner == report["selected_variant"], winner)
    extra = check_candidate(cfg, lock, report, ppr_params) if winner else {}
    ind.check_repository(cfg)
    failed = [c["check"] for c in ind.checks if not c["passed"]]
    write_result(RESULTS_DIR / "adaptive_verification.json", {
        "post_kaggle_feedback": True, "exploratory": True, "passed": len(ind.checks) - len(failed), "failed": failed,
        "candidate": extra, "checks": ind.checks,
        "execution": {"seconds": round(time.perf_counter() - started, 1),
                      "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20), 1)},
    })
    print("checks", len(ind.checks), "failed", len(failed), "candidate", extra)
    if failed:
        raise SystemExit(f"verification failed: {failed}")


if __name__ == "__main__":
    main()
