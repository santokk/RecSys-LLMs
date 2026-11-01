"""Independently verify the post-Kaggle experiment from the raw CSVs.

Writes only post-Kaggle results, so every earlier result stays exactly as locked.
"""
import hashlib
import json
import os

for name in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[name] = "1"

import math
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import independent as ind
from independent import check
from post_kaggle.repeat import LOCK_FILE, PROTECTED_CANDIDATES, lock_hash, positive_rows, repeat_scores
from sweettv.artifacts import load_arrays, load_manifest, stage_dir, write_result
from sweettv.config import EXAM_DIR, REPO_ROOT, RESULTS_DIR, data_dir, load_config
from sweettv.split import TRAIN

STAGES = ("preprocess", "split", "popularity", "models", "hybrid", "refit", "test_pool", "kaggle_candidates", "post_kaggle")
FAMILY_HALF_LIFE = {"repeat_seconds": None, "repeat_decay_4w": 28, "repeat_decay_8w": 56, "repeat_decay_8w_pop_rrf": 56}


def load(name):
    return json.loads((RESULTS_DIR / name).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_lock(cfg):
    lock, report, summary = load(LOCK_FILE), load("post_kaggle_validation.json"), load("post_kaggle_candidates.json")
    check("lock hash is intact", lock["lock_sha256"] == lock_hash(lock), lock["lock_sha256"])
    check("all results labelled post-Kaggle", all(d["post_kaggle_feedback"] is True for d in (lock, report, summary)))
    check("results belong to this lock", report["lock_sha256"] == summary["lock_sha256"] == lock["lock_sha256"])
    check("observed scores recorded as supplied",
          lock["observed_kaggle_scores"]["ppr"] == {"public_map_at_5": 0.30616, "private_map_at_5": 0.27220}
          and lock["observed_kaggle_scores"]["popularity"] == {"public_map_at_5": 0.28078, "private_map_at_5": 0.27750})
    check("exactly four repeat variants declared", [v["name"] for v in lock["repeat_family"]] == list(FAMILY_HALF_LIFE))
    check("lock predates the results", (RESULTS_DIR / LOCK_FILE).stat().st_mtime <= (RESULTS_DIR / "post_kaggle_validation.json").stat().st_mtime)
    scripts = [EXAM_DIR / "post_kaggle" / "repeat.py", *EXAM_DIR.glob("scripts/2[1-2]_*.py")]
    check("selection code never reads Kaggle scores", not [p.name for p in scripts if "observed_kaggle_scores" in p.read_text(encoding="utf-8")])

    changed = [name for name, value in lock["pre_existing_result_sha256"].items() if sha(RESULTS_DIR / name) != value]
    check("every pre-existing result is unchanged", not changed, changed)
    folder = stage_dir(cfg, "kaggle_candidates")
    check("protected candidates unchanged", all(sha(folder / name) == lock["protected_candidate_sha256"][name] for name in PROTECTED_CANDIDATES))
    test_name = cfg["files"]["schedule" + "_test"]
    for stage in STAGES:
        manifest = load_manifest(cfg, stage)
        if stage not in ("test_pool",):
            check(f"test schedule not parsed: {stage}", test_name not in manifest["parsed_sources"])
    check("test schedule not parsed: evaluation", test_name not in report["parsed_sources"], report["parsed_sources"])
    check("test schedule not parsed: candidates", test_name not in summary["parsed_sources"] and not summary["test_schedule_reopened"], summary["parsed_sources"])
    new_code = [*EXAM_DIR.glob("post_kaggle/*.py"), *EXAM_DIR.glob("scripts/2[0-2]_*.py")]
    check("new pipeline code never names the test schedule", not [p.name for p in new_code if test_name in p.read_text(encoding="utf-8")])
    return lock, report, summary


def variant_lists(name, scores, cohort, pool, order):
    if name == "repeat_decay_8w_pop_rrf":
        return ind.repeat_fused_lists(scores, cohort, pool, order, 0.8, 10)
    return ind.repeat_lists(scores, cohort, pool, order)


def check_main(cfg, report, ppr_params):
    lower = ind.view_time(cfg["windows"]["train"]["start"])
    middle = ind.view_time(cfg["windows"]["validation"]["start"])
    upper = ind.view_time(cfg["windows"]["validation"]["end"])
    seconds, fraction = cfg["watched"]["primary_seconds"], cfg["watched"]["airing_fraction"]
    core = ind.rederive_interactions(cfg, lower, middle, upper)
    (train, validation), _ = ind.watched_pairs(core, seconds, fraction)
    pool = ind.rederive_pool(core)
    cohort = ind.read_cohort_users(cfg, core["users"])
    targets = ind.top5_targets(validation, cohort, pool)
    order, _ = ind.popularity_order(train, pool)
    records = {record["name"]: record for record in report["repeat_family_main_validation"]}
    values = {}
    for name, half_life in FAMILY_HALF_LIFE.items():
        scores, latest = ind.repeat_score_map(core, seconds, fraction, half_life, middle)
        check(f"{name}: every training airing starts before prediction", latest < middle)
        lists = variant_lists(name, scores, cohort, pool, order)
        stored = load_arrays(cfg, "post_kaggle", name)["lists"]
        same = sum(stored[row].tolist() == lists[user] for row, user in enumerate(cohort))
        check(f"{name}: main recommendations rederived", same == 1260, f"{same}/1260")
        all_users, nonempty = ind.map_at_5(lists, targets, cohort)
        primary = records[name]["primary_top5_preference"]
        check(f"{name}: main MAP@5 rederived", abs(all_users - primary["map_at_5_all_users"]) < 1e-12, all_users)
        check(f"{name}: main non-empty MAP@5 rederived", abs(nonempty - primary["map_at_5_nonempty_users"]) < 1e-12)
        values[name] = all_users
    check("selection rederived from main MAP@5", max(FAMILY_HALF_LIFE, key=lambda n: values[n]) == report["selected_variant"])
    popularity_lists = {user: order[:5] for user in cohort}
    popularity_value = ind.map_at_5(popularity_lists, targets, cohort)[0]
    check("main popularity reference rederived", abs(popularity_value - report["main_validation_references"]["popularity"]) < 1e-12, popularity_value)
    check_sample(cfg, core, cohort, pool, middle, seconds, fraction)
    scale = cfg["models"]["strength_scale_seconds"]
    items, known, probabilities = ind.exact_pagerank(train, cohort, ppr_params, scale)
    hybrid = ind.fused_lists(items, known, probabilities.T, pool, order, 0.8, 10, cohort)
    stored = load_arrays(cfg, "hybrid", "hybrid_w0.8_k10")["lists"]
    same = sum(stored[row].tolist() == hybrid[user] for row, user in enumerate(cohort))
    check("pre-Kaggle hybrid main recommendations rederived", same == 1260, f"{same}/1260")
    check("pre-Kaggle hybrid main MAP@5 rederived",
          abs(ind.map_at_5(hybrid, targets, cohort)[0] - report["main_validation_references"]["hybrid_w0.8_k10_pre_kaggle"]) < 1e-12)
    return popularity_value, values[report["selected_variant"]]


def check_sample(cfg, core, cohort, pool, middle, seconds, fraction):
    """Row-level repeat scores for heavy and light users against the pipeline."""
    airings = load_arrays(cfg, "preprocess", "airings")
    interactions = load_arrays(cfg, "preprocess", "interactions")
    rows = positive_rows(airings, interactions, TRAIN, seconds, fraction)
    lock_users = load("protocol_lock.json")["report_users"]
    counts = np.bincount(rows["user"], minlength=len(core["users"]))
    active = sorted([u for u in cohort if counts[u] > 0], key=lambda u: (-counts[u], u))
    sample = sorted({core["users"][lock_users["heavy"]["user_id"]], core["users"][lock_users["light"]["user_id"]], *active[:3], *active[-3:]})
    users = np.array(sample, dtype=np.int32)
    pool_array = np.array(pool)
    position = {item: column for column, item in enumerate(pool)}
    worst = 0.0
    for half_life in (None, 28, 56):
        stored, _ = repeat_scores(rows, users, pool_array, half_life, middle)
        scores, _ = ind.repeat_score_map(core, seconds, fraction, half_life, middle)
        for row, user in enumerate(sample):
            expected = np.zeros(len(pool))
            for (who, item), value in scores.items():
                if who == user and item in position:
                    expected[position[item]] = value
            worst = max(worst, float(np.abs(stored[row] - expected).max()))
    check("repeat scores rederived for heavy and light sample users", worst < 1e-6, {"users": len(sample), "largest_difference": worst})
    hand_user = sample[0]
    table = [(capped, start) for user, capped, start in zip(rows["user"].tolist(), rows["capped"].tolist(), rows["airing_start"].tolist()) if user == hand_user][:3]
    by_hand = sum(capped * 0.5 ** (((middle - start) / 86400) / 56) for capped, start in table)
    by_formula = sum(capped * math.exp(-math.log(2) * ((middle - start) / 86400) / 56) for capped, start in table)
    check("half-life form equals the locked exponential form", abs(by_hand - by_formula) < 1e-6, by_formula)


def check_fold(cfg, report, ppr_params):
    lower = ind.view_time(cfg["stability"]["train"]["start"])
    middle = ind.view_time(cfg["stability"]["validation"]["start"])
    upper = ind.view_time(cfg["stability"]["validation"]["end"])
    seconds, fraction = cfg["watched"]["primary_seconds"], cfg["watched"]["airing_fraction"]
    core = ind.rederive_interactions(cfg, lower, middle, upper)
    (train, validation), _ = ind.watched_pairs(core, seconds, fraction)
    pool = ind.rederive_pool(core)
    cohort = ind.read_cohort_users(cfg, core["users"])
    targets = ind.top5_targets(validation, cohort, pool)
    order, _ = ind.popularity_order(train, pool)
    name = report["selected_variant"]
    scores, _ = ind.repeat_score_map(core, seconds, fraction, FAMILY_HALF_LIFE[name], middle)
    lists = variant_lists(name, scores, cohort, pool, order)
    stored = load_arrays(cfg, "post_kaggle", "fold_selected")["lists"]
    same = sum(stored[row].tolist() == lists[user] for row, user in enumerate(cohort))
    check("fold recommendations rederived", same == 1260, f"{same}/1260")
    value = ind.map_at_5(lists, targets, cohort)[0]
    primary = report["stability_fold"]["selected_variant_record"]["primary_top5_preference"]["map_at_5_all_users"]
    check("fold MAP@5 rederived", abs(value - primary) < 1e-12, value)
    baseline = ind.map_at_5({user: order[:5] for user in cohort}, targets, cohort)[0]
    check("fold popularity rederived", abs(baseline - report["stability_fold"]["popularity"]) < 1e-12, baseline)
    return baseline, value


def check_final(cfg, lock, summary, report, ppr_params):
    start = ind.view_time(cfg["final"]["train"]["start"])
    end = ind.view_time(cfg["final"]["train"]["end"])
    seconds, fraction = cfg["watched"]["primary_seconds"], cfg["watched"]["airing_fraction"]
    core = ind.rederive_interactions(cfg, start, end, end)
    (train, hold_out), _ = ind.watched_pairs(core, seconds, fraction)
    check("final: no view stops after week 30", core["latest_stop"] <= end, core["latest_stop"])
    check("final: no view starts in the test window", core["latest_start"] < end, core["latest_start"])
    check("final: no hold-out rows", len(hold_out) == 0)
    table = load_arrays(cfg, "refit", "train_table")
    check("final: training input equals the verified refit", dict(zip(zip(table["user"].tolist(), table["item"].tolist()), table["seconds"].tolist())) == dict(train), len(train))

    pool = ind.rederive_test_pool(cfg, core["train_channels"])
    check("final: test pool rederived", pool == load_arrays(cfg, "test_pool", "pool")["candidates"].tolist(), len(pool))
    order, popularity = ind.popularity_order(train, pool)
    cohort = ind.read_cohort_users(cfg, core["users"])
    user_ids = {index: user for user, index in core["users"].items()}
    frozen = load_arrays(cfg, "refit", "ppr")
    check("final: frozen PPR cohort order", frozen["cohort"].tolist() == cohort)

    items, known, probabilities = ind.exact_pagerank(train, cohort, ppr_params, cfg["models"]["strength_scale_seconds"])
    check("final: PPR items rederived", frozen["item_ids"].tolist() == items, len(items))
    column = np.array([items.index(i) for i in pool if i in set(items)])
    row_of = {user: row for row, user in enumerate(cohort)}
    agreeing = 0
    largest = 0.0
    for row, user in enumerate(known):
        mine = frozen["probabilities"][:, row_of[user]]
        exact = probabilities[:, row]
        largest = max(largest, float(np.abs(mine - exact).max()))
        top_mine = column[np.argsort(-mine[column], kind="stable")[:50]]
        top_exact = column[np.argsort(-exact[column], kind="stable")[:50]]
        agreeing += int(np.array_equal(top_mine, top_exact))
    check("final: PPR component probabilities rederived", largest < 1e-9, largest)
    check("final: PPR top-50 component ranking rederived", agreeing == len(known), f"{agreeing}/{len(known)}")

    hybrid = ind.fused_lists(items, known, probabilities.T, pool, order, 0.8, 10, cohort)
    sample_user = known[0]
    values = {item: probabilities[items.index(item), 0] if item in set(items) else 0.0 for item in pool}
    ranked = sorted(pool, key=lambda item: (-(values[item] if values[item] > 1e-15 else -1.0), item))
    rank_of = {item: rank for rank, item in enumerate(ranked, 1)}
    popular_rank = {item: rank for rank, item in enumerate(order, 1)}
    fused = {item: 0.8 / (10 + rank_of[item]) + (1 - 0.8) / (10 + popular_rank[item]) for item in pool}
    check("final: RRF formula recomputed for one user", sorted(pool, key=lambda item: (-fused[item], item))[:5] == hybrid[sample_user], hybrid[sample_user])

    template_users = [line.split(",")[0] for line in (data_dir(cfg) / cfg["files"]["submission_template"]).read_text(encoding="utf-8").split("\n")[1:] if line]
    check("template has 1,260 unique users", len(set(template_users)) == 1260)
    folder = stage_dir(cfg, "kaggle_candidates")
    pool_set = set(pool)
    hybrid_by_id = {user_ids[user]: chosen for user, chosen in hybrid.items()}
    digest, size = ind.check_candidate_file(folder / lock["candidates"]["hybrid"]["file"], template_users, pool_set, hybrid_by_id)
    check("hybrid SHA-256 and size match the summary", (digest, size) == (summary["candidates"]["hybrid"]["sha256"], summary["candidates"]["hybrid"]["bytes"]), digest)

    name = report["selected_variant"]
    scores, latest = ind.repeat_score_map(core, seconds, fraction, FAMILY_HALF_LIFE[name], end)
    check("final: every repeat airing starts before week 31", latest < end, latest)
    repeat = variant_lists(name, scores, cohort, pool, order)
    repeat_by_id = {user_ids[user]: chosen for user, chosen in repeat.items()}
    path = folder / lock["candidates"]["repeat"]["file"]
    check("repeat file exists because the rule accepted it", report["acceptance"]["accepted"] and path.exists())
    digest, size = ind.check_candidate_file(path, template_users, pool_set, repeat_by_id)
    check("repeat SHA-256 and size match the summary", (digest, size) == (summary["candidates"]["repeat"]["sha256"], summary["candidates"]["repeat"]["bytes"]), digest)
    check("only the permitted candidate files exist", {p.name for p in folder.glob("*.csv")} ==
          {*PROTECTED_CANDIDATES, lock["candidates"]["hybrid"]["file"], lock["candidates"]["repeat"]["file"]})
    return popularity


def main():
    started = time.perf_counter()
    cfg = load_config()
    grid = load("model_grid.json")
    ppr_params = next(r["params"] for r in grid["runs"] if r["run_id"] == grid["selected"]["ppr"])
    ind.check_sources(cfg)
    lock, report, summary = check_lock(cfg)
    main_baseline, main_value = check_main(cfg, report, ppr_params)
    fold_baseline, fold_value = check_fold(cfg, report, ppr_params)
    check("acceptance rederived", (main_value > main_baseline and fold_value > fold_baseline) == report["acceptance"]["accepted"],
          {"main": [main_value, main_baseline], "fold": [fold_value, fold_baseline]})
    check_final(cfg, lock, summary, report, ppr_params)
    designated = EXAM_DIR / "submission" / "final_submission.csv"
    adaptive = stage_dir(cfg, "adaptive_candidates") / "history_adaptive_submission.csv"
    check("designated final file, once present, is the exact adaptive candidate",
          not designated.exists() or designated.read_bytes() == adaptive.read_bytes())
    ind.check_repository(cfg)
    failed = [c["check"] for c in ind.checks if not c["passed"]]
    write_result(RESULTS_DIR / "post_kaggle_verification.json", {
        "post_kaggle_feedback": True,
        "passed": len(ind.checks) - len(failed), "failed": failed, "checks": ind.checks,
        "execution": {"seconds": round(time.perf_counter() - started, 1),
                      "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20), 1)},
    })
    print("checks", len(ind.checks), "failed", len(failed), "seconds", round(time.perf_counter() - started, 1))
    if failed:
        raise SystemExit(f"verification failed: {failed}")


if __name__ == "__main__":
    main()
