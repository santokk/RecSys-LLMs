"""Run popularity and every declared MF and PPR configuration.

Each configuration runs in its own process so its peak memory is its own. BLAS
is pinned to one thread for reproducible floating-point results. The script
refuses to start if the grids differ from the locked protocol.
"""
import os

for name in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[name] = "1"

import json
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from sweettv.artifacts import load_arrays, save_stage, stage_dir, write_result
from sweettv.baseline import recommend
from sweettv.config import RESULTS_DIR, load_config
from sweettv.evaluation import evaluate_lists, validation_targets
from sweettv.runlog import RunTimer
from sweettv.runner import (fallback_usage, mf_checkpoints, mf_config_id, mf_grid, popularity_order, ppr_config_id,
                            ppr_grid, ppr_lists)
from sweettv.split import side
from sweettv.submission import validate_row

PRIMARY = ("primary_top5_preference", "map_at_5_all_users")


def peak_mib():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round((peak if sys.platform == "darwin" else peak * 1024) / (1 << 20), 1)


def declared(cfg):
    runs = [("popularity", {})]
    runs += [(mf_config_id(params), params) for params in mf_grid(cfg)]
    runs += [(ppr_config_id(params), params) for params in ppr_grid(cfg)]
    return runs


def check_lock(cfg):
    lock = json.loads((RESULTS_DIR / "protocol_lock.json").read_text(encoding="utf-8"))
    if lock["grids"] != cfg["models"]:
        raise SystemExit("config.json grids differ from the locked protocol")


def write_run(folder, run_id, record, recommendations, users, list_scores, personal_counts, ap):
    lists = np.array([recommendations[user] for user in users], dtype=np.int64)
    np.savez(folder / f"{run_id}.npz", lists=lists, list_scores=list_scores, personal_counts=personal_counts, ap=ap)
    (folder / f"{run_id}.json").write_text(json.dumps(record), encoding="utf-8")


def worker(config_id, folder):
    started = time.perf_counter()
    cfg = load_config()
    k = cfg["top_k"]
    primary = cfg["watched"]["primary_seconds"]
    tables = load_arrays(cfg, "split", f"tables_{primary}")
    pool_arrays = load_arrays(cfg, "split", "pool")
    pool, cohort = pool_arrays["candidates"], pool_arrays["cohort"]
    train, validation = side(tables, "train"), side(tables, "validation")
    users = cohort.tolist()
    pool_ids = set(pool.tolist())
    fallback = load_arrays(cfg, "popularity", "popularity")["fallback_order"].tolist()
    target, broad = validation_targets(validation, cohort, pool, k)
    baseline = recommend(users, {}, fallback, k)
    _, baseline_ap = evaluate_lists(baseline, users, target, broad, len(pool), cfg)
    setup_seconds = time.perf_counter() - started

    def finish(run_id, method, params, lists, list_scores, personal_counts, extra, fit_seconds):
        for user in users:
            validate_row(lists[user], pool_ids, k)
        report, ap = evaluate_lists(lists, users, target, broad, len(pool), cfg, None if method == "popularity" else baseline_ap)
        record = {"run_id": run_id, "method": method, "params": params, **report, **extra,
                  "fallback": fallback_usage(personal_counts, k),
                  "execution": {"fit_seconds": round(fit_seconds, 2), "setup_seconds": round(setup_seconds, 2),
                                "peak_rss_mib": peak_mib()}}
        write_run(folder, run_id, record, lists, users, list_scores, personal_counts, ap)

    if config_id == "popularity":
        begin = time.perf_counter()
        order, scores = popularity_order(train, pool)
        lists = recommend(users, {}, order.tolist(), k)
        score_of = dict(zip(pool.tolist(), scores.tolist()))
        list_scores = np.array([[score_of[item] for item in lists[user]] for user in users], dtype=np.float64)
        finish("popularity", "popularity", {}, lists, list_scores, np.zeros(len(users), dtype=np.int64), {}, time.perf_counter() - begin)
        return
    for params in mf_grid(cfg):
        if mf_config_id(params) == config_id:
            begin = time.perf_counter()
            checkpoints = cfg["models"]["mf"]["iteration_checkpoints"]
            for iteration, lists, list_scores, personal_counts, losses in mf_checkpoints(train, params, cfg, cohort, pool, fallback, checkpoints):
                extra = {"final_loss": losses[-1], "loss_never_increased": bool(np.all(np.diff(losses) <= 1e-6 * abs(losses[0])))}
                finish(f"{config_id}_i{iteration}", "mf", {**params, "iterations": iteration}, lists, list_scores,
                       personal_counts, extra, time.perf_counter() - begin)
            return
    for params in ppr_grid(cfg):
        if ppr_config_id(params) == config_id:
            begin = time.perf_counter()
            lists, list_scores, personal_counts, details = ppr_lists(train, params, cfg, cohort, pool, fallback)
            finish(config_id, "ppr", params, lists, list_scores, personal_counts, {"graph": details}, time.perf_counter() - begin)
            return
    raise SystemExit(f"unknown configuration {config_id}")


def main():
    timer = RunTimer("06_run_models")
    cfg = load_config()
    check_lock(cfg)
    folder = stage_dir(cfg, "models_runs.partial")
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    for config_id, _ in declared(cfg):
        subprocess.run([sys.executable, __file__, "--worker", config_id, str(folder)], check=True)
        print("finished", config_id, flush=True)

    run_ids = []
    for config_id, params in declared(cfg):
        if config_id.startswith("mf_"):
            run_ids += [f"{config_id}_i{i}" for i in cfg["models"]["mf"]["iteration_checkpoints"]]
        else:
            run_ids.append(config_id)
    records = [json.loads((folder / f"{run_id}.json").read_text(encoding="utf-8")) for run_id in run_ids]

    def primary_of(record):
        return record[PRIMARY[0]][PRIMARY[1]]

    def select(method):
        chosen = None
        for record in records:
            if record["method"] == method and (chosen is None or primary_of(record) > primary_of(chosen)):
                chosen = record
        return chosen["run_id"]

    arrays = {}
    for run_id in run_ids:
        with np.load(folder / f"{run_id}.npz", allow_pickle=False) as members:
            arrays[run_id] = {key: members[key] for key in members.files}
    tables = load_arrays(cfg, "split", f"tables_{cfg['watched']['primary_seconds']}")
    pool_arrays = load_arrays(cfg, "split", "pool")
    target, _ = validation_targets(side(tables, "validation"), pool_arrays["cohort"], pool_arrays["candidates"], cfg["top_k"])
    matrix = np.zeros((len(pool_arrays["cohort"]), cfg["top_k"]), dtype=np.int64)
    for row, user in enumerate(pool_arrays["cohort"].tolist()):
        matrix[row, :len(target[user])] = sorted(target[user])
    arrays["targets"] = {"cohort": pool_arrays["cohort"], "target_ids": matrix}
    save_stage(cfg, "models", arrays, depends_on=("split", "popularity"))
    write_result(RESULTS_DIR / "model_grid.json", {
        "selection_rule": "highest primary_top5_preference.map_at_5_all_users; ties to the first declared",
        "selected": {"mf": select("mf"), "ppr": select("ppr")},
        "blas_threads": 1,
        "runs": records,
    })
    shutil.rmtree(folder)
    timer.finish(runs=len(records))


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        worker(sys.argv[2], Path(sys.argv[3]))
    else:
        main()
