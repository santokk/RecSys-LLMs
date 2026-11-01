"""Ordered Top-5 tables for the two locked report users. Facts only."""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sweettv.artifacts import load_arrays, load_text, write_result
from sweettv.config import RESULTS_DIR, load_config
from sweettv.runlog import RunTimer
from sweettv.schedule import programme_labels


def main():
    timer = RunTimer("08_report_tables")
    cfg = load_config()
    lock = json.loads((RESULTS_DIR / "protocol_lock.json").read_text(encoding="utf-8"))
    grid = json.loads((RESULTS_DIR / "model_grid.json").read_text(encoding="utf-8"))
    targets = load_arrays(cfg, "models", "targets")
    user_ids = load_text(cfg, "preprocess", "users.txt").split()
    row_of = {user_ids[user]: row for row, user in enumerate(targets["cohort"].tolist())}

    runs = {"popularity": "popularity", "mf": grid["selected"]["mf"], "ppr": grid["selected"]["ppr"]}
    loaded = {method: load_arrays(cfg, "models", run_id) for method, run_id in runs.items()}
    users = {role: lock["report_users"][role]["user_id"] for role in ("heavy", "light")}
    ids = {int(i) for arrays in loaded.values() for user in users.values() for i in arrays["lists"][row_of[user]]}
    labels = programme_labels(cfg, ids)

    tables = {}
    for role, user_id in users.items():
        row = row_of[user_id]
        wanted = {int(i) for i in targets["target_ids"][row] if i}
        tables[role] = {"user_id": user_id, "training": lock["report_users"][role], "target_size": len(wanted), "methods": {}}
        for method, arrays in loaded.items():
            entries = []
            for rank, (item, score) in enumerate(zip(arrays["lists"][row].tolist(), arrays["list_scores"][row].tolist()), start=1):
                entries.append({"rank": rank, "tv_show_id": item, **labels[item],
                                "score": None if math.isnan(score) else score,
                                "source": "popularity_fallback" if math.isnan(score) else method,
                                "in_validation_top5_target": item in wanted})
            tables[role]["methods"][method] = {"run_id": runs[method], "user_ap_at_5": float(arrays["ap"][row]), "top_5": entries}
    write_result(RESULTS_DIR / "report_user_tables.json", tables)
    timer.finish()


if __name__ == "__main__":
    main()
