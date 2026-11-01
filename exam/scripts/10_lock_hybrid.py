"""Hash the hybrid grid and its selection rule before any hybrid is evaluated."""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sweettv.artifacts import write_result
from sweettv.config import RESULTS_DIR, load_config
from sweettv.runlog import RunTimer


def grid_hash(cfg):
    return hashlib.sha256(json.dumps(cfg["hybrid"], sort_keys=True).encode()).hexdigest()


def main():
    timer = RunTimer("10_lock_hybrid")
    cfg = load_config()
    grid = json.loads((RESULTS_DIR / "model_grid.json").read_text(encoding="utf-8"))
    if grid["selected"]["ppr"] != cfg["hybrid"]["ppr_run"]:
        raise SystemExit("config names a PPR run that was not selected in the model grid")
    lock = {
        "formula": "score(i) = w / (K + rank_ppr(i)) + (1 - w) / (K + rank_popularity(i))",
        "grid": cfg["hybrid"],
        "grid_sha256": grid_hash(cfg),
        "frozen_components": {"ppr": grid["selected"]["ppr"], "popularity": "popularity"},
        "rules": [
            "ranks start at 1 and every candidate receives one",
            "component and fused ties break by ascending numeric tv_show_id",
            "a seeded user's unreachable candidates rank after reachable ones by ascending id",
            "a user with no training positive takes the common popularity fallback",
        ],
        "selection_rule": "highest main-validation primary MAP@5 among PPR alone and the four hybrids; "
                          "a hybrid within 1e-12 of PPR loses to the simpler PPR strategy; "
                          "the stability fold cannot override this",
    }
    write_result(RESULTS_DIR / "hybrid_lock.json", lock)
    print(lock["grid_sha256"])
    timer.finish()


if __name__ == "__main__":
    main()
