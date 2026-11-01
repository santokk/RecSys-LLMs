"""Write the immutable lock for the exploratory adaptive experiment. Runs no model."""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptive.fusion import LOCK_FILE, lock_hash, read_lock
from sweettv.artifacts import stage_dir
from sweettv.config import RESULTS_DIR, load_config


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cfg = load_config()
    path = RESULTS_DIR / LOCK_FILE
    if path.exists():
        print("lock already exists; left unchanged", read_lock(RESULTS_DIR)["lock_sha256"])
        return
    folder = stage_dir(cfg, "kaggle_candidates")
    lock = {
        "experiment": "bounded exploratory history-adaptive PPR-popularity fusion",
        "post_kaggle_feedback": True,
        "locked_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "chronology": [
            "PPR and popularity were validated, selected and submitted first",
            "a bounded repeat experiment followed, and its two candidates were then submitted",
            "all four Kaggle score pairs were observed",
            "only then was this exploratory experiment requested, so nothing here is pre-Kaggle evidence",
        ],
        "observed_kaggle_scores": {
            "ppr": {"public_map_at_5": 0.30616, "private_map_at_5": 0.27220},
            "popularity": {"public_map_at_5": 0.28078, "private_map_at_5": 0.27750},
            "ppr_popularity_rrf": {"public_map_at_5": 0.31824, "private_map_at_5": 0.28728},
            "repeat_popularity_rrf": {"public_map_at_5": 0.28864, "private_map_at_5": 0.25512},
            "use": "context only; excluded from acceptance and selection",
        },
        "hypothesis": "PPR personalization is less reliable for sparse histories, so lowering its fusion weight "
                      "for those users may improve validation",
        "target_note": "local Top-5 preference and broad watched-set targets are assumptions; Kaggle's hidden target "
                       "is unknown, and neither local target changes here",
        "fixed_seed": cfg["seed"],
        "frozen_components": {"ppr": cfg["hybrid"]["ppr_run"], "popularity": "distinct training users with a watched airing"},
        "fusion": "score(i) = w_u / (10 + rank_ppr(i)) + (1 - w_u) / (10 + rank_popularity(i)), complete rankings, K = 10",
        "comparison": "fixed w = 0.8, K = 10 fusion on the same fold",
        "variants": [
            {"name": "adaptive_programmes", "label": "A", "history": "distinct positive training programmes",
             "lower": 5, "upper": 20, "weights": [0.4, 0.6, 0.8]},
            {"name": "adaptive_days", "label": "B",
             "history": "distinct Kyiv calendar dates of the airing starts of positive labelled training interactions",
             "lower": 3, "upper": 14, "weights": [0.4, 0.6, 0.8]},
        ],
        "rules": [
            "history is computed from the fold's training side only",
            "users with no training positive receive popularity directly",
            "ties break by ascending numeric id",
            "no other variant, threshold, weight or K may be added",
        ],
        "acceptance_rule": "a variant qualifies only if its primary MAP@5 strictly exceeds fixed w = 0.8 fusion on BOTH "
                           "the main split and the stability fold",
        "selection_rule": "if both qualify, take the larger main-split primary MAP@5; an exact tie goes to A",
        "no_qualifier": "stop with the results and generate no CSV",
        "pre_existing_result_sha256": {p.name: sha(p) for p in sorted(RESULTS_DIR.glob("*.json"))},
        "existing_candidate_sha256": {p.name: sha(p) for p in sorted(folder.glob("*.csv"))},
    }
    lock["lock_sha256"] = lock_hash(lock)
    path.write_text(json.dumps(lock, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("lock written", lock["lock_sha256"])


if __name__ == "__main__":
    main()
