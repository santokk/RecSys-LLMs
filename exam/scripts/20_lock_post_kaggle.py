"""Write the immutable post-Kaggle experiment lock. Runs no model.

An existing lock is never rewritten. Its hash is only re-checked.
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from post_kaggle.repeat import LOCK_FILE, lock_hash, read_lock
from sweettv.artifacts import stage_dir
from sweettv.config import RESULTS_DIR, load_config

CHRONOLOGY = [
    "PPR and popularity were selected, generated and submitted first",
    "their Public and Private Kaggle scores were then observed",
    "this bounded experiment was requested only after those scores were seen",
    "nothing produced here is pre-Kaggle evidence, except the w=0.8, K=10 hybrid's earlier validation result",
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cfg = load_config()
    path = RESULTS_DIR / LOCK_FILE
    if path.exists():
        print("lock already exists; left unchanged", read_lock(RESULTS_DIR)["lock_sha256"])
        return
    existing = {p.name: sha(p) for p in sorted(RESULTS_DIR.glob("*.json"))}
    candidates = stage_dir(cfg, "kaggle_candidates")
    lock = {
        "experiment": "bounded post-Kaggle robustness experiment",
        "post_kaggle_feedback": True,
        "locked_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "chronology": CHRONOLOGY,
        "observed_kaggle_scores": {
            "ppr": {"public_map_at_5": 0.30616, "private_map_at_5": 0.27220},
            "popularity": {"public_map_at_5": 0.28078, "private_map_at_5": 0.27750},
            "use": "reporting evidence only; excluded from acceptance and selection",
        },
        "motivation": "PPR led on Public but popularity led on Private by 0.00530, so one bounded test of a popularity-"
                      "stabilised hybrid and a simple personal-repeat family is made, with no further tuning",
        "fixed_seed": cfg["seed"],
        "candidates": {
            "hybrid": {
                "file": "ppr_pop_rrf_w08_k10_submission.csv",
                "definition": "w / (K + rank_ppr) + (1 - w) / (K + rank_popularity), w = 0.8, K = 10",
                "evaluated_before_kaggle": True,
                "pre_kaggle_run": "hybrid_w0.8_k10",
                "ppr_run": "ppr_d0.3_seconds",
            },
            "repeat": {"file": "repeat_submission.csv", "generated_only_if_accepted": True, "evaluated_before_kaggle": False},
        },
        "repeat_family": [
            {"name": "repeat_seconds", "half_life_days": None, "rrf_with_popularity": None},
            {"name": "repeat_decay_4w", "half_life_days": 28, "rrf_with_popularity": None},
            {"name": "repeat_decay_8w", "half_life_days": 56, "rrf_with_popularity": None},
            {"name": "repeat_decay_8w_pop_rrf", "half_life_days": 56,
             "rrf_with_popularity": {"repeat_weight": 0.8, "popularity_weight": 0.2, "rank_constant": 10}},
        ],
        "repeat_rules": [
            "scores use only training-side positive interactions under the 300-second watched rule",
            "age runs from the airing start to the prediction-window start, in days",
            "age and weights use airing timestamps from the training side only",
            "previously unseen programmes receive no repeat score",
            "cold users and short lists use the existing popularity fallback",
            "in the RRF variant, a user's candidates without a repeat score rank after scored ones by ascending id",
            "exactly five distinct ids; ties by ascending numeric id",
            "no additional variants, half-lives, weights or K values",
        ],
        "selection_rule": "highest main-validation Top-5-preference MAP@5 across all 1,260 users; ties to the first declared",
        "stability_rule": "only the selected repeat variant runs on the weeks 11-27 to 28-30 fold",
        "acceptance_rule": "generate repeat_submission.csv only if the selected variant's main MAP@5 is strictly above "
                           "main popularity AND its fold MAP@5 is strictly above fold popularity",
        "kaggle_scores_in_acceptance": False,
        "pre_existing_result_sha256": existing,
        "protected_candidate_sha256": {name: sha(candidates / name) for name in ("ppr_submission.csv", "popularity_submission.csv")},
    }
    lock["lock_sha256"] = lock_hash(lock)
    path.write_text(json.dumps(lock, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("lock written", lock["lock_sha256"])


if __name__ == "__main__":
    main()
