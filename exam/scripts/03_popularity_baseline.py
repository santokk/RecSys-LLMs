"""Fit popularity on weeks 11-20 and evaluate it on weeks 21-30.

Popularity of a programme is the number of distinct users with at least one
watched training airing of it. Validation data only supplies labels.

Primary result: MAP@5 against each user's Top-5 preference target.
The broad watched-set score is a secondary diagnostic and must not be read as
the selection result.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sweettv.artifacts import load_arrays, save_stage, write_result
from sweettv.baseline import recommend
from sweettv.config import RESULTS_DIR, load_config
from sweettv.evaluation import evaluate_lists, validation_targets
from sweettv.runlog import RunTimer
from sweettv.runner import popularity_order
from sweettv.schedule import programme_labels
from sweettv.split import side
from sweettv.submission import validate_row


def evaluate(tables, pool, cohort, cfg):
    k = cfg["top_k"]
    train, validation = side(tables, "train"), side(tables, "validation")
    order, scores = popularity_order(train, pool)
    users = cohort.tolist()
    recommendations = recommend(users, {}, order.tolist(), k)
    pool_ids = set(pool.tolist())
    for user in users:
        validate_row(recommendations[user], pool_ids, k)
    target, broad = validation_targets(validation, cohort, pool, k)
    report, _ = evaluate_lists(recommendations, users, target, broad, len(pool), cfg)
    report["secondary_broad_watched_diagnostic"]["hit_rate_by_rank"] = [
        sum(item in broad[user] for user in users) / len(users) for item in order[:k].tolist()]
    return {"order": order, "scores": scores, "report": report}


def main():
    timer = RunTimer("03_popularity_baseline")
    cfg = load_config()
    k = cfg["top_k"]
    pool_arrays = load_arrays(cfg, "split", "pool")
    pool, cohort = pool_arrays["candidates"], pool_arrays["cohort"]
    primary = cfg["watched"]["primary_seconds"]

    results = {seconds: evaluate(load_arrays(cfg, "split", f"tables_{seconds}"), pool, cohort, cfg)
               for seconds in sorted({primary, *cfg["watched"]["sensitivity_seconds"]})}
    best = results[primary]
    top = best["order"][:20]
    score_of = dict(zip(pool.tolist(), best["scores"].tolist()))
    labels = programme_labels(cfg, top[:k].tolist())
    report = {
        "fit_window": "weeks 11-20 only",
        "popularity_definition": "distinct training users with a watched airing; ties by ascending numeric id",
        "repeat_policy": "previously watched programmes may be recommended",
        "primary_target": "each user's top-5 watched pool ids by summed capped seconds, ties by ascending id",
        "selection_metric": "primary_top5_preference.map_at_5_all_users, empty targets score zero",
        "candidate_pool_size": int(len(pool)),
        **best["report"],
        "top_5": [{"rank": rank, "tv_show_id": show_id, "training_users": score_of[show_id], **labels[show_id]}
                  for rank, show_id in enumerate(top[:k].tolist(), start=1)],
        "top_20_scores": [[show_id, score_of[show_id]] for show_id in top.tolist()],
        "pool_items_with_zero_popularity": int((best["scores"] == 0).sum()),
        "threshold_sensitivity": {str(seconds): {"top_5_ids": result["order"][:k].tolist(), **result["report"]}
                                  for seconds, result in results.items()},
    }
    arrays = {"popularity": {"candidates": pool, "scores": best["scores"], "fallback_order": best["order"]}}
    save_stage(cfg, "popularity", arrays, depends_on=("split",))
    write_result(RESULTS_DIR / "popularity_validation.json", report)
    timer.finish()


if __name__ == "__main__":
    main()
