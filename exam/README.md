# sweet.tv recommender: exam implementation

Standard library and NumPy only. Run every command from the repository root.

## Chronology

The order of work matters for reading every result here.

1. Validation, model selection and the PPR and popularity candidates came first, with no Kaggle feedback.
2. Those two files were submitted, and their Kaggle scores were observed:

   | Submission | Public MAP@5 | Private MAP@5 |
   | --- | ---: | ---: |
   | PPR | 0.30616 | 0.27220 |
   | Popularity | 0.28078 | 0.27750 |

3. A bounded robustness experiment was requested only after those scores were seen.
4. Everything from scripts 20-23, and every `post_kaggle_*` result, was produced after that feedback and is not
   pre-Kaggle evidence. The w = 0.8, K = 10 hybrid was evaluated before Kaggle; its candidate file was created after.

## Data and artifacts

`exam/config.json` holds repository-relative paths, so nothing here depends on one machine.

| Location | Default | Override |
| --- | --- | --- |
| Competition CSVs (read-only) | `../A07_competition` | `SWEETTV_DATA_DIR` |
| Generated arrays and caches | `../A07_artifacts` | `SWEETTV_ARTIFACT_DIR` |

The artifacts folder is outside Git. Small audit outputs are tracked in `exam/results/`.

## Commands

Unit tests:

```bash
python3 -m unittest discover -s exam/tests
```

Set the two locations if they differ from the defaults:

```bash
export SWEETTV_DATA_DIR=../A07_competition SWEETTV_ARTIFACT_DIR=../A07_artifacts
```

Validation (interactions, split, baseline, verification):

```bash
python3 exam/scripts/01_build_interactions.py && python3 exam/scripts/02_build_split.py && python3 exam/scripts/03_popularity_baseline.py && python3 exam/scripts/04_verify_artifacts.py
```

Model comparison (lock, grids, fold, tables, verification):

```bash
python3 exam/scripts/05_lock_protocol.py && python3 exam/scripts/06_run_models.py && python3 exam/scripts/07_stability_fold.py && python3 exam/scripts/08_report_tables.py && python3 exam/scripts/09_verify_models.py
```

Hybrid lock and evaluation:

```bash
python3 exam/scripts/10_lock_hybrid.py && python3 exam/scripts/11_evaluate_hybrid.py
```

Full refit on weeks 11-30:

```bash
python3 exam/scripts/12_full_refit.py
```

Candidate generation, which is the first and only use of the test schedule:

```bash
python3 exam/scripts/13_test_availability.py && python3 exam/scripts/14_make_candidates.py
```

Final verification:

```bash
python3 exam/scripts/15_verify_final.py
```

Order matters: each stage checks the manifests of the stages before it, and any code or configuration change means
rerunning from script 01.

| Script | Produces |
| --- | --- |
| `01_build_interactions.py` | Airings and per-airing interactions; `results/preprocess_audit.json` |
| `02_build_split.py` | User-programme tables at 60/300/600 s, candidate pool; `results/split_audit.json` |
| `03_popularity_baseline.py` | Fit on weeks 11-20, score on weeks 21-30; `results/popularity_validation.json` |
| `04_verify_artifacts.py` | Plain-Python rederivation, schema and repository checks; `results/verification.json` |
| `05_lock_protocol.py` | Locked target, grids and report users; `results/protocol_lock.json` |
| `06_run_models.py` | Every declared configuration, one process each; `results/model_grid.json` |
| `07_stability_fold.py` | Selected configurations, weeks 11-27 to 28-30; `results/stability_fold.json` |
| `08_report_tables.py` | Top-5 tables for the locked users; `results/report_user_tables.json` |
| `09_verify_models.py` | Independent rebuild of targets, lists and scores; `results/verification_models.json` |
| `10_lock_hybrid.py` | Hashed hybrid grid and selection rule; `results/hybrid_lock.json` |
| `11_evaluate_hybrid.py` | Four hybrids, selection, stability fold; `results/hybrid_validation.json` |
| `12_full_refit.py` | Frozen models from weeks 11-30; `results/full_refit.json` |
| `13_test_availability.py` | Test candidate pool and labels; `results/test_pool.json` |
| `14_make_candidates.py` | CSVs in `kaggle_candidates/` under the artifacts folder; `results/candidate_summary.json` |
| `15_verify_final.py` | Independent check of the refit, pool and files; `results/verification_final.json` |

Each script records its runtime and peak memory in `results/runtime.json`.
`scripts/independent.py` is the second implementation used by the verifiers.

## Matching rules

1. **Clock.** Both sources store naive Europe/Kyiv wall time. Every timestamp becomes UTC before arithmetic.
   `start + duration` then runs in UTC, because `duraton` and schedule `duration` are true elapsed seconds.
   This handles the 2020-03-29 03:00 to 04:00 jump. A wall time in the skipped hour is counted and resolved with
   the pre-transition offset; the data contains none.
2. **Airings.** Rows with `duration <= 0` are dropped. For the same channel and start, the longest row wins, then
   the larger id. A row still running when the next starts ends at that start. Every case is counted in the audit.
3. **Matching.** Same channel and positive overlap on half-open intervals, so touching ends do not match.
4. **Windows.** Views are cut at window edges, so seconds after 2020-05-18 00:00 Kyiv belong to validation.
5. **Aggregation.** Overlap is summed per (user, airing, window), then capped at the airing's duration.
6. **Watched.** `overlap >= min(T, 0.5 * airing duration)`, with `T = 300`. `T = 60` and `T = 600` are sensitivity runs.
7. **Items.** `tv_show_id = 0` is never an item. It stays only in the view-coverage audit.

## Evaluation protocol

- Train on weeks 11-20. Validate on weeks 21-30 with the 1,260 users in `submission.csv`.
- Candidates are labelled ids airing in the target window on a channel with a training-window view.
- Previously watched programmes may be recommended.

**Primary target, locked before any model ran.** Take a user's validation programmes that satisfy the watched rule
and sit in the candidate pool. Rank by summed capped seconds, descending, then ascending numeric id. Keep at most
five. This follows the competition wording "5 most preferred" and its five-id format; the exact Kaggle rule is
unpublished.

**Selection metric.** `primary_top5_preference.map_at_5_all_users`: MAP@5 across all 1,260 users, with an empty
target scoring zero. `AP@5 = sum(P@r * rel(r)) / min(5, |R|)`.

**Secondary diagnostic only.** Fields under `secondary_broad_watched_diagnostic` score against every watched
programme. That target is far easier, so those values are not comparable with the selection metric and are never
used for selection.

## Models

All methods use the training table only and the same path from scores to lists: descending score, ties by ascending
numeric id, topped up from the popularity order. A user with no training positive gets the popularity Top-5.

- **Popularity.** Distinct training users with a watched airing.
- **Weighted ALS** (`sweettv/mf.py`). Confidence `1 + alpha * log(1 + seconds / 300)`. Only programmes with a
  training interaction have factors, so cold candidates are never scored.
- **Personalized PageRank** (`sweettv/ppr.py`). Nodes are training programmes. Edges are co-views divided by each
  user's total strength. Restart and dangling mass return to the user's normalised training history.

The grids live in `config.json`. `05_lock_protocol.py` records them, and `06_run_models.py` stops if they differ.
BLAS is pinned to one thread in scripts 06, 07 and 09 so results repeat exactly.

## Hybrid

Reciprocal-rank fusion of the frozen PPR ranking and the popularity ranking:
`score(i) = w / (K + rank_ppr(i)) + (1 - w) / (K + rank_popularity(i))`, with `w` in {0.6, 0.8} and `K` in {10, 60}.
Ranks start at 1 and cover every candidate; ties break by ascending numeric id. A seeded user's unreachable candidates
rank after reachable ones. A user with no training positive takes the common popularity fallback. The strategy with
the highest main-validation primary MAP@5 among PPR alone and the four hybrids is selected, and a hybrid within 1e-12
of PPR loses to PPR. The stability fold is a diagnostic only.

## Full refit and candidates

`12_full_refit.py` rebuilds interactions from all views in weeks 11-30 with the same rules and no hold-out, then
freezes popularity, the selected PPR and the selected MF. MF is a report artifact and feeds no candidate.

The final pool holds labelled ids airing in weeks 31-42 on a channel with a weeks 11-30 view. `14_make_candidates.py`
fits nothing: it combines frozen models with that pool. Candidate files are byte-identical on rerun; only the
timestamp in the manifest changes.

## Post-Kaggle experiment

Run after Kaggle feedback, and labelled `post_kaggle_feedback: true` in every result. `20_lock_post_kaggle.py` writes
an immutable, hashed lock before any evaluation: observed scores, four fixed repeat variants, and the selection and
acceptance rules. Kaggle scores take no part in selection or acceptance.

```bash
python3 exam/scripts/20_lock_post_kaggle.py && python3 exam/scripts/21_post_kaggle_evaluate.py && python3 exam/scripts/22_post_kaggle_candidates.py && python3 exam/scripts/23_verify_post_kaggle.py
```

| Script | Produces |
| --- | --- |
| `20_lock_post_kaggle.py` | `results/post_kaggle_improvement_lock.json`; an existing lock is never rewritten |
| `21_post_kaggle_evaluate.py` | Repeat family, fold and acceptance; `results/post_kaggle_validation.json` |
| `22_post_kaggle_candidates.py` | New candidates from frozen models; `results/post_kaggle_candidates.json` |
| `23_verify_post_kaggle.py` | Independent recompute; `results/post_kaggle_verification.json` |

Repeat scores sum capped watched seconds over a user's training airings, optionally weighted by
`exp(-ln(2) * age_days / half_life)`, with age measured from the airing start to the prediction-window start.
This code lives in `exam/post_kaggle/`, outside the hashed `sweettv` code, so earlier artifacts stay valid.
`14_make_candidates.py` rebuilds the candidate folder, so rerun script 22 after it.

## Exploratory adaptive fusion

Run after all four Kaggle score pairs were observed (PPR 0.30616/0.27220, popularity 0.28078/0.27750, PPR-popularity
RRF 0.31824/0.28728, repeat-popularity RRF 0.28864/0.25512, Public/Private). Every result is labelled exploratory and
post-Kaggle. Scores are context only and take no part in acceptance. Local targets remain assumptions; Kaggle's hidden
target is unknown.

```bash
python3 exam/scripts/30_lock_adaptive.py && python3 exam/scripts/31_evaluate_adaptive.py && python3 exam/scripts/32_adaptive_candidate.py && python3 exam/scripts/33_verify_adaptive.py
```

Two locked variants lower the PPR fusion weight for sparse training histories: A by distinct positive programmes,
B by distinct Kyiv dates of positive airing starts. Output stays in `exam/adaptive/`, `results/adaptive_*.json` and the
artifact folder `adaptive_candidates/`, so earlier stages and candidates are unaffected.

## Final submission

`exam/submission/final_submission.csv` is a byte-for-byte copy of `adaptive_candidates/history_adaptive_submission.csv`,
SHA-256 `334ea1a56901b31e98eb178030f0f0e0b798e97d9a0ad427f9067fcc74e97ace`. It was chosen after all five Kaggle scores
were seen, so it is not a pre-Kaggle selection. Validate it with:

```bash
python3 exam/scripts/40_verify_final_submission.py
```

## Leakage guards

- `csvio.iter_fit_schedule` refuses the test schedule.
- `csvio.iter_test_availability` is the only route to the test schedule. It requires a clean refit manifest and
  returns labelled rows with availability and lookup fields only.
- Once a process has parsed the test schedule, `require_test_schedule_unread` stops any further fit in it.
- Each artifact manifest stores the sources parsed, plus hashes of sources, configuration and code.
- A manifest mismatch stops the next stage, so a stale cache cannot be reused silently.
- Pinned source checksums live in `config.json`.
