# Graph-Based Recommendations with Personalized PageRank

Course: LLM4Rec, HSE University
Instructor: Seungmin Jin
Task type: Individual assignment
Runtime: JavaScript and TensorFlow.js in a web browser

## Status and scope

This document defines the assignment. It does not claim that a new starter application, automated tests, or reference results on MovieLens have already been implemented.

Reuse the MovieLens 100K files and evaluation conventions from the previous matrix-factorization assignment. Do not overwrite the original `week5/` files. The directory and deadline for the new submission will be announced by the instructor.

GCN, GAT, temporal attention, and purchase-probability modeling are outside the required implementation.

## 1. Learning goals

- Construct a user-movie bipartite graph from positive training interactions.
- Explain row-normalized transitions, probability propagation, restart distributions, and stationary distributions.
- Implement PageRank and Personalized PageRank (PPR) with TensorFlow.js tensor operations.
- Turn movie-node scores into eligible Top-10 recommendations.
- Compare popularity, global PageRank, PPR, and your previous MF recommender under a shared retrieval protocol.
- Make an evidence-based deployment recommendation rather than assuming that a newer or more personalized algorithm is better.

## 2. Dataset and continuity with the previous assignment

Use `week5/u.data` and `week5/u.item`, and keep the previous per-user chronological 80%/10%/10% split. Do not overwrite the original `week5/` files. Treat missing ratings as unobserved, not as zero ratings.

Report the counts actually loaded by your application and all subsequent exclusions. The split is chronological within each user, not globally time-consistent; disclose that limitation and note that MovieLens records rating times, not viewing sessions.

## 3. Required implementation

### 3.1 Data preparation

Report users, movies, ratings, sparsity, positive training interactions, zero-degree users and movies, and the split record counts. The graph's node universe is the set of user IDs and movie IDs observed in the training split; map IDs to contiguous indices explicitly and never add a padding node to the graph or the recommendation pool.

```text
R[u, i] = 1 if user u rated movie i >= 4.0 in TRAINING
R[u, i] = 0 otherwise
```

A zero means no positive training edge; it is not a negative preference. Connect users and movies in both directions; there are no user-user or movie-movie edges.

### 3.2 Transition convention

```text
A = [[0, R],
     [R^T, 0]]
```

Normalize each nonzero row to obtain link-following probabilities. Notation: `alpha` is the probability of following a link, `1 - alpha` is the probability of restart, `v` is the normalized restart distribution, `p` is the column vector of node probabilities.

```text
p_next = alpha * P^T * p + (1 - alpha) * v
```

The transpose collects incoming probability for each destination; it does not reverse the graph's movement rules.

### 3.3 Zero-degree nodes

Do not divide by zero or silently discard probability mass. Implement the equivalent update without materializing dense dangling rows:

```text
p_next = alpha * (Q^T * p + d * v) + (1 - alpha) * v
```

where `Q` has normalized rows for positive-degree nodes and zero rows otherwise, and `d` is the current mass on zero-degree nodes. Report zero-degree counts and the policy; it must be the same rule in global PageRank and PPR.

### 3.4 Global PageRank and PPR

Global PageRank: `v` is uniform over all graph nodes. PPR for user `u`: `v` is one-hot at that user's node. Initialize `p` to `v`. Stop when the L1 difference between consecutive distributions is below `1e-6`, or after 500 iterations. Report the iteration count, final residual, and whether convergence was reached; an iteration cap is not proof of convergence. Do not force a user's final score to equal the restart probability.

### 3.5 TensorFlow.js requirements

Use TensorFlow.js for the core batched propagation. JavaScript loops are acceptable for parsing, ID mapping, formatting, and small fixtures; they must not replace the main propagation with per-edge loops. You may avoid the full square adjacency matrix by implementing its two off-diagonal blocks with `R` and `R^T`, vector degree normalization, and `tf.matMul`. Await `tf.ready()`, report `tf.getBackend()`, manage tensors with `tf.tidy()` and dispose long-lived tensors, read results asynchronously where practical, never put an async callback inside `tf.tidy()`, and make sure repeated runs do not accumulate tensors. Record runtime and backend without comparing timings across devices. PPR is an iterative calculation, not neural-network training; no optimizer is required. Write code, comments and JSDoc in English and document inputs, outputs, indexing, assumptions and failure cases.

## 4. Verification before MovieLens experiments

### 4.1 Reference graph

Node order A, B, C, D; directed edges `A -> B`, `A -> C`, `B -> C`, `C -> A`, `D -> C`. A divides its link-following probability equally between B and C; B, C and D each have one outgoing link; there are no dangling nodes.

For alpha = 0.85 and uniform restart:

```text
p = [0.3725268513, 0.1958239118, 0.3941492369, 0.0375000000]
ranking: C > A > B > D
```

For alpha = 0.85 and restart only at A:

```text
p = [0.4522328999, 0.1921989825, 0.3555681176, 0.0000000000]
ranking: A > C > B > D
```

These are numerical reference values for this specific test graph, not expected MovieLens outcomes.

### 4.2 Additional tests

Verify and report: normalized rows sum to 1; every restart distribution is nonnegative and sums to 1; iterations preserve total probability to numerical tolerance; stationary scores are nonnegative; one extra update produces approximately the same distribution; a separate zero-degree-node fixture preserves mass; graph direction and matrix orientation match; recommendation filtering removes the correct movies. Do not claim to have run checks you did not run.

## 5. From scores to Top-10 recommendations

Use all movies appearing in the training split as the candidate pool. For final TEST recommendations: extract movie-node scores only; exclude all movies the user rated in training or validation regardless of value; sort descending by score with ties broken by ascending movie ID; return up to 10; display IDs, titles, genres and scores. Report if fewer than 10 eligible candidates exist, and whether zero-score movies had to be included; do not claim that tie-broken zero-score candidates have demonstrated relevance. For VALIDATION recommendations exclude training-rated movies only. PPR scores are long-run graph-walk probabilities: not purchase probabilities, not predicted ratings, and not comparable in scale to MF scores.

## 6. Methods and experimental controls

Compare four methods: (1) Popularity — count distinct users with a positive training rating per movie; (2) Global PageRank with uniform restart; (3) PPR with restart at the target user; (4) MF — reuse your previous implementation and selected model. Do not alter the MF scoring definition to resemble PPR; state the model and checkpoint used.

### 6.1 Alpha selection

For PPR compare `alpha` in `{0.50, 0.70, 0.85, 0.95}` on validation Recall@10 and select one shared alpha for the full evaluation population; do not use test outcomes. Break exact ties by the smaller alpha. Use the same alpha for the primary global PageRank comparison and disclose this control; this is not a claim that global PageRank was independently optimized. MF keeps the previous validation-RMSE checkpoint rule; disclose that MF and PPR use different validation objectives.

### 6.2 One-factor-at-a-time demonstration

Choose two users from the common test cohort and keep their IDs fixed. Show three PPR settings: baseline (user A, alpha = 0.85), change seed only (user B, alpha = 0.85), change alpha only (user A, alpha = 0.50). Compare both changed settings with the baseline and never attribute a difference to one factor after changing both. Do not assume larger alpha always improves novelty, or that the restart seed must be the highest-scoring node.

## 7. Retrieval evaluation and popularity evidence

### 7.1 Shared evaluation cohort

Use the same candidate rules and the same users for all four methods. A user is evaluable when the user appears in training with at least one positive training edge, the selected MF implementation can score the required training-movie candidates, and at least one relevant validation/test movie remains eligible under the appropriate filtering rules. Apply the shared intersection for the primary comparison, disclose any additional restriction, and report how many users are excluded. This is a warm-user evaluation; it does not establish cold-start performance.

### 7.2 Recall@10

A relevant movie has rating >= 4.0 in the evaluation split and must belong to the training-movie candidate pool and pass that split's previously-rated exclusion rule.

```text
Recall@10 = relevant eligible movies retrieved / all relevant eligible movies
```

Report macro-average Recall@10 over the shared cohort, the cohort size, and exclusion counts; separately count relevant movies absent from the training candidate pool and users without eligible relevant movies. Do not compute rating RMSE for PPR.

### 7.3 Recommendation popularity

```text
popularity[i] = number of distinct users with training rating >= 4.0 for movie i
```

For each reported Top-10 list include each movie's popularity and the list's mean popularity, and for the two selected users report the number of common movies in their Top-10 lists within each method (descriptive evidence, not proof of better personalization). Do not treat unknown test movies as dislikes or popularity as platform-wide impressions.

## 8. Business & Algorithmic Analysis - exactly ONE question

### Should the platform keep MF, replace it with PPR, or use both?

Recommend one deployment strategy — keep MF as the main recommender, replace it with PPR, or use the two methods in different recommendation sections — supported by Recall@10 on the common cohort, the two users' Top-10 lists with titles and genres, their training histories, the training-set popularity of the recommended movies, and the observed effects of changing the restart user and alpha separately.

Do not assume a different list is better personalization, that less popular movies are automatically better discoveries, or that PPR scores are purchase probabilities. If you propose separate sections, name each section's purpose and explain why your results support the assignment. State what the offline results cannot establish about current-session intent, viewing completion, retention or revenue, and propose one online experiment before rollout specifying the intervention, primary outcome, at least one guardrail, and how user-level assignment avoids contamination. No extra business essay is required.

## 9. Deliverables

Submit: the JavaScript application and its source files with TensorFlow.js used in the core propagation; `README.md` with exact run instructions and dependency versions actually used; `verification.json` or an equivalent machine-readable test-result export; `results.csv` with retrieval metrics, common-cohort size, exclusions and selected settings; `recommendations.csv` with user ID, method, alpha where applicable, rank, movie ID, title, genres, score and training popularity; a course-format `report.pdf` covering implementation evidence and the single business question; the unmodified AI-session export required by the course, if that requirement remains in force. Preserve the original dataset and previous assignment files, and state any incomplete component, failed test, nonconverged run or unavailable prior MF output.

Confirm the current grading guide, submission directory, deadline and AI-export requirement with the instructor before publishing the final assignment. Do not invent an official numerical rubric or a deadline.

## 10. Evidence and citation integrity

Cite the original dataset and verify all paper titles, authors, years and URLs. Do not fabricate runtime measurements, experiment results, tests or online business effects. Make sure tables match the application's actual exports, and separate measured observations, proposed explanations, and proposed deployment actions.

Dataset provenance: MovieLens 100K, GroupLens, University of Minnesota. The previous assignment provides the citation: F. Maxwell Harper and Joseph A. Konstan, "The MovieLens Datasets: History and Context," ACM Transactions on Interactive Intelligent Systems, 2015. Verify the bibliographic record before including it in the submitted report.

## 11. Final checklist

- [ ] Same MovieLens files and split as the previous assignment.
- [ ] Graph built from positive training ratings only.
- [ ] All observed training/validation ratings used for final exclusion.
- [ ] Alpha consistently means link-following probability.
- [ ] Matrix orientation and dangling-mass handling tested.
- [ ] No unintended padding node or dataset mutation.
- [ ] TensorFlow.js propagation and memory management implemented.
- [ ] Shared evaluation cohort and candidates across methods.
- [ ] Validation used for selection; test used only for final evaluation.
- [ ] Popularity calculated from training data only.
- [ ] One setting changed per demonstration.
- [ ] PPR scores not presented as ratings or purchase probabilities.
- [ ] One evidence-based business decision, with limitations and an online experiment.
- [ ] All missing work and verification failures disclosed.

---

## Local run instructions (added for this starter)

TensorFlow.js is loaded from the CDN in `index.html`; d3 is vendored locally.

Serve the folder over HTTP (the data layer uses `fetch`, so `file://` will not work):

    cd week6
    python3 -m http.server 8000

Then open `http://localhost:8000/index.html`.

---

## Submission run instructions (HW6 submission, appended by the student)

Everything above this line is the unchanged assignment text. This file is named `readme.md`; on the case-insensitive macOS file system used for this submission, `README.md` and `readme.md` are the same path, so the required run instructions are appended here instead of being placed in a second file.

**Dependencies actually used.** TensorFlow.js 4.22.0 (CDN script in `index.html`), D3 7.9.0 (vendored `vendor/d3.min.js`), Python 3 only for the static file server. Tested in desktop Chrome; the expected backend is `webgl` with 32-bit float textures. No build step and no package installation.

**Serve and open.**

    cd week6
    python3 -m http.server 8000

Open `http://localhost:8000/index.html` (opening the file directly with `file://` does not work).

**Button sequence used for the submitted exports.**

1. Build graph — expect 2,554 nodes (943 users + 1,611 movies), 45,602 positive training edges, 1 zero-degree user, 253 zero-degree movies.
2. Run tests — expect `pass 12, fail 0, pending 0`.
3. Run alpha sweep (validation) — validation Recall@10 is reported for alpha 0.50, 0.70, 0.85 and 0.95.
4. Note that alpha = 0.50 is selected (highest validation Recall@10; an exact tie would go to the smaller alpha).
5. Set the alpha slider to 0.50 explicitly. The evaluation button uses the current slider value; it does not apply the sweep result by itself.
6. Evaluate Recall@10 — all four methods. The first click also trains the supplied MF reference.
7. Run two-user demo (alpha / seed) — users 1 and 2: user 1 at 0.85, user 2 at 0.85, user 1 at 0.50.
8. For users 1 and 2, select each method (popularity, pagerank, ppr, mf) and press Recommend Top-10, so that both users' lists are recorded.
9. Export verification, Export results, Export recommendations.

**Approximate runtimes (one macOS machine, Chrome, WebGL; measurements from one machine, not benchmarks).** Validation sweep about 6 s, of which alpha 0.95 takes about 4 s; first full evaluation including MF training about 25–30 s; repeated evaluation about 9 s; all-user PPR batch 0.2 s at alpha 0.50 and 3.7 s at alpha 0.95.

**Exports.**

- `verification.json` — the 12 self-checks with status and detail, pass/fail/pending counts, backend, TensorFlow.js and D3 versions, the MF reference configuration and checkpoint, cohort-level PageRank/PPR convergence summaries, and propagation/evaluation runtimes.
- `results.csv` — one row per test method and per validation alpha: Recall@10, cohort size, excluded unseen movies (count and IDs), users without an eligible relevant movie, mean recommended-list popularity, convergence fields, split, selected alpha, MF checkpoint and validation RMSE, cohort PPR convergence (`ppr_runs`, `ppr_nonconverged`, `ppr_max_iterations`, `ppr_max_residual`) and an `error` column. The legacy `iterations`/`residual`/`converged` columns on PPR rows describe one seed user's run; use the `ppr_*` columns for the cohort.
- `recommendations.csv` — user ID, method, alpha, rank, movie ID, title, genres, score and training popularity for every recorded Top-10 list (100 rows).

**Disclosures.**

- The instructor-supplied MF reference (16 factors, Adam 0.01, L2 0.05, batch 512, 30 epochs, seed 20261013) collapses to the global-mean predictor under its supplied regularization, which adds a full-matrix L2 sum on every mini-batch. Its selected checkpoint is epoch 1 (validation RMSE 1.1829944264078285), all candidate scores are equal, and its Top-10 order is the ascending movie-ID tie-break. This describes this supplied configuration only, not matrix factorization in general.
- At alpha = 0.95, 71 of the 846 validation-cohort PPR runs reach the 500-iteration cap without meeting the 1e-6 L1 tolerance under float32 WebGL arithmetic; they are exported as non-converged. The selected alpha = 0.50 converged for all 861 evaluated test users (maximum 22 iterations).
- PPR and PageRank scores are long-run graph-walk scores. They are not predicted ratings, purchase probabilities or viewing probabilities, and they are not comparable in scale to MF scores.
