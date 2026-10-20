# HW6: Personalized PageRank Movie Recommendations

**Student:** Tokkozhin Sanzhar  **Year:** 2  **Submission date:** 20 October 2026  **Assignment:** HW6 — Personalized PageRank Movie Recommendations

## 1. Abstract

This HW6 report [1] ranks movies with global PageRank [4] and Personalized PageRank (PPR) [5] on a graph of positive MovieLens 100K training ratings [2], [3], using TensorFlow.js [6]. Validation Recall@10 selected the link-following probability α = 0.50. On a shared 861-user test cohort, PPR reached Recall@10 of 0.1122, against 0.0797 for global PageRank, 0.0760 for popularity and 0.0211 for the instructor-supplied matrix-factorization (MF) reference. That reference collapsed to its global mean, so its lists follow movie-ID order rather than a trained model. All 12 automated checks passed. For this prototype I recommend replacing the collapsed MF ranking with α = 0.50 PPR, subject to a user-level online experiment, because offline recall cannot establish engagement or revenue effects.

## 2. Problem, Data, and Method

**Problem.** Rank each user's unrated movies by graph proximity to liked movies, comparing PageRank and PPR with popularity and MF under one protocol [1].

**Data.** MovieLens 100K holds 100,000 unique ratings (1–5) by 943 users on 1,682 movies, each user with at least 20 ratings [2]. Over the real 943 × 1,682 matrix, without padding, density is 6.3047% and sparsity 93.6953%. Each user's ratings are sorted by timestamp and split 80/10/10, giving 79,619 training, 9,596 validation and 10,785 test records. Timestamps record rating entry [2], not viewing sessions, and the split is per user, not global. The candidate pool is the 1,611 training movies; a rating of at least 4 is relevant.

**Graph.** $R_{ui} = 1$ when user $u$ rated movie $i$ at least 4 in training. The 943 users precede the 1,611 movies: 2,554 nodes, no padding node. Each of the 45,602 positive interactions gives a user-to-movie and a movie-to-user edge. User 685 and 253 movies have no positive training edge and therefore zero degree. $Q$ row-normalizes the adjacency matrix; zero-degree rows stay zero:

$$A=\begin{bmatrix}0 & R\\ R^{\top} & 0\end{bmatrix},\qquad p_{t+1}=\alpha\left(Q^{\top}p_t+d_t\,v\right)+(1-\alpha)\,v,\qquad d_t=\sum_{j\in Z}p_t[j].$$

Here α is the link-following probability, $1-\alpha$ the restart probability, $v$ the restart distribution and $d_t$ the mass on the zero-degree set $Z$, redistributed through $v$ by the same rule in both methods. Global PageRank uses a uniform $v$, following the random-surfer model [4]. PPR uses a one-hot $v$ at the target user's node: Brin and Page note that restricting the random jump to one page allows personalization [4], and Jeh and Widom define the personalized PageRank vector through a teleportation probability, here $1-\alpha$ [5]. Runs start at $p_0 = v$ and stop when $\lVert p_{t+1}-p_t\rVert_1 < 10^{-6}$ or after 500 iterations; capped runs count as non-converged.

**Implementation.** Propagation runs in TensorFlow.js on the WebGL backend, which stores tensors as textures and computes in shaders [6]. The 2,554 × 2,554 matrix is never built: $Q^{\top}p$ uses two cached blocks, $R\,\mathrm{diag}(1/\deg_M)$ and $R^{\top}\mathrm{diag}(1/\deg_U)$, with `tf.matMul`. PPR for all 943 users runs as one batch; each user column stops at its own tolerance. Tensors are released with `tf.tidy` and explicit disposal [6]. A Top-10 list ranks movie-node scores after removing movies rated in training or validation (test) or in training (validation), breaking ties by ascending movie ID.

$$\text{Recall@10}=\frac{1}{|U|}\sum_{u\in U}\frac{|\mathrm{Top10}(u)\cap \mathrm{Rel}(u)|}{|\mathrm{Rel}(u)|}$$

**Methods.** Popularity counts distinct users with a positive training rating. Global PageRank reuses the α selected for PPR, a disclosed control rather than separate tuning. MF is the instructor-supplied reference: 16 factors, Adam with learning rate 0.01, L2 setting 0.05, batch size 512, 30 epochs, seed 20261013, checkpoint by validation RMSE, so MF and PPR use different selection objectives.

## 3. Verification and Results

**Automated checks.** The supplied harness reports 12 passed, 0 failed, 0 pending (`verification.json`).

| Check | Exported evidence |
|---|---|
| Reference graph, uniform restart | C > A > B > D, 28 iterations |
| Reference graph, restart at A | A > C > B > D, D stays 0 |
| Row normalization | 6 positive-degree rows sum to 1; 2 dangling rows are zero |
| Restart distributions | uniform and one-hot, nonnegative, sum 1 |
| Probability conservation | sum = 1.0000002682 |
| Nonnegative scores | 3 of 3 nonnegative |
| One extra update | L1 = 7.972e-7 |
| Zero-degree fixture | mass 1.00000022 / 1.00000000 |
| Matrix orientation | p = [0.486486, 0.256757, 0.256757] |
| Filtering and tie-break | 10 eligible items for user 1 |
| Graph mapping, no padding | 2554 nodes = 943 users + 1611 movies |
| Public interface | 10 functions + ALPHA_GRID + MF_INFO |

The fixtures gave p = [0.3725269437, 0.1958237588, 0.3941493332, 0.0375000015] (uniform) and [0.4522329569, 0.1921990961, 0.3555681705, 0] (restart at A), within 1.6e-7 of the README values [1]. Live tensors numbered 0 before the first complete run and 13 after it and after a repeat (three graph, ten MF). Application error listeners recorded nothing; two extension exceptions also appear on a plain CSS page, so they come from tooling. A rerun after page reload, with MF retrained, reproduced `results.csv` and `recommendations.csv` byte for byte.

**Model selection.** α was chosen by validation Recall@10 on 846 users; an exact tie would go to the smaller α.

| α | 0.50 | 0.70 | 0.85 | 0.95 |
|---|---|---|---|---|
| Validation Recall@10 | **0.11448315913208877** | 0.1047969507641043 | 0.09002077155336492 | 0.07375030387909676 |

α = 0.50 was selected and set explicitly in the UI before the test run.

**Test comparison.** All methods share 861 warm users and the same candidates; 13 relevant test movies outside the training pool and 82 users without an eligible relevant movie are excluded.

| Method | Recall@10 | Mean list popularity |
|---|---|---|
| Popularity | 0.07602291343770759 | 271.99686411149816 |
| Global PageRank, α = 0.50 | 0.07969324149418144 | 257.820325203252 |
| PPR, α = 0.50 | **0.11218853219486066** | 244.20673635307733 |
| MF reference, epoch 1 | 0.021137488852036153 | 96.76085946573689 |

**Convergence and runtime.** Global PageRank at α = 0.50 converged in 19 iterations (residual 5.508336471393704e-7), and all 861 test PPR runs at α = 0.50 converged within 22 iterations. At α = 0.95, 71 of 846 validation PPR runs hit the 500-iteration cap (maximum residual 2.1856476450921036e-6) and are exported as non-converged; α = 0.95 has the lowest validation recall, so it does not affect the selection. On one machine, all-user PPR took 229 ms at α = 0.50 and 3,701 ms at α = 0.95.

**MF disclosure.** *Observation:* the supplied reference selected epoch 1 with validation RMSE 1.1829944264078285. Train and validation RMSE stayed at the global-mean baseline for all 30 epochs, factors stayed below 6e-5, and every exported MF score equals 3.5806026458740234, so its Top-10 is the ascending movie-ID tie-break and its Recall@10 measures no learned ranking. *Proposed explanation:* the supplied code adds the L2 sum of both full factor matrices to every mini-batch loss, which likely overwhelms the prediction error. The code was used unchanged; this says nothing general about matrix factorization.

## 4. Business and Algorithmic Analysis

**Should the platform keep MF, replace it with PPR, or use both?**

**Training histories.** User 1 has 217 training ratings, 131 positive, led by Drama (61), Comedy (39), Action (36) and Sci-Fi (27); five-star titles include *The Godfather*, *Star Wars* and *Alien*. User 2 has 49 training ratings, 32 positive, led by Drama (18), Romance (11) and Comedy (11), with five stars for *Sense and Sensibility*, *Fargo* and *Star Wars*.

PPR Top-10 at α = 0.50 (score × 10⁴, rounded; Pop. = distinct positive training users):

| # | User 1 movie | Genres | Score | Pop. |
|---|---|---|---|---|
| 1 | 100 Fargo (1996) | Crime, Drama, Thriller | 6.8552 | 377 |
| 2 | 258 Contact (1997) | Drama, Sci-Fi | 4.3987 | 307 |
| 3 | 286 English Patient, The (1996) | Drama, Romance, War | 4.3102 | 279 |
| 4 | 318 Schindler's List (1993) | Drama, War | 4.0382 | 224 |
| 5 | 357 One Flew Over the Cuckoo's Nest (1975) | Drama | 3.7717 | 202 |
| 6 | 483 Casablanca (1942) | Drama, Romance, War | 3.5969 | 197 |
| 7 | 222 Star Trek: First Contact (1996) | Action, Adventure, Sci-Fi | 3.3279 | 189 |
| 8 | 475 Trainspotting (1996) | Drama | 3.2809 | 164 |
| 9 | 275 Sense and Sensibility (1995) | Drama, Romance | 3.2019 | 178 |
| 10 | 423 E.T. the Extra-Terrestrial (1982) | Children's, Drama, Fantasy, Sci-Fi | 3.0567 | 167 |

| # | User 2 movie | Genres | Score | Pop. |
|---|---|---|---|---|
| 1 | 181 Return of the Jedi (1983) | Action, Adventure, Romance, Sci-Fi, War | 6.9734 | 339 |
| 2 | 9 Dead Man Walking (1995) | Drama | 4.9766 | 182 |
| 3 | 7 Twelve Monkeys (1995) | Drama, Sci-Fi | 4.7437 | 234 |
| 4 | 117 Rock, The (1996) | Action, Adventure, Thriller | 4.5636 | 215 |
| 5 | 98 Silence of the Lambs, The (1991) | Drama, Thriller | 4.4651 | 308 |
| 6 | 121 Independence Day (ID4) (1996) | Action, Sci-Fi, War | 4.3071 | 207 |
| 7 | 174 Raiders of the Lost Ark (1981) | Action, Adventure | 4.0285 | 312 |
| 8 | 268 Chasing Amy (1997) | Drama, Romance | 4.0181 | 146 |
| 9 | 475 Trainspotting (1996) | Drama | 4.0161 | 164 |
| 10 | 15 Mr. Holland's Opus (1995) | Drama | 4.0092 | 160 |

| Setting (movie IDs) | User 1 | User 2 |
|---|---|---|
| Popularity | 100 258 286 300 288 318 313 302 357 483 | 181 174 98 56 172 7 79 64 318 173 |
| Global PageRank, α = 0.50 | 258 286 100 313 300 288 302 294 275 222 | 181 174 98 7 56 172 117 121 268 222 |
| MF reference | 5 6 18 20 32 74 100 102 111 129 | 2 3 4 5 6 7 8 9 11 12 |
| PPR, α = 0.85 | 100 258 286 318 357 483 288 300 222 313 | 181 98 174 7 56 117 172 9 121 64 |

**Measured observations.** User 1's PPR list is mostly drama (*Schindler's List*, *Casablanca*), matching the leading liked genre, while *Contact*, *Star Trek: First Contact* and *E.T.* extend the Sci-Fi side of the history; three of user 1's 19 relevant test movies appear (*Fargo*, *Contact*, *Star Trek: First Contact*). User 2's list opens with *Return of the Jedi* and adds dramas such as *Dead Man Walking*, but also *The Rock* and *Independence Day*, although only 5 of this user's positives are Action; neither of user 2's two relevant test movies appears. Every PPR title has at least 146 positive training users. User 1's MF list holds seven relevant movies only because many held-out movies have low IDs.

In the controlled comparison, the baseline (user 1, α = 0.85) shares 0 movies with the seed-only change (user 2, α = 0.85) and 7 with the alpha-only change (user 1, α = 0.50); the two changed settings share 0. User 1's mean list popularity is 245.5 at α = 0.85 and 228.4 at α = 0.50. Users 1 and 2 share 1 movie under popularity, global PageRank and PPR, and 2 under MF. These counts are descriptive: a different list does not prove better personalization, nor is a less popular title automatically a discovery.

**Proposed explanation.** Restarting with probability 0.50 keeps the walk near the user's liked movies and their co-likers, while popular movies still gather mass through many edges.

**Recommendation.** For this prototype, replace the collapsed MF ranking with the validation-selected α = 0.50 PPR ranking, subject to an online experiment. PPR had the highest offline Recall@10; keeping both is not supported, because the collapsed MF adds only an ID-ordered list. This does not show that PPR beats a correctly trained MF model and establishes nothing causal about engagement, retention or revenue.

**Online experiment.** Assign users persistently, by a hash of the user ID, to the current production recommender (control) or α = 0.50 PPR Top-10 (treatment), so each user sees one arm everywhere and arms cannot contaminate each other. The primary outcome is the share of users who start at least one recommended title within 14 days of first exposure. Guardrails are catalogue coverage (distinct titles recommended per arm) and the rate of "not interested" actions per recommendation shown. The analysis window is fixed in advance at four weeks.

## 5. Limitations, AI Assistance, and References

**Limitations.** At α = 0.95, float32 WebGL arithmetic left 71 of 846 validation runs at the cap, and probability sums deviate from 1 by under 1e-6. The evaluation is warm-user only. Ratings are not session intent, and offline Recall@10 does not measure viewing completion, satisfaction, retention or revenue. Popularity uses only positive training ratings. PPR and MF scores are not comparable in scale, and the MF comparison rests on a collapsed supplied reference. `u.item` is Latin-1 but is decoded as UTF-8, so nine accented catalogue titles display incorrectly; none appears in the exports. On the case-insensitive macOS file system `README.md` and `readme.md` are one path, so run instructions were appended to `readme.md`.

**AI assistance.** Claude Code implemented the seven required stub functions and the minimal exporter additions, ran the automated and independent checks, verified the references and drafted this report. I manually ran the application controls and confirmed that all tests passed; all other checks were performed by Claude Code. I take responsibility for this submission.

**References**

[1] S. Jin, "Graph-Based Recommendations with Personalized PageRank," HW6 assignment, `dryjins/RecSys-LLMs`, HSE University, 2026. https://github.com/dryjins/RecSys-LLMs/tree/main/week6

[2] GroupLens Research, "MovieLens 100K Dataset," README file, n.d. https://files.grouplens.org/datasets/movielens/ml-100k-README.txt

[3] F. M. Harper and J. A. Konstan, "The MovieLens datasets: History and context," *ACM Transactions on Interactive Intelligent Systems*, vol. 5, no. 4, pp. 1–19, 2015. doi: 10.1145/2827872. https://doi.org/10.1145/2827872

[4] S. Brin and L. Page, "The anatomy of a large-scale hypertextual Web search engine," *Computer Networks and ISDN Systems*, vol. 30, no. 1–7, pp. 107–117, 1998. doi: 10.1016/S0169-7552(98)00110-X. http://infolab.stanford.edu/~backrub/google.html

[5] G. Jeh and J. Widom, "Scaling personalized web search," in *Proc. 12th International World Wide Web Conference (WWW '03)*, 2003. doi: 10.1145/775152.775191. https://doi.org/10.1145/775152.775191

[6] TensorFlow, "Platform and environment," TensorFlow.js guide. https://www.tensorflow.org/js/guide/platform_environment
