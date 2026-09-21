# A02 Evidence — 04. Profile Check

Heterogeneous triple used throughout the evidence suite:

| id | Title | Genres | Shared genres |
|---|---|---|---|
| 1 | Toy Story (1995) | Animation, Children's, Comedy | none |
| 296 | Promesse, La (1996) | Drama | none |
| 670 | Body Snatchers (1993) | Horror, Sci-Fi, Thriller | none |

Pairwise similarities are **all 0** (`watchedPairwise`: `1|296 = 0`, `1|670 = 0`, `296|670 = 0`) — the three movies share no genres, which is why this triple is a good, honest stress test for averaging.

## Averaged profile (element-wise mean of the three vectors)

`[0, 0, 1/3, 1/3, 1/3, 0, 0, 1/3, 0, 0, 1/3, 0, 0, 0, 1/3, 1/3, 0, 0]`

Seven coordinates carry a value, each at 1/3:

`Animation, Children's, Comedy, Drama, Horror, Sci-Fi, Thriller`

## Similarity of each watched movie to the averaged profile

| Watched movie | cosine(profile, watched) |
|---|---|
| Toy Story (1995) | 0.6547 |
| Promesse, La (1996) | **0.378** |
| Body Snatchers (1993) | 0.6547 |

## Coherence interpretation

The profile is a blend, not a clone. Toy Story and Body Snatchers each share 3 of 7 active profile dimensions with the centroid (7/3 = 2.333 active dimensions each), so they score 0.6547. The drama-only _Promesse, La_ shares 1 of 7 (its single Drama coordinate diluted to 1/3), so it scores 0.378 — roughly half the affinity of the other two.

**Interpretation:** averaging three genre-disjoint movies produces a centroid that is furthest from the most distinctive single-genre movie. Distinct tastes are diluted rather than deleted: the centroid still ranks candidates, but no coordinate reaches 1, and the profile is generic by construction. This is the averaging-collapse effect predicted by the assignment: with perfectly heterogeneous inputs the profile favors nothing in particular. For coherent triples (e.g., three Crime/Drama titles) the same centroid tracks well (see 05).

Verification output: `mission: 4. profile check — assertions passed, 0 failed`.