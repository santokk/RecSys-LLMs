# A02 Evidence — 06. Bias and Discovery

## Raw dot product vs cosine (assertion 6.1)

Reference vector = Toy Story (3 active genres: Animation, Children's, Comedy).

Two candidates share the **identical raw dot product** with Toy Story:

| id | Title | Total genres | dot(ref) | norm | cosine |
|---|---|---|---|---|---|
| 422 | Aladdin and the King of Thieves (1996) | 3 | 3 | 1.732 | **1.000** |
| 95 | Aladdin (1992) | 4 | 3 | 2.000 | **0.866** |

Both contain exactly the same three shared genres, so the raw dot product is 3 for both. Cosine divides by different vector lengths and separates them: the 3-genre movie scores 1.000, the 4-genre movie 0.866. **This is the vector-length advantage that normalization removes**: raw dot rewards movies simply for being denser (having more genre tags), independent of direction.

The effect is visible in whole-list rankings:

| Ranking key | Top-5 ids |
|---|---|
| Raw dot product | `[95, 422, 1219, 993, 820]` |
| Cosine (application) | `[422, 95, 1219, 225, 261]` |

The 5-genre movies **Hercules (993)** and **Space Jam (820)** rank inside the raw-dot Top-5 (high norms, same 3 shared genres) but fall out of the cosine Top-5 entirely, replaced by the 2-genre `{Children's, Comedy}` titles 225 and 261.

## Why cosine does not eliminate popularity bias

Cosine normalization operates only on genre-vector magnitude. Popularity enters the system nowhere in `data.js`/`script.js` as a score feature: `ratingCount` is computed from `u.data`, attached for display, and never multiplied, added, or sorted on. A high-visibility title that sits near the population's genre centroid is still recommended whenever it is directionally close to the query/profile — normalization cannot detect or correct that. Popularity can only be *measured* (as the rating-count diagnostic), not *removed*, by a genre-only cosine recommender.

## Popularity threshold (documented, catalog-based)

Threshold definition in the app: **head = top 20% of movies by rating count**, taken as the top-337 rows of the deterministic order (count desc → title asc → id asc).

| Quantity | Value |
|---|---|
| Head count | 337 (of 1,682; `Math.ceil(1682 × 0.20)`) |
| Boundary | 100 ratings (the 337th-highest count) |
| Long-tail count | 1,345 |

Classification counts applied to each Top-5:

| Mode | head | long-tail |
|---|---|---|
| Item-to-Item | 2 | 3 |
| Profile-Based | 3 | 2 |

Both methods surface long-tail items in this example (3 and 2 of 5 respectively), so neither is "mainstream-only" here. Because this is a single profile, I make no general claim about discovery performance.

## Limits of this analysis

- One profile (three movies) and one human-chosen triple; no aggregate evaluation.
- Only binary genre metadata (18 flags) is used; no release year, cast, or text features.
- Popularity is measured by rating *count* from `u.data`; recency and rating *value* are ignored.

Verification output: `mission: 6. bias and discovery — assertions passed, 0 failed`.