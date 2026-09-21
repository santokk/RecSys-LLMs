# A02 Report — Cosine-Based Content-Based Movie Recommendations on MovieLens 100K

**Student Name:** Tokkozhin Sanzhar

**Year of Study:** 2

**Submission Date:** 21 September 2026

## 1. Title

**Implementation, Verification, and Analysis of a Cosine-Similarity Content-Based Movie Recommender on MovieLens 100K**

---

## 2. Abstract

This report presents a content-based movie recommender implemented in vanilla HTML/CSS/JavaScript for the MovieLens 100K dataset. The system represents each of 1,682 movies as an 18-dimensional binary genre vector and replaces the starter Jaccard index with cosine similarity. A user profile is built by averaging the vectors of exactly three watched movies; two deterministic Top-5 rankings are produced, one item-to-item and one profile-based, with all three watched movies excluded from both. A heterogeneous test triple produced 1-of-5 overlap between the two lists, while a coherent crime triple agreed in 5-of-5 titles. A 32-assertion verification suite passed with zero failures under a 1e-12 floating-point tolerance. I conclude that cosine normalization removes vector-length bias but does not remove popularity bias, and that the two recommendation modes serve different product purposes.

---

## 3. Problem and Dataset

The assignment [1] provides a single-page MovieLens recommender whose similarity step is the Jaccard index over one selected movie. The task was to replace Jaccard with cosine similarity, represent each movie with the 18 meaningful genre flags, build a profile from exactly three watched movies, and produce two comparable Top-5 lists.

MovieLens 100K [2][3] contains exactly **1,682 movies** (`u.item`) and **100,000 ratings** (`u.data`) from **943 users**, with ratings on a 1–5 scale. Each `u.item` row carries 19 binary genre flags; the first flag is `unknown`, and the remaining 18 flags (`Action` through `Western`) form the recommendation features used here. Two movies (id 267 and id 1373) have no recognized genres after excluding `unknown`; they remain in the catalog, and the cosine function returns zero for any pair involving them so the pipeline never produces `NaN` or `Infinity`. The catalog also contains duplicate rows for the same title (ids 670 and 573 are both `Body Snatchers (1993)`); both are kept, and the recommender must therefore exclude watched movies by canonical title in addition to id (Sections 4–5).

---

## 4. Implementation

All files live in `week2/` and run statically over HTTP with no dependencies.

- **Corrected genre parsing.** The starter code read `fields.slice(5, 24)` against an 18-name list, shifting every label one column left and never reading the real Western column. The correction indexes `fields[6 + index]` for `index = 0..17`, aligning `Action…Western` with the official column order (`data.js`). `u.item` is fetched as an `ArrayBuffer` and decoded with `TextDecoder('iso-8859-1')`, preserving the eight Latin-1 titles.
- **Cosine formula.** `cosine(A, B) = dot(A, B) / (norm(A)·norm(B))` [4], normalized, zero-norm-safe, and clamped to the mathematically valid range [-1, 1] so rounding can never yield a self-similarity above 1 (`script.js`).
- **Three-movie averaged profile.** `buildUserProfile` computes the element-wise mean of exactly three movie vectors; the result is the query for the profile-based mode.
- **Watched-item exclusion, two levels.** A `Set` of the three watched IDs and a `Set` of their **canonical titles** (trim → lowercase → collapse whitespace; year preserved) both filter the candidate loop. Record-level exclusion removes the exact rows; title-level exclusion removes duplicate catalog rows for the same movie, which the dataset demonstrably contains (ids 670 and 573 are both `Body Snatchers (1993)`).
- **Deterministic Top-5.** Both modes sort by score descending, title ascending, then id ascending, and return exactly five unique recommendations whenever enough candidates exist.

---

## 5. Verification

Verification was executed against the real served code, not a copy.

- **Manual cosine example.** For A = Toy Story (id 1) and B = Aladdin (id 95): dot(A,B) = 3, ‖A‖ = 1.7320508, ‖B‖ = 2, so cosine = 0.8660254037844387; the application returned the identical value. Symmetry held and cosine(A,A) = 1 exactly.
- **Data integrity.** 1,682 movies and 100,000 ratings parsed; every vector has length 18; genre labels match the official order; Toy Story and Unforgiven pass as known-row checks.
- **No leakage.** For the test triple (1, 296, 670), none of the three watched movies appears in either Top-5 — checked by **id and by canonical title**. The 573/670 duplicate was a real leak: with ID-only exclusion, id 573 (`Body Snatchers (1993)`) ranked 3rd in the profile list even though it is the same title as watched id 670. A regression test (3.9) proves title-level exclusion removes it and that the removal is title-driven, not hardcoded.
- **Deterministic results.** Re-running with identical inputs produced byte-identical rankings; the runner's 32 assertions all passed (`passed = 32, failed = 0, total = 32, tolerance = 1e-12`).
- **Browser check.** A headless-Chrome load over a local HTTP server returned HTTP 200 for all six assets, populated all three selects with 1,682 IDs each, and reported zero page console errors. I additionally performed a manual in-browser pass covering page load, three distinct selections, both five-item Top-5 lists, leakage absence (checked by title as well as id), responsive stacking, and score rendering. My manual pass did not cover duplicate-selection validation or keyboard-only navigation; those remain documented as unchecked in that observation.

---

## 6. Item-to-Item vs Profile-Based Results

Watched triple: **Toy Story (1995)**, **Promesse, La (1996)**, **Body Snatchers (1993)** — curated so the three movies share no genres pairwise. Item-to-item basis: Toy Story.

### Item-to-Item Top-5

| Rank | Movie | Cosine |
|---|---|---|
| 1 | Aladdin and the King of Thieves (1996) | 1.000 |
| 2 | Aladdin (1992) | 0.866 |
| 3 | Goofy Movie, A (1995) | 0.866 |
| 4 | 101 Dalmatians (1996) | 0.816 |
| 5 | Air Bud (1997) | 0.816 |

### Profile-Based Top-5

| Rank | Movie | Cosine |
|---|---|---|
| 1 | Aladdin and the King of Thieves (1996) | 0.655 |
| 2 | Babe (1995) | 0.655 |
| 3 | Close Shave, A (1995) | 0.655 |
| 4 | Gattaca (1997) | 0.655 |
| 5 | House of Yes, The (1997) | 0.655 |

> **Correction after rerun:** the first release listed `Body Snatchers (1993)` at rank 3 — that is id 573, a duplicate catalog row for watched id 670, and it is now excluded by canonical title. The corrected list promotes `House of Yes, The (1997)` into the fifth slot via the deterministic tie-break. No result marked as verified predates the rerun.

**Overlap: 1 of 5** (Aladdin and the King of Thieves, id 422). The item-to-item list ranks exact or near-exact clones of Toy Story's genre triple, while the profile list admits films that touch three of the seven activation coordinates of the averaged centroid but do not resemble Toy Story (Babe, Close Shave, Gattaca, House of Yes). The profile's ceiling is 0.655 because no single candidate matches a seven-dimensional blend. For contrast, the coherent crime triple (Pulp Fiction, GoodFellas, Godfather) produced identical lists in both modes (overlap 5/5). Full tables with genres and rating counts are in `week2/evidence/05-item-to-item-vs-profile.md`.

---

## 7. Business and Algorithmic Analysis

- **Profile averaging trade-off.** Averaging combines the active genre dimensions across the watched movies into a single centroid. For the heterogeneous triple, the centroid's similarity to each watched movie was 0.6547, 0.378, 0.6547; the drama-only movie was diluted hardest because its single genre became one of seven at 1/3 each. A profile from diverse movies is generic by construction; the same centroid can still rank candidates the item-to-item mode would never surface.
- **Multi-genre vector normalization.** Two candidates (Aladdin and the King of Thieves, 3 genres; Aladdin, 4 genres) share the identical raw dot product (3) with Toy Story, yet receive cosines 1.000 and 0.866: normalization removes the vector-length advantage of denser movies. In list terms, the 5-genre films Hercules and Space Jam appear in a raw-dot Top-5 but vanish from the cosine Top-5.
- **Vector-length bias vs popularity bias.** These are distinct effects. Vector-length bias is a magnitude artifact of the dot product and is removed by cosine normalization. Popularity bias is different: rating counts are not part of the genre-vector formula at all, so a mainstream film that sits near the population's genre centroid is still recommended whenever it is directionally close. Cosine does not remove popularity bias. The separate `ratingCount` diagnostic makes popularity measurable without adding popularity to the ranking score.
- **Long-tail discovery.** Applying the documented top-20% threshold (head = 337 movies by rating count, boundary = 100 ratings; long-tail = 1,345 movies), the item-to-item list contained 3 long-tail and 2 head titles, and the profile list contained 2 long-tail and 3 head titles. Both modes surfaced non-mainstream titles in this single example; I make no general claim about either mode's discovery performance.
- **Product implications.** Fatigue: a pure item-to-item loop over the same watched genre set can reinforce one cluster, inviting repetition; a profile-based list diversifies across the watched genres. Discovery: the long-tail counts above show the system can reach low-visibility titles when they are directionally close. Retention: the toggle between modes gives users two distinct intents — "more like this" vs "a blend of what I like" — which is a reasonable basis for UI experimentation.

---

## 8. Limitations

- **Genre-only representation.** Similarity uses 18 binary flags; titles, decade, cast, and text signals are ignored, so stylistic nuance within a genre is lost.
- **One case study.** The comparison rests on a small set of hand-picked triples; the overlap figures are descriptive, not a ranking evaluation, and neither mode is claimed to be universally superior.
- **Offline evaluation protocol not constructed.** MovieLens ratings could support an offline evaluation after defining a relevance threshold and a train/test protocol [5]. This assignment did not construct such a protocol because the implemented ranking uses movie genres rather than user-rating histories; therefore precision, recall, and ranking metrics are not reported.
- **Possible profile averaging collapse.** For heterogeneous inputs the averaged profile dilutes distinctive tastes (Section 7). This is inherent to centroid averaging and was observed, not worked around, in this implementation.

---

## 9. Preventive Guardrails

The verification runner asserts seven structural guarantees that would fail loudly on regression: (1) every genre vector has length 18 with 0/1 entries across all 1,682 movies; (2) cosine returns 0 for zero-norm vectors (never NaN/Infinity); (3) watched IDs are excluded from both result sets; (4) watched **canonical titles** are excluded from both result sets (covers duplicate catalog rows such as ids 670/573 `Body Snatchers (1993)`); (5) deterministic sorting (score, title, id) reproduces byte-identical rankings; (6) self-similarity is clamped to the valid range; (7) genre labels match the official column order. These map directly to the historical failure modes of the starter code (label off-by-one, leakage, and magnitude bias).

---

## 10. AI Assistance Disclosure

I implemented and verified the assignment with assistance from an AI coding assistant (OpenCode). The assistant generated draft code and wrote the automated verification runners; I reviewed the implementation, inspected the evidence it produced, and performed the documented browser checks, including the **recommendation interaction and responsive layout checks**. All numbers quoted in this report come from the reproducible evidence in `week2/evidence/` or from my own manual observation; I did not accept an AI claim that a check ran without inspecting its evidence.

---

## 11. References

1. Repo of record — Instructor, *RecSys-LLMs* (Week 2 starter repository): https://github.com/dryjins/RecSys-LLMs
2. GroupLens Research, *MovieLens 100K Dataset* (official documentation): https://grouplens.org/datasets/movielens/100k/
3. Harper, F. M., and Konstan, J. A. (2015). *The MovieLens Datasets: History and Context*. ACM Transactions on Interactive Intelligent Systems, 5(4), Article 19. https://doi.org/10.1145/2827872
4. Salton, G., and McGill, M. J. (1983). *Introduction to Modern Information Retrieval*. McGraw-Hill.
5. Herlocker, J. L., Konstan, J. A., Terveen, L. G., and Riedl, J. (2004). *Evaluating Collaborative Filtering Recommender Systems*. ACM Transactions on Information Systems, 22(1), 5–53. https://doi.org/10.1145/963770.963772