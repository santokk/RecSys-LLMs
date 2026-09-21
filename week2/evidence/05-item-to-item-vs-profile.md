# A02 Evidence — 05. Item-to-Item vs Profile-Based

Same diverse triple as section 04 (Toy Story / Promesse, La / Body Snatchers). Item-to-item basis = **first** selected movie, Toy Story. Both lists exclude all three watched ids **and their canonical titles**. Data from `a02-verification.json`.

## Item-to-Item Top-5

| # | id | Title | cosine | Genres | Ratings | Class |
|---|---|---|---|---|---|---|
| 1 | 422 | Aladdin and the King of Thieves (1996) | 1.000 | Animation, Children's, Comedy | 26 | long-tail |
| 2 | 95 | Aladdin (1992) | 0.866 | Animation, Children's, Comedy, Musical | 219 | head |
| 3 | 1219 | Goofy Movie, A (1995) | 0.866 | Animation, Children's, Comedy, Romance | 20 | long-tail |
| 4 | 225 | 101 Dalmatians (1996) | 0.816 | Children's, Comedy | 109 | head |
| 5 | 261 | Air Bud (1997) | 0.816 | Children's, Comedy | 43 | long-tail |

## Profile-Based Top-5

| # | id | Title | cosine | Genres | Ratings | Class |
|---|---|---|---|---|---|---|
| 1 | 422 | Aladdin and the King of Thieves (1996) | 0.655 | Animation, Children's, Comedy | 26 | long-tail |
| 2 | 8 | Babe (1995) | 0.655 | Children's, Comedy, Drama | 219 | head |
| 3 | 408 | Close Shave, A (1995) | 0.655 | Animation, Comedy, Thriller | 112 | head |
| 4 | 270 | Gattaca (1997) | 0.655 | Drama, Sci-Fi, Thriller | 136 | head |
| 5 | 337 | House of Yes, The (1997) | 0.655 | Comedy, Drama, Thriller | 18 | long-tail |

> **Correction vs. first release:** id 573 ("Body Snatchers (1993)") previously appeared at rank 3. It is a duplicate catalog row for watched id 670 (same title), so it was removed by the title-level exclusion. The rerun promotes 270 and 337 one rank and drops Goofy Movie further via the deterministic tie-break.

## Overlap

**1 of 5**: id 422 (Aladdin and the King of Thieves), the only candidate whose genre set is both an exact Toy Story clone and aligned with the 7-dimension centroid.

## Concrete ranking differences (explained by the vectors)

- **Item-to-item collapses around the exact query.** Toy Story's vector is pure `{Animation, Children's, Comedy}`. Cosine ranks exact supersets first (422 = the identical triple, 1.000; then the 4-genre supersets 95 and 1219 at 0.866), then 2-genre subsets `{Children's, Comedy}` (225, 261 at 0.816). Candidates that do not touch Toy Story's genres never appear.
- **Profile spreads affinity across seven coordinates.** The centroid activates `{Animation, Children's, Comedy, Drama, Horror, Sci-Fi, Thriller}` at 1/3 each. Movies that touch three of these seven dimensions — Babe (Children's/Comedy/Drama), Close Shave (Animation/Comedy/Thriller), Gattaca (Drama/Sci-Fi/Thriller), House of Yes (Comedy/Drama/Thriller) — tie with the Toy Story–like 422 and enter the list, even though none of them resembles Toy Story directly. A Toy Story–specific follower such as Goofy Movie shares only 3 of the 7 dimensions and lands outside the Top-5 (it ties at 0.655 but is pushed below rank 5 by the title tie-break).
- **Score ceiling.** Item-to-item can reach the theoretical maximum 1.0; the averaged profile cannot, because no single movie matches a 7-dimensional blend. This is expected, not a defect.

## Single-case caution

This is one triple, chosen because it has zero pairwise genre overlap. It demonstrates *how* the two methods diverge; it does not demonstrate that either method is generally superior. For a coherent Crime/Drama triple (Pulp Fiction / GoodFellas / Godfather) both modes returned the same five Crime/Drama titles (overlap 5/5). I report both examples rather than generalizing from either.

Verification output: `mission: 5. item-to-item vs profile — assertions passed, 0 failed`.