# A02 Evidence — 01. Data Integrity

Source of truth: `week2/evidence/a02-verification.json` (produced by `run-verification.js`, which executes the real `week2/data.js` and `week2/script.js` in JavaScriptCore).

## Confirmed facts

| Check | Result |
|---|---|
| Number of movies parsed from `u.item` | **1,682** (assertion 1.1) |
| Number of ratings parsed from `u.data` | **100,000** (assertion 1.2) |
| Every movie `genreVector` length | **18** (assertion 1.3) |
| Vector entries are all 0/1 and finite | **true** (assertion 1.4) |

## Genre-label alignment (assertion 1.5)

`genreNames` in `data.js` equals, in order:

`Action, Adventure, Animation, Children's, Comedy, Crime, Documentary, Drama, Fantasy, Film-Noir, Horror, Musical, Mystery, Romance, Sci-Fi, Thriller, War, Western`

Because `parseItemData` reads `fields[6 + index]` for `index = 0..17`, each name is bound to the corresponding official `u.item` column (`fields[6]` = Action … `fields[23]` = Western). The `unknown` flag at `fields[5]` is excluded from the vector.

## Known-row sanity check (assertion 1.6)

`id 1 — Toy Story (1995)`

- vector: `[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`
- genres: `Animation, Children's, Comedy`
- ratingCount: 452

## Western-flag sanity check (assertion 1.7)

`id 203 — Unforgiven (1992)` has `genreVector[17] === 1` and `genreNames[17] === "Western"`.

## Unknown-only / zero-vector movies (assertion 1.8)

Exactly two movies have no recognized genre flags after excluding `unknown`:

- `id 267` — `unknown`
- `id 1373` — `Good Morning (1971)`

Their 18-vectors are all zeros. They are retained in the catalog; `cosineSimilarity` returns 0 for any pair involving a zero vector (see 08-guardrails).

## Duplicate catalog records (canonical titles)

The catalog contains rows that repeat the same title with different ids. A canonical title key is stored on every movie (`data.js` `canonicalizeTitle`: trim → lowercase → collapse repeated whitespace; the year is preserved, so remakes with a different year remain distinct). Relevant for this analysis:

- `id 670` **Body Snatchers (1993)** = `id 573` **Body Snatchers (1993)** → canonical `body snatchers (1993)` (duplicate data rows, both retained in the catalog; no source row deleted).

Title-level exclusion therefore complements ID-level exclusion in `script.js`; the duplicate pair above is exactly what made ID-only filtering insufficient (regression 3.9).

## Verification output

```
mission: 1. data integrity — 8 assertions, 0 failed
```