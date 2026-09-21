# A02 Evidence — 03. Recommendation Invariants

Run on watched triple **Toy Story (1), Promesse, La (296), Body Snatchers (670)** — a deliberately diverse triple — with all three watched ids **and their canonical titles** excluded from both candidate sets. Data from `week2/evidence/a02-verification.json` (checks 3.1–3.9).

## Invariants

| Check | Result |
|---|---|
| 3.1 item-to-item returns exactly 5 unique ids | ✓ ids `[422, 95, 1219, 225, 261]` |
| 3.2 profile-based returns exactly 5 unique ids | ✓ ids `[422, 8, 408, 270, 337]` |
| 3.3 no watched movie ({1, 296, 670}) in either Top-5 | ✓ |
| 3.4 all candidate scores finite | ✓ 1,678 scores per mode |
| 3.5 all candidate scores within **[0, 1]** | ✓ |
| 3.6 all 10 returned scores within [0, 1] | ✓ |
| 3.7 deterministic: identical ordering on repeat | ✓ |
| 3.8 no watched canonical title in either Top-5 | ✓ |
| 3.9 duplicate-title regression (watched 670 excludes same-title 573) | ✓ |

## Candidate score sweep

| Mode | Candidates | min | max |
|---|---|---|---|
| item-to-item (Toy Story) | 1,678 | 0 | 1 |
| profile-based (mean of 3) | 1,678 | 0 | 0.6546536707079771 |

Candidates = 1,682 − 3 watched ids − 1 duplicate-title row (id 573, "Body Snatchers (1993)" is the same title as watched id 670). The profile mode's ceiling is lower than 1 because no single candidate exactly matches the averaged centroid; zero scores come from the zero-vector movies 267/1373 and from candidates sharing no genre with the reference.

## Title-level exclusion (checks 3.8–3.9)

The catalog contains duplicate records: **id 670 and id 573 are both "Body Snatchers (1993)"**, verified to have the identical canonical key `body snatchers (1993)` (trimmed, lowercased, repeated-whitespace collapsed; the year is preserved, so a remake with a different year would stay distinct). ID-only exclusion let 573 through; check 3.9 regression proves the fix: 573 is absent from both Top-5 **and** reappears as rank 3 when the title filter is disabled (`recommendFromVector(profile, exclude, new Set(), 5)` → includes 573), i.e. the exclusion is title-driven, not hardcoded.

## Determinism

Deterministic sort key implemented in `script.js` (`compareCandidates`): **score descending, then title ascending, then id ascending**. Check 3.7 re-ran `recommendFromVector` for both modes with identical inputs and confirmed byte-identical `id→score→genres→ratingCount` ordering. No randomness is used anywhere in the pipeline.

## Verification output

```
mission: 3. recommendation invariants — 9 assertions, 0 failed
```