# A02 Evidence — 08. Preventive Guardrails

Regression checks that fail loudly if the implementation drifts. All verified in `run-verification.js` (assertions 8.1–8.4 and supporting checks).

| Guardrail | Check | Where enforced | Pass |
|---|---|---|---|
| Vector-length assertion | all 1,682 `genreVector`s have length 18; entries only 0/1 | `data.js` (`parseItemData` builds via `genreNames.map`); asserted (1.3/8.1) | ✓ |
| Zero-norm handling | cosine with a zero vector returns `0`, never NaN/Infinity: `cosine(ref, 267) = 0`, `cosine(ref, 1373) = 0`, `cosine(267, 1373) = 0` | `script.js` `cosineSimilarity` (`Math.max(-1, Math.min(1, …))` after the zero-norm early return) | ✓ |
| Watched-item filtering | watched ids ({1, 296, 670}) absent from both Top-5s; function filters the whole candidate loop | `script.js` `recommendFromVector` (excludeIds set) | ✓ |
| Watched-title filtering | canonical titles of the 3 watched movies absent from both Top-5s; catches duplicate catalog rows (e.g. id 573 = id 670 "Body Snatchers (1993)") skipped by id-only checks | `script.js` `recommendFromVector` + `data.js` `canonicalizeTitle` (trim, lowercase, collapse whitespace; year preserved) | ✓ |
| Deterministic sorting | identical inputs → byte-identical `id→score→genres→ratingCount` ordering on repeat; sort key = score desc → title asc → id asc | `script.js` `compareCandidates` | ✓ |
| Self-similarity clamp | `cosine(A, A) = 1` exactly (raw ratio can round to 1.0000000000000002; clamp restores the valid range) | `script.js` `cosineSimilarity` | ✓ |
| Genre alignment | `genreNames` equals official order; Western sanity via Unforgiven | `data.js` (`fields[6 + index]`) | ✓ |

## Why these specific checks prevent regression

- **Vector length.** If a parser edit reintroduced the old `slice(5, 24)` off-by-one, the vector length or label alignment check fails immediately.
- **Zero-norm.** Without the guard, a user selecting `unknown`/`Good Morning (1971)` (or a future all-zero row) would crash the score sweep with `0/0` → NaN.
- **Watched filtering.** This is the "already-watched leakage" failure from the HW1 feedback (test 1); the check makes any reintroduced leakage a failing assertion instead of a silent bug. After the duplicate-title finding, filtering is applied at **two** levels: watched **ids** and watched **canonical titles**, so duplicate catalog rows for the same movie cannot slip through (regression 3.9 proves id 573 is excluded only because its title equals watched id 670's).
- **Deterministic sort.** Removes dependence on engine sort stability and makes the two modes reproducible across machines/runs — the checked Top-5 tables can be compared verbatim.
- **Self-similarity clamp.** Spectacularly visible symptom (self-cosine > 1) is fixed at the source; the tolerance-based float comparisons in the runner prevent the test harness itself from misreporting rounding noise as a bug.

## Verification output

```
mission: 8. preventive guardrails — assertions passed, 0 failed
```

## Reproduce everything

```
osascript -l JavaScript week2/evidence/run-verification.js    # 32 assertions + JSON artifact
osascript -l JavaScript week2/evidence/interaction-check.js   # interaction code paths
git diff --check                                              # whitespace sanity
```