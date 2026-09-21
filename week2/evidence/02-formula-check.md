# A02 Evidence — 02. Formula Check

Every number below comes from `week2/evidence/a02-verification.json` (checks 2.1–2.6), computed by executing the real `data.js` + `script.js` cosine pipeline.

## Procedure

Real pair: **A = Toy Story (id 1)**, **B = Aladdin (id 95)**.

Vector A (`Toy Story`, Animation/Children's/Comedy):

`[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`

Vector B (`Aladdin`, Animation/Children's/Comedy/Musical):

`[0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]`

## Computed values

| Quantity | Value |
|---|---|
| dot(A, B) | 3 |
| norm(A) | √3 = 1.7320508075688772 |
| norm(B) | 2 |
| manual cosine = 3 / (√3 · 2) | 0.8660254037844387 |
| `cosineSimilarity(A, B)` | 0.8660254037844387 |

| Assertion | Pass |
|---|---|
| 2.1 dot = 3 (within 1e-12) | ✓ |
| 2.2 norms = √3 and 2 (within 1e-12) | ✓ |
| 2.3 manual vs application agree within **tolerance 1e-12** | ✓ |
| 2.4 symmetry: cosine(A,B) == cosine(B,A) (within 1e-12) | ✓ |
| 2.5 cosine(A, A) == 1 exactly (non-zero vector) | ✓ |
| 2.6 cosine(A, A) within 1e-12 of 1 | ✓ |

## Note on floating-point rounding

`dot(A,B)/(norm(A)·norm(B))` for identical vectors can round to
`1.0000000000000002`. The implementation clamps the raw ratio to the
mathematically valid range with `Math.max(-1, Math.min(1, value))`
(`script.js`, `cosineSimilarity`), so check 2.5 records `1` exactly. All
float comparisons in this evidence run use the documented tolerance
**1e-12** (`near(a, b) = |a − b| ≤ 1e-12`).

## Verification output

```
mission: 2. formula check — 6 assertions, 0 failed
```