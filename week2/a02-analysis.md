# A02 — Cosine-Similarity Content-Based Movie Recommender: Analysis

Branch: `a02-cosine-profile` · Files: `week2/index.html`, `week2/style.css`, `week2/data.js`, `week2/script.js`, `week2/a02-analysis.md` · Evidence: `week2/evidence/`

Every number cited here was produced by **executing the real application code** (`week2/data.js` + `week2/script.js`) in JavaScriptCore, plus a real headless-Chrome load over HTTP. The runner and artifacts are in `week2/evidence/`. I did not rely on a mirror implementation for the final numbers; the Python mirror I used during the audit was retired once the JS runner existed.

---

## 1. Data integrity

- `u.item` parses to exactly **1,682** movies; `u.data` parses to exactly **100,000** ratings (`evidence/01-data-integrity.md`).
- Every movie carries an **18-element** binary vector aligned to the official genre order `Action … Western` (`fields[6]` through `fields[23]`). The `unknown` flag at `fields[5]` is excluded from the vector.
- Parsing sanity: **Toy Story (1995)** → vector `[0,0,1,1,1,0,…]`, genres `Animation, Children's, Comedy`, 452 ratings. **Unforgiven (1992)** → Western flag at index 17.
- Unknown-only / zero-vector movies: exactly **id 267** (`unknown`) and **id 1373** (`Good Morning (1971)`). They are kept in the catalog; the cosine function returns 0 for any pair involving them.
- **Duplicate catalog records:** `id 670` and `id 573` are both `Body Snatchers (1993)` — separate data rows for the same title, with distinct rating counts (both retained; no source row deleted). Every movie therefore stores a **canonical title key** (`canonicalizeTitle`: trim → lowercase → collapse whitespace; the year is preserved so a remake with a different year stays distinct). Because watched-id 670's canonical title equals id 573's, ID-only exclusion silently let a watched movie back in; see §3 and §8.
- The 8 titles with non–UTF-8 bytes restore correctly under `ArrayBuffer` + `TextDecoder('iso-8859-1')` (verified in real Chrome: `Misérables, Les (1995)` and `C'est arrivé près de chez vous (1992)` render with accents).

## 2. Formula check

For **A = Toy Story**, **B = Aladdin (1992)**: dot(A,B) = 3, ‖A‖ = √3, ‖B‖ = 2, so cosine = 3/(√3·2) = 0.8660254037844387, exactly what `cosineSimilarity` returns (agreement within tolerance 1e-12). Symmetry holds, and cosine(A,A) = 1 exactly. One honest implementation detail: the raw IEEE-754 ratio for identical vectors can round to 1.0000000000000002, so `cosineSimilarity` clamps the ratio to [-1, 1] (`evidence/02-formula-check.md`). This is a guard, not a mask — the application returns the mathematically valid value, and the tolerance-based equality in the runner (|a−b| ≤ 1e-12) keeps the evidence honest.

## 3. Recommendation invariants

For the triple (1, 296, 670): both modes return **five unique** ids; no watched **id** and no watched **canonical title** appears in either list; every one of the 2·1,678 candidate scores is finite and within **[0, 1]**; and repeated runs produce byte-identical ordering. Sorting is **score descending → title ascending → id ascending**, so results are reproducible across machines (`evidence/03-recommendation-invariants.md`).

**Duplicate-title correction.** Watched id 670 (`Body Snatchers (1993)`) has a duplicate catalog row at id 573 (same title, same canonical key). With ID-only exclusion, 573 ranked 3rd in the profile Top-5 — a watched movie leaking back in through a different id. The fix excludes candidates by **both** ID and canonical title. Regression 3.9 proves the behavior is title-driven, not hardcoded: with the title filter disabled, 573 reappears at rank 3; with it enabled, it is gone. This is why dataset duplicates forced a *semantic* (title-level) exclusion on top of the *record-level* (ID) one — both retain their role because legitimate distinct-rows (e.g. remakes) must still be recommendable, so the canonical key deliberately keeps the year.

## 4. Profile check

The heterogeneous triple shares **zero** genres pairwise. Its averaged vector activates seven dimensions at 1/3 each. Similarity of each watched movie to the centroid: Toy Story 0.6547, Body Snatchers 0.6547, **Promesse, La (Drama-only) 0.378**. Averaging therefore diluted the distinctive single-genre movie most: the centroid is bounded, generic, and favorites no single taste. For this adversarial input the collapse is expected and visible; it is the cost of a profile, not a defect (`evidence/04-profile-check.md`).

## 5. Item-to-item vs profile-based

Item-to-item (Toy Story) returns exact-clone titles (Aladdin and the King of Thieves 1.000; 4-genre supersets 0.866; 2-genre subsets 0.816). The profile list, tied at 0.655, admits films that touch three of the seven centroid dimensions — Babe, Close Shave, Gattaca, House of Yes — films item-to-item would never surface; Comedy-only followers (101 Dalmatians, Air Bud) drop out. (The first release wrongly listed Body Snatchers here; id 573 is the duplicate row for watched id 670 and is now excluded by canonical title, see §3.) Overlap is **1 of 5**. The same comparison on a coherent Crime/Drama triple produced identical lists (overlap 5/5). I report both as observations of *how the methods trade off*, not as evidence that one is better (`evidence/05-item-to-item-vs-profile.md`).

## 6. Bias and discovery

Two candidates can share the same raw dot product (3) with Toy Story yet receive cosines 1.000 (3-genre) and 0.866 (4-genre). Raw dot rewards density; cosine rewards direction. In list terms, the 5-genre films **Hercules** and **Space Jam** sit in the raw-dot Top-5 but vanish from the cosine Top-5. Normalization thus removes the multi-genre magnitude advantage — but it **cannot** remove popularity bias, because rating counts are not part of the similarity formula at all; a mainstream film near the genre centroid is still recommended whenever it is directionally close. Apply the documented 80th-percentile threshold (head = top 337 movies by count, boundary 100 ratings; long-tail = 1,345): item-to-item returned 3 long-tail / 2 head, profile 2 long-tail / 3 head. These are counts for one profile, and I limit the claim to that (`evidence/06-bias-and-discovery.md`).

## 7. Browser check

A real headless-Chrome load over `python3 -m http.server` returned HTTP 200 for all six assets, populated all three selects with 1,682 distinct ids each, rendered the Latin-1 titles correctly, and produced **zero** page console errors. The interaction code paths (three distinct or duplicate selections, zero-genre first movie, both Top-5 panels, summary) were exercised by executing the real `script.js` against DOM stubs, since headless `--dump-dom` cannot click. The remaining visual/responsive checks (panel stacking, keyboard focus) are provided as explicit manual steps rather than fabricated results (`evidence/07-browser-check.md`).

## 8. Preventive guardrails

Seven regression guards are asserted: vector length (18), zero-norm cosine → 0, watched-**ID** filtering, watched-**canonical-title** filtering, deterministic sorting, the self-similarity clamp, and genre-label alignment. Each maps to a specific historical failure mode (see `evidence/08-preventive-guardrails.md`). The two-level watched filter exists because the dataset has duplicate rows (`Body Snatchers (1993)` = ids 670 and 573): ID exclusion alone could not enforce "do not recommend a movie the user has already watched" — only the title key can. The runner also cross-checks the checked Top-5 tables against themselves on repeat.

---

## Student Manual Verification

Manual browser pass (local HTTP server, real browser):

- The page loaded successfully: yes.
- Three unique movie selections worked: yes.
- Item-to-item Top-5 contained five results: yes.
- Profile-based Top-5 contained five results: yes.
- None of the three watched movies appeared in either list: yes.
- The comparison summary reported zero overlap for this selected example.
- None of the three watched movies appeared in either list (verified by title, not only by id): yes.
- Genre labels appeared semantically correct: yes.
- The responsive narrow layout stacked the result panels vertically: yes.
- Recommendation scores rendered correctly within the expected range.
- Duplicate-selection validation was not manually checked in this observation.
- Keyboard-only navigation was not manually checked in this observation.
- Browser console errors were covered by the automated browser evidence; I did not manually inspect the console in this observation.

The independent verification runner reported: **passed = 32, failed = 0, total = 32, tolerance = 1e-12**.

---

## Test results

```
run-verification.js    : passed=32 failed=0 total=32 tolerance=1e-12
interaction-check.js   : all scenarios behaved as expected (messages/text verified)
headless Chrome (HTTP) : all assets 200; selects populated; console errors 0
```

## Evidence file list

```
week2/evidence/01-data-integrity.md
week2/evidence/02-formula-check.md
week2/evidence/03-recommendation-invariants.md
week2/evidence/04-profile-check.md
week2/evidence/05-item-to-item-vs-profile.md
week2/evidence/06-bias-and-discovery.md
week2/evidence/07-browser-check.md
week2/evidence/08-preventive-guardrails.md
week2/evidence/a02-verification.json     <- raw assertion + data artifact
week2/evidence/run-verification.js      <- reproducible runner (32 assertions)
week2/evidence/interaction-check.js     <- reproducible interaction-path runner
```

## Remaining limitations

1. Visual/responsive clicks and keyboard flow were not automated (no webdriver here); the manual checklist in `07-browser-check.md` covers them.
2. Headless Chrome's `--virtual-time-budget` load is one snapshot; a slow network or a browser policy change could alter timing, but assets are tiny.
3. One profile and two contrast triples is a worked demonstration, not a ranking evaluation; all quality claims stay inside that scope.
4. Only the 18 binary genre flags drive similarity — no text, cast, or decade signal.

Nothing has been committed or pushed.