# Audit results — Week 1 icon display issue (`a01-icon-fix`)

**File audited:** `week1/index.html` (202 lines, single-file HTML/CSS/JS app)
**Branch:** `a01-icon-fix` (clean; `index.html` was NOT modified)
**Date:** 2026-09-16
**Environment:** macOS, headless Chrome 147 (`--headless=new`), Python 3.13 + CDP via websocket-client.
**All evidence and command outputs:** `week1/evidence/`. No application file was edited.

---

## 1. Exact Font Awesome version and URL used

`week1/index.html:7` loads one and only one icon stylesheet:

```
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
```

**Font Awesome Free 6.4.0** from cdnjs. The page uses only `fas` (solid) glyphs. No
`img.src`, no `fetch`, no image requests exist anywhere in the source — icons are pure
font glyphs injected via `innerHTML` (`index.html:186`) with class strings like
`fas fa-pizza-slice`.

---

## 2. Icon availability in FA Free 6.4.0 — method

Two independent, non-substring checks (all on the real resources):

1. **Exact CSS selector scan** of the actual `all.min.css` bytes the page loads
   (102025 bytes, downloaded from that URL). A class counts as available only when a
   rule `.{class}(:before)` with that exact class token exists. `fa-bowl` was checked
   specifically and does NOT match `fa-bowl-food`/`fa-bowl-rice`/`fa-bowling-ball`
   (exact-token boundary — see `evidence/icon-runtime-matrix.txt`).
2. **Official metadata** `FortAwesome/Font-Awesome` tag `6.4.0`,
   `metadata/icons.json` (1856 entries) — verified the `free: ["solid"]` flag.

A third check cross-validated this at runtime in the real page (see §3).

---

## 3. Results table

| Menu item | Current icon class | Exists in FA Free 6.4.0? | Semantically suitable? | Evidence |
|-----------|--------------------|--------------------------|------------------------|----------|
| Pizza     | `fa-pizza-slice`   | **Yes** (`.fa-pizza-slice:before` = `\f818`) | Yes — a slice of pizza | matrix |
| Sushi     | `fa-fish`          | **Yes** (`\f578`) | Partial — fish ≠ sushi, but a fish is the core ingredient; acceptable | matrix |
| Burger    | `fa-hamburger`     | **Yes** (`\f805`, comma-alias with `fa-burger`) | Yes — a hamburger | matrix |
| Salad     | `fa-leaf`          | **Yes** (`\f06c`) | Yes — leafy/green suggests salad | matrix |
| Tacos     | `fa-utensil-spoon` | **Yes** (`\f2e5`, comma-alias with `fa-spoon`) | **No** — renders a spoon, not a taco; no free `taco` icon exists in 6.4.0 | matrix; metadata has no free taco |
| Ramen     | `fa-bowl-hot`      | **No** — class does not exist | N/A — does not render | matrix, metadata: key absent; runtime content:none |
| Sandwich  | `fa-bread-slice`   | **Yes** (`\f7ec`) | Partial — bread slice is a sandwich component | matrix |
| Pasta     | `fa-pasta`         | **No** — only literal substring of `.fa-pastafarianism`; no exact rule | N/A — does not render | matrix, metadata key absent |
| Curry     | `fa-mortar-pestle` | **Yes** (`\f5a7`) | Partial — pestle&mortar = spice grinding, evokes curry prep, not the dish | matrix |
| Steak     | `fa-drumstick-bite`| **Yes** (`\f6d7`) | **No** — renders a meat drumstick, not a steak | matrix; no free `steak`/`beef` icon |
| Soup      | `fa-bowl`          | **No** — only substrings `fa-bowl-food`, `fa-bowl-rice`, `fa-bowling-ball`; no exact `.fa-bowl` | N/A — does not render | matrix, metadata key absent |
| BBQ       | `fa-fire`          | **Yes** (`\f06d`) | Yes — fire/grill | matrix |

**Runtime confirmation** (headless Chrome, real page): `document.fonts.status = "loaded"`,
`fonts.check('900 16px "Font Awesome 6 Free"') = true`, and injected `<i class="fas …">`
elements for **Ramen, Pasta, Soup** return `::before content: none` — those three render
**no glyph at all**. The other nine return a real glyph codepoint
(`evidence/icon-glyph-runtime.txt`).

---

## 4. Network responses

All resources were fetched over real network and observed twice (curl + browser CDP):

| Resource | curl status | browser (CDP) status |
|----------|:-----------:|:--------------------:|
| `…/font-awesome/6.4.0/css/all.min.css` | **200** (text/css, 102025 B) | **200** (h2) |
| `…/webfonts/fa-solid-900.woff2` | **200** (150124 B) | **200** (h3) |
| `…/webfonts/fa-solid-900.ttf` / `fa-brands-400.woff2` / `fa-regular-400.woff2` | **200** each | — |
| `metadata/icons.json` (github raw 6.4.0) | 200 (4262904 B, 1856 entries) | — |

Full record in `evidence/network-status.txt`. **No 404 or failed response was observed.**
Network was available, so this is **not** "NOT TESTED".

---

## 5. Unsupported CSS class vs. HTTP 404

Distinct mechanisms, both verified separately:

- **HTTP 404** would mean the stylesheet/font fail to load → *every* icon disappears.
  NO 404 observed (all 200, fonts loaded, 9/9 valid icons render). **Not the cause.**
- **Unsupported CSS class** means the stylesheet loads fine, but for a given class there
  is no `.{class}:before{content:…}` rule, so the `::before` pseudo-element has
  `content: none` → that single icon renders nothing while all others work.
  **This is exactly what happens for `fa-bowl-hot`, `fa-pasta`, `fa-bowl`.**

So the failures are per-icon (3 of 12), not global — a class-name problem, **not** a 404.

---

## 6. Deterministic rapid-click test

Battery: `document.getElementById('generateBtn').click()` fired from the DevTools
console via CDP with `Math.random` overridden to a known sequence, so the expected
selection of each click is known:

- click #1 → `Math.random()=1.5/12` → `floor(0.125·12)=1` → **Sushi** (`fa-fish`)
- click #2 (at t≈203 ms) → `Math.random()=8.5/12` → `floor(0.708·12)=8` → **Curry** (`fa-mortar-pestle`)

Observed timeline (recorded in §9's pre-fix run; the raw log file was removed during
evidence cleanup — the values below are preserved inline from that run):

| t (ms) | DOM state |
|--------|-----------|
| 1    | spinner (`fa-spinner fa-spin`), name "Thinking..." |
| 203  | click #2 dispatched |
| 204  | spinner, "Thinking..." |
| 410  | spinner, "Thinking..." (both timers still pending; #1 fires at ~500, #2 at ~703) |
| 614  | **Sushi** shown — timer of the *older* click has replaced the pending "Thinking..." **before** the newer click’s timer fires |
| 1072 | **Curry** shown — final state |

Two sub-questions answered:

- **Does an older timer briefly replace the newer "Thinking…" state?** **YES.**
  At t=614 ms the display showed Sushi even though the latest click (Curry, at 203 ms)
  was still pending. Cause: `setTimeout` in `generateRandomLunch` (`index.html:185`)
  is not cancelled per click and the button isn’t disabled.
- **Does the final state belong to an earlier click?** **NO.** The final state
  (Curry, at 1072 ms) matches the *last* click. Timers fire in schedule order, so the
  last scheduled timer wins. The earlier hypothesis that "final result may belong to an
  earlier click" is **not supported** by this test.

---

## 7. Hypotheses — verdicts

| Hypothesis | Verdict | Basis |
|------------|---------|-------|
| **Missing icon (class not in loaded font)** | **CONFIRMED** | `fa-bowl-hot`, `fa-pasta`, `fa-bowl` have no `:before` rule in the loaded CSS, no entry in official 6.4.0 metadata, and render `content:none` at runtime. Affects Ramen, Pasta, Soup. |
| **HTTP 404 (stylesheet/font)** | **REJECTED** | All assets observed at HTTP 200 in both curl and browser CDP; fonts loaded; 9/9 valid icons render. A 404 was never observed. |
| **Semantic mismatch (icon loads but depicts the wrong thing)** | **PARTIALLY CONFIRMED** *(judgment-based)* | Two icons render but misrepresent the food: Tacos → a spoon (`fa-utensil-spoon`), Steak → a poultry drumstick (`fa-drumstick-bite`). No free `taco`/`steak`/`beef` icons exist in 6.4.0. This does **not** make icons disappear, so it is not a cause of the missing-icon bug. |
| **Rapid-click race causing wrong/blank final icon** | **PARTIALLY CONFIRMED** | Confirmed: older timers briefly replace the newer pending "Thinking…" (stale flash at t≈500–703 ms). Rejected sub-claim: final state does **not** belong to an earlier click (last click always wins). |

Notes on honesty:
- prompt.md is an example README conversation (per the correction in this thread), not a
  formal assignment spec; it is not treated as a requirement, only as context.
- "Semantically suitable?" is **human judgment** for every row (a machine cannot decide
  if a taco icon "should" be a taco); "Exists in FA Free 6.4.0?" is machine-verified.

---

## 8. Minimal proposed fix (APPLIED in this session)

Applied changes to `week1/index.html` only (no new files, no build, no dependencies):

| Menu item | Old class | New visual | Basis |
|-----------|-----------|-----------|-------|
| Ramen | `fa-bowl-hot` | `fa-bowl-food` | confirmed unsupported class; replacement verified free-solid |
| Pasta | `fa-pasta` | `fa-plate-wheat` | confirmed unsupported class; only plated-dish free option in 6.4.0 |
| Soup | `fa-bowl` | `fa-bowl-rice` | confirmed unsupported class; replacement verified free-solid |
| Tacos | `fa-utensil-spoon` → `fa-pepper-hot` → 🌮 | native emoji U+1F32E | see manual-finding note below |
| Steak | `fa-drumstick-bite` | `fa-cow` | semantic mismatch (drumstick ≠ steak); candidate verified free-solid (`\f6c8`) |

The four FA classes were verified two ways before editing: exact `.fa-{name}:before`
selector present in the FA 6.4.0 `all.min.css`, and `free: ["solid"]` in the official
6.4.0 `metadata/icons.json`.

### Student Manual Verification

Manual finding (student manual visual verification): `fa-pepper-hot` technically
resolves to a valid free-solid glyph, but during personal visual inspection it was
**rejected** because a pepper does not clearly represent tacos. It is therefore
replaced with the native Unicode taco emoji 🌮 (U+1F32E) rather than an image, another
CDN, a Pro icon, or a new dependency. Rendering supports both forms per menu item: a
Font Awesome class (`icon`) or a native emoji (`emoji`); the emoji is output inside
`<span class="food-emoji" role="img" aria-label="…">`.

Student-performed observations (opened `http://localhost:8000/week1/` in Chrome):

- The page loaded successfully.
- Regular button clicks worked correctly.
- The initial pepper icon rendered, but I rejected it because it did not clearly represent tacos.
- After the correction, Tacos displayed the native 🌮 emoji.
- Five rapid clicks did not allow an older result to replace the latest Thinking state.
- The final food name matched the displayed visual.
- Chrome DevTools Console showed no errors.

Rapid-click stale flash fix (`generateRandomLunch`, `index.html`):

- added `let pendingLunchTimeout = null;`
- `clearTimeout(pendingLunchTimeout)` runs before each new `setTimeout`
- the new timer id is stored back into `pendingLunchTimeout`

The 12-item menu, `Math.random` selection, 500 ms "Thinking…" delay, appearance CSS, and
GitHub Pages compatibility are unchanged.

---

## 9. Post-fix verification (2026-09-16, headless Chrome 147 via CDP)

Re-ran the deterministic audit after the emoji change (`evidence/cdp_audit.py`, output
in `evidence/postfix_verification.txt` + `evidence/cdp_results.json`). Full results:

**Icons — 11 / 11 Font Awesome classes resolve to a non-empty `::before` glyph; the
remaining Tacos visual is the native emoji 🌮 (U+1F32E), not a Font Awesome glyph:**

```
Pizza fa-pizza-slice OK   | Ramen fa-bowl-food OK    | Curry fa-mortar-pestle OK
Sushi fa-fish OK          | Sandwich fa-bread-slice OK| Steak fa-cow OK
Burger fa-hamburger OK    | Pasta fa-plate-wheat OK   | Soup fa-bowl-rice OK
Salad fa-leaf OK          | BBQ fa-fire OK            | Tacos -> 🌮 (U+1F32E)
```

The 11 FA classes that exist in FA Free 6.4.0 all render; only Tacos uses a native
emoji. Verified at runtime via a real click pinned to index 4 (Tacos):
`codes = {found: true, text: "\u{1F32E}", codePoint: "1f32e", role: "img",
aria-label: "Tacos"}`, i.e. the emoji renders as a non-empty native taco and is
exposed accessibly. Measured rendered boxes: FA Pizza `i` 64×64 px vs emoji
`span.food-emoji` 56×68 px (font-size 3.5rem vs 4rem) — approximately the same visual
size.

**Network** — stylesheet `…/font-awesome/6.4.0/css/all.min.css` **200** (h2) and font
`…/webfonts/fa-solid-900.woff2` **200** (h3), observed by the browser; `fonts.status =
loaded`.

**Rapid-click race, deterministic (`Math.random` pinned: click1→Sushi, click2→Curry at
t≈200 ms):**

| t (ms) | DOM state | Expectation |
|--------|-----------|-------------|
| 2 | spinner, "Thinking..." | ok |
| 209 | spinner, "Thinking..." | ok (pre-click 2) |
| 211 | spinner, "Thinking..." | ok (post-click 2) |
| 415 | spinner, "Thinking..." | ok |
| 651 | **spinner, "Thinking..."** | **click1's timer window — was NOT replaced by Sushi** |
| 857 | Curry | click2 resolved |
| 1280 | Curry | **final belongs to latest click** |

- `first_click_cannot_replace_thinking = True`
- `final_belongs_to_latest_click = True`
- console errors / uncaught exceptions: **none**

Pre-fix behavior (from this doc §3/§6) showed `Sushi` displayed at t≈614 ms while the
latest click (Curry) was still pending; after the fix the same instant stays
"Thinking..." and only the latest click resolves. Pre-fix findings above are preserved
unmodified.

Screenshots: `evidence/shot_tacos_emoji.png`, `evidence/shot_final_curry.png`.
Evidence directory was cleaned per §10: `icons.json`, `all.min.css`, the raw
flawed-probe log (`cdp_output.txt`), and the pre-fix screenshots were removed.