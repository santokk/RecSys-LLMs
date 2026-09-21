# A02 Evidence — 07. Browser Check

Two layers:

1. **Real headless Chrome over HTTP** — loading, script/data fetch, select population, Latin-1 titles, console errors.
2. **DOM-stub execution of the real `script.js` interaction code** — duplicate validation, zero-genre validation, both Top-5 renderers, comparison summary.

## 1. Real browser load (headless Chrome, HTTP)

Commands actually run:

```
python3 -m http.server 8912            # serving week2/
curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8912/index.html   -> 200
"Google Chrome" --headless=new --disable-gpu \
  --enable-logging=stderr --v=0 --virtual-time-budget=8000 \
  --dump-dom http://127.0.0.1:8912/index.html
```

Verified from the server log — every asset returned **200**:
`/index.html`, `/style.css`, `/data.js`, `/script.js`, `/u.item`, `/u.data`.

Verified from the dumped DOM:

| Check | Result |
|---|---|
| Page console errors / uncaught JS exceptions | **0** (`grep -c CONSOLE` on Chrome log = 0) |
| `movie-select-1` option count | 1,683 (1 placeholder + 1,682 movies) |
| `movie-select-2` option count | 1,683 |
| `movie-select-3` option count | 1,683 |
| Distinct numeric option values | 1,682 |
| Status message | `Data loaded. 1682 movies, 100000 ratings...` |
| Result panels initially hidden | yes (`class="panels hidden"`) |
| Latin-1 title `Misérables, Les (1995)` | rendered with accent intact |
| Latin-1 title `C'est arrivé près de chez vous (1992)` | rendered with accent intact |

## 2. Interaction logic (real `script.js` executed against DOM stubs)

Runner: `week2/evidence/interaction-check.js` (JavaScriptCore). It imported styles/scripts exactly as `index.html` does (data.js before script.js), populated the same globals, and drove `getRecommendations()` directly.

| Scenario | Result |
|---|---|
| Select Toy Story in first **and** second box, valid third | `Please choose three distinct movies. Duplicate: "Toy Story (1995)".` |
| First movie = `267` (no recognized genres) | `"unknown" has no recognized genres, so item-to-item similarity cannot be calculated for it...` |
| Valid triple (1, 296, 670) | both lists render 5 items; panels unhidden |
| Item-to-item heading | `Item-to-Item Top-5 (based on "Toy Story (1995)")` |
| Summary (overlap, leakage, popularity) | overlap `1 of 5` (Aladdin and the King of Thieves); leakage `None of the 10 recommendations is one of the 3 watched movies (checked by id and by title)`; item-to-item `3 long-tail / 2 head`, profile `2 long-tail / 3 head`; threshold line present |

> Interaction evidence was re-captured after the duplicate-title fix: the profile Top-5 is now `[422, 8, 408, 270, 337]` (id 573 "Body Snatchers (1993)" removed because it is a duplicate catalog row for watched id 670), and the leakage line in the UI states that exclusion is checked by id *and* title.

## What automation could not do (and the manual steps instead)

The headless `--dump-dom` mode cannot click or resize, so the following visual checks were **not** executed by an automated browser and are **not** claimed as performed. To verify manually with the same served copy:

1. Serve: `cd week2 && python3 -m http.server 8000`, open `http://localhost:8000/`.
2. Select three **different** movies and press **Get Recommendations**; confirm both panels appear side by side (stacked when the viewport is narrower than 768px) and the summary renders below.
3. Select the **same** movie twice; confirm the duplicate error appears without hiding the page.
4. Select `unknown` (or `Good Morning (1971)`) as the **first** movie; confirm the zero-genre validation message.
5. Resize the window / use a phone preview to confirm the `768px` media query stacks the panels and controls remain usable at the keyboard (tab order, focus ring).
6. Open the browser console on the data-loaded page and confirm no errors/warnings other than browser-internal noise.

## Verification output

```
mission: 7. browser check — load OK (real Chrome), interaction paths OK (DOM-stub execution),
           visual/responsive checks: see manual steps above (not automated)
```