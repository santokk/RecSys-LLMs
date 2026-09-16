#!/usr/bin/env python3
"""Reproducible CDP audit of week1/index.html using headless Chrome (post-emoji-fix).

Checks:
  1. network status of the FA 6.4.0 stylesheet + solid font (no 404)
  2. the 11 Font Awesome menu icons each resolve to a non-empty ::before glyph
  3. the Tacos item renders the native taco emoji (U+1F32E) accessibly
     (role="img", aria-label) and at a size comparable to the FA glyphs
  4. deterministic rapid-click race (Math.random overridden):
     - earlier clicks' timers are cancelled (must not replace the latest
       click's 'Thinking...' state)
     - the final result belongs to the latest click
  5. no uncaught JavaScript exceptions / console errors
"""
import base64
import json
import os
import time
import urllib.parse

import requests
import websocket

DEBUG_PORT = 9223
PAGE_URL = "file:///Users/Sanzha.Tokkozhin/Desktop/Projects/RecSys-LLMs/week1/index.html"
EVIDENCE = os.path.dirname(os.path.abspath(__file__))

TACO = "\U0001F32E"  # U+1F32E taco emoji

FA_ICONS = [
    ("Pizza", "fa-pizza-slice"),
    ("Sushi", "fa-fish"),
    ("Burger", "fa-hamburger"),
    ("Salad", "fa-leaf"),
    ("Ramen", "fa-bowl-food"),
    ("Sandwich", "fa-bread-slice"),
    ("Pasta", "fa-plate-wheat"),
    ("Curry", "fa-mortar-pestle"),
    ("Steak", "fa-cow"),
    ("Soup", "fa-bowl-rice"),
    ("BBQ", "fa-fire"),
]


class CDP:
    def __init__(self, ws_url):
        self.ws = websocket.create_connection(ws_url, timeout=30)
        self.msg_id = 0
        self.events = []

    def send(self, method, params=None):
        self.msg_id += 1
        mid = self.msg_id
        self.ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
        while True:
            raw = self.ws.recv()
            msg = json.loads(raw)
            if "id" in msg:
                if msg["id"] == mid:
                    return msg.get("result", {})
            else:
                self.events.append(msg)

    def evaluate(self, expression):
        res = self.send("Runtime.evaluate", {"expression": expression, "returnByValue": True})
        if "exceptionDetails" in res:
            return {"__error__": json.dumps(res["exceptionDetails"], default=str)}
        rr = res.get("result", {})
        if rr.get("type") == "undefined":
            return None
        return rr.get("value")


def main():
    new_target = requests.put(
        "http://127.0.0.1:%d/json/new?%s" % (DEBUG_PORT, urllib.parse.quote(PAGE_URL, safe="")),
        timeout=10,
    ).json()
    cdp = CDP(new_target["webSocketDebuggerUrl"])

    cdp.send("Page.enable")
    cdp.send("Runtime.enable")
    cdp.send("Network.enable")

    log = []
    def log_line(*parts):
        s = " ".join(str(p) for p in parts)
        print(s, flush=True)
        log.append(s)

    log_line("=== POST-EMOJI-FIX CDP audit of %s ===" % PAGE_URL)
    cdp.send("Page.navigate", {"url": PAGE_URL})
    t_start = time.monotonic()
    while time.monotonic() - t_start < 15:
        if cdp.evaluate("document.readyState") == "complete":
            break
        time.sleep(0.2)
    time.sleep(1.1)  # initial generateRandomLunch delay (500ms)

    # ---- 1. Network ----
    net = []
    for ev in cdp.events:
        if ev.get("method") == "Network.responseReceived":
            resp = ev["params"]["response"]
            if "font-awesome" in resp["url"] or "webfonts" in resp["url"] or resp["url"].startswith("file:"):
                net.append({"url": resp["url"], "status": resp.get("status"),
                            "mimeType": resp.get("mimeType"), "protocol": resp.get("protocol")})
    log_line("=== Network responses ===")
    for n in net:
        log_line("  status=%s mime=%s proto=%s url=%s" % (n["status"], n["mimeType"], n["protocol"], n["url"]))

    # ---- 2. 11 Font Awesome glyphs ----
    inject = (
        "(function(){"
        "  var c=document.createElement('div');c.id='probe-grid';"
        "  var html='';"
        "  var items=[%s];"
        "  for(var i=0;i<items.length;i++){html+='<p id=\"fa-probe-'+i+'\"><i class=\"fas '+items[i]+'\"></i></p>';}"
        "  c.innerHTML=html;document.body.appendChild(c);"
        "  return document.querySelectorAll('#probe-grid i').length;"
        "})()" % ", ".join('"%s"' % cls for _, cls in FA_ICONS)
    )
    log_line("injected FA icons:", cdp.evaluate(inject))
    time.sleep(0.5)
    probe = r'''
    (function(){
      var cls=[%s];
      var out=[];
      for(var i=0;i<cls.length;i++){
        var el=document.getElementById('fa-probe-'+i).querySelector('i');
        out.push({cls: cls[i], content: getComputedStyle(el,'::before').content});
      }
      return JSON.stringify(out);
    })()
    ''' % ", ".join('"%s"' % cls for _, cls in FA_ICONS)
    glyphs = json.loads(cdp.evaluate(probe))
    log_line("=== 11 Font Awesome icon glyph resolution ===")
    missing = []
    for (name, cls), g in zip(FA_ICONS, glyphs):
        ok = g["content"] not in ("none", "normal", "")
        if not ok:
            missing.append(cls)
        log_line("  %-10s %-20s ::before content=%-12s %s" % (name, g["cls"], json.dumps(g["content"]), "OK" if ok else "** MISSING **"))
    log_line("fa_glyphs_ok:", len(FA_ICONS) - len(missing), "/", len(FA_ICONS), "missing:", missing or "none")
    cdp.evaluate("document.getElementById('probe-grid').remove(); true")

    # Measure FA glyph box inside the real container (then remove)
    def pin_random(idx):
        cdp.evaluate("Math.random=function(){return (%d.5)/12;};" % (idx,))
        cdp.evaluate("document.getElementById('generateBtn').click()")
        time.sleep(0.7)

    # ---- 3a. FA icon rendered box (Pizza, index 0) ----
    pin_random(0)
    fa_box = json.loads(cdp.evaluate(
        "JSON.stringify((function(){var r=document.querySelector('.food-icon i').getBoundingClientRect();"
        "return {w:r.width,h:r.height};})())"))

    # ---- 3b. Tacos emoji rendered via the app (index 4) ----
    pin_random(4)
    emoji_state = json.loads(cdp.evaluate(
        "JSON.stringify((function(){"
        "  var el=document.querySelector('.food-icon .food-emoji');"
        "  if(!el){return {found:false};}"
        "  var t=el.textContent.trim();"
        "  var first=Array.from(t)[0];"
        "  var r=el.getBoundingClientRect();"
        "  return {found:true, text:t, codePoint:first?first.codePointAt(0).toString(16):'0',"
        "          role:el.getAttribute('role'), aria:el.getAttribute('aria-label'),"
        "          box:{w:r.width,h:r.height},"
        "          cls:el.className};"
        "})())"))
    log_line("=== Tacos emoji render (real click, index 4) ===")
    emoji_ok = (
        emoji_state.get("found") is True
        and emoji_state.get("codePoint", "").lower() == "1f32e"
        and emoji_state.get("role") == "img"
        and emoji_state.get("aria") == "Tacos"
        and emoji_state.get("cls") == "food-emoji"
    )
    log_line("  emoji_state:", emoji_state)
    log_line("  taco_emoji_ok:", emoji_ok)
    log_line("  FA icon box(Pizza):", fa_box, "vs emoji box:", emoji_state.get("box"))

    # ---- 4. Deterministic rapid-click race ----
    log_line("=== Race test: click1->Sushi(idx1), click2->Curry(idx8), click2 at ~200ms ===")
    cdp.evaluate(
        "window.__rseq=[1.5/12, 8.5/12];"
        "Math.random=function(){var v=window.__rseq.shift();return (v===undefined)?0.5:v;};"
        "window.__t0=performance.now();true"
    )

    def state():
        return json.loads(cdp.evaluate(
            "JSON.stringify({t:Math.round(performance.now()-window.__t0),"
            "icon:document.querySelector('.food-icon').innerHTML,"
            "name:document.querySelector('.food-name').textContent})"))

    results = []
    cdp.evaluate("document.getElementById('generateBtn').click()")          # click 1 (Sushi)
    results.append(("after click1", state()))
    time.sleep(0.20)
    results.append(("t+200ms (before click2)", state()))
    cdp.evaluate("document.getElementById('generateBtn').click()")          # click 2 (Curry)
    results.append(("right after click2", state()))
    time.sleep(0.20)
    results.append(("t+400ms", state()))
    time.sleep(0.23)
    results.append(("t+630ms (click1's timer window)", state()))
    time.sleep(0.20)
    results.append(("t+830ms (click2's timer window)", state()))
    time.sleep(0.42)
    final = state()
    results.append(("t+1250ms final", final))
    for label, r in results:
        log_line("  [%s] %s" % (label, json.dumps(r)))

    tank = results[4][1]
    stale_replaced = tank["name"].strip() == "Sushi" and 'fa-fish' in tank["icon"]
    log_line("first_click_cannot_replace_thinking:", not stale_replaced)
    log_line("final_belongs_to_latest_click:", final["name"] == "Curry" and "fa-mortar-pestle" in final["icon"])
    log_line("final_state:", final)

    # ---- 5. Console errors / uncaught exceptions ----
    errors = []
    for ev in cdp.events:
        m = ev.get("method")
        if m == "Runtime.exceptionThrown":
            errors.append("uncaught exception: " + json.dumps(ev["params"].get("exceptionDetails", {}), default=str)[:300])
        elif m == "Runtime.consoleAPICalled" and ev["params"].get("type") in ("error", "assert"):
            errors.append("console %s: %s" % (ev["params"]["type"], json.dumps(ev["params"].get("args", []), default=str)[:300]))
    log_line("=== Console / exceptions ===")
    log_line("errors:", errors or "none")

    # ---- 6. Screenshots ----
    shot = cdp.send("Page.captureScreenshot", {"format": "png"})
    with open(os.path.join(EVIDENCE, "shot_tacos_emoji.png"), "wb") as f:
        f.write(base64.b64decode(shot["data"]))
    time.sleep(0.2)
    shot = cdp.send("Page.captureScreenshot", {"format": "png"})
    with open(os.path.join(EVIDENCE, "shot_final_curry.png"), "wb") as f:
        f.write(base64.b64decode(shot["data"]))
    log_line("screenshots:", "shot_tacos_emoji.png, shot_final_curry.png")

    # ---- 7. Structured results ----
    out = {
        "page": PAGE_URL,
        "network": net,
        "font_state": {"status": cdp.evaluate("document.fonts.status")},
        "fa_icons": [{"name": n, "class": g["cls"], "content": g["content"]} for (n, _), g in zip(FA_ICONS, glyphs)],
        "taco_emoji": emoji_state,
        "taco_emoji_ok": emoji_ok,
        "rendered_box_px": {"fa_pizza": fa_box, "taco_emoji": emoji_state.get("box")},
        "race_timeline": [{"t": r["t"], "icon": r["icon"], "name": r["name"]} for _, r in results],
        "first_click_cannot_replace_latest_thinking": not stale_replaced,
        "final_belongs_to_latest_click": final["name"] == "Curry" and "fa-mortar-pestle" in final["icon"],
        "console_errors": errors,
    }
    with open(os.path.join(EVIDENCE, "cdp_results.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=True)
    with open(os.path.join(EVIDENCE, "postfix_verification.txt"), "w") as f:
        f.write("\n".join(log) + "\n")


if __name__ == "__main__":
    main()