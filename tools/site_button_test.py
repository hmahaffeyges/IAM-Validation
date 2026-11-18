#!/usr/bin/env python3
"""Live-site button test (author 2026-11-17: 'someone will click those and they wont work and that looks really bad'). Runs ON A GITHUB MACHINE.
Downloads the PUBLISHED site (not the repo): checks/checks.json, the published verify_book.py and every data file it lists, and every chapter page.
(1) Runs every check the published buttons would run, from the published files only (what a reader's browser runs).
(2) For every 'Run this check' button on every chapter page, confirms its label exists in checks.json and that the button sits within the
    paragraph block that prints its value. Reports counts and the first failures as notices (at most 6 notice lines)."""
import os, sys, json, re, subprocess, urllib.request, urllib.parse, pathlib, collections
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "numpy", "scipy", "sympy", "pandas", "pyarrow", "beautifulsoup4", "mpmath"], check=True)
from bs4 import BeautifulSoup
SITE = "https://hmahaffeyges.github.io/IAM-Validation/"
W = pathlib.Path("site_copy"); W.mkdir(exist_ok=True)
def get(rel, binary=False):
    u = SITE + urllib.parse.quote(rel)
    with urllib.request.urlopen(u, timeout=120) as r: b = r.read()
    p = W / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b); return b if binary else b.decode("utf-8", "replace")
cj = json.loads(get("checks/checks.json")); checks = cj["checks"]; files = cj["data_files"]
get("checks/verify_book.py")
miss = []
for f in files:
    try:
        b = get("checks/data/" + f, binary=True); q = W / "checks" / f; q.parent.mkdir(parents=True, exist_ok=True); q.write_bytes(b)
    except Exception as e: miss.append((f, str(e)[:60]))
print(f"published: {len(checks)} checks, {len(files)} data files, {len(miss)} data files missing: {miss[:8]}")
# (1) run every check from the published copy
os.chdir(W / "checks")
r = subprocess.run([sys.executable, "verify_book.py", "--fails"], capture_output=True, text=True, timeout=7000)
fails = [l for l in r.stdout.splitlines() if l.startswith(("FAIL", "ERROR"))]
m = re.search(r"SUMMARY\s+PASS (\d+)\s+FAIL (\d+)", r.stdout + r.stderr)
os.chdir("../..")
summ = f"PASS {m.group(1)} FAIL {m.group(2)}" if m else "no summary: " + (r.stdout + r.stderr)[-300:]
# (2) buttons on the pages; printed values from the published verifier
sys.path.insert(0, str(W / "checks"))
src = (W / "checks" / "verify_book.py").read_text()
PRINTED = dict(re.findall(r"@check\(label='([^']+)'[^@]*?printed='([^']*)'", src))
idx = get("book/index.html") if True else ""
pages = sorted(set(re.findall(r'href="([^"#]+\.html)"', idx)))
btn, nolabel, far, perpage = 0, [], [], collections.Counter()
for pg in pages:
    rel = "book/" + pg if not pg.startswith(("book/", "../", "http")) else pg.replace("../", "")
    try: h = get(rel)
    except Exception: continue
    soup = BeautifulSoup(h, "html.parser")
    for b in soup.select("button.iam-mark[data-labels]"):
        labs = json.loads(b.get("data-labels")); btn += len(labs); perpage[rel] += len(labs)
        blk = b.find_parent("div", class_="iam-checks-after")
        blk = blk.find_previous_sibling() if blk is not None else b.find_parent(["p", "li", "td", "div", "table"])
        txt = blk.get_text(" ") if blk is not None else ""
        for lab in labs:
            if lab not in checks: nolabel.append((rel, lab)); continue
            pr = PRINTED.get(lab, "")
            if pr and pr not in txt: far.append((rel, lab, pr))
print(f"pages {len(pages)}, buttons {btn}, unknown labels {len(nolabel)}, value not in the button's block {len(far)}")
json.dump(dict(summary=summ, fails=fails, missing_files=miss, buttons=btn, unknown=nolabel, far=far), open("site_button_test.json", "w"), indent=1)
os.makedirs("xout", exist_ok=True); os.replace("site_button_test.json", "xout/site_button_test.json")
print(f"::notice::SITE CHECKS run from the published files: {summ}; first fails: " + " | ".join(l[:90] for l in fails[:4]))
print(f"::notice::SITE BUTTONS: {btn} buttons on {len(pages)} pages; {len(nolabel)} with no check behind them; {len(far)} whose printed value is not in their paragraph")
print("::notice::no-check examples: " + json.dumps(nolabel[:6])[:700])
print("::notice::off-paragraph examples: " + json.dumps(far[:8])[:900])
print(f"::notice::missing data files on the site: {len(miss)} " + json.dumps(miss[:5])[:400])
