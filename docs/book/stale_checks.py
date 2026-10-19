#!/usr/bin/env python3
"""stale_checks.py (2026-10-19): a check is stale when the number it holds as 'printed' no longer appears in its chapter, so it can pass
while the book says something else. Lists them; --check refuses any stale check not already in stale_checks_baseline.json (the list
can shrink, never grow). The baseline is the pre-existing set awaiting review; fix a chapter or its check and remove it from the baseline."""
import os, re, sys, json
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H); os.chdir(H)
import verify_book as V
def norm(t):   # what a reader sees: thousands separators dropped, 1.2\times10^{-7} read as 1.2e-7
    t = re.sub(r"(?<=\d)\{,\}(?=\d)", "", t); t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t); t = t.replace("\\,", " ").replace("~", " ")
    t = re.sub(r"\$?([\d.]+)\s*\\times\s*10\^\{?([-\u2212]?\d+)\}?\$?", lambda m: m.group(1) + "e" + m.group(2).replace("\u2212", "-"), t)
    t = re.sub(r"\$?10\^\{?([-]?\d+)\}?\$?", lambda m: "1e" + m.group(1), t); return t.replace("e+", "e").replace("e0", "e").replace("e-0", "e-")
def pnorm(p): return re.sub(r"(?<=\d)\{,\}(?=\d)", "", p).replace("e+", "e").replace("e0", "e").replace("e-0", "e-")
REVIEWED = {   # read 2026-10-19: the chapter prints the number in another form; the check is right
 "app:notation:L120": "the chapter prints 48,528 noise sites", "app:glossary:L484": "the glossary prints 48,528 sites",
 "app:glossary:L70:814": "the glossary prints about 814,000 CpGs (the check holds thousands)",
 "ch:baryon_chain:L166:957": "the table prints 14,957 rows; the check holds its last three digits (a split at the thousands separator)"}
_RAW = {}
def present(pr, normed, f):
    """the check's number is in its chapter: as printed after normalizing, or as its bare digits with TeX separators and wrappers ignored
    (2026-10-19: 72 of 74 'stale' checks were the chapter printing 0.8\\,\\% for 0.8\\%, 111\\,612 for 111612, 6.113\\times10^{-10} with spacing)."""
    if re.search(r"(?<![\d.])" + re.escape(pr) + r"(?!\d)", normed): return True
    if f not in _RAW: _RAW[f] = re.sub(r"(?<=\d)(\{,\}|\\,|\\ |~|,(?=\d{3}\b))(?=\d)", "", open(f + ".tex", encoding="utf-8").read())
    cores = [c for c in re.findall(r"\d*\.\d+|\d+", re.sub(r"(?<=\d)(\{,\}|\\,|,)(?=\d)", "", pr)) if c != "10"]
    return bool(cores) and all(re.search(r"(?<![\d.])0?" + re.escape(c) + r"(?!\d)", _RAW[f]) for c in cores[:1])
def stale():
    cache, out = {}, []
    for c in V.CHECKS:
        m = c.meta; f = m.get("file"); pr = norm(pnorm(str(m.get("printed", "")).strip())).replace("$", "").strip()
        if not f or not pr or not re.search(r"\d", pr) or not os.path.exists(f + ".tex"): continue
        if f not in cache: cache[f] = norm(open(f + ".tex", encoding="utf-8").read())
        if c.label not in REVIEWED and not present(pr, cache[f], f): out.append(dict(label=c.label, file=f, printed=pr))
    return out
if __name__ == "__main__":
    S = stale(); B = os.path.join(H, "stale_checks_baseline.json")
    if "--baseline" in sys.argv: json.dump(sorted(S, key=lambda x: x["label"]), open(B, "w"), indent=0); print("stale_checks: baseline written,", len(S)); sys.exit(0)
    base = {x["label"] for x in json.load(open(B))} if os.path.exists(B) else set()
    new = [x for x in S if x["label"] not in base]
    if "--check" in sys.argv:
        print("stale_checks: " + ("ok (%d awaiting review in the baseline)" % len(S) if not new else "REFUSED: %d check(s) hold a number their chapter no longer prints: %s" % (len(new), "; ".join(f"{x['label']} ({x['printed']})" for x in new[:12]))))
        sys.exit(1 if new else 0)
    for x in S: print(x["label"], "|", x["file"], "|", x["printed"])
