#!/usr/bin/env python3
"""Gauge words (author 2026-10-15: "you need to add this to the chain, and ONLY use the chain, and make sure the report SAYS this").
The gauge's reading, its misreading and the forbidden phrasings live in ONE file, the chain's
Biological_Physics/MethylPhys/chain/Runtime Matrices/Gauge/cell_gauge_marks_v1.json. This check refuses, on every presentation surface
(book, page builder and sources, cards, day logs, catalogue, board, READMEs, STATUS, SOP, OM, the chain's report), the two misreadings:
the floor described as the most a cell can hold, and cancer described as losing its identity, unless a negation stands just before.
It also requires the report to print the reading and the misreading from the marks file, and the book's gauge chapter to carry the passage.
Usage: gauge_words.py check | controls"""
import re, os, sys, json, glob
MARKS = "Biological_Physics/MethylPhys/chain/Runtime Matrices/Gauge/cell_gauge_marks_v1.json"
NEG = re.compile(r"(?i)\b(?:not|never|no longer|retired|wrong|isn't|is not|does not|doesn't)\b")
def surfaces(root="."):
    J = lambda *p: os.path.join(root, *p)
    F = (glob.glob(J("docs/book/**/*.tex"), recursive=True) + glob.glob(J("website/build/*.py")) + glob.glob(J("website/**/*.md"), recursive=True)
         + glob.glob(J("CANON/daylogs/*.json")) + glob.glob(J("CANON/cards/*")) + glob.glob(J("website/cards/*"))
         + [J(x) for x in ("CANON/catalogue.json", "CANON/instruments.json", "CANON/now_running.json", "README.md", "REFERENCE.md",
                           "Biological_Physics/MethylPhys/README.md", "Biological_Physics/MethylPhys/STATUS.md", "Biological_Physics/MethylPhys/INSTRUMENTS.md",
                           "Biological_Physics/MethylPhys/chain/MethylPhys_Interface/report_v3.py", "Biological_Physics/MethylPhys/sop/MethylPhys_CPG_SOP_v3.md",
                           "Biological_Physics/MethylPhys/sop/MethylPhys_CPG_SOP_v3_full.md", "Biological_Physics/MethylPhys/manual/OM_v3_neutrophil_chain.md")])
    return sorted({f for f in F if os.path.isfile(f)})
def scan_text(text, G):
    hits = []
    for name, r in G["forbidden"].items():
        if name.startswith("_"): continue
        for m in re.finditer(r["pattern"], text):
            lo, hi = max(0, m.start() - r["window"]), m.end() + r["window"]
            if not re.search(r["near"], text[lo:hi]): continue
            if NEG.search(text[max(0, m.start() - 40):m.start()]): continue
            if r.get("unless_before") and re.search(r["unless_before"], text[max(0, m.start() - 70):m.start()]): continue
            hits.append((name, re.sub(r"\s+", " ", text[max(0, m.start() - 80):m.end() + 40])))
    return hits
def check(root="."):
    G = json.load(open(os.path.join(root, MARKS), encoding="utf-8")); P = []
    for k in ("reading", "misreading", "forbidden", "book_reading_anchor"):
        if k not in G: P.append(f"[gauge words] {MARKS} has no '{k}'")
    if P: return P
    for f in surfaces(root):
        for name, ctx in scan_text(open(f, encoding="utf-8", errors="replace").read(), G):
            P.append(f"[gauge words] {os.path.relpath(f, root)}: {name}: ...{ctx}... (the floor is the least holding that keeps identity; cancer moves right and has not lost its identity: {MARKS})")
    rp = os.path.join(root, "Biological_Physics/MethylPhys/chain/MethylPhys_Interface/report_v3.py")
    if os.path.isfile(rp):
        s = open(rp, encoding="utf-8").read()
        if 'G["reading"]' not in s or "G['misreading']" not in s: P.append("[gauge words] report_v3.py must print the reading and the misreading from the marks file under every gauge")
    bk = os.path.join(root, G.get("book", ""))
    if os.path.isfile(bk) and G["book_reading_anchor"] not in open(bk, encoding="utf-8").read(): P.append(f"[gauge words] {G['book']} lacks the passage '{G['book_reading_anchor']}'")
    return P
def controls():
    G = json.load(open(MARKS, encoding="utf-8"))
    cases = [("C1 tonight's sentence 1", "**Left end:** the floor, derived. The most a cell can hold.", True),
             ("C2 tonight's sentence 2", "0 = the floor: the most a cell can hold its book, about 0.115 bits per site.", True),
             ("C3 tonight's sentence 3", "Its left edge is H_min: the most a cell can hold.", True),
             ("C4 cancer loses identity", "In the tumor the cell is losing its identity as the marks wash out.", True),
             ("C5 floor as most order", "The floor is the most order a site can hold.", True),
             ("C6 the reading passage", " ".join(G["reading"]), False),
             ("C7 the misreading passage", G["misreading"], False),
             ("C8 cancer has not lost identity", "Cancer has not lost its identity; it is over-writing.", False),
             ("C10 Astro-Genetics breach sentence", "Breach, the point past which a cell can no longer hold its identity, lies to the right of Normal and short of the full surface.", True),
             ("C11 identity lost left of the floor", "Going left past the floor, a cell can no longer hold its identity.", False),
             ("C9 an unrelated 'most'", "The ceiling is the most error a site can carry, a coin flip; the floor is the least holding that keeps identity.", False)]
    bad = []
    for name, text, refuse in cases:
        got = bool(scan_text(text, G)); ok = got == refuse
        print(f"  {name}: {'REFUSED' if got else 'ACCEPTED'} (want {'REFUSE' if refuse else 'ACCEPT'})"); bad += [] if ok else [name]
    print("gauge_words controls: " + ("ok (%d of %d)" % (len(cases), len(cases)) if not bad else "FAILED: " + ", ".join(bad))); return not bad
if __name__ == "__main__":
    if sys.argv[1:] == ["controls"]: sys.exit(0 if controls() else 1)
    P = check(); print("gauge_words: " + ("ok" if not P else "REFUSED (%d)" % len(P))); [print("  " + p) for p in P]; sys.exit(1 if P else 0)
