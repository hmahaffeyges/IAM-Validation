#!/usr/bin/env python3
"""CANON/physics_ties.py - writes each book chapter's physics tie (CANON/physics_ties.json) into a marked block right after the chapter's
\\chapter or first \\section heading (with its \\label). The block is replaced on every run, so the JSON is the one source.
Usage: python3 CANON/physics_ties.py [--check]   (--check writes nothing; exit 1 naming each chapter whose block differs)"""
import os, re, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); BOOK = os.path.join(ROOT, "docs", "book", "part6")
BEGIN, END = "% physics-tie: begin (CANON/physics_ties.py; edit CANON/physics_ties.json)", "% physics-tie: end"

def apply(tex, sentence):
    tex = re.sub(r"\n+" + re.escape(BEGIN) + r".*?" + re.escape(END) + r"\n", "\n", tex, flags=re.S)
    m = re.search(r"^\\(?:chapter|section)\*?\{[^\n]*\}(?:\\label\{[^}]+\})?[^\n]*\n", tex, re.M)
    if not m: raise SystemExit("physics_ties: no heading found")
    return tex[:m.end()] + "\n" + BEGIN + "\n" + sentence.strip() + "\n" + END + "\n" + tex[m.end():]

if __name__ == "__main__":
    T = {k: v for k, v in json.load(open(os.path.join(ROOT, "CANON", "physics_ties.json"), encoding="utf-8")).items() if not k.startswith("_")}
    check = "--check" in sys.argv; bad = []
    for name, s in T.items():
        p = os.path.join(BOOK, name + ".tex")
        if not os.path.exists(p): bad.append(f"{name}: no such chapter"); continue
        cur = open(p, encoding="utf-8").read(); new = apply(cur, s)
        if new != cur:
            if check: bad.append(f"{name}: physics tie differs from CANON/physics_ties.json")
            else: open(p, "w", encoding="utf-8").write(new)
    if bad: print("physics_ties: " + "; ".join(bad)); sys.exit(1)
    print(f"physics_ties: {'ok' if check else 'wrote'} ({len(T)} chapters)")
