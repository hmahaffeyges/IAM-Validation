#!/usr/bin/env python3
"""EARLIER_RESULTS.md: every results-register entry that is not on the front of the page (CANON/chain_front.json 'front'), newest first,
with its numbers computed from its record (results_to_tex.compute), exactly as the page prints them. Author 2026-10-17: the front page carries
only the commissioned chain; earlier results move to the repository, which the site renders as its own page.
Usage: python3 CANON/earlier_results.py [--check]"""
import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, "CANON")); import results_to_tex as RT
OUT = os.path.join(ROOT, "EARLIER_RESULTS.md")
def vals(e):
    try: v = dict(RT.compute(e))
    except Exception: v = {}
    if "n" in v: v["exp"] = round(0.05 * v["n"])
    if "k" in v and v.get("exp"): v["xchance"] = round(v["k"] / (0.05 * v["n"]))
    return v
def text():
    F = json.load(open(os.path.join(ROOT, "CANON/chain_front.json"), encoding="utf-8")); R = json.load(open(os.path.join(ROOT, "CANON/results_register.json"), encoding="utf-8"))
    E = sorted([e for e in R["entries"] if e["id"] not in F["front"] and e.get("status") != "withdrawn"], key=lambda e: e["date"], reverse=True)
    L = [f"# {F['earlier']}", "", "*Generated from CANON/results_register.json by CANON/earlier_results.py; do not edit by hand. The front page carries only the",
         "commissioned chain (CANON/chain_front.json). These results came before it: most were development tests, read before the chain that now",
         "runs every sample existed. Each keeps its record; none is withdrawn.*", ""]
    for e in E:
        w, v = e["win"], vals(e); f = lambda t: t.format(**v)
        st = {"commissioned": "commissioned", "published": "published", "reproduced": "reproduced"}.get(e.get("status"), "development")
        if e.get("sealed") is False: st += ", early result (data already read)"
        L += [f"## {f(w['head'])}", "", f"**{e['date']}** · {st} · {f(w['stat'])}", "", f(w["sub"]), "", f"Record: [{os.path.basename(e['record'])}]({e['record']})", ""]
    return "\n".join(L)
if __name__ == "__main__":
    t = text()
    if "--check" in sys.argv:
        ok = os.path.exists(OUT) and open(OUT, encoding="utf-8").read() == t
        print("earlier_results: " + ("ok" if ok else "REFUSED: EARLIER_RESULTS.md differs (run python3 CANON/earlier_results.py)")); sys.exit(0 if ok else 1)
    open(OUT, "w", encoding="utf-8").write(t); print("earlier_results: written")
