#!/usr/bin/env python3
"""Stage 7 — ONE tier function, read from Runtime Matrices/Tier_breakpoints/tier_breakpoints.json (PROC-TIER-01, 2026-09-21).
Before this module the engine carried three definitions (report builder x2 with 1.04 onsets, JSON v1.3 with 1.01). Every tier word
in the chain now comes from tier_of(). Rules: reportable=False -> tier None (s108: no tier without a reportable gauge);
2026-09-27: the AT_CEILING word was removed by the author's ruling - the tier scale is the five bands and two lines, nothing else. The scorer's own bound A <= 1/H_min stays as an internal arithmetic guard, never printed."""
import os, json
HERE=os.path.dirname(os.path.abspath(__file__)); _P=os.path.join(HERE,"Runtime Matrices","Tier_breakpoints","tier_breakpoints.json"); _S=None
def scheme(path=None):
    global _S
    if _S is None or path:
        j=json.load(open(path or _P)); ts=j["tier_system_v1_2"]
        bands=[(t["tier_id"],t["a_score_range"]["min"],t["a_score_range"].get("max_exclusive",t["a_score_range"].get("max_inclusive"))) for t in ts["tiers"] if t.get("a_score_range")]
        _S={"bands":sorted(bands,key=lambda b:b[1]),"warburg":ts["warburg_line_value"],"breach":ts["breach_line_value"],"version":j["_meta"]["version"]}
    return _S
def tier_of(A, reportable=True, h_min=None, path=None):
    """Return (tier_id, note). tier_id None when not reportable or A is None."""
    if not reportable or A is None: return None, "no tier: gauge not reportable" if not reportable else "no tier: no A"
    s=scheme(path)
    for tid,lo,hi in s["bands"]:
        if lo<=A<hi: return tid, f"tier_breakpoints.json {s['version']}"
    return s["bands"][-1][0], f"tier_breakpoints.json {s['version']}"
