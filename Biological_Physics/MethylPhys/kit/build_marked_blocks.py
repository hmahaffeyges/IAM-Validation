#!/usr/bin/env python3
"""build_marked_blocks.py - the chain-as-it-runs block inside doors/RUNBOOK.md and README.md, between
<!-- GENERATED:chain start --> and <!-- GENERATED:chain end -->. Everything outside the markers is the author's prose and is
left alone; everything inside is rewritten from chain_sequence.json and the generated-documents register on every build."""
import os, json, glob, sys, time
K = os.path.dirname(os.path.abspath(__file__)); MP = os.path.dirname(K)
sys.path.insert(0, os.path.join(MP, "chain")); import build_all as BA
seq = json.load(open(os.path.join(MP, "chain", "chain_sequence.json")))
S, E = "<!-- GENERATED:chain start -->", "<!-- GENERATED:chain end -->"
def block(up):
    rows = ["| # | step | file | what it implements |", "|---|---|---|---|"]
    for i, s in enumerate(seq["live_path"], 1):
        rows.append("| %d | `%s` | `%s` | %s |" % (i, s["step"], s["where"], s.get("implements", "").replace("|", "/")[:140]))
    gen = ", ".join("`%s%s`" % (up, a.split(" (")[0]) for a, _ in BA.GENERATED)
    return (S + "\n**The chain as the code runs it** (generated %s by `%skit/build_marked_blocks.py` from `%schain/chain_sequence.json`; %d steps; "
            "%d chain-named files no path calls). One command regenerates every document that reports the chain: `python3 %schain/build_all.py` - "
            "run by `guarded_push.sh` on every push.\n\n%s\n\n**Generated documents** (never edit; rerun build_all): %s\n%s"
            % (time.strftime("%Y-%m-%d"), up, up, len(seq["live_path"]), len(seq.get("not_in_live_path", [])), up, "\n".join(rows), gen, E))
for rel, anchor in (("doors/RUNBOOK.md", "## The order of steps"), ("README.md", "## One command")):
    p = os.path.join(MP, rel); t = open(p, encoding="utf-8").read(); b = block("../" if "/" in rel else "")
    if S in t and E in t: t = t[:t.index(S)] + b + t[t.index(E) + len(E):]
    elif anchor in t: i = t.index(anchor); t = t[:i] + b + "\n\n" + t[i:]
    else: t = t.rstrip() + "\n\n" + b + "\n"
    open(p, "w", encoding="utf-8").write(t); print("%s: block written (%d rows)" % (rel, len(seq["live_path"])))
