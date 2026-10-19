"""RETIRED 2026-10-13: ROADMAP.md is frozen; the road ahead is CANON/manifest.json (MANIFEST.md, TODO.md). This script no longer writes."""
import sys
if __name__ == "__main__":
    print("roadmap: retired - see CANON/manifest.py"); sys.exit(0)
"""Public roadmap (author 2026-10-12: show followers and researchers the road ahead and why support is needed). Writes ROADMAP.md from
CANON/next_jobs.json (the ranked job list the Master Inventory renders): each job in plain words, what it needs, its estimate, where it
stands and its record; then the gaps public data cannot close ("where a researcher could help"). --check (run by status_check) refuses a
ROADMAP.md that differs, a job without plain/needs/estimate/record, a record that does not exist, and the voice and hype words of the field
notes (CANON/now_running.py). Usage: python3 CANON/roadmap.py [--check]"""
import os, sys, json, importlib.util
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); OUT = os.path.join(ROOT, "archive", "retired_root_2026-10-19", "ROADMAP.md")   # retired 2026-10-13; out of the root 2026-10-19
spec = importlib.util.spec_from_file_location("nr", os.path.join(ROOT, "CANON", "now_running.py")); NR = importlib.util.module_from_spec(spec); spec.loader.exec_module(NR)
KIND = {"replication": "Replicate it yourself", "commissioning": "Making the instruments trustworthy", "atlas": "The map of healthy cells", "public result": "Cancer results",
        "physics": "Testing the physics", "research": "Filling gaps in the reference"}
def problems(J):
    P = []
    for j in J["jobs"]:
        for k in ("plain", "needs", "estimate", "stands", "record"):
            if not j.get(k): P.append(f"job {j['todo']}: no {k}")
        if j.get("record") and not os.path.exists(os.path.join(ROOT, j["record"])): P.append(f"job {j['todo']}: record not found")
    for x, (n_, gb) in later(J):
        if n_ == 0: P.append(f"later: no data-register entry matches '{x['match']}'")
    for h in J.get("help_wanted", []):
        if not os.path.exists(os.path.join(ROOT, h["record"])): P.append(f"help: record not found {h['record']}")
    txt = " ".join([j.get("plain", "") + " " + j.get("needs", "") + " " + j.get("stands", "") for j in J["jobs"]] + [h["ask"] + " " + h["why"] for h in J.get("help_wanted", [])] + [x["plain"] for x in J.get("later", [])] + [x["plain"] for x in J.get("ideas", [])])
    for rx, w in ((NR.VOICE, "voice"), (NR.HYPE, "hype word")):
        m = rx.search(txt)
        if m: P.append(f"{w} '{m.group(0)}'")
    return P
def later(J):
    import re, pandas as pd
    D = pd.read_csv(os.path.join(ROOT, "Biological_Physics/MethylPhys/doors/DATA_REGISTER.csv"), keep_default_na=False); D["GB"] = pd.to_numeric(D.GB, errors="coerce").fillna(0)
    out = []
    for x in J.get("later", []):
        m = D[D.planned_tests.str.contains(x["match"], case=False, regex=True)]; out.append((x, (len(m), float(m.GB.sum()))))
    return out
def render(J):
    L = ["# The road ahead", "", "*Every job still in front of us, in order, in plain words: what it is, what it needs, what it is expected to cost and where it",
         "stands. Generated from CANON/next_jobs.json by CANON/roadmap.py; do not edit by hand. What is running right now: the Now running board",
         "on the [main page](README.md); every experiment's story: the [field notes](Biological_Physics/MethylPhys/FIELD_NOTES.md); every machine run and its cost: the",
         "[run ledger](Biological_Physics/MethylPhys/RUNS.md).*", ""]
    # One list, in the author's priority order (CANON/next_jobs.json is ordered; PRIORITY_2026-10-13.md says why)
    L += ["## The order of work", "", "*In priority order: each job is what most strengthens the two instruments and the law at that point.*", ""]
    for n, j in enumerate(J["jobs"], 1):
        L += [f"{n}. **{j['plain']}** Needs {j['needs']}; {j['estimate']}. Where it stands: {j['stands']}. [Record]({j['record']})"]
    L += [""]
    L += ["## Further down the road", "", "*Planned, not yet designed. The data for each is already on our storage (sizes from the data register).*", ""]
    for x, (n_, gb) in later(J): L += [f"- **{x['plain']}** {n_} data sets, {gb:,.0f} GB in hand."]
    L += [""]
    if J.get("ideas"):
        L += ["## Ideas we want to test", "", "*Not designed yet. Each goes through a written design, a synthetic test and, where it reads data, a null test before anything it shows is claimed.*", ""]
        L += [f"- **{x['plain'].split('. ')[0]}.** {'. '.join(x['plain'].split('. ')[1:])} Needs {x['needs']}; {x['stands']}." for x in J["ideas"]]
        L += [""]
    L += ["## Where a researcher could help", "", "*These are the gaps public data cannot close. If you hold data like this, or can point to it, open an issue or",
          "a discussion on the repository. You do not need to introduce yourself to follow along.*", ""]
    L += [f"- **{h['ask']}** {h['why']} [Record]({h['record']})" for h in J.get("help_wanted", [])]
    return "\n".join(L) + "\n"
if __name__ == "__main__":
    J = json.load(open(os.path.join(ROOT, "CANON", "next_jobs.json"), encoding="utf-8")); P = problems(J)
    if "--check" in sys.argv:
        if not os.path.exists(OUT) or open(OUT, encoding="utf-8").read() != render(J): P.append("ROADMAP.md differs from CANON/next_jobs.json (run python3 CANON/roadmap.py)")
        print("roadmap: " + ("ok" if not P else "REFUSED: " + "; ".join(P))); sys.exit(1 if P else 0)
    if P: raise SystemExit("roadmap: " + "; ".join(P))
    open(OUT, "w", encoding="utf-8").write(render(J)); print("roadmap: ROADMAP.md written")
