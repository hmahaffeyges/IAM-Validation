"""Public roadmap (author 2026-10-12: show followers and researchers the road ahead and why support is needed). Writes ROADMAP.md from
CANON/next_jobs.json (the ranked job list the Master Inventory renders): each job in plain words, what it needs, its estimate, where it
stands and its record; then the gaps public data cannot close ("where a researcher could help"). --check (run by status_check) refuses a
ROADMAP.md that differs, a job without plain/needs/estimate/record, a record that does not exist, and the voice and hype words of the field
notes (CANON/now_running.py). Usage: python3 CANON/roadmap.py [--check]"""
import os, sys, json, importlib.util
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); OUT = os.path.join(ROOT, "ROADMAP.md")
spec = importlib.util.spec_from_file_location("nr", os.path.join(ROOT, "CANON", "now_running.py")); NR = importlib.util.module_from_spec(spec); spec.loader.exec_module(NR)
KIND = {"commissioning": "Making the instruments trustworthy", "atlas": "The map of healthy cells", "public result": "Cancer results",
        "physics": "Testing the physics", "research": "Filling gaps in the reference"}
def problems(J):
    P = []
    for j in J["jobs"]:
        for k in ("plain", "needs", "estimate", "stands", "record"):
            if not j.get(k): P.append(f"job {j['todo']}: no {k}")
        if j.get("record") and not os.path.exists(os.path.join(ROOT, j["record"])): P.append(f"job {j['todo']}: record not found")
    for h in J.get("help_wanted", []):
        if not os.path.exists(os.path.join(ROOT, h["record"])): P.append(f"help: record not found {h['record']}")
    txt = " ".join([j.get("plain", "") + " " + j.get("needs", "") + " " + j.get("stands", "") for j in J["jobs"]] + [h["ask"] + " " + h["why"] for h in J.get("help_wanted", [])])
    for rx, w in ((NR.VOICE, "voice"), (NR.HYPE, "hype word")):
        m = rx.search(txt)
        if m: P.append(f"{w} '{m.group(0)}'")
    return P
def render(J):
    L = ["# The road ahead", "", "*Every job still in front of us, in order, in plain words: what it is, what it needs, what it is expected to cost and where it",
         "stands. Generated from CANON/next_jobs.json by CANON/roadmap.py; do not edit by hand. What is running right now: the Now running board",
         "on the [main page](README.md); every experiment's story: the [field notes](FIELD_NOTES.md); every machine run and its cost: the",
         "[run ledger](Biological_Physics/MethylPhys/RUNS.md).*", ""]
    seen = []
    for j in J["jobs"]:
        if j["impact"] not in seen: seen.append(j["impact"])
    n = 0
    for k in seen:
        L += [f"## {KIND.get(k, k)}", ""]
        for j in [x for x in J["jobs"] if x["impact"] == k]:
            n += 1; L += [f"{n}. **{j['plain']}** Needs {j['needs']}; {j['estimate']}. Where it stands: {j['stands']}. [Record]({j['record']})"]
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
