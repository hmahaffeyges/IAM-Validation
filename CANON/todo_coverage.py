"""Every item of the author's 2026-10-12 to-do list is either a numbered job, a long-range track, an idea, housekeeping, or
closed - and this script refuses if any of them is none of those (run by status_check). The universe of ids is the list itself;
closed items are named in CANON/todo_closed.json with the date they closed, so nothing can quietly disappear.
Usage: python3 CANON/todo_coverage.py [--check]"""
import json, os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNIVERSE = ([f"N{i}" for i in range(1, 6)] + [f"A{i}" for i in range(1, 12)] + [f"M{i}" for i in range(1, 7)] +
            [f"AT{i}" for i in range(1, 14)] + [f"R{i}" for i in range(1, 8)] + [f"B{i}" for i in range(1, 15)] +
            [f"G{i}" for i in range(1, 5)] + [f"C{i}" for i in range(1, 12)] + [f"O{i}" for i in range(1, 6)])
def ids(text): return re.findall(r"\b(?:N|A|M|AT|R|B|G|C|O)\d{1,2}\b", text or "")
def coverage():
    J = json.load(open(os.path.join(ROOT, "CANON", "next_jobs.json"), encoding="utf-8"))
    closed = json.load(open(os.path.join(ROOT, "CANON", "todo_closed.json"), encoding="utf-8"))
    where = {}
    for n, j in enumerate(J["jobs"], 1):
        for i in ids(j.get("todo")): where.setdefault(i, f"job {n}")
    for x in J.get("later", []) + J.get("ideas", []) + J.get("housekeeping", []):
        for i in ids(x.get("todo", "")): where.setdefault(i, "later/idea/housekeeping")
    for i, when in closed.get("closed", {}).items(): where.setdefault(i, f"closed {when}")
    missing = [i for i in UNIVERSE if i not in where]
    unknown = [i for i in where if i not in UNIVERSE]
    return where, missing, unknown
if __name__ == "__main__":
    w, missing, unknown = coverage()
    P = []
    if missing: P.append("to-do items in no job, track, idea or closed list: " + ", ".join(missing))
    if unknown: P.append("ids not in the 2026-10-12 list: " + ", ".join(unknown))
    print("todo_coverage: " + ("ok (" + str(len(w)) + " items placed)" if not P else "REFUSED: " + "; ".join(P)))
    sys.exit(1 if P else 0)
