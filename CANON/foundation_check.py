"""The foundation must stay sound. Two checks, run by status_check on every push.

1. Every entry of CANON/foundation.json still has its anchor in the file that derives it. If an anchor has moved, the
   foundation changed and this file did not: refuse until both are changed deliberately.
2. No retired phrasing appears in a file this push changed. A historical record may keep one by marking the line
   `[retired wording kept for the record]`.

Why: on 2026-10-13 the lever audit was written with the old per-copy wording, two days after the book derived that the held
error is a balance of restore and loss. The author caught it twice. A file to read before writing physics, plus a gate that
refuses the old wording, is the structural answer.

Usage: python3 CANON/foundation_check.py [--changed f1 f2 ...] [--print]
"""
import json, os, re, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = json.load(open(os.path.join(ROOT, "CANON", "foundation.json"), encoding="utf-8"))
SKIP = ("CANON/foundation.json", "CANON/foundation_check.py", "LESSONS.md", "METHYLPHYS_DEVELOPMENT_LOG.md", "CLASS_HISTORY.md")
def changed_files():
    base = subprocess.run(["git", "merge-base", "HEAD", "origin/main"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    if not base: return []
    return [l for l in subprocess.run(["git", "diff", "--name-only", base, "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.split("\n") if l.strip()]
def problems(changed=None):
    P = []
    for e in F["entries"]:
        p = os.path.join(ROOT, e["derived_in"])
        if not os.path.isfile(p): P.append(f"{e['id']}: its source {e['derived_in']} is gone"); continue
        if e["anchor"] not in open(p, encoding="utf-8", errors="replace").read():
            P.append(f"{e['id']}: the anchor is no longer in {e['derived_in']} - the foundation moved and CANON/foundation.json did not")
    for r in F.get("retired_foundations", []):
        if r.get("origin_record") and not os.path.isfile(os.path.join(ROOT, r["origin_record"])):
            P.append(f"retired {r['id']}: its origin record {r['origin_record']} is gone - a retired foundation must keep its history")
    for f in (changed if changed is not None else changed_files()):
        if any(s in f for s in SKIP) or not f.endswith((".md", ".tex", ".py", ".sh", ".json")): continue
        p = os.path.join(ROOT, f)
        if not os.path.isfile(p): continue
        for n, line in enumerate(open(p, encoding="utf-8", errors="replace").read().split("\n"), 1):
            if "[retired wording kept for the record]" in line: continue
            RN = F.get("retired_numbers") or {}
            if not any(x in f for x in RN.get("exempt_files", [])):
                for v in RN.get("values", []):
                    if v in line:
                        P.append(f"{f}:{n}: class-era floor value {v} - never derived, never measured; nothing current uses it "
                                 f"(CANON/foundation.json, retired_numbers)")
            for r in F["retired_phrasings"]:
                if re.search(re.escape(r["phrase"]), line, re.I):
                    P.append(f"{f}:{n}: retired wording \"{r['phrase']}\" - {r['why']}; say instead: {r['use_instead']}")
    return P
if __name__ == "__main__":
    if "--print" in sys.argv:
        print(F["_read_this_first"], "\n")
        for r in F.get("retired_foundations", []): print(f"[{r['status']}] {r['id']}: {r['statement']}\n   origin: {r['origin_record']}\n")
        for e in F["entries"]: print(f"[{e['status']}] {e['id']}: {e['statement']}\n   source: {e['derived_in']} ({e.get('commit','')})\n")
        sys.exit(0)
    ch = sys.argv[sys.argv.index("--changed") + 1:] if "--changed" in sys.argv else None
    P = problems(ch)
    print("foundation_check: " + ("ok (" + str(len(F["entries"])) + " entries anchored)" if not P else "REFUSED:\n  " + "\n  ".join(P)))
    sys.exit(1 if P else 0)
