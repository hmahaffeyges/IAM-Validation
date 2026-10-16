#!/usr/bin/env python3
"""KISS check (author 2026-10-16: "Where can you write KISS so you will remember it during our commissioning and official pre-reg testing??").
Every doors record to which a push ADDS a sealed rule (a seal line or a scored bar) must carry, in that record:
 a. '**The instrument's job:**' - one plain sentence, at least 6 words, of what the instrument must do (e.g. a healthy sample reads Normal on the
    gauge). Bars test that job and nothing tighter; the author sees this line when he seals.
 b. at most 3 scored bars (lines '- **S1:**', '- **B2:**', '- **Bar 3:**' ...); 'Reported, not scored' lines do not count.
 c. '**Sources:**' - every data source, said to be public, with what it provides; a source under controlled access (EGA, dbGaP, 'controlled',
    'on request') refuses the seal (lesson C19).
Applies to rules added from 2026-10-16. CANON/kiss_check_controls.py must pass first. Usage: python3 CANON/kiss_check.py check [base]"""
import os, sys, re, subprocess
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEAL = re.compile(r"(?i)^(\*\*Sealed\b|\*\*Step \d+: the rule|\*\*Status: rule|## (Rule|Bars?)\b|\*\*Bars?:?\*\*|- \*\*(S|B|Bar ?)\d+[a-z]?:\*\*)")
BAR = re.compile(r"(?i)^- \*\*(S|B|Bar ?)\d+[a-z]?:\*\*")
JOB = re.compile(r"(?m)^\*\*The instrument's job:\*\*\s*(.+)$"); SRC = re.compile(r"(?m)^\*\*Sources:\*\*\s*(.+)$")
CTRL = re.compile(r"(?i)\b(EGA|EGAS\d+|dbGaP|controlled[- ]access|controlled|on request|data access committee)\b")
def findings(name, old, new):
    added = [l for l in new.split("\n") if l not in set(old.split("\n"))]
    if not any(SEAL.search(l) for l in added): return []
    F = []; j = JOB.search(new); s = SRC.search(new)
    if not j or len(j.group(1).split()) < 6: F.append(f"[a] {name}: a sealed rule needs '**The instrument's job:**' in one plain sentence (at least 6 words)")
    n = sum(bool(BAR.search(l)) for l in new.split("\n"))
    if n > 3: F.append(f"[b] {name}: {n} scored bars; KISS allows at most 3")
    if not s or "public" not in s.group(1).lower(): F.append(f"[c] {name}: a sealed rule needs '**Sources:**' naming each source as public, with what it provides")
    elif CTRL.search(s.group(1)): F.append(f"[c] {name}: a source is under controlled access ({CTRL.search(s.group(1)).group(0)}); not public (lesson C19)")
    return F
def git(*a): return subprocess.run(["git", "-C", R] + list(a), capture_output=True, text=True).stdout
if __name__ == "__main__" and sys.argv[1:2] == ["check"]:
    base = sys.argv[2] if len(sys.argv) > 2 else "origin/main"; F = []
    for p in [x for x in git("diff", "--name-only", base, "HEAD").split("\n") if "/doors/" in x and x.endswith(".md") and os.path.isfile(os.path.join(R, x))]:
        F += findings(os.path.basename(p), git("show", f"{base}:{p}"), open(os.path.join(R, p), encoding="utf-8").read())
    print("kiss_check: " + ("ok" if not F else "REFUSED (%d)" % len(F))); [print("  " + f) for f in F]; sys.exit(1 if F else 0)
