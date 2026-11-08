#!/usr/bin/env python3
"""KISS check (author 2026-10-16: "Where can you write KISS so you will remember it during our commissioning and official pre-reg testing??").
Every doors record to which a push ADDS a sealed rule (a seal line or a scored bar) must carry, in that record:
 a. '**The instrument's job:**' - one plain sentence, at least 6 words, of what the instrument must do (e.g. a healthy sample reads Normal on the
    gauge). Bars test that job and nothing tighter; the author sees this line when he seals.
 b. at most 3 scored bars (lines '- **S1:**', '- **B2:**', '- **Bar 3:**' ...); 'Reported, not scored' lines do not count.
 c. '**Sources:**' - every data source, said to be public, with what it provides; a source under controlled access (EGA, dbGaP, 'controlled',
    'on request') refuses the seal (lesson C19).
 d. (2026-11-06, author: 'How is it that your synthetics always seem to pass but we have a LOT of trouble with the real thing?') a push that
    ADDS a synthetic written before it runs must add, with it, '**Range of the data:**' - the real data's measured conditions (depth, coverage,
    copies, conversion, mix; never its readings) beside the range the synthetic tries, with numbers, the words 'real' and 'synthetic', and the
    synthetic covering the real range including its tails. A synthetic that meets its bars in conditions the data never has is not evidence.
e. (2026-11-07, author: 'healthy for comparisons MUST always come from the atlas's ... never tare our instruments with anything other than
    what we can be certain is healthy'; 'we need to make sure this doesnt happen again') '**Reference:**' - the healthy reference, which must be
    IAM-Atlas v2.1 or the IAM Sequencing Atlas and nothing else: a reference built from a study's 'healthy', 'normal' or control samples, a
    laboratory's or run's own samples (a tare), or a person's non-tumor tissue is refused.
 f. (2026-11-07: the same-laboratory tare met its median while 38 to 52 % of people read inside Normal) at least one scored bar asks it of each
    person or sample (each, every, at least N of, a share of the samples); bars on medians alone are refused.
 g. (2026-11-07 evening, author: 'there should be no reason for us to put a plus or minus 5 percent band on anything. We just know that 1 is
    in the middle'; replaces the 9-26 ruling 'Healthy is A=1 +/-5') no fixed healthy band: a bar may not set Normal by a chosen interval (0.95 to
    1.05, +/- 5 %, within 0.05 of 1). A rule whose bars say Normal or healthy must carry '**Normal:**' naming where that cell's healthy width is
    measured (the atlas's own donors read against the atlas, or the A-score database), and each reading is recorded as its distance from 1 in
    units of that width. A synthetic's tolerance against its own planted truth is not a healthy band and stays allowed.
 A sealed rule is also a test written before it runs whose bars are in a sentence ('Bars ...: 1. ... 2. ...'), not only bullet bars.
Applies to rules added from 2026-10-16 (d: from 2026-11-06; e, f, g and sentence bars: from 2026-11-07). CANON/kiss_check_controls.py must pass first. Usage: python3 CANON/kiss_check.py check [base]"""
import os, sys, re, subprocess
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEAL = re.compile(r"(?i)^(\*\*Sealed\b|\*\*Step \d+: the rule|\*\*Status: rule|## (Rule|Bars?)\b|\*\*Bars?:?\*\*|- \*\*(S|B|Bar ?)\d+[a-z]?:\*\*)")
BAR = re.compile(r"(?i)^- \*\*(S|B|Bar ?)\d+[a-z]?:\*\*")
JOB = re.compile(r"(?m)^\*\*The instrument's job:\*\*\s*(.+)$"); SRC = re.compile(r"(?m)^\*\*Sources:\*\*\s*(.+)$")
SYN = re.compile(r"(?i)\bsynthetic\b[^.]{0,160}\b(written before|before it runs|before (it|they) (reads?|runs?|opens?))|\b(written before|before it runs)[^.]{0,160}\bsynthetic\b")
RANGE = re.compile(r"(?ms)^\*\*Range of the data:\*\*\s*(.+?)(?:\n\s*\n|\Z)")   # the whole paragraph, which may wrap
REF = re.compile(r"(?m)^\*\*Reference:\*\*\s*(.+)$")
ATLAS = re.compile(r"(?i)IAM[- ]Atlas v2\.1|IAM Sequencing Atlas")
OTHER_REF = re.compile(r"(?i)\btar(e|ed|ing)\b|same[- ]run|same[- ]lab|own (healthy|normal)|non-tumou?r|adjacent|\bcontrols?\b|healthy (samples|donors|arrays|bloods) of (the|its|this)|the (study|series|laboratory)'s (own )?(healthy|normal)")
NOT_USED = re.compile(r"(?i)\b(never|not|no)\b[^.;]{0,40}\b(tare|same[- ]run|same[- ]lab|controls?|normal)")
SENT_BARS = re.compile(r"(?is)\bBars?\b[^\n]{0,120}?:\s*(1\..+?)(?:\n\s*\n|\Z)")
WRITTEN = re.compile(r"(?i)written (and committed )?before|before (any|it|they)[^.]{0,40}(read|run|open|count)")
PERSON = re.compile(r"(?i)\b(each|every)\b(?![^.;]{0,30}\bmedian)|\bat least \d+ of\b|\b\d+ (of|in) (the )?\d+\b|\b\d+ ?(%|percent) of\b|\bshare of (the )?(samples|people|arrays|bloods)")
FIXED = re.compile(r"(?i)0\.95\s*(to|-|–|and)\s*1\.05|\+/-\s*0?\.?0?5\s*(%|percent)?\b|±\s*0?\.?0?5\b|\b5 ?(%|percent) band|within 0\.05 of 1\b")
NORMAL = re.compile(r"(?m)^\*\*Normal:\*\*\s*(.+)$"); WIDTH = re.compile(r"(?i)atlas'?s? (own )?donors|A-score database|measured (healthy )?width")
SAYS_NORMAL = re.compile(r"(?i)\b(inside|within|reads?) Normal\b|\bhealthy (band|width|range)\b")
CTRL = re.compile(r"(?i)\b(EGA|EGAS\d+|dbGaP|controlled[- ]access|controlled|on request|data access committee)\b")
def findings(name, old, new):
    added = [l for l in new.split("\n") if l not in set(old.split("\n"))]
    F = []; addt = "\n".join(added)
    if SYN.search(addt):   # rule d
        rg = RANGE.search(addt)
        if not rg or not (re.search(r"(?i)\breal\b", rg.group(1)) and re.search(r"(?i)\bsynthetic\b", rg.group(1)) and re.search(r"\d", rg.group(1))):
            F.append(f"[d] {name}: a synthetic written before it runs needs '**Range of the data:**' with the real data's measured range beside the synthetic's, in numbers")
    sent = SENT_BARS.search(addt) if WRITTEN.search(addt) else None   # 2026-11-07: bars written in a sentence
    if not any(SEAL.search(l) for l in added) and not sent: return F
    j = JOB.search(new); s = SRC.search(new)
    if not j or len(j.group(1).split()) < 6: F.append(f"[a] {name}: a sealed rule needs '**The instrument's job:**' in one plain sentence (at least 6 words)")
    n = sum(bool(BAR.search(l)) for l in new.split("\n"))
    if n > 3: F.append(f"[b] {name}: {n} scored bars; KISS allows at most 3")
    r = REF.search(new)   # e
    if not r or not ATLAS.search(r.group(1)): F.append(f"[e] {name}: a sealed rule needs '**Reference:**' naming the healthy reference, IAM-Atlas v2.1 or the IAM Sequencing Atlas")
    elif OTHER_REF.search(r.group(1)) and not NOT_USED.search(r.group(1)): F.append(f"[e] {name}: the reference uses something other than an atlas ('{OTHER_REF.search(r.group(1)).group(0)}'); the atlases are the only healthy reference")
    bars = [l for l in added if BAR.search(l)] + ([x for x in re.split(r"(?:^|\s)\d\.\s", sent.group(1)) if x.strip()] if sent else [])   # f
    if bars and not any(PERSON.search(x) for x in bars): F.append(f"[f] {name}: no bar asks it of each person or sample (bars on medians alone are refused)")
    if bars and any(FIXED.search(x) for x in bars): F.append(f"[g] {name}: a bar sets Normal by a fixed band ('{next(FIXED.search(x).group(0) for x in bars if FIXED.search(x))}'); "
                                                     "1 is the middle and the width is measured per cell (atlas donors, A-score database)")
    nl = NORMAL.search(new)
    if bars and any(SAYS_NORMAL.search(x) for x in bars) and (not nl or not WIDTH.search(nl.group(1)) or FIXED.search(nl.group(1))):
        F.append(f"[g] {name}: bars say Normal; a '**Normal:**' line must name where the healthy width is measured (the atlas's own donors or the A-score database), with no fixed band")
    if not s or "public" not in s.group(1).lower(): F.append(f"[c] {name}: a sealed rule needs '**Sources:**' naming each source as public, with what it provides")
    elif CTRL.search(s.group(1)): F.append(f"[c] {name}: a source is under controlled access ({CTRL.search(s.group(1)).group(0)}); not public (lesson C19)")
    return F
def git(*a): return subprocess.run(["git", "-C", R] + list(a), capture_output=True, text=True).stdout
if __name__ == "__main__" and sys.argv[1:2] == ["check"]:
    base = sys.argv[2] if len(sys.argv) > 2 else "origin/main"; F = []
    for p in [x for x in git("diff", "--name-only", base, "HEAD").split("\n") if "/doors/" in x and x.endswith(".md") and os.path.isfile(os.path.join(R, x))]:
        F += findings(os.path.basename(p), git("show", f"{base}:{p}"), open(os.path.join(R, p), encoding="utf-8").read())
    print("kiss_check: " + ("ok" if not F else "REFUSED (%d)" % len(F))); [print("  " + f) for f in F]; sys.exit(1 if F else 0)
