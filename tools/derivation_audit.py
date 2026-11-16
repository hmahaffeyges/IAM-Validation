#!/usr/bin/env python3
"""Derivation propagation audit (2026-11-16; author: 'investigate each and every single piece of the chain ... what has been updated with the
new derivations over the past few days'). Runs ON A GITHUB MACHINE. For each derived or measured quantity of the cell work, lists where it
appears in each public place (chain code, chain runtime files, Decoder, canon, cell simulator, cosmos simulator, book, website, other docs),
and inventories every file changed since 2026-11-12 and every file with a date in its name. Writes xout/audit_hits.csv,
xout/audit_recent.csv, xout/audit_dated.csv, xout/audit_summary.txt (small)."""
import os, re, csv, subprocess, collections
os.makedirs("xout", exist_ok=True)
subprocess.run("git fetch --unshallow -q || true", shell=True)
PLACES = [
    ("chain code", r"^Biological_Physics/MethylPhys/chain/.*\.py$"),
    ("chain runtime files", r"^Biological_Physics/MethylPhys/chain/(Runtime Matrices|FROZEN_INPUTS).*"),
    ("Decoder", r"^Biological_Physics/MethylPhys/manual/decoder_text\.md$"),
    ("canon", r"^CANON/iam_canon\.json$"),
    ("cell simulator", r"^Biological_Physics/MethylPhys/minicell/.*\.py$"),
    ("cosmos simulator", r"^Biological_Physics/MethylPhys/minicell/minicosmos.*\.py$"),
    ("book", r"^docs/book/.*\.(tex|bib)$"),
    ("website", r"^website/.*\.(py|html|md)$"),
    ("other docs", r"^(INSTRUMENTS|README|Biological_Physics/MethylPhys/[A-Z_]+)\.md$|^Biological_Physics/MethylPhys/METHYLPHYS_DEVELOPMENT_LOG\.md$"),
]
TERMS = [
    ("A1 floor", r"1\s*/\s*\(1\s*\+\s*e\^?\{?M|H_min_cell|2\.5e-08|1\.24\s*\\?times\s*10"),
    ("A2 two coins per restore", r"2[- ]ATP|two ATP|two coins|2M\s*\\?kB|41\.88"),
    ("A4 held error from M (0.04454)", r"eps0_candidate|0\.04454"),
    ("P4 phi from M (0.1464)", r"phi_candidate|0\.1464|phi_derived"),
    ("E_hold derived 3.066 kT", r"3\.066|3\.07\s*k"),
    ("A5 eraser drops out in mature blood", r"drops? out|from M alone.{0,60}(eraser|no measured)|median 0\.997|0\.997 of"),
    ("A5 eraser shares measured (0.8642/0.905)", r"0\.8642|0\.905\b"),
    ("A6 blind test PHY-084", r"PHY-084"),
    ("line constants measured (0.0403/0.0346)", r"0\.0402[0-9]*|0\.0403\b|0\.0346[0-9]*|0\.03462"),
    ("A7 slope within-sample pump", r"cooperative|within-sample slope|slope_pump"),
    ("A8 each type's own level", r"\bA8\b|each (cell )?type's own level"),
    ("Met-A H_ref measured 0.2593", r"H_ref_cell|0\.2593|0\.259\b"),
    ("M2 derived array level 0.9545", r"0\.9545|0\.2627"),
    ("England bound (E7)", r"England"),
    ("Sanchez-Mackenzie", r"Sanchez"),
    ("CP1 run lengths", r"loss_run_ratio|loss_runs"),
    ("TERR tile line", r"held_err_firm_tiles|_tile_line"),
    ("A8H 5hmC", r"hmC_share|5hmC"),
    ("T1 islands = meter (PHY-093)", r"PHY-093"),
    ("C3 clock sites (PHY-094)", r"PHY-094|epitoc2"),
    ("meter false marks (PHY-091)", r"PHY-091"),
]
files = subprocess.run(["git", "ls-files"], capture_output=True, text=True).stdout.splitlines()
hits = []
for place, rx in PLACES:
    R = re.compile(rx)
    for f in (x for x in files if R.search(x)):
        try: txt = open(f, encoding="utf-8", errors="replace").read()
        except Exception: continue
        lines = txt.splitlines()
        for term, trx in TERMS:
            T = re.compile(trx, re.I)
            ln = [i + 1 for i, l in enumerate(lines) if T.search(l)]
            if ln:
                first = lines[ln[0] - 1].strip()[:200]
                hits.append(dict(term=term, place=place, file=f, n_lines=len(ln), lines=" ".join(map(str, ln[:12])), first=first))
with open("xout/audit_hits.csv", "w", newline="") as fo:
    w = csv.DictWriter(fo, fieldnames=list(hits[0])); w.writeheader(); w.writerows(hits)
# matrix: term x place
M = collections.defaultdict(int)
for h in hits: M[(h["term"], h["place"])] += 1
out = ["TERM x PLACE (number of files mentioning it)", "term | " + " | ".join(p for p, _ in PLACES)]
for term, _ in TERMS: out.append(term + " | " + " | ".join(str(M[(term, p)]) for p, _ in PLACES))
# files changed since 2026-11-12, with commit counts and last date
log = subprocess.run(["git", "log", "--since=2026-11-12", "--name-only", "--format=@@%cI"], capture_output=True, text=True).stdout.splitlines()
last, cnt, d = {}, collections.Counter(), None
for l in log:
    if l.startswith("@@"): d = l[2:18]; continue
    if l.strip():
        cnt[l] += 1; last.setdefault(l, d)
with open("xout/audit_recent.csv", "w", newline="") as fo:
    w = csv.writer(fo); w.writerow(["file", "commits_since_11_12", "last_commit", "exists"])
    for f, n in sorted(cnt.items(), key=lambda kv: kv[0]): w.writerow([f, n, last[f], os.path.exists(f)])
dated = [f for f in files if re.search(r"20\d\d-\d\d-\d\d|20\d\d_\d\d_\d\d|\d{1,2}-\d{1,2}-\d\d", os.path.basename(f))]
with open("xout/audit_dated.csv", "w", newline="") as fo:
    w = csv.writer(fo); w.writerow(["file"]); [w.writerow([f]) for f in dated]
top = collections.Counter(os.path.dirname(f) for f in cnt).most_common(25)
out += ["", f"files tracked: {len(files)}; changed since 11-12: {len(cnt)} (still present {sum(os.path.exists(f) for f in cnt)}); dated names: {len(dated)}",
        "most-changed folders: " + "; ".join(f"{k} {v}" for k, v in top)]
open("xout/audit_summary.txt", "w").write("\n".join(out) + "\n")
for l in out[:30]: print("::notice::" + l[:900])
