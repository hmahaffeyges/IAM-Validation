"""Every instrument of the default chain on spent healthy arrays (2026-11-09; a working check, nothing scored; author: 'make sure it all
works first, check all the instruments including the C scores and territories'). SAMPLE p (0 to 3): the first spent healthy whole blood of
GSE318173 (EPIC), of GSE87571 and of GSE67393 (450K), and an atlas neutrophil donor (GSE118144, EPIC), fetched from GEO and read by the one
command with its defaults (the full development chain). For each instrument the bundle's section is checked: present, holding a reading, and
no error text. Output: instrument_check_p{p}.json and .txt in OUTDIR. Usage: instrument_check.py OUTDIR SAMPLE"""
import os, sys, json, subprocess, urllib.request, re, csv
H = os.path.dirname(os.path.abspath(__file__)); RS = os.path.abspath(os.path.join(H, "..", "..", "..", "chain", "MethylPhys_Interface", "run_sample.py"))
out = os.path.abspath(sys.argv[1]); P = int(sys.argv[2]); os.makedirs(out, exist_ok=True); wd = os.path.join(out, f"tmp{P}"); os.makedirs(wd, exist_ok=True)
SRC = [("realblood_GSE318173_samples.csv", "whole blood"), ("realblood_GSE87571_samples.csv", "whole blood"), ("realblood_GSE67393_samples.csv", "whole blood"), (None, "sorted neutrophils")]
lst, spec = SRC[P]
g = next(csv.DictReader(open(os.path.join(H, lst))))["gsm"] if lst else next(r["gsm"] for r in csv.DictReader(open(os.path.join(H, "gmp_source_donors.csv"))) if r["series"] == "GSE118144")
u = f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{g[:-3]}nnn/{g}/suppl/"
try:
    names = sorted(set(re.findall(rf"{g}_[^\"<>]*_(?:Grn|Red)\.idat\.gz", urllib.request.urlopen(u, timeout=120).read().decode())))
except Exception as e:
    open(os.path.join(out, f"instrument_check_p{P}.txt"), "w").write(f"{g}: the raw files could not be listed: {e}\n"); sys.exit(1)
fs = {}
for n in names:
    f = os.path.join(wd, n); urllib.request.urlretrieve(u + n, f); fs["grn" if "_Grn" in n else "red"] = f
bj = os.path.join(wd, "b.json")
p = subprocess.run([sys.executable, RS, "--grn", fs["grn"], "--red", fs["red"], "--specimen", spec, "--id", g, "--out", os.path.join(wd, "r.html"), "--bundle", bj],
                   cwd=os.path.dirname(RS), capture_output=True, text=True)
B = json.load(open(bj)) if os.path.exists(bj) else {}
import shutil
if os.path.exists(bj): shutil.copy(bj, os.path.join(out, f"instrument_check_p{P}_bundle.json"))   # the full bundle kept (2026-11-09)
open(os.path.join(out, f"instrument_check_p{P}_chain.log"), "w").write(p.stdout[-20000:] + "\n--- stderr ---\n" + p.stderr[-8000:])
SECT = ["intake", "cell_shares", "cells_found", "label_check", "met_a_each_cell", "met_a_whole", "cscore_each_cell", "cscore_sky", "territories", "iam_a",
        "iam_a_mix", "coupling_a", "info_sphere", "globe", "atlas_ruler", "reference", "full_chain", "development"]
BAD = re.compile(r"(?i)traceback|error|exception|not made|could not|failed|withheld")
rows = []
for k in SECT:
    v = B.get(k); txt = json.dumps(v, default=str) if v is not None else ""
    nums = len(re.findall(r":\s*-?\d+\.\d+", txt))
    bad = sorted(set(m.group(0).lower() for m in BAD.finditer(txt)))
    rows.append(dict(section=k, present=v is not None, numbers=nums, problem_words=bad, sample=(txt[:160] if bad else "")))
res = dict(gsm=g, specimen=spec, rc=p.returncode, stderr=p.stderr[-600:] if p.returncode else "", sections=rows,
           other_sections=sorted(set(B) - set(SECT)), red_flags=B.get("red_flags"))
json.dump(res, open(os.path.join(out, f"instrument_check_p{P}.json"), "w"), indent=1, default=str)
L = [f"{g} ({spec}): rc {p.returncode}"] + [f"  {r['section']:18} {'present' if r['present'] else 'MISSING':8} numbers {r['numbers']:5} {('PROBLEM WORDS ' + ', '.join(r['problem_words']) + ' | ' + r['sample']) if r['problem_words'] else ''}" for r in rows]
open(os.path.join(out, f"instrument_check_p{P}.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
