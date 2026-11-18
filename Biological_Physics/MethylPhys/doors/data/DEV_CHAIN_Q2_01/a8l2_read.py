#!/usr/bin/env python3
"""A8L2: location (register A8, LOCATION LEAD; plan '### A8L2 plan', prediction written 2026-11-16 before the train). Runs ON A MACHINE (Google or
GitHub), one or more spent IAM Sequencing Atlas libraries (GSE186458) per call: python3 a8l2_read.py <outdir> GSM [GSM ...].
For each library: (1) the full chain, one button, defaults (run_sample.py --geo-pat GSM, specimen 'sorted cells', no cell label, info sphere on): the
library's own reading; (2) its molecule file is split by replication timing into early / mid / late thirds by the committed splitter
DEV_METHIONINE_01/lag_split.py (constitutive timing = mean of the 15 ENCODE UW Repli-seq wavelet tracks, CpG numbering checked against the chain's
own index ranges); (3) each third goes through the same one command with its defaults. If the chain refuses a third (for example for genome
coverage), the refusal is recorded as the chain's answer; no development flag is used.
Writes <outdir>/a8l2_<GSM>.json with every reading's IAM-A scalars and the chain's refusal code, if any. No reading is computed here: every number
comes from a chain bundle."""
import os, sys, json, glob, subprocess, time, shutil, re
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
IF = os.path.join(REPO, "Biological_Physics/MethylPhys/chain/MethylPhys_Interface")
SPLIT = os.path.join(REPO, "Biological_Physics/MethylPhys/doors/data/DEV_METHIONINE_01/lag_split.py")
OUT = os.path.abspath(sys.argv[1]); os.makedirs(OUT, exist_ok=True)
WORK = os.path.abspath(os.environ.get("IAM_WORK", os.path.join(OUT, "work"))); os.makedirs(WORK, exist_ok=True)
UCSC = "https://hgdownload.soe.ucsc.edu/goldenPath/hg19"
def sh(cmd, **k): return subprocess.run(cmd, shell=True, **k)
def setup():
    """the splitter's inputs: $IAM_WORK/iamrepo -> this repository, $IAM_WORK/rt/hg19.2bit and the 15 UW Repli-seq wavelet bigWigs"""
    sh(f"{sys.executable} -m pip install -q py2bit pyBigWig", check=True)
    link = os.path.join(WORK, "iamrepo")
    if not os.path.exists(link): os.symlink(REPO, link)
    rt = os.path.join(WORK, "rt"); os.makedirs(rt, exist_ok=True)
    if not os.path.exists(os.path.join(rt, "hg19.2bit")): sh(f"curl -fsSL --retry 5 -o {rt}/hg19.2bit {UCSC}/bigZips/hg19.2bit", check=True)
    lst = subprocess.run(f"curl -fsSL --retry 5 {UCSC}/encodeDCC/wgEncodeUwRepliSeq/", shell=True, capture_output=True, text=True).stdout
    names = sorted(set(re.findall(r'href="(wgEncodeUwRepliSeq\w+WaveSignalRep1\.bigWig)"', lst)))
    for n in names:
        if not os.path.exists(os.path.join(rt, n)): sh(f"curl -fsSL --retry 5 -o {rt}/{n} {UCSC}/encodeDCC/wgEncodeUwRepliSeq/{n}", check=True)
    assert len(names) == 15, f"expected 15 UW Repli-seq wavelet tracks, found {len(names)}"
    return names
def scal(d, depth=0):
    if not isinstance(d, dict): return None
    o = {}
    for k, v in d.items():
        if isinstance(v, (int, float, bool)) or v is None: o[k] = v
        elif isinstance(v, str) and len(v) < 160: o[k] = v
        elif isinstance(v, dict) and depth < 1:
            s = scal(v, depth + 1)
            if s: o[k] = s
    return o
def reading(args, tag, wdir):
    b = os.path.join(wdir, f"{tag}.json"); lg = os.path.join(wdir, f"{tag}.log")
    with open(lg, "w") as fh:
        rc = subprocess.run([sys.executable, "run_sample.py", *args, "--specimen", "sorted cells", "--id", tag, "--out", os.path.join(wdir, tag + ".html"),
                             "--bundle", b], cwd=IF, stdout=fh, stderr=subprocess.STDOUT).returncode
    B = json.load(open(b)) if os.path.exists(b) else {}
    tail = [l.strip()[:240] for l in open(lg, errors="replace").read().splitlines()[-6:]]
    ia = B.get("iam_a") or {}
    return {"rc": rc, "iam_a": scal(ia), "refusal": ia.get("refusal") or B.get("refusal"), "refusal_code": ia.get("refusal_code") or B.get("refusal_code"),
            "cell": ia.get("cell"), "log_tail": tail}
tracks = setup()
for g in [a for a in sys.argv[2:] if a.startswith("GSM")]:
    res_f = os.path.join(OUT, f"a8l2_{g}.json")
    if os.path.exists(res_f): print(f"{g} already read"); continue
    t0 = time.time(); w = os.path.join(WORK, g); os.makedirs(w, exist_ok=True); gd = os.path.join(w, "geo")
    R = {"gsm": g, "tracks": tracks, "readings": {}}
    R["readings"]["whole"] = reading(["--geo-pat", g, "--geo-dir", gd], g + "_whole", w)
    pats = sorted(glob.glob(os.path.join(gd, "**", "*.pat.gz"), recursive=True), key=os.path.getsize)
    if not pats: R["error"] = "no molecule file kept by the whole-library reading"; json.dump(R, open(res_f, "w"), indent=1); continue
    pat = os.path.join(w, f"{g}.pat.gz"); shutil.move(pats[-1], pat)
    sp = subprocess.run([sys.executable, SPLIT, pat], capture_output=True, text=True, env={**os.environ, "IAM_WORK": WORK})
    R["split_log"] = [l[:240] for l in (sp.stdout + sp.stderr).splitlines()[-8:]]
    for k in ("early", "mid", "late"):
        part = os.path.join(w, f"{g}_{k}.pat.gz")
        if not os.path.exists(part): R["readings"][k] = {"error": "splitter wrote no part"}; continue
        r = reading(["--pat", part], f"{g}_{k}", w); r["flag"] = "defaults (a refusal, e.g. for genome coverage, is recorded as the chain's own answer)"
        R["readings"][k] = r
    R["hours"] = round((time.time() - t0) / 3600, 3)
    json.dump(R, open(res_f, "w"), indent=1, default=str)
    shutil.rmtree(w, ignore_errors=True)
    print(f"{g} read in {R['hours']} h")
