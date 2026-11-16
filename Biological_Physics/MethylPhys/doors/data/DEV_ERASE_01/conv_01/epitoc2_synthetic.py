#!/usr/bin/env python3
"""C3 synthetic test (PHY-094; runs on GitHub explore.yml; written 2026-11-15 before any library is read for C3). The one command,
run_sample.py --pat ... --dev-count-only --dev-count-sites epitoc2, must recover a stray-mark rate planted only at the mitotic clock's sites
(the 163 epiTOC2 CpGs, Runtime Matrices/Clock_Sites/epitoc2_sites.csv, mapped to hg19 CpG indices by the atlas's sites_sequencing_key, with
25 CpGs either side in genome order) and keep it apart from a different rate planted elsewhere. Constructed .pat (hg19 CpG numbering): molecules of 10 calls held unmarked (every call T), with a lone C planted
independently at each call at rate r_isl on molecules over clock sites and r_out on molecules at least 50 CpGs from any clock site.
Planted pairs: (0.004, 0.015), (0.012, 0.012), (0.0, 0.015). Checks:
  (1) the clock sites' stray-mark rate (count_sites.gain_err) recovered within 15 % relative of r_isl (or below 3e-4 at 0);
  (2) the genome-wide rate (count.gain_err) lands between the two planted rates, weighted by opportunities, within 15 %;
  (3) the clock-sites count has opportunities (the site mapping worked).
Writes xout/epitoc2_synthetic_summary.txt; exit 1 if any check fails."""
import os, sys, json, gzip, random, bisect, subprocess
H = os.path.dirname(os.path.abspath(__file__)); MP = os.path.abspath(os.path.join(H, "..", "..", "..", ".."))
CH = os.path.join(MP, "chain"); IF = os.path.join(CH, "MethylPhys_Interface"); sys.path.insert(0, CH)
import pandas as pd, pyarrow.parquet as pq, stage_metaa_atlas as SA, stage_q2_selfref as Q2
OUT = os.path.abspath("xout"); W = os.path.join(OUT, "clk_syn"); os.makedirs(W, exist_ok=True)
K = pq.read_table(os.path.join(MP, "atlas", "v2.1", "sites_sequencing_key.parquet")).to_pandas()
isl = set(pd.read_csv(os.path.join(CH, "Runtime Matrices", "Clock_Sites", "epitoc2_sites.csv"), comment="#").probe)
B0 = set(K[K.probe.isin(isl)].loyfer_cpg_index.dropna().astype(int))
I = sorted({i + d for i in B0 for d in range(-25, 26) if i + d >= 1})
CH_START, acc = [], 0
for c, n in Q2.HG19_CPGS: CH_START.append((acc + 1, c)); acc += n
starts = [s for s, _ in CH_START]
def chrom_of(i): return CH_START[bisect.bisect_right(starts, i) - 1][1]
rng = random.Random(20261115)
anchors = rng.sample(I, min(4000, len(I)))
Iset = sorted(I)
def far(i):
    k = bisect.bisect_left(Iset, i)
    return all(abs(i - Iset[j]) >= 50 for j in (k - 1, k) if 0 <= j < len(Iset))
others = []
while len(others) < 4000:
    i = rng.randrange(1, acc - 20)
    if far(i) and far(i + 10) and chrom_of(i) == chrom_of(i + 10): others.append(i)
def make_pat(path, r_isl, r_out, per=10):
    rows = []
    for anchors_, r in ((anchors, r_isl), (others, r_out)):
        for a in anchors_:
            for m in range(per):
                s = a - rng.randrange(1, 9)
                if chrom_of(s) != chrom_of(s + 9): continue
                pat = "".join("C" if rng.random() < r else "T" for _ in range(10))
                rows.append((chrom_of(s), s, pat))
    rows.sort(key=lambda x: (x[0], x[1]))
    with gzip.open(path, "wt") as f:
        for c, s, p in rows: f.write(f"{c}\t{s}\t{p}\t1\n")
L, bad = ["C3 clock-sites synthetic (run_sample.py --pat --dev-count-only --dev-count-sites epitoc2), 2026-11-15",
          f"  clock sites: {len(isl)} probes, {len(B0)} with an hg19 CpG index, {len(I)} CpGs with their neighborhoods; {len(anchors)} clock anchors, {len(others)} far anchors"], 0
for r_isl, r_out in ((0.004, 0.015), (0.012, 0.012), (0.0, 0.015)):
    pat = os.path.join(W, f"syn_{r_isl}_{r_out}.pat.gz"); make_pat(pat, r_isl, r_out); b = os.path.join(W, f"syn_{r_isl}_{r_out}.json")
    p = subprocess.run([sys.executable, "run_sample.py", "--pat", pat, "--specimen", "cultured cells", "--seq-cell", "syn", "--id", "syn",
                        "--dev-count-only", "--dev-allow-partial-genome", "--dev-count-genome", "hg19", "--dev-count-sites", "epitoc2",
                        "--out", os.path.join(W, "r.html"), "--bundle", b], cwd=IF, capture_output=True, text=True)
    if p.returncode != 0 or not os.path.exists(b):
        L.append(f"  planted clock sites {r_isl}, elsewhere {r_out}: RUN FAILED (exit {p.returncode}) {p.stdout[-600:]} {p.stderr[-600:]}"); bad += 1; continue
    c = json.load(open(b))["iam_a_count_only"]; cs = c.get("count_sites") or {}; cg = c.get("count") or {}
    gi, go = cs.get("gain_err"), cs.get("gain_opp") or 0
    ok1 = gi is not None and ((r_isl == 0 and gi < 3e-4) or (r_isl > 0 and abs(gi / r_isl - 1) <= 0.15))
    ok3 = go > 0
    gg = cg.get("gain_err"); ok2 = gg is not None and min(r_isl, r_out) * 0.85 <= gg <= max(r_isl, r_out) * 1.15
    bad += (not ok1) + (not ok2) + (not ok3)
    L.append(f"  planted clock sites {r_isl}, elsewhere {r_out}: clock sites read {gi} on {go} opportunities ({'PASS' if ok1 and ok3 else 'FAIL'}); "
             f"genome-wide {gg} ({'PASS' if ok2 else 'FAIL'})")
L.append("SYNTHETIC MET" if not bad else f"SYNTHETIC NOT MET ({bad} checks failed)")
open(os.path.join(OUT, "epitoc2_synthetic_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L)); sys.exit(1 if bad else 0)
