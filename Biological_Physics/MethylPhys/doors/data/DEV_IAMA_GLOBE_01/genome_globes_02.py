"""DEV-IAMA-GLOBE-01, step 2 (written 2026-10-22; development, pictures, nothing scored). The full-genome IAM-A globe on real healthy molecules.
Two healthy CD8 T-cell samples of the committed self-reference line (GSM5652282, GSM5652283; public .pat files on GEO; spent: they built
the line, so these pictures show the method and test nothing), read by chain Stage Q2 (count_pat, for each sample's loose share) and
chain/stage_q2_globe.py (count_pat_regions, the same count kept per region of 100 CpGs), unchanged. Three globes, one Explorer page:
  1. one sample against itself: each region's held error against what the sample's own loose share predicts (no other person);
  2. the same sample with held-mark loss planted (CONSTRUCTED, not a person): on held molecules in one stretch of chr2 (CpG 2,500,001 to
     2,800,000), every interior methylated call whose neighbors are methylated is turned unmethylated with probability 0.04, which about
     doubles the held error there; the methylation calls (even lines) are untouched, so the loose share and the prediction do not move;
  3. two people: the first sample against the second, region by region, only where they differ left lit.
Usage: python3 genome_globes_02.py OUTDIR   (writes OUTDIR/explorer/iama-genome.json and its pictures, genome_globes_02_summary.txt, a .tar.gz)"""
import os, sys, re, time, gzip, json, tarfile, urllib.request, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); CH = os.path.join(H, "..", "..", "..", "chain"); sys.path[:0] = [CH, os.path.join(CH, "MethylPhys_Interface")]
import stage_q2_selfref as Q2, stage_q2_globe as S, globe_viewer as V
out = sys.argv[1]; E = os.path.join(out, "explorer"); os.makedirs(E, exist_ok=True); L = []
def fetch(g):
    base = f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{g[:7]}nnn/{g}/suppl/"
    for a in range(5):
        try:
            name = sorted(set(re.findall(rf"{g}_[^\"<>]*\.pat\.gz", urllib.request.urlopen(base, timeout=120).read().decode())))[0]
            f = os.path.join(out, name); urllib.request.urlretrieve(base + name, f); return f
        except Exception as e: print(g, "retry", type(e).__name__, flush=True); time.sleep(60 * (a + 1))
    raise RuntimeError(g + ": download failed")
PL = (2284470 + 2_500_001, 2284470 + 2_800_000)   # chr2's CpG 2,500,001 to 2,800,000 in the genome-wide index
def planted(lines, seed=1):
    g = np.random.default_rng(seed)
    for k, line in enumerate(lines):
        p = line.split("\t")
        if k % 2 == 1 and len(p) >= 4 and PL[0] <= int(p[1]) <= PL[1]:
            pat = list(p[2]); cl = [q for q, x in enumerate(pat) if x != "."]
            if len(cl) >= Q2.QUAL_MIN_CALLS:
                for j in range(1, len(cl) - 1):
                    a, b, c = pat[cl[j - 1]], pat[cl[j]], pat[cl[j + 1]]
                    if a == b == c == "C" and g.random() < 0.04: pat[cl[j]] = "T"
                p[2] = "".join(pat); line = "\t".join(p)
        yield line
R, LS = {}, {}
for g in ("GSM5652282", "GSM5652283"):
    t0 = time.time(); f = fetch(g); c = Q2.count_pat(gzip.open(f, "rt"), Q2.hg19_tiles()); LS[g] = c["loose_share"]
    R[g] = S.count_pat_regions(gzip.open(f, "rt"))
    if g == "GSM5652282": R["planted"] = S.count_pat_regions(planted(gzip.open(f, "rt")))
    os.remove(f); L.append(f"{g}: loose share {c['loose_share']:.4f}, held error {c['held_err']:.5f} (all chromosomes), autosomal regions counted in {time.time() - t0:.0f} s"); print(L[-1], flush=True)
specs = [("one healthy sample against itself", *S.self_globe(R["GSM5652282"], LS["GSM5652282"], "a", "x")),
         ("the same sample, loss planted in one stretch (CONSTRUCTED, not a person)", *S.self_globe(R["planted"], LS["GSM5652282"], "b", "x")),
         ("two healthy people against each other", *S.pair_globe(R["GSM5652282"], R["GSM5652283"], "c", "x", "the other person"))]
groups = []
for name, spec, rec in specs:
    for gr in spec["groups"]: gr["name"] = name; groups.append(gr)
    L.append(f"{name}: {rec}")
    if "planted" in name:
        z, ok = S._z_self(R["planted"], rec["predicted"]); reg = np.arange(len(z)) * S.REGION + 1; ins = (reg >= PL[0]) & (reg + S.REGION - 1 <= PL[1]); ch = ok & (np.abs(z) > rec["line"])
        L.append(f"   planted stretch: {int((ch & ins).sum())} of {int((ok & ins).sum())} drawn regions inside it changed; changed outside it {int((ch & ~ins).sum())}")
about = (S.ABOUT.replace("__R__", str(S.REGION)).replace("__WHAT__", "Three views: one healthy sample against its own prediction; the same sample with loss planted in one stretch, to show what a "
         "change looks like; and two healthy people against each other.") + " These samples built IAM-A's self-reference line, so the pictures show the method; they test nothing.")
spec = V.dedupe(dict(id="iama-genome", name="IAM-A on the whole genome: held marks, molecule by molecule (development)", groups=groups, about=about))
s = V.write_files(spec, E, "../art/explorer"); json.dump(s, open(os.path.join(E, "iama-genome.json"), "w"), indent=1)
open(os.path.join(out, "genome_globes_02_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
with tarfile.open(os.path.join(out, "genome_globes_02.tar.gz"), "w:gz") as t: t.add(E, arcname="explorer")
