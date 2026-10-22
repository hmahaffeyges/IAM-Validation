"""DEV-IAMA-GLOBE-01, step 5 (written 2026-10-22 after step 4; step 4 unchanged except that every pixel of a region now carries it, so the globe is
filled instead of one dot per region, and the region counts are saved so a picture can be redrawn without recounting; development, pictures, nothing scored).
Step 3 showed: two people against each other are quiet (6 of 25,555 regions), one sample against its genome-wide prediction leaves 33 of
25,595 lit, and the constructed plant lit only 11 of its 299 regions. Step 4 adds the view that matters most for one individual over time:
the SAME sample read twice. Its molecules are split in two halves by a fixed rule (each pair of lines, the unit count_pat reads, goes to half
1 or half 2 by a seeded coin, seed 20261022), so the two halves differ only by counting; they stand for one individual sampled twice.
Globes (regions of 1,000 CpGs, chain/stage_q2_globe.py, unchanged; healthy CD8 T cells GSM5652282 and GSM5652283, spent):
  1. the sample against its own predicted held error (as step 3);
  2. half 2 against half 1: one individual sampled twice, nothing changed;
  3. half 2 with held-mark loss planted (CONSTRUCTED: on held molecules in chr2's CpG 2,500,001 to 2,800,000, each interior methylated call
     between two methylated ones turned unmethylated with probability 0.15) against half 1: one individual sampled twice, with a change;
  4. the two people against each other (as step 3).
Reported: each globe's changed regions; the median held-error chances per region; for 3, the planted stretch's regions lit and the regions
lit outside it. Usage: python3 genome_globes_05.py OUTDIR"""
import os, sys, re, time, gzip, json, tarfile, urllib.request, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); CH = os.path.join(H, "..", "..", "..", "chain"); sys.path[:0] = [CH, os.path.join(CH, "MethylPhys_Interface")]
import stage_q2_selfref as Q2, stage_q2_globe as S, globe_viewer as V
S.REGION = 1000
out = sys.argv[1]; E = os.path.join(out, "explorer"); os.makedirs(E, exist_ok=True); L = []
def fetch(g):
    base = f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{g[:7]}nnn/{g}/suppl/"
    for a in range(5):
        try:
            name = sorted(set(re.findall(rf"{g}_[^\"<>]*\.pat\.gz", urllib.request.urlopen(base, timeout=120).read().decode())))[0]
            f = os.path.join(out, name); urllib.request.urlretrieve(base + name, f); return f
        except Exception as e: print(g, "retry", type(e).__name__, flush=True); time.sleep(60 * (a + 1))
    raise RuntimeError(g + ": download failed")
PL = (2284470 + 2_500_001, 2284470 + 2_800_000)
def half(lines, h, seed=20261022):
    g = np.random.default_rng(seed); side = None
    for k, line in enumerate(lines):
        if k % 2 == 0: side = int(g.random() < 0.5)
        if side == h: yield line
def planted(lines, prob=0.15, seed=1):
    g = np.random.default_rng(seed)
    for k, line in enumerate(lines):
        p = line.split("\t")
        if k % 2 == 1 and len(p) >= 4 and PL[0] <= int(p[1]) <= PL[1]:
            pat = list(p[2]); cl = [q for q, x in enumerate(pat) if x != "."]
            if len(cl) >= Q2.QUAL_MIN_CALLS:
                for j in range(1, len(cl) - 1):
                    if pat[cl[j - 1]] == pat[cl[j]] == pat[cl[j + 1]] == "C" and g.random() < prob: pat[cl[j]] = "T"
                p[2] = "".join(pat); line = "\t".join(p)
        yield line
R, LS = {}, {}
for g in ("GSM5652282", "GSM5652283"):
    t0 = time.time(); f = fetch(g); op = lambda: gzip.open(f, "rt")
    c = Q2.count_pat(op(), Q2.hg19_tiles()); LS[g] = c["loose_share"]; R[g] = S.count_pat_regions(op())
    if g == "GSM5652282":
        R["h1"] = S.count_pat_regions(half(op(), 1)); R["h2"] = S.count_pat_regions(half(op(), 0)); R["h2p"] = S.count_pat_regions(planted(half(op(), 0)))
    os.remove(f); L.append(f"{g}: loose share {c['loose_share']:.4f}, held error {c['held_err']:.5f}; counted in {time.time() - t0:.0f} s"); print(L[-1], flush=True)
np.savez_compressed(os.path.join(out, "genome_globes_05_counts.npz"), loose=json.dumps(LS), **{f"{k}_{f}": v[f] for k, v in R.items() for f in ("err", "opp", "meth", "calls")})
o = R["GSM5652282"]["opp"]; L.append(f"median held-error chances per drawn region: whole sample {np.median(o[o >= S.MIN_OPP]):.0f}; one half {np.median(R['h1']['opp'][R['h1']['opp'] >= S.MIN_OPP]):.0f}")
specs = [("one healthy sample against its own prediction", *S.self_globe(R["GSM5652282"], LS["GSM5652282"], "a", "x")),
         ("the same sample read twice, nothing changed (its molecules split in two halves)", *S.pair_globe(R["h2"], R["h1"], "b", "x", "the same sample's other half")),
         ("the same sample read twice, a change planted in one stretch (CONSTRUCTED)", *S.pair_globe(R["h2p"], R["h1"], "c", "x", "the same sample's other half")),
         ("two healthy people against each other", *S.pair_globe(R["GSM5652282"], R["GSM5652283"], "d", "x", "the other person"))]
groups = []
for name, spec, rec in specs:
    for gr in spec["groups"]: gr["name"] = name; groups.append(gr)
    L.append(f"{name}: {rec}")
    if "planted" in name:
        z, ok = S._z_pair(R["h2p"], R["h1"]); z = z / rec["own_spread"]; reg = np.arange(len(z)) * S.REGION + 1
        ins = (reg >= PL[0]) & (reg + S.REGION - 1 <= PL[1]); ch = ok & (np.abs(z) > rec["line"])
        L.append(f"   planted stretch: {int((ch & ins).sum())} of {int((ok & ins).sum())} drawn regions inside it changed; changed outside it {int((ch & ~ins).sum())}")
about = (S.ABOUT.replace("__R__", str(S.REGION)).replace("__WHAT__", "Four views: one healthy sample against its own prediction; the same sample read twice (its molecules split "
         "in two halves), first with nothing changed and then with a loss planted in one stretch between the two readings; and two healthy people against each other.")
         + " These samples built IAM-A's self-reference line, so the pictures show the method; they test nothing.")
spec = V.dedupe(dict(id="iama-genome", name="IAM-A on the whole genome: held marks, molecule by molecule (development)", groups=groups, about=about))
s = V.write_files(spec, E, "../art/explorer"); assert s["assets"], "no pictures written"; json.dump(s, open(os.path.join(E, "iama-genome.json"), "w"), indent=1)
open(os.path.join(out, "genome_globes_05_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
with tarfile.open(os.path.join(out, "genome_globes_05.tar.gz"), "w:gz") as t: t.add(E, arcname="explorer")
