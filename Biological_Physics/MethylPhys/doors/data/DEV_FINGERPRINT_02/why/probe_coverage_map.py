"""DEV-FINGERPRINT-02 'why' diagnostics, step 1 (descriptive; verdicts are not re-scored). For every 450K and EPIC v1 cg probe: its hg19 CpG
index (wgbstools CpG.bed.gz), local CpG density (CpGs within +-500 bp), and how many normal RRBS molecules (Box Run 9 normal .pat files) cover it.
Probe positions: methylprep's pinned manifests (build 37 columns). Usage (box):
  python3 probe_coverage_map.py CPG_BED_GZ MANIFEST_DIR NORMAL_PAT_DIR OUT.csv.gz"""
import sys, os, gzip, glob, numpy as np, pandas as pd
CPG, MAN, PATD, OUT = sys.argv[1:5]
cp = pd.read_csv(CPG, sep="\t", header=None, names=["chr", "pos", "idx"], dtype={"chr": str, "pos": np.int64, "idx": np.int64})
cp["chr"] = cp.chr.str.replace("chr", "", regex=False)
def man(f, chrcol, poscol):
    rows = []
    with gzip.open(os.path.join(MAN, f), "rt") as h:
        hdr = None
        for l in h:
            p = l.rstrip("\n").split(",")
            if hdr is None:
                if p[0] == "IlmnID": hdr = p
                continue
            if not p[0].startswith("cg") or len(p) < len(hdr): continue
            d = dict(zip(hdr, p)); rows.append((p[0], d.get(chrcol, ""), d.get(poscol, "")))
    m = pd.DataFrame(rows, columns=["probe", "chr", "pos"]); m = m[(m.chr != "") & (m.pos != "")]
    m["pos"] = m.pos.astype(float).astype(np.int64); return m
M = []
for f in ("HumanMethylation450k_15017482_v3.csv.gz", "MethylationEPIC_v-1-0_B4.CoreColumns.csv.gz"):
    with gzip.open(os.path.join(MAN, f), "rt") as h:
        for l in h:
            if l.startswith("IlmnID"): hdr = l.rstrip("\n").split(","); break
    chrcol = [c for c in hdr if c.upper() in ("CHR_37", "CHR")][-1 if "CHR_37" not in [c.upper() for c in hdr] else 0]
    poscol = [c for c in hdr if c.upper() in ("MAPINFO_37", "MAPINFO")][-1 if "MAPINFO_37" not in [c.upper() for c in hdr] else 0]
    if "CHR_37" in [c.upper() for c in hdr]: chrcol = hdr[[c.upper() for c in hdr].index("CHR_37")]; poscol = hdr[[c.upper() for c in hdr].index("MAPINFO_37")]
    print(f, "columns", chrcol, poscol, flush=True); M.append(man(f, chrcol, poscol))
M = pd.concat(M).drop_duplicates("probe"); M["chr"] = M.chr.astype(str).str.replace("chr", "", regex=False)
out = []
for ch, g in M.groupby("chr"):
    c = cp[cp.chr == ch]
    if c.empty: continue
    pos = c.pos.values; idx = c.idx.values
    k = np.searchsorted(pos, g.pos.values); k1 = np.clip(k, 0, len(pos) - 1); k0 = np.clip(k - 1, 0, len(pos) - 1)
    best = np.where(np.abs(pos[k1] - g.pos.values) <= np.abs(pos[k0] - g.pos.values), k1, k0); dist = np.abs(pos[best] - g.pos.values)
    dens = np.searchsorted(pos, g.pos.values + 500) - np.searchsorted(pos, g.pos.values - 500)
    out.append(pd.DataFrame(dict(probe=g.probe.values, chr=ch, pos=g.pos.values, cpg_idx=np.where(dist <= 1, idx[best], -1), cpg_density_1kb=dens)))
P = pd.concat(out); print("probes mapped", int((P.cpg_idx >= 0).sum()), "of", len(P), flush=True)
cov = np.zeros(int(cp.idx.max()) + 2, np.int64)
for f in sorted(glob.glob(os.path.join(PATD, "*.pat.gz"))):
    d = pd.read_csv(f, sep="\t", header=None, usecols=[1, 2, 3], names=["s", "p", "n"])
    L = d.p.str.len().values; s = d.s.values; n = d.n.values
    np.add.at(cov, s, n); np.add.at(cov, s + L, -n)
cov = np.cumsum(cov)
P["normal_rrbs_molecules"] = np.where(P.cpg_idx >= 0, cov[np.clip(P.cpg_idx, 0, len(cov) - 1)], 0)
P.to_csv(OUT, index=False); print(P.describe().round(1).to_string())
