#!/usr/bin/env python3
"""Neutrophil sky maps for the cellular book, Tools from the Sky chapter (2026-10-02).
Six physical Salas EPIC neutrophil arrays (GSE110554; GSE167998 re-deposits the same six, so it is not read). Our Stage 1 betas.
Each array is held out in turn and read against the OTHER FIVE.
Map sites: reference rule of the chain (across-array SD <= 0.05 among the five, mean beta 0.75-0.95 or 0.05-0.25), all that qualify.
Per site: z = (H(beta) - mean H of the five) / s, s = SD of H among the five shrunk toward the pooled SD (k = 10), as in the chain.
Sky: all EPIC probes in genome order (chr1 -> chrY, then position) laid onto HEALPix NSIDE 64 RING pixels in sequence; pixel value =
sum z / sqrt(n) over the measured map sites in that pixel; empty pixels masked.
C-score (chain form, conductor_v3._clustering): z ordered along the genome at the chain's 6,000 frozen identity sites, block 50,
clustering = var(block means x sqrt(50)) / var(z), divided by the healthy baseline 1.1104 (neutrophil_reference_v1_1.json).
Maps written: healthy (each of the 6), null (healthy minus healthy, two arrays vs the other four), 2 % blur everywhere,
5 % blur inside 10 contiguous genomic blocks (5 % of sites)."""
import glob, json, numpy as np, pandas as pd
R = "/home/ubuntu/data/atlas_sources/blood/GSE110554/shards"; NSIDE = 64; NPIX = 12 * NSIDE * NSIDE; rng = np.random.default_rng(20261002)
F = json.load(open("metA_floors_v1_3.json"))["platforms"]["EPIC"]["neutrophils"]; REF = json.load(open("neutrophil_reference_v1_1.json"))
ARR = F["refs"]; ID = pd.Index(F["sites"]); BASE = REF["healthy_clustering_median"]; W = int(REF["clustering_block"])
B = pd.concat([pd.read_parquet(glob.glob(f"{R}/{a}*.parquet")[0]).iloc[:, 0].rename(a.split("_")[0]) for a in ARR], axis=1).astype("float64")
M = pd.read_csv("/home/ubuntu/data/EPIC.hg38.manifest.gencode.v36.tsv.gz", sep="\t", usecols=["probeID", "CpG_chrm", "CpG_beg"]).dropna()
M = M[M.CpG_chrm.str.match(r"^chr([0-9]+|X|Y)$")]
ck = M.CpG_chrm.str.replace("chr", "").replace({"X": "23", "Y": "24"}).astype(int)
M = M.assign(ck=ck).sort_values(["ck", "CpG_beg", "probeID"], kind="mergesort").reset_index(drop=True)
M["pix"] = (np.arange(len(M)) * NPIX // len(M)).astype(int); M["order"] = np.arange(len(M)); M = M.set_index("probeID")
B = B.loc[B.index.intersection(M.index)]
IDX = ID.intersection(B.index)
def Hb(x):
    x = np.clip(x, 1e-6, 1 - 1e-6); return -(x * np.log2(x) + (1 - x) * np.log2(1 - x))
cols = list(B.columns)
def sites_for(ref):
    mu = B[ref].mean(axis=1); sd = B[ref].std(axis=1)
    ok = (sd <= 0.05) & (((mu >= 0.75) & (mu <= 0.95)) | ((mu >= 0.05) & (mu <= 0.25))); return ok[ok].index
def zmap(x, ref, idx):
    Hr = Hb(B.loc[idx, ref]); m = Hr.mean(axis=1); s = Hr.std(axis=1); n = len(ref); sp = np.sqrt(np.nanmedian(s ** 2))
    s2 = np.sqrt(((n - 1) * s ** 2 + 10 * sp ** 2) / (n - 1 + 10)); return (Hb(x.loc[idx]) - m) / s2
def sky(z):
    p = M.loc[z.index, "pix"]; g = pd.DataFrame({"p": p.values, "z": z.values}).groupby("p").z.agg(["sum", "size"])
    v = np.full(NPIX, np.nan); v[g.index.values] = (g["sum"] / np.sqrt(g["size"])).values; return v
def clus(z):
    o = z.reindex(M.loc[z.index].sort_values("order").index).dropna().values; nb = len(o) // W
    b = o[:nb * W].reshape(nb, W).mean(axis=1) * np.sqrt(W); return float(np.var(b) / np.var(o))
def blur(x, e): return x + e * (0.5 - x)
def metA(x, ref):
    return float(Hb(x.loc[IDX]).mean() / Hb(B.loc[IDX, ref]).mean(axis=0).mean())
out = {}; stats = []
for h in cols:
    ref = [c for c in cols if c != h]; idx = sites_for(ref); x = B[h]
    o = M.loc[idx].sort_values("order").index; L = len(o); bl = int(0.005 * L); starts = rng.choice(L - bl, 10, replace=False)
    loc = pd.Index(np.unique(np.concatenate([o[s:s + bl] for s in starts])))
    xl = x.copy(); xl.loc[loc] = blur(x.loc[loc], 0.05)
    xl_id = x.copy(); oi = M.loc[IDX].sort_values("order").index; Li = len(oi); bi = int(0.005 * Li)
    si = rng.choice(Li - bi, 10, replace=False); loci = pd.Index(np.unique(np.concatenate([oi[s:s + bi] for s in si])))
    xl_id.loc[loci] = blur(x.loc[loci], 0.05)
    for nm, xx, xc in (("healthy", x, x), ("blur2pct", blur(x, 0.02), blur(x, 0.02)), ("local5pct", xl, xl_id)):
        z = zmap(xx, ref, idx); zid = zmap(xc, ref, IDX)
        out[f"{h}_{nm}"] = sky(z)
        stats.append(dict(array=h, map=nm, n_map_sites=len(idx), MetA_6000=metA(xc, ref), C_6000=clus(zid) / BASE,
                          map_clustering=clus(z), z_mean=float(z.mean()), z_sd=float(z.std())))
a, b = cols[0], cols[1]; ref = [c for c in cols if c not in (a, b)]; idx = sites_for(ref)
zd = (zmap(B[a], ref, idx) - zmap(B[b], ref, idx)) / np.sqrt(2); out["null"] = sky(zd)
stats.append(dict(array=f"{a}-{b}", map="null", n_map_sites=len(idx), MetA_6000=np.nan, C_6000=np.nan, map_clustering=clus(zd),
                  z_mean=float(zd.mean()), z_sd=float(zd.std())))
chrom = np.full(NPIX, -1); g = M.groupby("pix").ck.first(); chrom[g.index.values] = g.values; out["pix_chrom"] = chrom
out["probes_per_pixel"] = np.bincount(M["pix"].values, minlength=NPIX)
np.savez_compressed("sky_neut6_maps.npz", nside=NSIDE, n_probes=len(M), **out)
T = pd.DataFrame(stats); T.to_csv("sky_neut6_stats.csv", index=False)
print("arrays", cols, "| EPIC probes on sky", len(M), "| identity sites on sky", len(IDX)); print(T.round(4).to_string(index=False)); print("DONE")
