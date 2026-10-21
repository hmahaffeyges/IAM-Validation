"""Noise maps of IAM-Atlas v2.1 for the globe (2026-10-21, to-do step 28; author: "watch in real time as the noise dampens").

Every atlas site carries 100 posterior draws per cell (4 sampling chains of 25), published in Zenodo record 10.5281/zenodo.23422688.
Each draw is one equally likely version of the cell's healthy map. Averaging more draws beats the sampling noise down (about as one over
the square root of the number of draws) and the map settles onto the atlas's value. This builds, for the six blood panel cells, the average
of the first k draws at every site for k = 1, 2, 4, 8, 16, 32, 64, 100, taking draws in turn from the four chains (draw j of the sequence is
chain j mod 4, its (j div 4)-th draw), so every step mixes all chains. A picture of the atlas's own uncertainty: nothing is fitted or scored.

  block TAR OUT      one Zenodo bundle (70 blocks): download, check its MD5 against the record, check every draws file against
                     output_manifest_v21.json (a file missing from it, or differing, stops the job), write OUT/noise_<first>-<last>.npz (cpg, k, cells, mean[k, cell, site])
  combine DIR OUT    every bundle's output -> genome-order sky (atlas/tools/sky_order_v21's frozen layout), one texture per cell per k
                     (OUT/noise_<cell>_k<k>.jpg, the atlas pages' parchment scale), and OUT/noise.json with, per cell and k, the mean
                     |average of k draws - average of 100| over measured sites (the number beside the slider)"""
import os, sys, io, json, hashlib, tarfile, numpy as np
REC = "23422688"; K = (1, 2, 4, 8, 16, 32, 64, 100)
CELLS = ("neutrophils", "monocytes", "b cells", "cd4 t cells", "cd8 t cells", "nk cells")
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.abspath(os.path.join(H, "..", "..", "..", ".."))

def order_index(chain):
    """Draw indices in the stated interleaved order: chain 0's 1st, chain 1's 1st, ..., chain 0's 2nd, ..."""
    by = [list(np.where(chain == c)[0]) for c in sorted(set(chain.tolist()))]
    return np.array([by[j % len(by)][j // len(by)] for j in range(sum(len(b) for b in by))])

def block(tarname, out):
    import requests
    J = requests.get(f"https://zenodo.org/api/records/{REC}", timeout=120).json(); F = {x["key"]: x for x in J["files"]}
    man = requests.get(F["output_manifest_v21.json"]["links"]["self"], timeout=120).json()
    man = man.get("files", man) if isinstance(man, dict) else {}
    want = F[tarname]["checksum"].split(":")[1]; path = os.path.join(out, tarname); md5 = hashlib.md5()
    os.makedirs(out, exist_ok=True)
    with requests.get(F[tarname]["links"]["self"], stream=True, timeout=600) as r, open(path, "wb") as f:
        r.raise_for_status()
        for ch in r.iter_content(1 << 22): f.write(ch); md5.update(ch)
    assert md5.hexdigest() == want, f"{tarname}: MD5 {md5.hexdigest()} is not the record's {want}"
    print(tarname, "MD5 matches the record", flush=True)
    cpg, means, checked = [], [], 0
    with tarfile.open(path) as T:
        for m in T:
            if not m.name.endswith("_draws.npz"): continue
            b = T.extractfile(m).read(); base = os.path.basename(m.name)
            exp = man.get(m.name) or man.get(base) or (man.get("blocks/" + base) if isinstance(man, dict) else None)
            if isinstance(exp, dict): exp = exp.get("sha256")
            assert exp, f"{m.name}: not listed in output_manifest_v21.json"
            assert hashlib.sha256(b).hexdigest() == exp, f"{m.name}: SHA-256 differs from output_manifest_v21.json"; checked += 1
            Z = np.load(io.BytesIO(b), allow_pickle=True); cells = list(map(str, Z["cells"])); j = [cells.index(c) for c in CELLS]
            d = Z["draws"][:, j, :].astype(np.float32)[order_index(Z["chain"])]   # (100, 6, sites), interleaved
            cs = np.nancumsum(d, axis=0); cn = np.cumsum(np.isfinite(d), axis=0)
            means.append(np.stack([np.where(cn[k - 1] > 0, cs[k - 1] / np.maximum(cn[k - 1], 1), np.nan) for k in K]).astype(np.float16))
            cpg += list(map(str, Z["cpg"]))
    os.remove(path)
    lo, hi = tarname.split("_")[-1].replace(".tar", "").split("-")
    np.savez_compressed(os.path.join(out, f"noise_{lo}-{hi}.npz"), cpg=np.array(cpg, dtype=object), k=np.array(K), cells=np.array(CELLS), mean=np.concatenate(means, axis=2))
    print(f"{tarname}: {len(cpg)} sites, {checked} draws files checked against the manifest", flush=True)

RES = 0.01   # the residual frames' color scale, in methylation (0-1)

def combine(d, out):
    import glob, pyarrow.parquet as pq
    from matplotlib.colors import ListedColormap
    from PIL import Image
    sys.path[:0] = [os.path.join(R, "Biological_Physics", "MethylPhys", "chain", "MethylPhys_Interface")]
    import report_globe_v4 as G
    os.makedirs(out, exist_ok=True)
    ids = pq.read_table(os.path.join(G.ATLAS, "sites.parquet")).column("cpg").to_pylist(); ix = {c: i for i, c in enumerate(ids)}
    M = np.full((len(K), len(CELLS), len(ids)), np.nan, np.float32); n = 0
    for f in sorted(glob.glob(os.path.join(d, "**", "noise_*.npz"), recursive=True)):
        Z = np.load(f, allow_pickle=True); r = np.array([ix[c] for c in Z["cpg"]]); M[:, :, r] = Z["mean"].astype(np.float32); n += len(r)
    assert n == len(ids), f"combined {n} sites, the atlas has {len(ids)}"
    O = np.load(G.ORDER); order = O["order"]; nside = int(O["nside"]); npix = 12 * nside * nside; look = G._lookup(nside)
    cm = ListedColormap(np.loadtxt(os.path.join(R, "website", "art", "data", "Planck_Parchment_RGB.txt")) / 255.0)
    info = dict(record=f"10.5281/zenodo.{REC}", k=list(K), cells=list(CELLS), order="draws taken in turn from the 4 chains",
                layout="IAM-Atlas v2.1 genome-order sky, NSIDE 256 NESTED (sky_order_v21.npz)", settle={}, residual_scale=RES, source="atlas/tools/noise_maps_v21.py")
    for c, cell in enumerate(CELLS):
        info["settle"][cell] = {}
        for a, k in enumerate(K):
            v = M[a, c][order]; m = G._to_sky(v, npix)[look].reshape(G.HT, G.W)
            x = (cm(np.clip(np.nan_to_num(m, nan=0), 0, 1))[..., :3] * 255).astype(np.uint8); x[~np.isfinite(m)] = (42, 45, 51)
            Image.fromarray(x).save(os.path.join(out, f"noise_{cell.replace(' ', '_')}_k{k}.jpg"), quality=72, optimize=True, progressive=True)
            info["settle"][cell][str(k)] = round(float(np.nanmean(np.abs(M[a, c] - M[-1, c]))), 5)
            # 2026-10-21: the noise itself, visible: the k-draw map minus the 100-draw map on a fixed scale of +/- RES (it shrinks about as
            # one over the square root of k; on the 0-1 methylation scale it is under one percent and cannot be seen)
            d = G._to_sky((M[a, c] - M[-1, c])[order], npix)[look].reshape(G.HT, G.W); z = np.clip(np.nan_to_num(d, nan=0) / RES, -1, 1)
            f = np.abs(z) ** 0.7; y = np.full(z.shape + (3,), 22.0)   # dark where the draws agree; red above, blue below the 100-draw map
            y[..., 0] += np.where(z > 0, 233 * f, 30 * f); y[..., 1] += 45 * f; y[..., 2] += np.where(z < 0, 233 * f, 30 * f)
            y = y.astype(np.uint8); y[~np.isfinite(d)] = (42, 45, 51)
            Image.fromarray(y).save(os.path.join(out, f"resid_{cell.replace(' ', '_')}_k{k}.jpg"), quality=72, optimize=True, progressive=True)
        print(cell, info["settle"][cell], flush=True)
    json.dump(info, open(os.path.join(out, "noise.json"), "w"), indent=1)

UNC_K = [1, 2, 4, 8, 16, 32, 64, 100]

def uncertainty(out):
    """2026-10-21 (author: "make it stay lighter longer if it is less certain"): per blood cell, the noise expected in an average of k draws at
    every site, the site's posterior spread over the square root of k (spread = the atlas's committed 95 % range / 3.92; draws treated as
    independent, so this is the smallest the noise can be). One fixed scale for every frame and cell (the 95th percentile of the spread over
    the autosomes) on a log scale from S/10 (dark) to S (full): every doubling of the draws dims every site by the same step, so a sure
    site falls below the dark line early and an unsure one stays lit to the end (an even scale only dims all sites together). Each cell twice: every
    site, and the X and Y hidden (the atlas is least sure there because its donors are women and men). Light: built from the committed atlas,
    no draws needed. Writes unc_{cell}_k{k}.jpg, unc_{cell}_noXY_k{k}.jpg and uncertainty.json."""
    import json as _j
    from PIL import Image
    from matplotlib import colormaps
    sys.path[:0] = [os.path.join(R, "Biological_Physics", "MethylPhys", "chain", "MethylPhys_Interface")]
    import report_globe_v4 as G, pyarrow.parquet as pq
    os.makedirs(out, exist_ok=True)
    ids = np.array(pq.read_table(os.path.join(G.ATLAS, "sites.parquet")).column("cpg").to_pylist(), dtype=object)
    O = np.load(G.ORDER); order = O["order"]; nside = int(O["nside"]); npix = 12 * nside * nside; look = G._lookup(nside)
    xy = np.isin(ids, list(G.sex_sites())); lo, hi = G._mu(CELLS, "mu_lo"), G._mu(CELLS, "mu_hi")
    SD = {c: (hi[c] - lo[c]) / 3.92 for c in CELLS}; S = float(np.nanpercentile(np.concatenate([SD[c][~xy] for c in CELLS]), 95))
    cm = colormaps["inferno"]; info = dict(k=UNC_K, cells=list(CELLS), scale=round(S, 5), source="atlas/tools/noise_maps_v21.py uncertainty",
                                          rule="noise in a k-draw average = the site's posterior spread / sqrt(k); spread = (mu_hi - mu_lo) / 3.92; drawn on a log scale, dark at scale/10, full at scale", median_spread={}, xy_median_spread={})
    for c in CELLS:
        info["median_spread"][c] = round(float(np.nanmedian(SD[c][~xy])), 5); info["xy_median_spread"][c] = round(float(np.nanmedian(SD[c][xy])), 5)
        for hide, tag in ((False, ""), (True, "_noXY")):
            v = SD[c].copy()
            if hide: v[xy] = np.nan
            for k in UNC_K:
                m = G._to_sky((v / np.sqrt(k))[order], npix)[look].reshape(G.HT, G.W); z = np.clip((np.log10(np.maximum(np.nan_to_num(m, nan=0), 1e-9)) - np.log10(S / 10)), 0, 1)   # log scale: S lights fully, S/10 and below dark
                y = (cm(z)[..., :3] * 255).astype(np.uint8); y[~np.isfinite(m)] = (42, 45, 51)
                Image.fromarray(y).save(os.path.join(out, f"unc_{c.replace(' ', '_')}{tag}_k{k}.jpg"), quality=78, optimize=True, progressive=True)
        print(c, info["median_spread"][c], "X and Y", info["xy_median_spread"][c], flush=True)
    _j.dump(info, open(os.path.join(out, "uncertainty.json"), "w"), indent=1)

if __name__ == "__main__":
    if sys.argv[1] == "uncertainty": uncertainty(sys.argv[2]); sys.exit()
    {"block": lambda: block(sys.argv[2], sys.argv[3]), "combine": lambda: combine(sys.argv[2], sys.argv[3])}[sys.argv[1]]()
