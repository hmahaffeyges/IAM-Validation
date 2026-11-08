"""reference_globes.py - the reference globes, drawn by the chain (2026-10-22; author: "if we are going to use the globes as references and visual
representations of real samples run by the chain, then the chain must be able to create the globes"). Moved here unchanged from the website's
scripts (website/art/make_explorer.py purified and noise, make_sphere_3d.py, make_sphere_masks.py), which now only run the one command:
  run_sample.py --build explorer-purified --build-out SPEC.json          Met-A's purified-array reference on the atlas's layout
  run_sample.py --build explorer-noise --build-input NOISE_DIR --build-out SPEC.json   the atlas's noise, beaten down by averaging draws
  run_sample.py --build atlas-sphere --build-out DIR [--build-input ATLAS_DIR]          the 79 cells' globe textures (checked against the cell pages)
  run_sample.py --build atlas-masks --build-out DIR                       the value pictures for the atlas globe's sliders
Pictures of references: nothing is read, fitted or scored here."""
import os, sys, re, json, glob, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); CH = os.path.dirname(HERE); REPO = os.path.abspath(os.path.join(CH, "..", "..", ".."))
ATLAS = os.path.join(CH, "..", "atlas", "v2.1"); sys.path[:0] = [CH, HERE, os.path.join(CH, "..", "atlas", "tools")]
import report_globe_v4 as G, globe_viewer as V
GROUP = {"NEU": "neutrophils", "MONO": "monocytes", "B": "B cells", "CD4T": "CD4 T cells", "CD8T": "CD8 T cells", "NK": "NK cells", "BASO": "basophils",
         "EOS": "eosinophils", "BAS": "basophils", "TREG": "regulatory T cells", "GRAN": "granulocytes"}
W, HT = 1024, 512; GREY = np.array([0x2a, 0x2d, 0x33], dtype=np.uint8); HW_MAX = 0.15
def slug(c): return re.sub(r"[^a-z0-9]+", "-", c.lower()).strip("-")

def purified():
    import pyarrow.parquet as pq
    from matplotlib.colors import ListedColormap
    B = json.load(open(os.path.join(CH, "Runtime Matrices", "Met_A_Floors", "blood_composition_EPIC_v1.json")))
    ids = np.array(pq.read_table(os.path.join(G.ATLAS, "sites.parquet")).column("cpg").to_pylist(), dtype=object)
    Z = np.load(G.ORDER); order = Z["order"]; nside = int(Z["nside"]); npix = 12 * nside * nside; look = G._lookup(nside)
    cm = ListedColormap(np.loadtxt(os.path.join(HERE, "data", "Planck_Parchment_RGB.txt")) / 255.0); L = []
    for g, prof in B["profiles_at_neutrophil_sites"].items():
        v = pd.Series(np.asarray(prof, float), index=B["neutrophil_sites"]).reindex(ids[order]).to_numpy()
        t = G._dots(G._to_sky(v, npix)[look].reshape(G.HT, G.W)); x = (cm(np.clip(np.nan_to_num(t, nan=0), 0, 1))[..., :3] * 255).astype(np.uint8); x[~np.isfinite(t)] = (42, 45, 51)
        L.append(dict(name=GROUP.get(g, g), kind="image", src="data:image/jpeg;base64," + G._jpg_b64(x), mask=G._vmask(t), **G.MV,
                      note=f"The purified healthy {GROUP.get(g, g)} profile at Met-A's {int(np.isfinite(v).sum()):,} identity sites placed on the atlas's layout, 0 to 1."))
    spec = V.dedupe(dict(id="purified", name="Met-A's purified-array reference", groups=[dict(name="purified healthy blood groups", layers=L)],
        about=("The reference Met-A and the C-Score read against: purified healthy cell arrays, at Met-A's 6,000 neutrophil identity sites only, "
               "so it is drawn as dots, each at the same place as on the atlas globes. A blood sample is compared with these profiles mixed in its own "
               "proportions. IAM-Atlas v2.1 is a different reference, built from 21 sources over the whole genome; the two agree closely value by value.")))
    return spec

def noise(d):
    """The noise maps of IAM-Atlas v2.1 (atlas/tools/noise_maps_v21.py): per blood cell, how sure the atlas is at every site, as the noise left
    in an average of k draws (the site's posterior spread over the square root of k, on one log scale), with every site and with the X and Y
    hidden; and the atlas's best value. Rebuilt 2026-10-21 (author: "make it stay lighter longer if it is less certain"): the earlier frames,
    each k-draw average minus the 100-draw average, faded every site at the same rate."""
    import base64
    J = json.load(open(os.path.join(d, "noise.json"))); U = json.load(open(os.path.join(d, "uncertainty.json"))); K = U["k"]; groups = []
    b64 = lambda f: "data:image/jpeg;base64," + base64.b64encode(open(os.path.join(d, f), "rb").read()).decode()
    from matplotlib import colormaps
    cm = colormaps["inferno"]; css = "linear-gradient(to right, " + ", ".join(f"rgb({int(255*cm(t)[0])},{int(255*cm(t)[1])},{int(255*cm(t)[2])}) {int(100*t)}%" for t in np.linspace(0, 1, 11)) + ")"
    key = ("<b>Color key</b><div style='height:12px;margin:4px 0;border:1px solid #555;background:" + css + "'></div><div style='display:flex;justify-content:space-between'>"
           f"<span>dark: noise {U['scale']/10:.4f} or less (sure)</span><span>bright: {U['scale']:.4f} or more (least sure)</span></div>"
           "<div style='margin-top:4px'>The noise is in methylation (0 to 1). Each doubling of the draws dims every site by the same step, so what stays lit "
           "longest is where the atlas is least sure. Charcoal: no site there, or the X and Y hidden.</div>")
    for cell in U["cells"]:
        c = cell.replace(" ", "_"); m, mx = U["median_spread"][cell], U["xy_median_spread"][cell]
        lay = lambda tag, nm, extra: dict(name=nm, kind="steps", srcs=[b64(f"unc_{c}{tag}_k{k}.jpg") for k in K], key_html=key,
                 hint="Slide right to average more draws: sure sites go dark first; the least sure stay lit to the end.",
                 fixed=f"How sure the atlas is at each site: the noise left in an average of k draws. Median spread {m:.4f} on the autosomes, {mx:.4f} on the X and Y.",
                 labels=[f"{k} draw{'s' if k > 1 else ''} averaged" for k in K],
                 note="Each frame is the site's posterior spread (from the atlas's 95 % range) over the square root of the number of draws averaged. " + extra)
        groups.append(dict(name=cell, layers=[
            lay("", "how sure the atlas is, every site", "The X and Y stay lit longest: the atlas's donors are women and men, and a woman holds one X methylated."),
            lay("_noXY", "how sure the atlas is, X and Y hidden", "With the X and Y hidden, the rest of the genome is easier to read."),
            dict(name="the atlas's best value (100 draws)", kind="image", src=b64(f"noise_{c}_k100.jpg"), key_css=None,
                 **({"mask": "data:image/png;base64," + base64.b64encode(open(os.path.join(d, f"noise_{c}_k100_m.png"), "rb").read()).decode(), **G.MV} if os.path.exists(os.path.join(d, f"noise_{c}_k100_m.png")) else {}),
                 note="The mean of all 100 draws, 0 to 1: the atlas's map of this cell.")]))
    about = ("IAM-Atlas v2.1 was fitted the way a sky map is: every site carries 100 equally likely versions of each cell's healthy map (posterior "
             f"draws, Zenodo {J['record']}). Averaging more draws beats the noise down, about as one over the square root of their number, which is "
             "how the microwave sky's noise is beaten down by more observing. Slide from 1 draw to 100: sure places go dark first, and what stays lit "
             "longest is where the atlas is least sure. Why it is here: before trusting a change a sample shows against the atlas, check that the atlas "
             "is sure at that place.")
    return V.dedupe(dict(id="noise", name="How sure the atlas is: its noise, beaten down by averaging", groups=groups, about=about))

def manifests():
    """methylprep's EPIC and 450K manifests under the names sky_maps_v21.genome_order reads."""
    from methylprep.files import Manifest
    from methylprep.models import ArrayType
    for t in ("epic", "450k"):
        try: Manifest(ArrayType(t))
        except Exception as e: print("methylprep Manifest", t, "->", type(e).__name__, str(e)[:200], flush=True)
    try:   # the EPIC v1 B4 manifest straight from methylprep's own source, by methylprep's own file table (2026-10-21)
        import methylprep.files.manifests as MM, urllib.request
        tab = {str(k): v for k, v in getattr(MM, "ARRAY_TYPE_MANIFEST_FILENAMES", {}).items()}; remote = getattr(MM, "MANIFEST_REMOTE_PATH", "")
        print("methylprep manifest table:", tab, "| remote:", remote, flush=True)
        dd = os.path.expanduser("~/.methylprep_manifest_files"); os.makedirs(dd, exist_ok=True)
        for fn in tab.values():
            if ("B4" in fn or "450k" in fn) and not os.path.exists(os.path.join(dd, fn)) and remote:
                urllib.request.urlretrieve(remote + fn, os.path.join(dd, fn)); print("downloaded", fn, flush=True)
    except Exception as e: print("manifest table route:", type(e).__name__, str(e)[:200], flush=True)
    d = os.path.join("/tmp", "iam_manifests"); os.makedirs(d, exist_ok=True)
    roots = [os.path.expanduser("~"), os.getcwd(), "/tmp"]   # methylprep's download folder differs between installs: find the files by name
    # 2026-10-21: the cell pages' maps were laid out from Illumina's B4 manifest itself; methylprep's own EPIC table moves ~170 pixels in 27 cells,
    # so only the B4 file is accepted, fetched from methylprep's manifest store and checked against Stage 1's pin (MANIFEST_SHA256)
    import hashlib, urllib.request, stage_1_idat_calibration as S1
    b4 = "MethylationEPIC_v-1-0_B4.CoreColumns.csv.gz"; dd = os.path.expanduser("~/.methylprep_manifest_files"); os.makedirs(dd, exist_ok=True)
    if not os.path.exists(os.path.join(dd, b4)): urllib.request.urlretrieve("https://s3.amazonaws.com/array-manifest-files/" + b4, os.path.join(dd, b4))
    h = hashlib.sha256(open(os.path.join(dd, b4), "rb").read()).hexdigest(); assert h == S1.MANIFEST_SHA256[b4], f"{b4}: sha256 {h[:16]} is not Stage 1's pin"
    print("manifest", b4, "matches Stage 1's pin", flush=True)
    alt = {b4: (b4,),
           "HumanMethylation450k_15017482_v3.csv.gz": ("HumanMethylation450k_15017482_v3.csv.gz",)}   # the EPIC v1 B4 manifest under the names methylprep versions use
    for name in alt:
        f = [os.path.join(dp, n) for n in alt[name] for r in roots for dp, _, fs in os.walk(r) if n in fs and not dp.startswith(d)]
        assert f, f"manifest {name} not found under {roots}; methylprep folder holds: {sorted(glob.glob(os.path.expanduser('~/.methylprep_manifest_files/*')))}"
        if not os.path.exists(os.path.join(d, name)): os.symlink(f[0], os.path.join(d, name))
        print("manifest", name, "from", f[0], flush=True)
    return d

def atlas_sphere(out, atlas, man, cells_index=None, name="IAM-Atlas v2.1", pages=True):   # 2026-11-01: name and pages (the IAM Sequencing Atlas has no cell pages)
    import pyarrow.parquet as pq, sky_maps_v21 as SM
    from stage_v4 import nest_pix2ang
    from scipy.spatial import cKDTree
    from matplotlib.colors import ListedColormap
    import matplotlib.pyplot as plt
    from PIL import Image
    os.makedirs(out, exist_ok=True)
    ids = pq.read_table(os.path.join(atlas, "sites.parquet")).column("cpg").to_pylist(); order, unplaced = SM.genome_order(man, ids)
    th, ph = nest_pix2ang(SM.NSIDE, np.arange(SM.NPIX)); xyz = np.c_[np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]
    tt = np.pi * (np.arange(HT) + 0.5) / HT; pp = 2 * np.pi * (np.arange(W) + 0.5) / W; T, P = np.meshgrid(tt, pp, indexing="ij")
    look = cKDTree(xyz).query(np.c_[(np.sin(T) * np.cos(P)).ravel(), (np.sin(T) * np.sin(P)).ravel(), np.cos(T).ravel()])[1]
    CM = ListedColormap(np.loadtxt(os.path.join(HERE, "data", "Planck_Parchment_RGB.txt")) / 255.0); UC = plt.get_cmap("magma"); cells = []
    def tex(m, cmap, lo, hi, path):
        v = m[look].reshape(HT, W); rgb = (cmap(np.clip((np.nan_to_num(v, nan=lo) - lo) / (hi - lo), 0, 1))[..., :3] * 255).astype(np.uint8)
        rgb[~np.isfinite(v)] = GREY; Image.fromarray(rgb).save(path, quality=86, optimize=True)
    for k in (1, 2, 3):
        a, lo, hi = (pq.read_table(os.path.join(atlas, f"{t}_part{k}.parquet")) for t in ("mu", "mu_lo", "mu_hi"))
        for c in a.column_names:
            mu = SM.to_sky(SM.decode(a.column(c).to_numpy())[order]); hw = SM.to_sky(((SM.decode(hi.column(c).to_numpy()) - SM.decode(lo.column(c).to_numpy())) / 2)[order])
            s = slug(c); tex(mu, CM, 0.0, 1.0, os.path.join(out, s + "_v.jpg")); tex(hw, UC, 0.0, HW_MAX, os.path.join(out, s + "_u.jpg"))
            _f = np.isfinite(mu); _h = hw[np.isfinite(hw)]   # 2026-11-03: the numbers each cell page prints, from the same maps (the v2.1 index fields)
            cells.append(dict(cell=c, slug=s, pixels_measured=int(_f.sum()), pixels=int(SM.NPIX), mean_methylation=round(float(np.nanmean(mu)), 3) if _f.any() else None,
                              halfwidth_median=round(float(np.median(_h)), 4) if len(_h) else None, halfwidth_p90=round(float(np.percentile(_h, 90)), 4) if len(_h) else None)); print(c, flush=True)
    cells.sort(key=lambda x: x["cell"].lower())
    # the layout must be the cell pages' own: every cell's measured-pixel count equal to website/art/out/cells/index.json
    ix = {c["slug"]: c.get("pixels_measured") for c in json.load(open(cells_index or os.path.join(REPO, "website", "art", "out", "cells", "index.json")))["cells"]} if pages else {}
    bad = [(c["slug"], c["pixels_measured"], ix.get(c["slug"])) for c in cells if ix.get(c["slug"]) != c["pixels_measured"]]
    if pages: print(f"layout check against the cell pages: {len(cells) - len(bad)} of {len(cells)} cells identical" + (f"; DIFFER: {bad[:5]}" if bad else ""), flush=True)
    if bad and pages: sys.exit("make_sphere_3d: the layout differs from the cell pages' maps; textures not written as the record")
    json.dump(dict(atlas=name, nside=SM.NSIDE, order="NESTED", pixels=SM.NPIX, sites=len(ids), unplaced=unplaced, texture=[W, HT],
                   value_scale=[0, 1], uncertainty_scale=[0, HW_MAX], source="run_sample.py --build atlas-sphere (chain/MethylPhys_Interface/reference_globes.py)", cells=cells),
              open(os.path.join(out, "sphere.json"), "w"), indent=1)
    print(f"sphere: {len(cells)} cells written to {out}")

def noise_sphere(out, cols, man, lim=0.3):
    """2026-11-01 (checklist C3, development): the IAM Sequencing Atlas's two noise maps on the sky sphere, per cell (stage_seq_atlas.noise_maps):
    {slug}_n1.jpg (donor half A minus half B) and {slug}_n2.jpg (read half A minus read half B), a diverging scale -lim to +lim, grey where blank;
    noise_sphere.json holds each cell's summary row and the known-answer check (stage_seq_atlas.noise_check)."""
    import pyarrow.parquet as pq, sky_maps_v21 as SM, stage_seq_atlas as Q
    from stage_v4 import nest_pix2ang
    from scipy.spatial import cKDTree
    import matplotlib.pyplot as plt
    from PIL import Image
    os.makedirs(out, exist_ok=True)
    ids = pq.read_table(os.path.join(cols, "sites.parquet") if os.path.exists(os.path.join(cols, "sites.parquet")) else os.path.join(ATLAS, "sites.parquet")).column("cpg").to_pylist(); order, unplaced = SM.genome_order(man, ids)
    th, ph = nest_pix2ang(SM.NSIDE, np.arange(SM.NPIX)); xyz = np.c_[np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]
    tt = np.pi * (np.arange(HT) + 0.5) / HT; pp = 2 * np.pi * (np.arange(W) + 0.5) / W; T, P = np.meshgrid(tt, pp, indexing="ij")
    look = cKDTree(xyz).query(np.c_[(np.sin(T) * np.cos(P)).ravel(), (np.sin(T) * np.sin(P)).ravel(), np.cos(T).ravel()])[1]; CM = plt.get_cmap("RdBu_r")
    def tex(m, path):
        v = m[look].reshape(HT, W); rgb = (CM(np.clip((np.nan_to_num(v, nan=0) + lim) / (2 * lim), 0, 1))[..., :3] * 255).astype(np.uint8)
        rgb[~np.isfinite(v)] = GREY; Image.fromarray(rgb).save(path, quality=86, optimize=True)
    M, rows = Q.noise_maps(cols)
    for c, (m1, m2) in M.items():
        s = slug(c); tex(SM.to_sky(m1[order]), os.path.join(out, s + "_n1.jpg")); tex(SM.to_sky(m2[order]), os.path.join(out, s + "_n2.jpg")); print(c, flush=True)
    rep = dict(atlas="IAM Sequencing Atlas (development)", scale=[-lim, lim], unplaced=unplaced, check=Q.noise_check(cols), cells=rows,
               source="run_sample.py --build seq-noise-sphere (chain/MethylPhys_Interface/reference_globes.py)")
    json.dump(rep, open(os.path.join(out, "noise_sphere.json"), "w"), indent=1); print("noise sphere:", len(M), "cells; check", rep["check"]); return rep

def atlas_masks(OUT, A=None):
    import pyarrow.parquet as pq, sky_maps_v21 as SM
    from PIL import Image
    A = A or ATLAS
    O = np.load(G.ORDER); order = O["order"]; nside = int(O["nside"]); npix = 12 * nside * nside; look = G._lookup(nside)
    S = {c["slug"]: c for c in json.load(open(os.path.join(OUT, "sphere.json")))["cells"]}
    def code(v, hi):
        c = np.where(np.isfinite(v), 1 + np.round(np.clip(v / hi, 0, 1) * 254), 0).astype(np.uint8); return c[::2, ::2]
    bad = []; n = 0
    for k in (1, 2, 3):
        a, lo, hi = (pq.read_table(os.path.join(A, f"{t}_part{k}.parquet")) for t in ("mu", "mu_lo", "mu_hi"))
        for c in a.column_names:
            s = slug(c); mu = SM.decode(a.column(c).to_numpy()); hw = (SM.decode(hi.column(c).to_numpy()) - SM.decode(lo.column(c).to_numpy())) / 2
            sky = G._to_sky(mu[order], npix); vm = sky[look].reshape(G.HT, G.W); um = G._to_sky(hw[order], npix)[look].reshape(G.HT, G.W)
            if s not in S or int(np.isfinite(sky).sum()) != S[s]["pixels_measured"]: bad.append((s, int(np.isfinite(sky).sum()), S.get(s, {}).get("pixels_measured"))); continue   # sky pixels, as the record counts them
            Image.fromarray(code(vm, 1.0)).save(os.path.join(OUT, s + "_vm.png"), optimize=True); Image.fromarray(code(um, HW_MAX)).save(os.path.join(OUT, s + "_um.png"), optimize=True); n += 1
    print(f"masks: {n} cells written" + (f"; LAYOUT DIFFERS for {len(bad)}: {bad[:4]}" if bad else ""))
    return n, bad


def wg_sphere(out, tex_files):
    """2026-11-07 (author: the Sequencing Atlas Globe on the website must be the new one, at every CpG): the Globe Viewer's pictures of the
    Sequencing Atlas Globe at every CpG, drawn from the published textures (Zenodo 10.5281/zenodo.24029332: globe_wg_textures_n256.npz and the
    megakaryocyte column's), by the same drawing rule as atlas_sphere: NSIDE 256, NESTED, every pixel the average of the CpGs it holds (about 36
    of the 28,217,448 a pixel), the level 0 to 1 and the half-width of the 95 % range 0 to 0.15. A picture of the atlas: nothing is read or scored
    from it. Writes <slug>_v.jpg, <slug>_u.jpg and index.json (each cell's measured pixels and CpGs, the source files)."""
    from stage_v4 import nest_pix2ang
    from scipy.spatial import cKDTree
    from matplotlib.colors import ListedColormap
    import matplotlib.pyplot as plt
    from PIL import Image
    os.makedirs(out, exist_ok=True); NS = 256; NP = 12 * NS * NS; BL = 65535
    th, ph = nest_pix2ang(NS, np.arange(NP)); xyz = np.c_[np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]
    tt = np.pi * (np.arange(HT) + 0.5) / HT; pp = 2 * np.pi * (np.arange(W) + 0.5) / W; T, P = np.meshgrid(tt, pp, indexing="ij")
    look = cKDTree(xyz).query(np.c_[(np.sin(T) * np.cos(P)).ravel(), (np.sin(T) * np.sin(P)).ravel(), np.cos(T).ravel()])[1]
    CM = ListedColormap(np.loadtxt(os.path.join(HERE, "data", "Planck_Parchment_RGB.txt")) / 255.0); UC = plt.get_cmap("magma")
    def tex(m, cmap, lo, hi, path):
        v = m[look].reshape(HT, W); rgb = (cmap(np.clip((np.nan_to_num(v, nan=lo) - lo) / (hi - lo), 0, 1))[..., :3] * 255).astype(np.uint8)
        rgb[~np.isfinite(v)] = GREY; Image.fromarray(rgb).save(path, quality=86, optimize=True)
    dec = lambda a: np.where(a == BL, np.nan, a.astype("float64") / 1e4); cells = []
    for f in tex_files:
        Z = np.load(f); names = sorted({k.split("|")[0] for k in Z.files if k.endswith("|mu")})
        for c in names:
            mu, lo, hi = dec(Z[f"{c}|mu"]), dec(Z[f"{c}|lo"]), dec(Z[f"{c}|hi"]); n = Z[f"{c}|n"].astype(np.int64)
            assert len(mu) == NP, (c, len(mu)); hw = (hi - lo) / 2; s = slug(c)
            tex(mu, CM, 0.0, 1.0, os.path.join(out, s + "_v.jpg")); tex(hw, UC, 0.0, HW_MAX, os.path.join(out, s + "_u.jpg"))
            _f = np.isfinite(mu); _h = hw[np.isfinite(hw)]
            cells.append(dict(cell=c, slug=s, pixels_measured=int(_f.sum()), pixels=NP, cpgs_with_reads=int(n.sum()), mean_methylation=round(float(np.nanmean(mu)), 3) if _f.any() else None,
                              halfwidth_median=round(float(np.median(_h)), 4) if len(_h) else None, source=os.path.basename(f))); print(c, cells[-1]["pixels_measured"], flush=True)
    cells.sort(key=lambda x: x["cell"].lower())
    json.dump(dict(atlas="IAM Sequencing Atlas at every CpG (the Sequencing Atlas Globe)", nside=NS, order="NESTED", pixels=NP, cpgs=28217448, texture=[W, HT],
                   value_scale=[0, 1], uncertainty_scale=[0, HW_MAX], doi="10.5281/zenodo.24029332",
                   source="run_sample.py --build seq-wg-sphere (chain/MethylPhys_Interface/reference_globes.py wg_sphere), from the published textures", cells=cells),
              open(os.path.join(out, "index.json"), "w"), indent=1)
    print(f"whole-genome sphere: {len(cells)} cells written to {out}"); return len(cells)


NZ_LIM = 0.05   # display scale of the whole-genome noise maps, the same for every cell and both maps so they compare: each NSIDE 256 pixel averages
                # about 36 CpGs, so its half-difference is near 0.03 at the 1st and 99th percentiles (globe_wg_noise_maps_n256.npz); a picture's scale only


def wg_layers(out, tex_files, noise_files, seed=20261108):
    """2026-11-08 (author: 'I want to see a layer that we can toggle between for each of the different tools we use with it ... Imagine its the CMB').
    The Globe Viewer's layers of the Sequencing Atlas Globe beyond the level and range pictures (wg_sphere), each the picture of a tool already
    measured at every CpG; nothing new is read or scored here. For each cell: the value codes the viewer's slider reads (<slug>_vm.png, level 0 to 1;
    <slug>_um.png, half-width 0 to HW_MAX; one byte a pixel, 0 = no measured CpG, 1 to 255 the value, every other row and column, as atlas_masks);
    noise map 1, donor halves (<slug>_n1.jpg) and noise map 2, read halves (<slug>_n2.jpg), from the noise tools' maps (stage seq_globe_wg.noise_join,
    globe_wg_noise_maps_n256.npz), on a fixed scale of +/-NZ_LIM; and the null (<slug>_null.jpg): map 2's own values shuffled among the pixels the
    tools use, seeded, one of the shuffles the null test compares against. One mask for the globe (mask.jpg): the sex chromosomes the tools leave out
    (the pixels after the last pixel any map holds; genome order puts X and Y last), the pixels no cell measured, and the pixels used.
    Writes layers.json (what each layer is, its scale, its source)."""
    from stage_v4 import nest_pix2ang
    from scipy.spatial import cKDTree
    import matplotlib.pyplot as plt
    from PIL import Image
    os.makedirs(out, exist_ok=True); NS = 256; NP = 12 * NS * NS; BL = 65535
    th, ph = nest_pix2ang(NS, np.arange(NP)); xyz = np.c_[np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]
    tt = np.pi * (np.arange(HT) + 0.5) / HT; pp = 2 * np.pi * (np.arange(W) + 0.5) / W; T, P = np.meshgrid(tt, pp, indexing="ij")
    look = cKDTree(xyz).query(np.c_[(np.sin(T) * np.cos(P)).ravel(), (np.sin(T) * np.sin(P)).ravel(), np.cos(T).ravel()])[1]
    NC = plt.get_cmap("RdBu_r")
    def tex(m, path):
        v = m[look].reshape(HT, W); rgb = (NC(np.clip((np.nan_to_num(v, nan=0) + NZ_LIM) / (2 * NZ_LIM), 0, 1))[..., :3] * 255).astype(np.uint8)
        rgb[~np.isfinite(v)] = GREY; Image.fromarray(rgb).save(path, quality=86, optimize=True)
    def code(m, hi, path):
        v = m[look].reshape(HT, W); c = np.where(np.isfinite(v), 1 + np.round(np.clip(v / hi, 0, 1) * 254), 0).astype(np.uint8)
        Image.fromarray(c[::2, ::2]).save(path, optimize=True)
    dec = lambda a: np.where(a == BL, np.nan, a.astype("float64") / 1e4)
    N = {}
    for f in noise_files:
        Z = np.load(f); N.update({k: f for k in Z.files})
    have = np.zeros(NP, bool)
    for k, f in N.items(): have |= np.isfinite(np.load(f)[k])
    last = int(np.flatnonzero(have).max()); sex = np.zeros(NP, bool); sex[last + 1:] = True
    rng = np.random.default_rng(seed); cells = []
    for f in tex_files:
        Z = np.load(f)
        for c in sorted({k.split("|")[0] for k in Z.files if k.endswith("|mu")}):
            s = slug(c); mu, lo, hi = dec(Z[f"{c}|mu"]), dec(Z[f"{c}|lo"]), dec(Z[f"{c}|hi"]); assert len(mu) == NP, (c, len(mu))
            code(mu, 1.0, os.path.join(out, s + "_vm.png")); code((hi - lo) / 2, HW_MAX, os.path.join(out, s + "_um.png")); row = dict(cell=c, slug=s, slider=True)
            for k, name in ((1, "map1 donor halves"), (2, "map2 read halves")):
                key = f"{c}|{name}"
                if key not in N: continue
                m = np.load(N[key])[key].astype("float64"); tex(m, os.path.join(out, f"{s}_n{k}.jpg")); row[f"n{k}"] = True
                if k == 2:
                    g = np.isfinite(m) & ~sex; sh = np.full(NP, np.nan); sh[g] = rng.permutation(m[g]); tex(sh, os.path.join(out, s + "_null.jpg")); row["null"] = True
            cells.append(row); print(c, {k: v for k, v in row.items() if k not in ("cell", "slug")}, flush=True)
    MK = np.full(NP, 0.5); MK[~have] = 0.0; MK[sex] = 1.0
    v = MK[look].reshape(HT, W); rgb = np.zeros((HT, W, 3), np.uint8); rgb[v == 0.5] = (70, 92, 120); rgb[v == 0.0] = GREY; rgb[v == 1.0] = (235, 170, 60)
    Image.fromarray(rgb).save(os.path.join(out, "mask.jpg"), quality=90, optimize=True)
    rep = dict(atlas="IAM Sequencing Atlas at every CpG (the Sequencing Atlas Globe)", nside=NS, order="NESTED", pixels=NP, noise_scale=[-NZ_LIM, NZ_LIM],
               value_code="one byte a pixel: 0 no measured CpG, 1 + round(value / top * 254), every other row and column", value_top=1.0, halfwidth_top=HW_MAX,
               mask=dict(sex_chromosome_pixels=int(sex.sum()), unmeasured_pixels=int((~have & ~sex).sum()), used_pixels=int((have & ~sex).sum()), first_sex_pixel=last + 1),
               null=dict(rule="map 2's own values shuffled among the pixels the tools use (sex chromosomes and unmeasured pixels left out)", seed=seed),
               source="run_sample.py --build seq-wg-layers (chain/MethylPhys_Interface/reference_globes.py wg_layers), from the published textures and the noise tools' maps",
               cells=cells)
    json.dump(rep, open(os.path.join(out, "layers.json"), "w"), indent=1); print("whole-genome layers:", len(cells), "cells; mask", rep["mask"]); return rep
