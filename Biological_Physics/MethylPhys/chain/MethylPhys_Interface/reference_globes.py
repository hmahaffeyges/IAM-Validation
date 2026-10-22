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

def atlas_sphere(out, atlas, man, cells_index=None):
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
            cells.append(dict(cell=c, slug=s, pixels_measured=int(np.isfinite(mu).sum()))); print(c, flush=True)
    cells.sort(key=lambda x: x["cell"].lower())
    # the layout must be the cell pages' own: every cell's measured-pixel count equal to website/art/out/cells/index.json
    ix = {c["slug"]: c.get("pixels_measured") for c in json.load(open(cells_index or os.path.join(REPO, "website", "art", "out", "cells", "index.json")))["cells"]}
    bad = [(c["slug"], c["pixels_measured"], ix.get(c["slug"])) for c in cells if ix.get(c["slug"]) != c["pixels_measured"]]
    print(f"layout check against the cell pages: {len(cells) - len(bad)} of {len(cells)} cells identical" + (f"; DIFFER: {bad[:5]}" if bad else ""), flush=True)
    if bad: sys.exit("make_sphere_3d: the layout differs from the cell pages' maps; textures not written as the record")
    json.dump(dict(atlas="IAM-Atlas v2.1", nside=SM.NSIDE, order="NESTED", pixels=SM.NPIX, sites=len(ids), unplaced=unplaced, texture=[W, HT],
                   value_scale=[0, 1], uncertainty_scale=[0, HW_MAX], source="run_sample.py --build atlas-sphere (chain/MethylPhys_Interface/reference_globes.py)", cells=cells),
              open(os.path.join(out, "sphere.json"), "w"), indent=1)
    print(f"sphere: {len(cells)} cells written to {out}")

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
