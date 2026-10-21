"""The 3D sphere of IAM-Atlas v2.1 for the public page (to-do step 13, 2026-10-21). The same maps as every cell page, wrapped on a globe a reader
can turn: every atlas site keeps its one place on the HEALPix sphere (NSIDE 256, NESTED, genome order; atlas/tools/sky_maps_v21.py, whose
layout, decoding and pixel rule are imported here unchanged), drawn as an equirectangular texture per cell: the atlas's best value (the
Planck parchment scale, 0 to 1) and how sure it is (half-width of the 95 % range, magma, the cell pages' shared 0 to 0.15). Unmeasured
pixels grey. The page (website/build/atlas_pages.py, atlas/sphere.html) turns the texture on a sphere in the browser, no outside library.
A picture of the atlas: nothing is read, fitted or scored from it.
Usage: python3 website/art/make_sphere_3d.py OUT [ATLAS_DIR MANIFEST_DIR]   (needs the Illumina EPIC v1 B4 and 450K v3 manifests for genome order;
when MANIFEST_DIR is not given, methylprep's own downloads are used). Writes OUT/<slug>_v.jpg, OUT/<slug>_u.jpg and OUT/sphere.json."""
import os, sys, re, json, glob, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.abspath(os.path.join(H, "..", ".."))
sys.path[:0] = [os.path.join(R, "Biological_Physics", "MethylPhys", "atlas", "tools"), os.path.join(R, "Biological_Physics", "MethylPhys", "chain")]
import sky_maps_v21 as SM
from stage_v4 import nest_pix2ang
from scipy.spatial import cKDTree
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from PIL import Image
W, HT = 1024, 512; GREY = np.array([0x2a, 0x2d, 0x33], dtype=np.uint8); HW_MAX = 0.15
def slug(c): return re.sub(r"[^a-z0-9]+", "-", c.lower()).strip("-")
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
    alt = {"MethylationEPIC_v-1-0_B4.CoreColumns.csv.gz": ("MethylationEPIC_v-1-0_B4.CoreColumns.csv.gz", "MethylationEPIC_v-1-0_B4.csv.gz", "HumanMethylationEPIC_manifest_v1.csv.gz"),
           "HumanMethylation450k_15017482_v3.csv.gz": ("HumanMethylation450k_15017482_v3.csv.gz",)}   # the EPIC v1 B4 manifest under the names methylprep versions use
    for name in alt:
        f = [os.path.join(dp, n) for n in alt[name] for r in roots for dp, _, fs in os.walk(r) if n in fs and not dp.startswith(d)]
        assert f, f"manifest {name} not found under {roots}; methylprep folder holds: {sorted(glob.glob(os.path.expanduser('~/.methylprep_manifest_files/*')))}"
        if not os.path.exists(os.path.join(d, name)): os.symlink(f[0], os.path.join(d, name))
        print("manifest", name, "from", f[0], flush=True)
    return d
def main(out, atlas, man):
    import pyarrow.parquet as pq
    os.makedirs(out, exist_ok=True)
    ids = pq.read_table(os.path.join(atlas, "sites.parquet")).column("cpg").to_pylist(); order, unplaced = SM.genome_order(man, ids)
    th, ph = nest_pix2ang(SM.NSIDE, np.arange(SM.NPIX)); xyz = np.c_[np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]
    tt = np.pi * (np.arange(HT) + 0.5) / HT; pp = 2 * np.pi * (np.arange(W) + 0.5) / W; T, P = np.meshgrid(tt, pp, indexing="ij")
    look = cKDTree(xyz).query(np.c_[(np.sin(T) * np.cos(P)).ravel(), (np.sin(T) * np.sin(P)).ravel(), np.cos(T).ravel()])[1]
    CM = ListedColormap(np.loadtxt(os.path.join(H, "data", "Planck_Parchment_RGB.txt")) / 255.0); UC = plt.get_cmap("magma"); cells = []
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
    json.dump(dict(atlas="IAM-Atlas v2.1", nside=SM.NSIDE, order="NESTED", pixels=SM.NPIX, sites=len(ids), unplaced=unplaced, texture=[W, HT],
                   value_scale=[0, 1], uncertainty_scale=[0, HW_MAX], source="website/art/make_sphere_3d.py", cells=cells),
              open(os.path.join(out, "sphere.json"), "w"), indent=1)
    print(f"sphere: {len(cells)} cells written to {out}")
if __name__ == "__main__":
    out = sys.argv[1]; atlas = sys.argv[2] if len(sys.argv) > 2 else os.path.join(R, "Biological_Physics", "MethylPhys", "atlas", "v2.1")
    main(out, atlas, sys.argv[3] if len(sys.argv) > 3 else manifests())
