import numpy as np
"""The Globe Explorer's globes beyond the atlas (2026-10-21, to-do step 28; author: "the webpage to show ALL the versions").
Writes website/art/out/explorer/<id>.json (a globe spec, chain/MethylPhys_Interface/globe_viewer.py) and its pictures beside it:
  purified   Met-A's own healthy reference: the purified-array group profiles at its 6,000 neutrophil identity sites
             (Runtime Matrices/Met_A_Floors/blood_composition_EPIC_v1.json), one layer per group, drawn as dots on the atlas's layout
  samples    the report globe of each example sample, from the one command's own globe module (report_globe_v4.build), unchanged
Usage: python3 website/art/make_explorer.py purified
       python3 website/art/make_explorer.py noise NOISE_DIR   (output of atlas/tools/noise_maps_v21.py combine)
       python3 website/art/make_explorer.py sample ID "TITLE" BETAS.csv BUNDLE.json [planted]"""
import os, sys, json, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.abspath(os.path.join(H, "..", ".."))
CH = os.path.join(R, "Biological_Physics", "MethylPhys", "chain"); sys.path[:0] = [CH, os.path.join(CH, "MethylPhys_Interface")]
import report_globe_v4 as G, globe_viewer as V
OUT = os.path.join(H, "out", "explorer"); URL = "../art/explorer"
GROUP = {"NEU": "neutrophils", "MONO": "monocytes", "B": "B cells", "CD4T": "CD4 T cells", "CD8T": "CD8 T cells", "NK": "NK cells", "BASO": "basophils",
         "EOS": "eosinophils", "BAS": "basophils", "TREG": "regulatory T cells", "GRAN": "granulocytes"}

def purified():
    import pyarrow.parquet as pq
    from matplotlib.colors import ListedColormap
    B = json.load(open(os.path.join(CH, "Runtime Matrices", "Met_A_Floors", "blood_composition_EPIC_v1.json")))
    ids = np.array(pq.read_table(os.path.join(G.ATLAS, "sites.parquet")).column("cpg").to_pylist(), dtype=object)
    Z = np.load(G.ORDER); order = Z["order"]; nside = int(Z["nside"]); npix = 12 * nside * nside; look = G._lookup(nside)
    cm = ListedColormap(np.loadtxt(os.path.join(H, "data", "Planck_Parchment_RGB.txt")) / 255.0); L = []
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

def sample(sid, title, betas, bundle=None, planted=False):
    """A sample globe exactly as the one command draws it (2026-10-22, rule 23b-f): runs chain/MethylPhys_Interface/run_sample.py on the
    sample's values with --explorer-spec and only renames the spec; nothing is read or drawn here. bundle is kept so recorded commands run."""
    import subprocess, tempfile
    d = tempfile.mkdtemp(); sp = os.path.join(d, "spec.json")
    rs = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Biological_Physics", "MethylPhys", "chain", "MethylPhys_Interface", "run_sample.py")
    r = subprocess.run([sys.executable, rs, "--betas", betas, "--specimen", "whole blood", "--id", sid, "--out", os.path.join(d, sid + ".html"),
                        "--explorer-spec", sp] + (["--banner", "PLANTED: a constructed example, not a person."] if planted else []), capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(sp): sys.exit(f"run_sample.py did not write the globe ({r.returncode}): {r.stdout[-400:]} {r.stderr[-400:]}")
    s = json.load(open(sp)); s["id"] = sid; s["name"] = title
    if planted: s["about"] = "<b>PLANTED: a constructed example, not a person.</b> A clustered loss was planted in a healthy array. " + s["about"]
    return s

if __name__ == "__main__":
    spec = purified() if sys.argv[1] == "purified" else noise(sys.argv[2]) if sys.argv[1] == "noise" else sample(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], len(sys.argv) > 6)
    os.makedirs(OUT, exist_ok=True); s = V.write_files(spec, OUT, URL)
    json.dump(s, open(os.path.join(OUT, spec["id"] + ".json"), "w"), indent=1); print("wrote", spec["id"], len(s["assets"]), "pictures")
