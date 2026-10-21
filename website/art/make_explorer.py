"""The Globe Explorer's globes beyond the atlas (2026-10-21, to-do step 28; author: "the webpage to show ALL the versions").
Writes website/art/out/explorer/<id>.json (a globe spec, chain/MethylPhys_Interface/globe_viewer.py) and its pictures beside it:
  purified   Met-A's own healthy reference: the purified-array group profiles at its 6,000 neutrophil identity sites
             (Runtime Matrices/Met_A_Floors/blood_composition_EPIC_v1.json), one layer per group, drawn as dots on the atlas's layout
  samples    the report globe of each example sample, from the one command's own globe module (report_globe_v4.build), unchanged
Usage: python3 website/art/make_explorer.py purified
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
        L.append(dict(name=GROUP.get(g, g), kind="image", src="data:image/jpeg;base64," + G._jpg_b64(x),
                      note=f"The purified healthy {GROUP.get(g, g)} profile at Met-A's {int(np.isfinite(v).sum()):,} identity sites placed on the atlas's layout, 0 to 1."))
    spec = V.dedupe(dict(id="purified", name="Met-A's purified-array reference", groups=[dict(name="purified healthy blood groups", layers=L)],
        about=("The reference Met-A and the C-Score read against: purified healthy cell arrays, at Met-A's 6,000 neutrophil identity sites only, "
               "so it is drawn as dots, each at the same place as on the atlas globes. A blood sample is compared with these profiles mixed in its own "
               "proportions. IAM-Atlas v2.1 is a different reference, built from 21 sources over the whole genome; the two agree closely value by value.")))
    return spec

def sample(sid, title, betas, bundle, planted=False):
    import conductor_v3 as C
    b = pd.read_csv(betas, index_col=0).iloc[:, 0]; o = C.run_neutrophil(b, specimen="whole blood")
    sh = {c["cell"]: c["share"] for c in json.load(open(bundle))["cell_shares"]["cells"]}
    r = G.build(b, "whole blood", sh, None, z=o["_cscore_z"], fractions=(o.get("composition") or {}).get("fractions"))
    s = r["spec"]; s["id"] = sid; s["name"] = title
    if planted: s["about"] = "<b>PLANTED: a constructed example, not a person.</b> A clustered loss was planted in a healthy array. " + s["about"]
    return s

if __name__ == "__main__":
    spec = purified() if sys.argv[1] == "purified" else sample(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], len(sys.argv) > 6)
    os.makedirs(OUT, exist_ok=True); s = V.write_files(spec, OUT, URL)
    json.dump(s, open(os.path.join(OUT, spec["id"] + ".json"), "w"), indent=1); print("wrote", spec["id"], len(s["assets"]), "pictures")
