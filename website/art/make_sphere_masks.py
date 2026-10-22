"""Value pictures for the atlas globe's map layers (2026-10-21; author: "the other options dont even have a slider"). For each of IAM-Atlas
v2.1's 79 cells: the best value (0 to 1) and the half-width of the 95 % range (0 to HW_MAX) at every texture pixel, stored losslessly as
an 8-bit PNG at half the texture's size (code 0: no site; 1 to 255: the value on its scale), so the globe page can keep only the sites above a
chosen value. Laid out by the chain's frozen sky order (chain/Runtime Matrices/Atlas_V21/sky_order_v21.npz, the cell pages' own layout); each
cell's measured sky pixels must equal its globe's count in website/art/out/sphere/sphere.json, or nothing is written.
Usage: python3 website/art/make_sphere_masks.py [OUT=website/art/out/sphere]"""
import os, sys, json, numpy as np, pyarrow.parquet as pq
from PIL import Image
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.dirname(os.path.dirname(H))
sys.path[:0] = [os.path.join(R, "Biological_Physics", "MethylPhys", "chain", "MethylPhys_Interface"), os.path.join(R, "Biological_Physics", "MethylPhys", "atlas", "tools"), H]
import report_globe_v4 as G, sky_maps_v21 as SM, make_sphere_3d as M3
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(H, "out", "sphere"); A = os.path.join(R, "Biological_Physics", "MethylPhys", "atlas", "v2.1")
O = np.load(G.ORDER); order = O["order"]; nside = int(O["nside"]); npix = 12 * nside * nside; look = G._lookup(nside)
S = {c["slug"]: c for c in json.load(open(os.path.join(OUT, "sphere.json")))["cells"]}
def code(v, hi):
    c = np.where(np.isfinite(v), 1 + np.round(np.clip(v / hi, 0, 1) * 254), 0).astype(np.uint8); return c[::2, ::2]
bad = []; n = 0
for k in (1, 2, 3):
    a, lo, hi = (pq.read_table(os.path.join(A, f"{t}_part{k}.parquet")) for t in ("mu", "mu_lo", "mu_hi"))
    for c in a.column_names:
        s = M3.slug(c); mu = SM.decode(a.column(c).to_numpy()); hw = (SM.decode(hi.column(c).to_numpy()) - SM.decode(lo.column(c).to_numpy())) / 2
        sky = G._to_sky(mu[order], npix); vm = sky[look].reshape(G.HT, G.W); um = G._to_sky(hw[order], npix)[look].reshape(G.HT, G.W)
        if s not in S or int(np.isfinite(sky).sum()) != S[s]["pixels_measured"]: bad.append((s, int(np.isfinite(sky).sum()), S.get(s, {}).get("pixels_measured"))); continue   # sky pixels, as the record counts them
        Image.fromarray(code(vm, 1.0)).save(os.path.join(OUT, s + "_vm.png"), optimize=True); Image.fromarray(code(um, M3.HW_MAX)).save(os.path.join(OUT, s + "_um.png"), optimize=True); n += 1
print(f"masks: {n} cells written" + (f"; LAYOUT DIFFERS for {len(bad)}: {bad[:4]}" if bad else ""))
if bad: sys.exit(1)
