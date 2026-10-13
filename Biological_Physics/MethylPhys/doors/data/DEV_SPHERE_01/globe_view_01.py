"""DEV-SPHERE-01: the globe viewer and the flat map, for any information-sphere map (a picture, not a reading).

Colors (author, 2026-10-13: pastel, CMB-like, the islands contrasted): a diverging scale like the Planck maps - soft blue for
unmethylated, cream in between, soft coral for methylated - and CpG islands marked in mint over that. Empty regions of the globe
(no site there) are pale grey, never colored as data. The map is smoothed the way sky maps are displayed (count-weighted Gaussian,
2 degrees), so a pixel holding few sites does not shout.
Writes OUT_globe.html (spin and zoom; three.js) and OUT_flat.png (Mollweide, the flat map). Usage: globe_view_01.py MAPS.npz OUT
MAPS.npz: count, mean_b, island (per HEALPix pixel), from sphere_construction_01.maps.
"""
import sys, os, base64, io, numpy as np, healpy as hp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
PASTEL = LinearSegmentedColormap.from_list("pastel_cmb", ["#8fb8de", "#c6dcef", "#f6efdc", "#f6c7b2", "#ec9a86"])
MINT = np.array([0.66, 0.90, 0.79]); EMPTY = np.array([0.93, 0.93, 0.92])
def smooth(m, w, deg=2.0):
    a = hp.smoothing(np.nan_to_num(m) * w, fwhm=np.radians(deg)); b = hp.smoothing(w, fwhm=np.radians(deg))
    out = np.where(b > 1e-3 * b.max(), a / np.maximum(b, 1e-12), np.nan); return out, b
def rgb_image(Z, I, cover):
    img = PASTEL(np.clip(np.nan_to_num(Z, nan=0.5), 0, 1))[..., :3]
    if I is not None:
        k = np.clip((np.nan_to_num(I) - 0.5) / 0.3, 0, 1)[..., None] * 0.85; img = img * (1 - k) + MINT * k
    img[~cover] = EMPTY; return img
def main(npz, out):
    d = np.load(npz); cnt = d["count"]; mb = d["mean_b"]; isl = d["island"] if "island" in d.files else None
    zb, cov = smooth(mb, cnt); zi = smooth(isl, cnt)[0] if isl is not None else None
    covm = cov > 0.02 * cov.max()
    # flat map (Mollweide)
    proj = hp.projector.MollweideProj(xsize=1800)
    f = lambda m: proj.projmap(np.nan_to_num(m, nan=0.0), lambda x, y, z: hp.vec2pix(hp.npix2nside(len(m)), x, y, z))
    Z = f(zb); C = f(covm.astype(float)) > 0.5; I = f(zi) if zi is not None else None
    img = rgb_image(Z, I, C); img[np.isinf(proj.projmap(np.zeros(len(mb)), lambda x, y, z: hp.vec2pix(hp.npix2nside(len(mb)), x, y, z)))] = 1.0
    fig, ax = plt.subplots(figsize=(12, 6.6)); ax.imshow(img, origin="lower"); ax.axis("off")
    ax.set_title("The information sphere (picture, not a reading): blue unmethylated, cream between, coral methylated; mint = CpG islands", fontsize=10)
    fig.savefig(out + "_flat.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    # globe texture (equirectangular), embedded in a self-contained viewer
    proj2 = hp.projector.CartesianProj(xsize=2048, ysize=1024)
    g = lambda m: proj2.projmap(np.nan_to_num(m, nan=0.0), lambda x, y, z: hp.vec2pix(hp.npix2nside(len(m)), x, y, z))
    tex = rgb_image(g(zb), g(zi) if zi is not None else None, g(covm.astype(float)) > 0.5)[::-1]
    buf = io.BytesIO(); plt.imsave(buf, tex, format="png"); b64 = base64.b64encode(buf.getvalue()).decode()
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>The information sphere</title>
<style>body{{margin:0;background:#101318;color:#e9e4d8;font:15px system-ui}}#t{{position:fixed;left:16px;top:12px;max-width:640px}}</style></head>
<body><div id="t"><b>The information sphere</b> - a picture, not a reading. Drag to turn, scroll to zoom.<br>Blue unmethylated, cream between,
coral methylated; mint marks CpG islands; pale grey is empty globe.</div>
<script type="importmap">{{"imports":{{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}}}</script>
<script type="module">
import * as THREE from "three"; import {{ OrbitControls }} from "three/addons/controls/OrbitControls.js";
const r = new THREE.WebGLRenderer({{antialias:true}}); r.setSize(innerWidth, innerHeight); document.body.appendChild(r.domElement);
const s = new THREE.Scene(), c = new THREE.PerspectiveCamera(40, innerWidth/innerHeight, 0.1, 100); c.position.set(0,0,3.4);
const t = new THREE.TextureLoader().load("data:image/png;base64,{b64}"); t.colorSpace = THREE.SRGBColorSpace;
s.add(new THREE.Mesh(new THREE.SphereGeometry(1, 128, 96), new THREE.MeshBasicMaterial({{map:t}})));
const o = new OrbitControls(c, r.domElement); o.enableDamping = true; o.autoRotate = true; o.autoRotateSpeed = 0.6;
addEventListener("resize", () => {{ c.aspect = innerWidth/innerHeight; c.updateProjectionMatrix(); r.setSize(innerWidth, innerHeight); }});
(function a() {{ requestAnimationFrame(a); o.update(); r.render(s, c); }})();
</script></body></html>"""
    open(out + "_globe.html", "w").write(html); print("wrote", out + "_flat.png", out + "_globe.html")
if __name__ == "__main__": main(sys.argv[1], sys.argv[2])
