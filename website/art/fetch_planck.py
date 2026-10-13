"""The measured microwave sky for the front-page art: Planck 2018 (PR3) SMICA CMB temperature map, COM_CMB_IQU-smica_2048_R3.00_full.fits,
from the NASA/IPAC Infrared Science Archive mirror of the Planck Legacy Archive. Downloads it (2.0 GB, not kept in the repository), records its
size and SHA-256, reads the intensity field (K_CMB), converts to microkelvin, and reduces it from NSIDE 2048 to NSIDE 256 (the atlas art's
resolution) by averaging, in RING order. Writes website/art/data/planck_smica_pr3_nside256.npz and its provenance JSON.
Credit: ESA and the Planck Collaboration; Planck 2018 results IV, A&A 641, A4 (2020). Usage: python3 website/art/fetch_planck.py WORKDIR"""
import os, sys, json, hashlib, urllib.request, numpy as np
os.environ.setdefault("ASTROPY_CONFIGDIR", os.path.join(os.getcwd(), ".astro")); os.makedirs(os.environ["ASTROPY_CONFIGDIR"], exist_ok=True)
import healpy as hp
URL = "https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/component-maps/cmb/COM_CMB_IQU-smica_2048_R3.00_full.fits"
HERE = os.path.dirname(os.path.abspath(__file__)); W = sys.argv[1]; F = os.path.join(W, os.path.basename(URL))
if not os.path.exists(F) or os.path.getsize(F) != 2013312960:
    urllib.request.urlretrieve(URL, F + ".part"); os.replace(F + ".part", F)
h = hashlib.sha256()
with open(F, "rb") as f:
    for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
m, hdr = hp.read_map(F, field=0, h=True, nest=False, dtype=np.float64); H = dict(hdr)
m256 = hp.ud_grade(m, 256, order_in="RING", order_out="RING") * 1e6   # K_CMB -> microkelvin
os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
np.savez_compressed(os.path.join(HERE, "data", "planck_smica_pr3_nside256.npz"), map=m256.astype(np.float32), nside=256, order="RING", unit="uK_CMB")
prov = {"source": URL, "bytes": os.path.getsize(F), "sha256": h.hexdigest(), "field": "I_STOKES", "unit_in": H.get("TUNIT1"),
        "ordering_in": H.get("ORDERING"), "nside_in": H.get("NSIDE"), "reduced_to": "NSIDE 256, ring, mean of child pixels, microkelvin",
        "credit": "ESA and the Planck Collaboration; Planck 2018 results IV, A&A 641, A4 (2020)", "rms_uK": round(float(np.std(m256)), 2)}
json.dump(prov, open(os.path.join(HERE, "data", "planck_smica_pr3_nside256.provenance.json"), "w"), indent=1); print(prov)
