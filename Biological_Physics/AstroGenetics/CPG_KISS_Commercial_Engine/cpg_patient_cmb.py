#!/usr/bin/env python3
"""
cpg_patient_cmb.py — Stage 4.6 patient Cosmic Methylome Background (the real one).

For each architectural class, the patient's beta is compared CpG-by-CpG to the class healthy
reference and standardized by the atlas's per-CpG spread:

        z(CpG) = (beta_patient - mu_class) / max(sd_class, SD_FLOOR)

sd_class is the atlas per-CpG standard deviation, which scales with locus lability (small at
locked loci, larger at labile mu~0.5 loci) — i.e. how much a healthy person varies at that CpG.
That makes z a departure relative to healthy biological variation, not relative to reference
precision.

A class is ASSESSABLE from a whole-blood draw only if the patient's blood actually contains that
architecture. When it does not (stem, secretory, cycling, ... are not in blood), the whole panel
reads as a uniform offset and is labeled reference-only rather than shown as a patient finding.
Assessability is self-determined from the data: median|z| below ASSESS_MAX => assessable.

Pixel mapping is atlas row order (pixel = floor(i * npix / n)), provenance-compliant, no external
manifest. Same CpG -> same pixel for patient and reference, which is all the projection needs.
"""
from __future__ import annotations
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import healpy as hp

NSIDE = 128
NPIX = 12 * NSIDE ** 2
SD_FLOOR = 0.005          # guards divide-by-near-zero at ultra-locked loci
ASSESS_MAX = 3.0          # median|z| below this => the class is assessable from this sample
CLASSES = ["stem_pluri","stem_adult","progenitor","stromal","cycling","secretory","immune","terminal"]


def load_atlas_mean_sd(atlas_csv, classes=CLASSES):
    want = ["cpg_id"] + [f"{c}_mean" for c in classes] + [f"{c}_sd" for c in classes]
    a = pd.read_csv(atlas_csv, usecols=lambda c: c in set(want)).set_index("cpg_id")
    return a


def atlas_order_mapping(n, npix=NPIX):
    return (np.arange(n) * npix // n).astype(np.int32)


def compute_patient_cmb(beta, atlas, classes=CLASSES):
    """Return {class: {z, pixval, median_abs_z, assessable}} plus the pixel map."""
    beta = beta.reindex(atlas.index)
    bv = beta.to_numpy(float)
    pix = atlas_order_mapping(len(atlas))
    out = {}
    for c in classes:
        mu = atlas[f"{c}_mean"].to_numpy(float)
        sd = np.maximum(atlas[f"{c}_sd"].to_numpy(float), SD_FLOOR)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = (bv - mu) / sd
        m = np.isfinite(z)
        med = float(np.median(np.abs(z[m]))) if m.any() else float("nan")
        sums = np.bincount(pix[m], weights=z[m], minlength=NPIX)
        cnts = np.bincount(pix[m], minlength=NPIX)
        pv = np.full(NPIX, hp.UNSEEN); nz = cnts > 0; pv[nz] = sums[nz] / cnts[nz]
        out[c] = dict(z=z, pixval=pv, median_abs_z=med, assessable=(med < ASSESS_MAX))
    return out, pix


def render_patient_cmb(cmb, out_path, patient_id="patient", zlim=3.0):
    fig, axes = plt.subplots(2, 4, figsize=(20, 9)); fig.patch.set_facecolor("black")
    for ax, c in zip(axes.ravel(), CLASSES):
        d = cmb[c]; assess = d["assessable"]
        cmap = "RdBu_r" if assess else "Greys"
        tag = f"{c}  (median|z|={d['median_abs_z']:.2f})" if assess \
              else f"{c}  — reference-only (not in blood)"
        plt.axes(ax)
        hp.mollview(d["pixval"], fig=fig.number, hold=True, title=tag, cmap=cmap,
                    min=-zlim, max=zlim, cbar=False, bgcolor="black", notext=True)
        if not assess:
            ax.text(0.5, 0.5, "reference-only", transform=ax.transAxes, ha="center",
                    va="center", color="#888", fontsize=11, alpha=0.7)
    present = [c for c in CLASSES if cmb[c]["assessable"]]
    fig.suptitle(f"Personal Cosmic Methylome — per-class departure z vs healthy "
                 f"(assessable from blood: {', '.join(present)})  ·  {patient_id}",
                 color="white", fontsize=14)
    fig.savefig(out_path, dpi=90, facecolor="black", bbox_inches="tight"); plt.close(fig)
    return out_path, present


if __name__ == "__main__":
    import pickle, sys
    ATLAS = "/home/claude/work/FILES FOR AI/CPG_CMB_v5/IAM_Atlas/IAMAtlasREBUILD.csv"
    cache = pickle.load(open("/home/claude/work/FILES FOR AI/CPG_CMB_v5/TEST_DATA/betas_cache.pkl", "rb"))
    beta = cache["GSM1051525"]
    atlas = load_atlas_mean_sd(ATLAS)
    cmb, pix = compute_patient_cmb(beta, atlas)
    out, present = render_patient_cmb(cmb, "/mnt/user-data/outputs/CPG_CMB_vKISS/patient_cosmic_methylome_RA.png", "RA · GSM1051525")
    print("rendered:", out)
    print("assessable from blood:", present)
    for c in CLASSES:
        print(f"  {c:12s} median|z|={cmb[c]['median_abs_z']:.3f}  {'ASSESSABLE' if cmb[c]['assessable'] else 'reference-only'}")
