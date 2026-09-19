#!/usr/bin/env python3
"""render_cosmic_methylome.py — Generate 8-panel Mollweide CMM for one Hannum HC.

Uses the canonical HEALPix mapping (NSIDE=128, 196608 pixels, 483092 CpGs annotated).
For each architectural class, computes patient β departure z-score (β - HC_mean)/HC_sd,
projects onto the HEALPix grid, renders Mollweide.

Output: cosmic_methylome_example.png (the example patient's personal CMM)
"""
import json, sys
import numpy as np
import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RUNTIME = Path("/home/claude/IAM-Validation/Biological_Physics/atlas_vault/walther_clinical_runtime")
HEALPIX_NPY = "/home/claude/IAM-Validation/Biological_Physics/atlas_vault/IAMAtlas_v0_1/healpix_mapping/iamatlas_cpg_to_healpix_nside128.npy"
ATLAS_CSV = "/home/claude/iamatlas_rebuild/IAMAtlasREBUILD.csv"

CLASSES = ["stem_pluri","stem_adult","stromal","progenitor","cycling","secretory","immune","terminal"]
NSIDE = 128
NPIX = hp.nside2npix(NSIDE)  # 196608

# Load the example patient
with open("healpix_example_sample.json") as f:
    example = json.load(f)
patient_betas = example["betas"]  # {cpg: beta} dict, ~13K entries
print(f"Example patient: gsm={example['gsm']}, age={example['age']:.0f}, n_betas={len(patient_betas)}")

# Load atlas CpG list + per-class means/SDs (streaming)
print("\nLoading atlas CpGs + class means/sds (streaming)...")
import csv
class_mean_cols = {}
class_sd_cols = {}
atlas_cpgs = []
class_means = {cls: [] for cls in CLASSES}
class_sds = {cls: [] for cls in CLASSES}
with open(ATLAS_CSV) as f:
    rdr = csv.reader(f)
    header = next(rdr)
    idx = {n: i for i, n in enumerate(header)}
    cpg_idx = idx.get("cpg_id") or idx.get("cpg") or 0
    for cls in CLASSES:
        class_mean_cols[cls] = idx.get(f"{cls}_mean")
        class_sd_cols[cls] = idx.get(f"{cls}_sd")
    n_read = 0
    for row in rdr:
        atlas_cpgs.append(row[cpg_idx])
        for cls in CLASSES:
            mc = class_mean_cols[cls]
            sc = class_sd_cols[cls]
            try:
                class_means[cls].append(float(row[mc]) if mc is not None and row[mc] else np.nan)
                class_sds[cls].append(float(row[sc]) if sc is not None and row[sc] else np.nan)
            except (ValueError, IndexError):
                class_means[cls].append(np.nan)
                class_sds[cls].append(np.nan)
        n_read += 1
        if n_read % 100000 == 0:
            print(f"  {n_read} CpGs read...")
print(f"  Total CpGs: {len(atlas_cpgs)}")

# Convert to arrays
for cls in CLASSES:
    class_means[cls] = np.array(class_means[cls])
    class_sds[cls] = np.array(class_sds[cls])

# Load HEALPix mapping
print("\nLoading HEALPix CpG → pixel mapping...")
healpix_map = np.load(HEALPIX_NPY)
print(f"  mapping shape: {healpix_map.shape}")
print(f"  align with atlas: assuming order matches IAMAtlas REBUILD CpG order")

# For each CpG in atlas, get patient β and class means/sds
print("\nBuilding patient β vector aligned to atlas...")
patient_beta_arr = np.array([patient_betas.get(c, np.nan) for c in atlas_cpgs])
print(f"  patient β: {np.sum(~np.isnan(patient_beta_arr))} valid of {len(patient_beta_arr)}")

# Render 8-panel CMM
print("\nRendering 8-panel Cosmic Methylome...")
fig = plt.figure(figsize=(20, 10), facecolor="black")
fig.suptitle(
    f"Personal Cosmic Microwave Methylome\n"
    f"Patient {example['gsm']} (HC, age {example['age']:.0f}, GSE40279 Hannum) — z-score departure from IAMAtlas HC mean",
    color="white", fontsize=14, y=0.97
)

for panel_idx, cls in enumerate(CLASSES):
    # Compute z-score departure: (patient β - HC mean) / HC sd
    mean = class_means[cls]
    sd = class_sds[cls]
    # Only valid where patient + HC ref are both valid
    z = np.full(len(atlas_cpgs), np.nan)
    valid = (~np.isnan(patient_beta_arr)) & (~np.isnan(mean)) & (~np.isnan(sd)) & (sd > 1e-6)
    z[valid] = (patient_beta_arr[valid] - mean[valid]) / sd[valid]

    # Project onto HEALPix grid
    grid = np.full(NPIX, np.nan)
    for cpg_i in range(len(atlas_cpgs)):
        pix = healpix_map[cpg_i]
        if pix < NPIX - 1 and not np.isnan(z[cpg_i]):  # skip sentinel
            # If multiple CpGs land on same pixel, average them
            if np.isnan(grid[pix]):
                grid[pix] = z[cpg_i]
            else:
                grid[pix] = (grid[pix] + z[cpg_i]) / 2

    n_pixels_lit = np.sum(~np.isnan(grid))
    z_clipped = np.where(np.isnan(grid), hp.UNSEEN, np.clip(grid, -3, 3))

    # Render
    ax = fig.add_subplot(2, 4, panel_idx + 1)
    hp.mollview(z_clipped, sub=(2, 4, panel_idx + 1), title=f"{cls}", coord="C",
                cmap="RdBu_r", min=-3, max=3, unit="z", cbar=True, notext=True,
                bgcolor="black", badcolor="black", margins=(0, 0.02, 0, 0.04))
    # Add panel info
    plt.gca().text(0.02, 0.02, f"n_pix lit: {n_pixels_lit}",
                   transform=plt.gca().transAxes, color="white",
                   fontsize=8, va="bottom", ha="left")

plt.savefig("cosmic_methylome_example.png", dpi=130, facecolor="black",
            bbox_inches="tight", pad_inches=0.2)
plt.close()
print(f"  Saved cosmic_methylome_example.png")

# Also save numerical departure summary
departure_summary = {}
for cls in CLASSES:
    mean = class_means[cls]
    sd = class_sds[cls]
    z = np.full(len(atlas_cpgs), np.nan)
    valid = (~np.isnan(patient_beta_arr)) & (~np.isnan(mean)) & (~np.isnan(sd)) & (sd > 1e-6)
    z[valid] = (patient_beta_arr[valid] - mean[valid]) / sd[valid]
    z_valid = z[~np.isnan(z)]
    if len(z_valid) > 0:
        departure_summary[cls] = {
            "n_cpgs_valid": int(len(z_valid)),
            "mean_z": float(np.mean(z_valid)),
            "median_z": float(np.median(z_valid)),
            "abs_z_p95": float(np.percentile(np.abs(z_valid), 95)),
            "n_cpgs_z_above_2": int(np.sum(np.abs(z_valid) > 2)),
            "n_cpgs_z_above_3": int(np.sum(np.abs(z_valid) > 3)),
        }
with open("cosmic_methylome_z_summary.json", "w") as f:
    json.dump(departure_summary, f, indent=2)
print(f"  Saved cosmic_methylome_z_summary.json")
print("\nPer-class z-departure summary:")
for cls, s in departure_summary.items():
    print(f"  {cls:12s}: n_valid={s['n_cpgs_valid']:5d}, "
          f"mean_z={s['mean_z']:+.3f}, "
          f"|z|>3 count={s['n_cpgs_z_above_3']}")
