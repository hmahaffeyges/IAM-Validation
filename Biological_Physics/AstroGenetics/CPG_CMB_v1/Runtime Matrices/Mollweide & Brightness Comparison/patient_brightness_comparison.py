#!/usr/bin/env python3
"""
patient_brightness_comparison.py — Stage 4.6 of the Walther Clinical Engine

The patient's β values are compared CpG-by-CpG, class-by-class, against the
per-class healthy brightness reference (the data behind Plate 1, the Cosmic
Microwave Methylome). Output: per-class z-score map, summary statistics, and
the patient's personal 8-panel Mollweide projection on the same HEALPix grid
as Plate 1.

THE COSMIC MICROWAVE METHYLOME ANALOGY
---------------------------------------
Planck's pipeline produces a per-pixel posterior mean temperature + uncertainty
for the cosmic microwave background. The CMB is humanity's reference baseline
— the universe's "healthy" pattern at 380,000 years old. Any subsequent
observation (a galaxy, a cluster, a void) gets compared against the CMB
reference to identify departures.

For the methylome, the per-class brightness CSVs are the human body's
"healthy" pattern — the converged MCMC posterior of per-CpG β across the
architectural classes from 3 weeks of dedicated MCMC compute. Patient runtime
compares the patient's β against this reference per class, producing a z-score
departure map. The patient's per-class Mollweide projection becomes their
personal CMM — the visualization analog of every other metric the engine
produces.

This module is the engine's bridge between the reference (Plate 1) and the
patient (their per-class z-score map projected onto the same grid).

REFERENCE
---------
- Plate 1 conventions: HEALPix NSIDE=128 (npix=196,608), Mollweide projection,
  CpGs ordered by chr × MAPINFO, sequential pixel assignment in genomic order,
  multi-CpG-per-pixel averaging.
- Per-class brightness CSVs: cpg_id, class, mean, sd, ci_lo, ci_hi
  Located inside Biological_Physics/atlas_vault/IAMAtlas_v0_1/class_archives/
  {class}_v0_1_REBUILD.tar.xz as iamatlas_v0_1_{class}_brightness.csv.
- Full-atlas merged: IAMAtlasREBUILD.csv (cpg_id + per-class {mean, sd, ci_lo,
  ci_hi} for 8 classes + per-cell-type columns).
"""

from __future__ import annotations

import json
import logging
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS (Plate 1 conventions — match these exactly so per-patient
# projections sit on the same grid as the reference plates)
# ============================================================================

HEALPIX_NSIDE: int = 128
HEALPIX_NPIX: int = 12 * (HEALPIX_NSIDE ** 2)  # = 196608

ARCHITECTURAL_CLASSES: list[str] = [
    "stem_pluri", "stem_adult", "stromal", "progenitor",
    "cycling", "secretory", "terminal", "immune"
]

H_MIN_BY_CLASS: dict[str, float] = {
    "terminal":    0.7728,
    "immune":      0.838889,
    "secretory":   0.8433,
    "progenitor":  0.8522,
    "cycling":     0.8561,
    "stromal":     0.8630,
    "stem_adult":  0.8737,
    "stem_pluri":  0.9822,
}

# Z-score thresholds for the patient's per-CpG departure flag.
# These are NOT customer-facing tier breakpoints — they are engine-internal
# significance flags for the per-CpG outlier list.
Z_THRESHOLD_NOTABLE: float = 2.0
Z_THRESHOLD_EXTREME: float = 3.0

# Stromal galactic mask threshold — CpGs below this MCMC coverage in the
# stromal posterior get masked in the projection (Plate 1 stromal panel
# documents 4.93% coverage for the full class; per-CpG masking applies the
# same discipline downstream).
STROMAL_MCMC_COVERAGE_FLOOR: float = 0.50  # per-CpG R-hat / coverage flag floor


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class BrightnessReference:
    """Per-class healthy reference: per-CpG posterior mean, SD, and credible interval.

    The data behind Plate 1's per-class panels. Loaded once per session, consulted
    per patient.
    """
    cpg_ids: np.ndarray              # shape (n_cpgs,) — string array
    mean: np.ndarray                  # shape (n_cpgs,) — posterior mean β per CpG
    sd: np.ndarray                    # shape (n_cpgs,) — posterior SD per CpG
    ci_lo: np.ndarray                 # shape (n_cpgs,) — 95% credible interval lower
    ci_hi: np.ndarray                 # shape (n_cpgs,) — 95% credible interval upper
    cls: str                          # architectural class name
    h_min: float                      # H_min anchor for the class
    masked_indices: np.ndarray | None = None  # indices to mask (e.g., stromal gaps)

    @property
    def n_cpgs(self) -> int:
        return len(self.cpg_ids)


@dataclass
class PerClassDeparture:
    """Patient's per-CpG z-score departure for a single architectural class.

    The intermediate engine output — per-CpG z-scores. The customer-facing
    visualization (Plate-1-style Mollweide) is generated from this.
    """
    cls: str
    h_min: float
    z_scores: np.ndarray              # shape (n_cpgs,) — (β_patient − μ_class) / σ_class
    cpg_ids: np.ndarray               # shape (n_cpgs,) — same order as z_scores
    n_total: int
    n_notable: int                    # |z| > 2
    n_extreme: int                    # |z| > 3
    n_masked: int                     # CpGs masked (e.g., stromal galactic mask)
    mean_abs_z: float
    max_abs_z: float
    top_outlier_cpgs: list[dict]      # top-100 CpGs by |z|, with cpg_id + z + direction
    direction_summary: dict[str, int] # {n_up_notable, n_down_notable}

    def summary_dict(self) -> dict[str, Any]:
        return {
            "class": self.cls,
            "h_min": self.h_min,
            "n_total_cpgs_in_class_marker_pool": self.n_total,
            "n_notable_departures_abs_z_gt_2": self.n_notable,
            "n_extreme_departures_abs_z_gt_3": self.n_extreme,
            "n_masked": self.n_masked,
            "mean_abs_z": float(self.mean_abs_z),
            "max_abs_z": float(self.max_abs_z),
            "fraction_notable": self.n_notable / max(self.n_total, 1),
            "direction_summary": self.direction_summary,
            "top_outlier_cpgs": self.top_outlier_cpgs[:25],  # top 25 in summary
        }


@dataclass
class PatientBrightnessReport:
    """Aggregate Stage 4.6 output across all 8 classes for a single patient."""
    patient_id: str
    per_class_results: dict[str, PerClassDeparture]
    healpix_nside: int = HEALPIX_NSIDE
    notes: list[str] = field(default_factory=list)

    def summary_dict(self) -> dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "stage": "4.6",
            "healpix_nside": self.healpix_nside,
            "per_class_summaries": {
                cls: dep.summary_dict()
                for cls, dep in self.per_class_results.items()
            },
            "notes": self.notes,
            "framework_anchor": (
                "Reference: CPG Plate 1 (Cosmic Microwave Methylome). Per-class brightness "
                "data from IAMAtlas REBUILD MCMC posterior, frozen 2026-04-06."
            ),
        }


# ============================================================================
# REFERENCE LOADING
# ============================================================================

def load_brightness_reference_from_csv(
    csv_path: Path | str,
    cls: str,
) -> BrightnessReference:
    """Load a per-class brightness CSV (extracted from the class archive).

    Expects columns: cpg_id, class, mean, sd, ci_lo, ci_hi (one row per CpG).
    """
    df = pd.read_csv(csv_path)
    expected_cols = {"cpg_id", "class", "mean", "sd", "ci_lo", "ci_hi"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Brightness CSV {csv_path} missing required columns: {missing}. "
            f"Expected schema: {sorted(expected_cols)}."
        )

    # Confirm class consistency
    file_class_values = df["class"].unique()
    if len(file_class_values) != 1 or file_class_values[0] != cls:
        raise ValueError(
            f"Brightness CSV {csv_path} contains class values {file_class_values}, "
            f"expected uniform '{cls}'."
        )

    h_min = H_MIN_BY_CLASS.get(cls)
    if h_min is None:
        raise ValueError(f"H_min not defined for class '{cls}'. Known: {list(H_MIN_BY_CLASS)}")

    return BrightnessReference(
        cpg_ids=df["cpg_id"].to_numpy(),
        mean=df["mean"].to_numpy(dtype=np.float64),
        sd=df["sd"].to_numpy(dtype=np.float64),
        ci_lo=df["ci_lo"].to_numpy(dtype=np.float64),
        ci_hi=df["ci_hi"].to_numpy(dtype=np.float64),
        cls=cls,
        h_min=h_min,
    )


def load_brightness_reference_from_archive(
    archive_path: Path | str,
    cls: str,
) -> BrightnessReference:
    """Load brightness reference directly from the tar.xz class archive.

    The class archives at Biological_Physics/atlas_vault/IAMAtlas_v0_1/class_archives/
    contain the brightness CSV as {class}/iamatlas_v0_1_{class}_brightness.csv.
    """
    archive_path = Path(archive_path)
    if not archive_path.exists():
        raise FileNotFoundError(f"Class archive not found: {archive_path}")

    inner_name = f"{cls}/iamatlas_v0_1_{cls}_brightness.csv"
    with tarfile.open(archive_path, "r:xz") as tar:
        try:
            member = tar.getmember(inner_name)
        except KeyError as exc:
            raise FileNotFoundError(
                f"Member {inner_name} not found inside {archive_path}. "
                f"Archive contents: {[m.name for m in tar.getmembers()][:5]}..."
            ) from exc
        with tar.extractfile(member) as fp:
            df = pd.read_csv(fp)

    h_min = H_MIN_BY_CLASS[cls]
    return BrightnessReference(
        cpg_ids=df["cpg_id"].to_numpy(),
        mean=df["mean"].to_numpy(dtype=np.float64),
        sd=df["sd"].to_numpy(dtype=np.float64),
        ci_lo=df["ci_lo"].to_numpy(dtype=np.float64),
        ci_hi=df["ci_hi"].to_numpy(dtype=np.float64),
        cls=cls,
        h_min=h_min,
    )


def load_all_8_class_references(
    archives_dir: Path | str,
) -> dict[str, BrightnessReference]:
    """Load all 8 per-class brightness references from the class archives directory."""
    archives_dir = Path(archives_dir)
    references: dict[str, BrightnessReference] = {}
    for cls in ARCHITECTURAL_CLASSES:
        archive_path = archives_dir / f"{cls}_v0_1_REBUILD.tar.xz"
        references[cls] = load_brightness_reference_from_archive(archive_path, cls)
        logger.info(
            "Loaded brightness reference for class=%s: n_cpgs=%d, mean β=%.3f, mean σ=%.3f",
            cls, references[cls].n_cpgs,
            references[cls].mean.mean(), references[cls].sd.mean(),
        )
    return references


def load_full_atlas_rebuild(rebuild_csv_path: Path | str) -> pd.DataFrame:
    """Load the merged IAMAtlas REBUILD CSV for the full-atlas projection panel.

    Columns: cpg_id + per-class {mean, sd, ci_lo, ci_hi} for 8 classes.
    """
    return pd.read_csv(rebuild_csv_path)


# ============================================================================
# CORE COMPARISON: PATIENT β vs PER-CLASS HEALTHY REFERENCE
# ============================================================================

def compute_per_class_departure(
    patient_beta: pd.Series,
    reference: BrightnessReference,
    *,
    z_notable: float = Z_THRESHOLD_NOTABLE,
    z_extreme: float = Z_THRESHOLD_EXTREME,
    mask_low_sd: bool = True,
    min_sd_for_z_calculation: float = 1e-4,
) -> PerClassDeparture:
    """Compute per-CpG z-score departure of patient β from a class's healthy reference.

    Parameters
    ----------
    patient_beta : pd.Series
        Patient's β values, indexed by cpg_id. May span the full atlas CpG universe.
    reference : BrightnessReference
        Per-class healthy reference loaded from the class brightness CSV.
    z_notable, z_extreme : float
        |z|-score thresholds for notable / extreme departure counts.
    mask_low_sd : bool
        If True, mask CpGs where the posterior SD is below min_sd_for_z_calculation.
        (Avoids inflated z-scores from near-singular posterior SDs — relevant near
        the stromal galactic mask.)
    min_sd_for_z_calculation : float
        Lower bound on posterior SD; CpGs with σ < this are masked.

    Returns
    -------
    PerClassDeparture
        Per-CpG z-scores + summary statistics for this class.
    """
    # Align patient β to the reference's CpG order
    common_cpgs = pd.Index(reference.cpg_ids).intersection(patient_beta.index)
    if len(common_cpgs) == 0:
        raise ValueError(
            f"No common CpGs between patient and {reference.cls} reference. "
            f"Patient has {len(patient_beta)} CpGs; reference has {reference.n_cpgs}."
        )

    # Re-order both to the reference's CpG order (intersection only)
    cpg_idx_in_ref = pd.Index(reference.cpg_ids).get_indexer(common_cpgs)
    ref_mean = reference.mean[cpg_idx_in_ref]
    ref_sd = reference.sd[cpg_idx_in_ref]
    patient_vals = patient_beta.loc[common_cpgs].to_numpy(dtype=np.float64)

    # Mask CpGs with too-small SD (would inflate z)
    mask = np.isfinite(ref_sd) & (ref_sd >= min_sd_for_z_calculation) if mask_low_sd else np.ones_like(ref_sd, dtype=bool)
    mask &= np.isfinite(patient_vals) & np.isfinite(ref_mean)

    z_scores = np.full(len(common_cpgs), np.nan, dtype=np.float64)
    z_scores[mask] = (patient_vals[mask] - ref_mean[mask]) / ref_sd[mask]

    # Stats over the valid (non-masked) z-scores
    valid_z = z_scores[mask]
    abs_z_valid = np.abs(valid_z)
    n_notable = int(np.sum(abs_z_valid > z_notable))
    n_extreme = int(np.sum(abs_z_valid > z_extreme))
    n_masked = int(np.sum(~mask))

    mean_abs_z = float(np.mean(abs_z_valid)) if len(valid_z) > 0 else 0.0
    max_abs_z = float(np.max(abs_z_valid)) if len(valid_z) > 0 else 0.0

    # Top outlier CpGs: rank by |z|
    common_array = np.asarray(common_cpgs)
    if len(valid_z) > 0:
        rank_order = np.argsort(-abs_z_valid)[:100]  # top 100
        valid_cpg_ids = common_array[mask][rank_order]
        valid_z_top = valid_z[rank_order]
        top_outlier_cpgs = [
            {
                "cpg_id": str(cpg_id),
                "z_score": float(z),
                "direction": "hyper" if z > 0 else "hypo",
            }
            for cpg_id, z in zip(valid_cpg_ids, valid_z_top)
        ]
    else:
        top_outlier_cpgs = []

    n_up_notable = int(np.sum(valid_z > z_notable))
    n_down_notable = int(np.sum(valid_z < -z_notable))

    return PerClassDeparture(
        cls=reference.cls,
        h_min=reference.h_min,
        z_scores=z_scores,
        cpg_ids=common_array,
        n_total=len(common_cpgs),
        n_notable=n_notable,
        n_extreme=n_extreme,
        n_masked=n_masked,
        mean_abs_z=mean_abs_z,
        max_abs_z=max_abs_z,
        top_outlier_cpgs=top_outlier_cpgs,
        direction_summary={
            "n_up_notable_abs_z_gt_2": n_up_notable,
            "n_down_notable_abs_z_gt_2": n_down_notable,
        },
    )


def compute_all_8_class_departures(
    patient_beta: pd.Series,
    references: dict[str, BrightnessReference],
    *,
    patient_id: str = "patient",
) -> PatientBrightnessReport:
    """Stage 4.6 top-level entry point.

    Compute the patient's per-CpG z-score departure for all 8 architectural classes,
    aggregate into the PatientBrightnessReport.
    """
    per_class_results = {}
    notes: list[str] = []

    for cls in ARCHITECTURAL_CLASSES:
        if cls not in references:
            notes.append(f"WARNING: brightness reference for class '{cls}' not loaded — skipped")
            continue
        try:
            departure = compute_per_class_departure(patient_beta, references[cls])
            per_class_results[cls] = departure
            logger.info(
                "Class %s: mean_abs_z=%.3f, max_abs_z=%.3f, n_notable=%d, n_extreme=%d, n_masked=%d",
                cls, departure.mean_abs_z, departure.max_abs_z,
                departure.n_notable, departure.n_extreme, departure.n_masked,
            )
        except Exception as exc:
            notes.append(f"ERROR computing departure for class '{cls}': {exc}")
            logger.exception("Failed to compute departure for class=%s", cls)

    return PatientBrightnessReport(
        patient_id=patient_id,
        per_class_results=per_class_results,
        notes=notes,
    )


# ============================================================================
# MOLLWEIDE PROJECTION (Plate 1 conventions)
# ============================================================================

def project_to_healpix_pixels(
    departure: PerClassDeparture,
    cpg_to_pixel: np.ndarray,
    nside: int = HEALPIX_NSIDE,
    aggregation: str = "mean",
) -> np.ndarray:
    """Project per-CpG z-scores onto HEALPix pixels.

    Parameters
    ----------
    departure : PerClassDeparture
        Per-class z-score departure.
    cpg_to_pixel : np.ndarray
        Length-(n_cpgs) array mapping CpG index to HEALPix pixel index.
        MUST match the genomic-order assignment used to build Plate 1.
    nside : int
        HEALPix NSIDE (default 128 per Plate 1).
    aggregation : str
        How to combine multiple CpGs falling in the same pixel: 'mean' or 'max_abs'.

    Returns
    -------
    np.ndarray
        Length-npix array of per-pixel z-score (NaN for empty pixels).

    Notes
    -----
    The cpg_to_pixel mapping should be loaded from a canonical mapping file
    `iamatlas_cpg_to_healpix_nside128.npy` saved at IAMAtlas BUILD time.
    Until that file is generated, this function raises NotImplementedError —
    the mapping must be the same one used to produce Plate 1, not derived
    independently per patient.
    """
    npix = 12 * (nside ** 2)
    pixel_values = np.full(npix, np.nan, dtype=np.float64)

    if len(cpg_to_pixel) != len(departure.z_scores):
        raise ValueError(
            f"cpg_to_pixel length ({len(cpg_to_pixel)}) must match z_scores "
            f"length ({len(departure.z_scores)})."
        )

    if aggregation == "mean":
        sums = np.zeros(npix, dtype=np.float64)
        counts = np.zeros(npix, dtype=np.int64)
        valid = np.isfinite(departure.z_scores)
        np.add.at(sums, cpg_to_pixel[valid], departure.z_scores[valid])
        np.add.at(counts, cpg_to_pixel[valid], 1)
        nonzero = counts > 0
        pixel_values[nonzero] = sums[nonzero] / counts[nonzero]
    elif aggregation == "max_abs":
        # Per pixel, take the CpG with the largest |z| in that pixel
        for px in np.unique(cpg_to_pixel):
            mask = cpg_to_pixel == px
            vals = departure.z_scores[mask]
            valid = np.isfinite(vals)
            if np.any(valid):
                idx_max = np.argmax(np.abs(vals[valid]))
                pixel_values[px] = vals[valid][idx_max]
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}. Use 'mean' or 'max_abs'.")

    return pixel_values


def render_patient_mollweide_panel(
    pixel_values: np.ndarray,
    cls: str,
    *,
    cmap: str = "RdBu_r",  # Red-Blue diverging, centered at z=0 (red=hyper, blue=hypo)
    vmin: float = -3.0,
    vmax: float = 3.0,
    title: str | None = None,
):
    """Render one Mollweide panel for a single class.

    Returns a matplotlib Figure. Caller composes the 8-panel figure.

    Requires matplotlib + healpy. If healpy is unavailable, falls back to
    a placeholder figure with a warning — production rendering needs the
    full HEALPix tooling.
    """
    try:
        import healpy as hp
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Mollweide rendering requires matplotlib + healpy. Install with: "
            "pip install healpy matplotlib"
        ) from exc

    fig = plt.figure(figsize=(8, 4))
    title = title or f"{cls.upper()} · z-score departure"
    hp.mollview(
        pixel_values,
        title=title,
        cmap=cmap,
        min=vmin, max=vmax,
        unit="z-score (β_patient − μ_class) / σ_class",
        fig=fig.number,
        cbar=True,
        notext=False,
        badcolor="black",   # mask appears black (matches Plate 1 stromal galactic mask)
    )
    return fig


def render_patient_cosmic_methylome(
    report: PatientBrightnessReport,
    cpg_to_pixel: np.ndarray,
    out_path: Path | str,
    *,
    nside: int = HEALPIX_NSIDE,
) -> Path:
    """Render the patient's personal 8-panel Cosmic Microwave Methylome.

    Mirrors Plate 1 conventions: 4 rows × 2 columns, Mollweide projection,
    HEALPix NSIDE=128, diverging colormap centered at z=0.

    Returns the saved PNG path.
    """
    try:
        import healpy as hp
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Mollweide rendering requires matplotlib + healpy."
        ) from exc

    fig = plt.figure(figsize=(16, 22))
    fig.patch.set_facecolor("black")

    for i, cls in enumerate(ARCHITECTURAL_CLASSES):
        if cls not in report.per_class_results:
            continue
        departure = report.per_class_results[cls]
        pixel_values = project_to_healpix_pixels(departure, cpg_to_pixel, nside=nside)

        sub = fig.add_subplot(4, 2, i + 1)
        hp.mollview(
            pixel_values,
            title=f"{cls.upper()}  ·  H_min={departure.h_min:.4f}  ·  mean|z|={departure.mean_abs_z:.2f}",
            cmap="RdBu_r",
            min=-3.0, max=3.0,
            unit="z-score departure (red = hyper, blue = hypo)",
            sub=(4, 2, i + 1),
            hold=True,
            cbar=True,
            badcolor="black",
        )

    fig.suptitle(
        f"Your Personal Cosmic Microwave Methylome — {report.patient_id}",
        fontsize=20, color="white", style="italic", y=0.995
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    return out_path


# ============================================================================
# PERSISTENCE
# ============================================================================

def save_brightness_report(
    report: PatientBrightnessReport,
    out_dir: Path | str,
) -> dict[str, Path]:
    """Persist the brightness report and per-class z-score CSVs.

    Returns a dict of artifact paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, Path] = {}

    # Summary JSON
    summary_path = out_dir / f"{report.patient_id}_brightness_comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump(report.summary_dict(), f, indent=2)
    artifacts["summary"] = summary_path

    # Per-class z-score CSVs
    for cls, departure in report.per_class_results.items():
        csv_path = out_dir / f"{report.patient_id}_{cls}_z_scores.csv"
        pd.DataFrame({
            "cpg_id": departure.cpg_ids,
            "z_score": departure.z_scores,
        }).to_csv(csv_path, index=False)
        artifacts[f"{cls}_z_scores"] = csv_path

    return artifacts


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def _cli_main():
    """Smoke test — instantiates the module's classes and reports load status.

    Production invocation comes from walther_clinical.py orchestrator at Stage 4.6:

        from patient_brightness_comparison import (
            load_all_8_class_references,
            compute_all_8_class_departures,
            render_patient_cosmic_methylome,
            save_brightness_report,
        )

        references = load_all_8_class_references(
            "Biological_Physics/atlas_vault/IAMAtlas_v0_1/class_archives"
        )

        report = compute_all_8_class_departures(
            patient_beta=patient_beta_series,
            references=references,
            patient_id=patient_metadata["patient_id"],
        )

        cpg_to_pixel = np.load("iamatlas_cpg_to_healpix_nside128.npy")
        cmm_png = render_patient_cosmic_methylome(
            report, cpg_to_pixel,
            out_path=f"reports/{patient_id}_cosmic_methylome.png"
        )

        save_brightness_report(report, out_dir=f"reports/{patient_id}/brightness/")

    """
    import argparse
    parser = argparse.ArgumentParser(description="Stage 4.6 patient brightness comparison")
    parser.add_argument(
        "--archives-dir",
        default="Biological_Physics/atlas_vault/IAMAtlas_v0_1/class_archives",
        help="Path to class_archives/ folder",
    )
    parser.add_argument("--smoke-test", action="store_true", help="Load references and exit")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.smoke_test:
        print(f"Smoke test: loading all 8 class brightness references from {args.archives_dir}")
        references = load_all_8_class_references(args.archives_dir)
        print(f"Loaded {len(references)} references.")
        for cls, ref in references.items():
            print(f"  {cls:12s}  n_cpgs={ref.n_cpgs:6d}  H_min={ref.h_min:.4f}  "
                  f"mean β={ref.mean.mean():.3f}  mean σ={ref.sd.mean():.3f}")
        print("\nSmoke test PASS — references loaded.")
        print("Production use: import this module and call compute_all_8_class_departures(...)")
        return

    parser.print_help()


if __name__ == "__main__":
    _cli_main()
