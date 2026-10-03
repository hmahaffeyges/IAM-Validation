#!/usr/bin/env python3
# Toolkit: not yet wired into chain v3; enters the chain at commissioning with its own pre-registered check.  (SOP v3 section 2b, stage 11b; chain/TOOLKIT.md)
"""toolkit_surface_brightness.py - stage 11b, surface brightness: a 95 % interval on a reading from the atlas's per-CpG
brightness posteriors (mean, sd, ci_lo, ci_hi per CpG, one CSV inside each per-class archive in
atlas/iamatlas_class_archives/*_REBUILD.tar.xz), propagated through A = H(beta_mean) / H_min by Monte Carlo over the scored loci.

Extracted unchanged on 2026-10-03 from the retired v1 clinical conductor (git history, commit a7588732), where it was the
"brightness CI" step of the class-era Stage 4; that conductor is archived privately. It reads class-era inputs
(class markers, H_min by class) that chain v3 does not produce: at commissioning it is rewired to a v3 per-cell reading
and its pre-registered check fixes which loci and which floor it uses.
"""
import csv
from pathlib import Path

import numpy as np
import pandas as pd

ARCHIVES = Path(__file__).resolve().parent.parent / "atlas" / "iamatlas_class_archives"


def _shannon_bit(b: float) -> float:
    """Binary Shannon entropy of a single beta value, in bits (NaN/edge-safe)."""
    eps = 1e-9
    b = min(max(float(b), eps), 1.0 - eps)
    return float(-b * np.log2(b) - (1.0 - b) * np.log2(1.0 - b))


def _load_class_brightness_sd(brightness_archives_dir):
    """Return {class_name: {cpg_id: sd}} from the 8 per-class brightness CSVs.
    The CSVs live inside per-class .tar.xz archives (cpg_id,class,mean,sd,ci_lo,ci_hi)."""
    import tarfile, io
    arch_dir = Path(brightness_archives_dir)
    out = {}
    for tar_path in sorted(arch_dir.glob("*_REBUILD.tar.xz")):
        cls = tar_path.name.split("_v0_1_REBUILD")[0]
        try:
            with tarfile.open(tar_path, "r:xz") as tf:
                member = next((m for m in tf.getmembers()
                               if m.name.endswith("_brightness.csv")), None)
                if member is None:
                    continue
                raw = tf.extractfile(member).read().decode("utf-8", "replace")
        except Exception:
            continue
        sd_map = {}
        rdr = csv.DictReader(io.StringIO(raw))
        for row in rdr:
            cpg = row.get("cpg_id")
            try:
                sd_map[cpg] = float(row.get("sd", "nan"))
            except (TypeError, ValueError):
                continue
        if sd_map:
            out[cls] = sd_map
    return out


def attach_brightness_ci(stage4_output, a_score_loci_path, brightness_archives_dir,
                         patient_beta, n_mc=200, seed=0):
    """Attach A_ci_lo / A_ci_hi (95%) to every scored class and cell-type A-score.
    The CI is computed over the SAME §41 markers the A-score used (handed through from
    stage_4 as class_markers / ct_markers), so the interval and the score can never diverge.
    Mutates stage4_output in place and also returns it."""
    rng = np.random.default_rng(seed)
    class_markers = stage4_output.get("class_markers", {}) or {}
    ct_markers = stage4_output.get("ct_markers", {}) or {}
    ct_to_class = stage4_output.get("celltype_to_class", {})
    h_min_by_class = stage4_output.get("h_min_by_class", {})
    bsd = _load_class_brightness_sd(brightness_archives_dir)

    if isinstance(patient_beta, pd.Series):
        beta = patient_beta
    else:
        beta = pd.Series(patient_beta)

    def _ci_for(loci_list, cls):
        h_min = h_min_by_class.get(cls)
        if not loci_list or h_min in (None, 0):
            return None, None
        present = [c for c in loci_list if c in beta.index]
        if not present:
            return None, None
        vals = beta.loc[present].astype(float).values
        beta_mean = float(np.nanmean(vals))
        sd_map = bsd.get(cls, {})
        sds = np.array([sd_map.get(c, np.nan) for c in present], dtype=float)
        sds = sds[~np.isnan(sds)]
        if sds.size == 0:
            return None, None
        # standard error of the panel mean under the atlas posterior SD at these loci
        se = float(np.sqrt(np.mean(sds ** 2) / len(present)))
        draws = beta_mean + rng.normal(0.0, se, size=n_mc)
        a_draws = np.array([_shannon_bit(b) / h_min for b in draws])
        return float(np.percentile(a_draws, 2.5)), float(np.percentile(a_draws, 97.5))

    for cls, rec in stage4_output.get("class_ascores", {}).items():
        if isinstance(rec, dict) and rec.get("A") is not None:
            lo, hi = _ci_for(class_markers.get(cls), cls)
            rec["A_ci_lo"], rec["A_ci_hi"] = lo, hi
            rec["ci_method"] = "brightness_posterior_mc_95"

    for ct, rec in stage4_output.get("celltype_ascores", {}).items():
        if isinstance(rec, dict) and rec.get("A") is not None:
            cls = rec.get("class") or ct_to_class.get(ct)
            lo, hi = _ci_for(ct_markers.get(ct), cls)
            rec["A_ci_lo"], rec["A_ci_hi"] = lo, hi
            rec["ci_method"] = "brightness_posterior_mc_95"

    return stage4_output
