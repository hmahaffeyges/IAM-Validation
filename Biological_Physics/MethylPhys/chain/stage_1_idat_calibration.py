"""Stage 1 - IDAT calibration to beta (SOP Stage 1, steps 1.1-1.2 + 1.5).

Turns a single patient's raw IDAT pair into calibrated beta values using the
standard per-sample preprocessing: dye-bias correction + probe-type
normalization (noob). This is SELF-CONTAINED per sample - it uses only the
patient's own on-chip control and out-of-band probes. No cohort, no reference
matrix, no external lookup of any kind enters this step.

Why this exists: the raw decoder (idat_decoder_pure) computes beta = M/(M+U+100)
with no dye-bias or probe-type normalization, which leaves the methylated peak
compressed (~0.85 instead of ~0.95) and inflates every downstream A-score. The
chain's run_pipeline is designed to receive ALREADY-CALIBRATED beta; this module
is the calibration that the SOP specifies and the 2026-06-10 audit flagged as the
outstanding Stage 1.

Array type (450k vs EPIC) is auto-detected from the IDAT probe count, so the
operator drops any IDAT pair and the step adapts.

Dependency: methylprep (already in the Anaconda biology stack). methylprep manages
its own manifest cache - no manifest file needs to be supplied by the operator.
"""

import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd


def _ensure_methylprep():
    """Make sure methylprep is importable; pip-install it once if it isn't.

    Installs into the SAME interpreter that's running the chain (sys.executable),
    so it lands in the active Anaconda env. Raises a clear, actionable error if
    the automatic install can't complete (e.g. no network)."""
    try:
        import methylprep  # noqa: F401
        return
    except ImportError:
        pass
    import subprocess
    import sys
    import importlib
    print("      methylprep not found - installing it now (one-time setup) ...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "methylprep"])
    except Exception as e:
        raise RuntimeError(
            "Stage 1 calibration needs the 'methylprep' package, and the automatic "
            "install failed (often no internet on the run machine). Install it once "
            "by hand and re-run:\n\n    pip install methylprep\n\n"
            f"(auto-install error: {e})")
    importlib.invalidate_caches()
    try:
        import methylprep  # noqa: F401
    except Exception as e:
        msg = str(e)
        if any(s in msg.lower() for s in ("numpy", "binary incompat", "dtype size changed")):
            raise RuntimeError(
                "methylprep installed, but its dependencies conflict with the numpy/pandas "
                "already in this environment. Fastest fix is a clean env for the chain:\n\n"
                "    conda create -n cpg python=3.11 -y\n"
                "    conda activate cpg\n"
                "    pip install methylprep numpy pandas scipy scikit-learn matplotlib\n\n"
                "then run the chain from that env. (Or simply close and re-run once - sometimes "
                "enough if numpy was only loaded earlier in the session.)\n"
                f"(detail: {e})")
        raise RuntimeError(
            "methylprep installed but could not be imported in this session. "
            "Close and re-run the chain once more.\n"
            f"(import error: {e})")
    print("      methylprep installed OK.")


def _detect_array_type(grn_path):
    """Read the IDAT bead-address count and map it to methylprep's array type."""
    from methylprep.files import IdatDataset
    from methylprep.models import Channel
    from methylprep.models.arrays import ArrayType
    d = IdatDataset(str(grn_path), Channel.GREEN)
    n = len(d.probe_means)
    at = ArrayType.from_probe_count(n)
    barcode = getattr(d, "barcode", None)
    return at, str(at), barcode, n


def calibrate_idat_to_beta(grn_path, red_path, array_type=None, verbose=True, mask_detection=True, return_mask=False):
    """Calibrate one IDAT pair to noob-normalized beta.

    Returns (beta_series, meta). beta_series is a pd.Series indexed by IlmnID
    (cgXXXX) of calibrated beta in [0,1]. meta carries barcode + array_type +
    n_cpgs + the calibration method string for the Stage 1 provenance.
    """
    grn_path, red_path = str(grn_path), str(red_path)
    _ensure_methylprep()
    at, at_str, barcode, n_addr = _detect_array_type(grn_path)
    if array_type is not None:
        at_str = array_type
    # methylprep wants its array_type string token; map from the detected enum
    at_token = {"450k": "450k", "epic": "epic", "epic+": "epic+",
                "27k": "27k", "mouse": "mouse"}.get(at_str.lower(), at_str.lower())
    if verbose:
        print(f"      array detected: {at_str} ({n_addr:,} bead addresses, barcode {barcode})")

    import methylprep

    # methylprep pairs by {barcode}_{position}_Grn/Red.idat; stage the pair in an
    # isolated temp dir with a one-row samplesheet so processing is fully per-sample.
    workdir = tempfile.mkdtemp(prefix="cpg_stage1_")
    try:
        bc = barcode or "patient"
        pos = "R01C01"
        g_dst = os.path.join(workdir, f"{bc}_{pos}_Grn.idat")
        r_dst = os.path.join(workdir, f"{bc}_{pos}_Red.idat")
        # decompress if gzipped, else copy
        _stage_idat(grn_path, g_dst)
        _stage_idat(red_path, r_dst)
        sheet = os.path.join(workdir, "samplesheet.csv")
        with open(sheet, "w") as fh:
            fh.write("Sample_Name,Sentrix_ID,Sentrix_Position\n")
            fh.write(f"{bc},{bc},{pos}\n")

        # 2026-09-27: per-probe detection (poobah, against THIS array's negative controls), the control probes and the
        # rs (SNP) probes come out with the betas. Stage 0.4/0.5/0.7 run on these numbers; probes at background are
        # removed before any stage reads the beta (no measurement at background); the SNP probes give the sky its
        # on-array noise term. FINDING_GSE125105_LOW_SIGNAL.md is why.
        import glob as _glob, pickle as _pickle, numpy as _np, pandas as _pd
        methylprep.run_pipeline(
            workdir, array_type=at_token, betas=True, export=True, save_control=True, poobah=True,
            sample_sheet_filepath=sheet)
        proc = _glob.glob(os.path.join(workdir, "**", "*_processed.csv"), recursive=True)[0]
        df = _pd.read_csv(proc, index_col=0)
        bcol = "beta_value" if "beta_value" in df.columns else [c for c in df.columns if "beta" in c.lower()][0]
        pcol = [c for c in df.columns if "poobah" in c.lower()]
        allb = df[bcol].astype(float)
        idx = allb.index.map(str)
        is_rs = _np.array([str(i).startswith("rs") for i in idx]); is_cg = _np.array([str(i).startswith("cg") for i in idx])
        rs = allb[is_rs].dropna()
        if pcol:
            pv = df[pcol[0]].astype(float)
            detected = (pv <= 0.05)                       # poobah p (per probe vs THIS array's negatives) at its own 0.05; the SOP's 0.01 was written for minfi's detectionP, a different statistic - 2026-09-27
            n_cg = int(is_cg.sum()); n_det = int((detected & is_cg).sum())
            beta = allb[is_cg & detected.values].dropna() if mask_detection else allb[is_cg].dropna()   # mask_detection=False is a TEST switch (PROC-INTAKE-01 B3); the chain always masks
            qc = {"detection_available": True, "n_probes": n_cg, "n_detected": n_det, "pct_detected": n_det / max(n_cg, 1),
                  "n_masked": n_cg - n_det}
            detected_mask = _pd.Series(detected.values[is_cg], index=idx[is_cg])   # per cg probe: poobah p <= 0.05 (not written to the bundle)
        else:
            beta = allb[is_cg].dropna(); qc = {"detection_available": False, "n_probes": int(is_cg.sum())}; detected_mask = None
        beta.name = "beta"
        ctrl = {}
        cp = _glob.glob(os.path.join(workdir, "**", "control_probes.pkl"), recursive=True)
        if cp:
            cdf = list(_pickle.load(open(cp[0], "rb")).values())[0]; ct = cdf["Control_Type"].astype(str).str.upper()
            gcol = [c for c in cdf.columns if "green" in c.lower()][0]; rcol = [c for c in cdf.columns if "red" in c.lower()][0]
            def med(t, col):
                x = cdf[ct.str.contains(t, na=False)][col]; return float(x.median()) if len(x) else None
            ctrl = {"bisulfite_conversion_I_median": med("BISULFITE CONVERSION I", gcol), "bisulfite_conversion_II_median": med("BISULFITE CONVERSION II", rcol),
                    "hybridization_G_median": med("HYBRIDIZATION", gcol), "negative_G_median": med("NEGATIVE", gcol), "negative_R_median": med("NEGATIVE", rcol),
                    "non_polymorphic_G_median": med("NON-POLYMORPHIC", gcol), "non_polymorphic_R_median": med("NON-POLYMORPHIC", rcol)}
            if ctrl["non_polymorphic_G_median"] and ctrl["negative_G_median"]:
                ctrl["signal_to_background_G"] = ctrl["non_polymorphic_G_median"] / max(ctrl["negative_G_median"], 1.0)
                ctrl["signal_to_background_R"] = (ctrl["non_polymorphic_R_median"] or 0.0) / max(ctrl["negative_R_median"] or 1.0, 1.0)
        # this array's own noise from its SNP probes: sigma^2(beta) = a + b*beta(1-beta)  (sky denominator)
        snp = None
        b_ = _np.asarray(rs, float); b_ = b_[~_np.isnan(b_)]
        if b_.size >= 20:
            ideal = _np.array([0.0, 0.5, 1.0]); lab = _np.argmin(_np.abs(b_[:, None] - ideal[None, :]), axis=1)
            cen = _np.array([_np.median(b_[lab == k]) if (lab == k).any() else ideal[k] for k in range(3)]); lab = _np.argmin(_np.abs(b_[:, None] - cen[None, :]), axis=1)
            sds = [float(_np.std(b_[lab == k], ddof=1)) if (lab == k).sum() >= 3 else float("nan") for k in range(3)]
            a_ = float(_np.nanmean([sds[0] ** 2, sds[2] ** 2])); bb_ = max(0.0, 4.0 * (sds[1] ** 2 - a_)) if not _np.isnan(sds[1]) else 0.0
            snp = {"a": a_, "b": bb_, "cluster_sd": sds, "n_rs": int(b_.size), "T_offset": float(cen[1] - 0.5), "T_scale": float(cen[2] - cen[0])}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    meta = {
        "barcode": barcode,
        "array_type": at_str,
        "pipeline": f"stage1_noob_{'450K' if at_str=='450k' else at_str.upper()}",   # LESSON-SCALE-01: the tag beta_scale_maps_v1.json is keyed by
        "n_cpgs": int(len(beta)),
        "calibration": "noob (dye-bias + probe-type normalization), per-sample; probes at background (poobah p > 0.05) removed" if mask_detection else "noob; DETECTION MASK WITHHELD (test only)",
        "stage": "SOP Stage 1 steps 1.1-1.2 + 1.5",
        "detection": qc, "controls": ctrl, "snp_noise": snp,
    }
    if return_mask:   # per cg probe poobah detection (bool Series), for run_sample's Stage-1 call rate; off by default so callers that dump meta are unchanged
        meta["_detected_mask"] = detected_mask
    if verbose:
        print(f"      calibrated: {len(beta):,} CpGs (noob, per-sample dye-bias + probe-type norm)")
    return beta, meta


def _stage_idat(src, dst):
    """Copy (or gunzip) an IDAT into the staging dir under the methylprep name."""
    if str(src).endswith(".gz"):
        import gzip
        with gzip.open(src, "rb") as fi, open(dst, "wb") as fo:
            shutil.copyfileobj(fi, fo)
    else:
        shutil.copyfile(src, dst)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("usage: python stage_1_idat_calibration.py <Grn.idat> <Red.idat>")
        sys.exit(1)
    b, m = calibrate_idat_to_beta(sys.argv[1], sys.argv[2])
    print(m)
    print(b.head())
