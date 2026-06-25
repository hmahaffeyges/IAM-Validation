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


def calibrate_idat_to_beta(grn_path, red_path, array_type=None, verbose=True):
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

        betas = methylprep.run_pipeline(
            workdir, array_type=at_token, betas=True,
            sample_sheet_filepath=sheet)
        beta = betas.iloc[:, 0].dropna().astype(float)
        beta.name = "beta"
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    meta = {
        "barcode": barcode,
        "array_type": at_str,
        "n_cpgs": int(len(beta)),
        "calibration": "noob (dye-bias + probe-type normalization), per-sample",
        "stage": "SOP Stage 1 steps 1.1-1.2 + 1.5",
    }
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
