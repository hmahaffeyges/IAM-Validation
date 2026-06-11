"""Shared IDAT decoder for the CPG clinical pipeline.

ONE decoder feeds two stages. It turns a raw Illumina IDAT pair (Grn + Red)
into the quantities Stage 0 (intake QC) and Stage 1 (calibration) need:

    beta          {cpg_id: beta}     noob-corrected            -> Stage 1 (and 1.5 cross-check)
    meth/unmeth   {cpg_id: M}/{U}    per-probe intensities     -> Stage 1.5 (beta = M/(M+U+100))
    detection_p   {cpg_id: p}        poobah p-value per probe  -> Stage 0.5 (detection-p gate)
    n_beads       {address: n}       bead count per address    -> Stage 0.6 (bead-count gate)
    controls      {ctrl_type: ...}   control-probe intensities -> Stage 0.4 (BS/hyb/extension)
    sex           {chrX/chrY/pred}   chrX/Y median intensity   -> Stage 0.8 (sex check)
    array_type / barcode / n_addresses                          -> Stage 0.1 / platform tag

================================  ENVIRONMENT  ================================
methylprep 1.7.1 is built for the pandas<2 / numpy<2 API and breaks on pandas2
(removed DataFrame.append) and numpy2 (dye-bias indexing). It therefore runs in
a PINNED environment, NOT on the host's Python 3.12:

    Python 3.10 or 3.11
    numpy<2  pandas<2  methylprep   scipy

The IdatDataset raw read and the manifest load DO work on any pandas; the full
noob/run_pipeline path is what needs the pin. Build/run this module inside that
pinned container (or image). Validate against a known IDAT, e.g. the real EPIC
pair used during development (GSE116339 / GSM3228562).

Lines marked  # VALIDATE-IN-CONTAINER  use methylprep internals whose exact
attribute/column names vary by version; confirm them on first run in the pinned
env and pin the methylprep version once confirmed.
=============================================================================
"""
from dataclasses import dataclass, field
from typing import Dict, Optional
import os
import glob


@dataclass
class DecodedSample:
    array_type: str = ""
    barcode: str = ""
    n_addresses: int = 0
    beta: Dict[str, float] = field(default_factory=dict)
    meth: Dict[str, float] = field(default_factory=dict)
    unmeth: Dict[str, float] = field(default_factory=dict)
    detection_p: Dict[str, float] = field(default_factory=dict)
    n_beads: Dict[str, int] = field(default_factory=dict)
    controls: dict = field(default_factory=dict)
    sex: dict = field(default_factory=dict)
    status: str = "OK"
    notes: list = field(default_factory=list)


# methylprep array_type string <-> our platform tag
_ARRAY_TO_PLATFORM = {"450k": "HM450K", "epic": "EPIC_v1", "epic+": "EPIC_v2"}


def _infer_array_type(grn_path: str) -> str:
    """nSNPsRead -> methylprep array_type string. HM450K ~622k, EPIC ~1.05M, EPIC v2 ~1.1M."""
    from methylprep.files.idat import IdatDataset
    from methylprep.models import Channel
    d = IdatDataset(grn_path, channel=Channel.GREEN)
    n = int(d.n_snps_read)
    if n < 800_000:
        return "450k"
    if n < 1_080_000:
        return "epic"
    return "epic+"


def preflight() -> dict:
    """Confirm the decode dependencies are present BEFORE attempting a decode.

    Returns {ok, python, methylprep, numpy, pandas, messages}. It does NOT install
    anything: a clinical pipeline's environment must be provisioned deliberately,
    not mutated at runtime. It reports exactly what is missing or version-incompatible
    and the one command that fixes it. (The Illumina manifest -- the probe map, which
    is DATA not software -- is a separate thing that methylprep auto-downloads and
    caches on first use; that does not need provisioning.)
    """
    import sys
    info = {"python": "%d.%d" % sys.version_info[:2],
            "methylprep": None, "numpy": None, "pandas": None,
            "messages": [], "ok": True}

    if sys.version_info[:2] not in ((3, 10), (3, 11)):
        info["ok"] = False
        info["messages"].append(
            f"Python {info['python']} unsupported for methylprep 1.7.1; use 3.10 or 3.11 "
            "(pandas<2 has no Python 3.12 wheels).")

    try:
        import methylprep
        info["methylprep"] = methylprep.__version__
    except Exception:
        info["ok"] = False
        info["messages"].append(
            "methylprep NOT installed. In the pinned env run: "
            "pip install 'numpy<2' 'pandas<2' methylprep")

    try:
        import numpy
        info["numpy"] = numpy.__version__
        if int(numpy.__version__.split(".")[0]) >= 2:
            info["ok"] = False
            info["messages"].append("numpy>=2 breaks methylprep 1.7.1; pin numpy<2.")
    except Exception as e:
        info["ok"] = False
        info["messages"].append(f"numpy import failed: {e}")

    try:
        import pandas
        info["pandas"] = pandas.__version__
        if int(pandas.__version__.split(".")[0]) >= 2:
            info["ok"] = False
            info["messages"].append("pandas>=2 breaks methylprep 1.7.1; pin pandas<2.")
    except Exception as e:
        info["ok"] = False
        info["messages"].append(f"pandas import failed: {e}")

    return info


def decode_idat_pair(grn_path: str,
                     red_path: str,
                     array_type: Optional[str] = None,
                     do_noob: bool = True) -> DecodedSample:
    """Decode one IDAT pair into a DecodedSample.

    grn_path / red_path : the *_Grn.idat / *_Red.idat pair (decompressed).
    array_type          : '450k' | 'epic' | 'epic+'; auto-inferred if None.
    do_noob             : run methylprep noob (matches the pipeline's chosen
                          normalization). Stage 1.5 may also recompute beta from
                          the returned meth/unmeth so the choice is auditable.

    Fails fast with status DECODE_ENV_NOT_READY (no exception) if the pinned
    methylprep environment is absent, so the caller can surface a clear message.
    """
    pf = preflight()
    if not pf["ok"]:
        return DecodedSample(status="DECODE_ENV_NOT_READY", notes=pf["messages"])

    import methylprep
    from methylprep.files.idat import IdatDataset
    from methylprep.models import Channel

    out = DecodedSample()
    if array_type is None:
        array_type = _infer_array_type(grn_path)
    out.array_type = _ARRAY_TO_PLATFORM.get(array_type, array_type)

    grn = IdatDataset(grn_path, channel=Channel.GREEN)
    out.barcode = str(getattr(grn, "barcode", ""))
    out.n_addresses = int(grn.n_snps_read)

    # methylprep wants a directory of IDATs; isolate this pair in its own dir.
    sample_dir = os.path.dirname(os.path.abspath(grn_path))

    # Full decode: noob beta + per-probe M/U + poobah detection p. export=True
    # writes a {barcode}_processed.csv we then read (stable file contract; more
    # robust than relying on in-memory attribute names across versions).
    # noob is methylprep's default correction (matches the pipeline's choice);
    # do_noob is accepted here for an explicit raw-only path if ever needed.
    run_kwargs = dict(array_type=array_type, export=True, betas=False,
                      save_uncorrected=True, poobah=True, make_sample_sheet=True)
    if not do_noob:
        run_kwargs["do_noob"] = False     # VALIDATE-IN-CONTAINER (param surface)
    methylprep.run_pipeline(sample_dir, **run_kwargs)

    proc = _find_processed_csv(sample_dir, out.barcode)
    if proc is None:
        out.status = "DECODE_NO_PROCESSED_OUTPUT"
        out.notes.append("methylprep produced no *_processed.csv; check pinned env")
        return out

    import pandas as pd
    df = pd.read_csv(proc, index_col=0)
    # Column names across methylprep versions: meth/unmeth and beta/detection.
    meth_col = _first_col(df, ["noob_meth", "meth", "methylated"])           # VALIDATE-IN-CONTAINER
    unmeth_col = _first_col(df, ["noob_unmeth", "unmeth", "unmethylated"])   # VALIDATE-IN-CONTAINER
    beta_col = _first_col(df, ["beta_value", "beta"])
    detp_col = _first_col(df, ["poobah_pval", "detection_pval", "pval"])

    if beta_col:
        out.beta = df[beta_col].dropna().to_dict()
    if meth_col and unmeth_col:
        out.meth = df[meth_col].dropna().to_dict()
        out.unmeth = df[unmeth_col].dropna().to_dict()
    if detp_col:
        out.detection_p = df[detp_col].dropna().to_dict()
    else:
        out.notes.append("no detection-p column found; Stage 0.5 will run DEFERRED")

    # Raw-channel byproducts (work without the noob path): controls, beads, sex.
    out.controls = _extract_controls(grn_path, red_path, array_type)   # VALIDATE-IN-CONTAINER
    out.n_beads = _extract_n_beads(grn)                                 # VALIDATE-IN-CONTAINER
    out.sex = _extract_sex(out.meth, out.unmeth, array_type)

    return out


def _find_processed_csv(sample_dir: str, barcode: str):
    cands = glob.glob(os.path.join(sample_dir, "*_processed.csv"))
    if not cands:
        cands = glob.glob(os.path.join(sample_dir, "**", "*_processed.csv"), recursive=True)
    if not cands:
        return None
    for c in cands:                       # prefer the one matching this barcode
        if barcode and barcode in os.path.basename(c):
            return c
    return cands[0]


def _first_col(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None


def _extract_controls(grn_path: str, red_path: str, array_type: str) -> dict:
    """Control-probe intensities by control type, from the manifest control table
    mapped onto the raw channel means. Feeds Stage 0.4 (BS conversion, hybridization,
    extension). VALIDATE-IN-CONTAINER: confirm manifest.control_data_frame columns."""
    try:
        from methylprep.files.idat import IdatDataset
        from methylprep.files.manifests import Manifest
        from methylprep.models import Channel, ArrayType
        at = {"450k": ArrayType.ILLUMINA_450K,
              "epic": ArrayType.ILLUMINA_EPIC,
              "epic+": ArrayType.ILLUMINA_EPIC_PLUS}.get(array_type, ArrayType.ILLUMINA_EPIC)
        man = Manifest(at)
        ctl = getattr(man, "control_data_frame", None)
        if ctl is None:
            return {"_note": "manifest exposes no control_data_frame in this version"}
        grn = IdatDataset(grn_path, channel=Channel.GREEN).probe_means
        red = IdatDataset(red_path, channel=Channel.RED).probe_means
        gmap = grn.iloc[:, 0].to_dict()
        rmap = red.iloc[:, 0].to_dict()
        by_type = {}
        # control_data_frame: Address_ID + Control_Type (+ Color). Group means by type.
        addr_col = _first_col(ctl, ["Address_ID", "AddressA_ID", "address"])
        type_col = _first_col(ctl, ["Control_Type", "Type", "control_type"])
        if not addr_col or not type_col:
            return {"_note": "control_data_frame columns unrecognized; confirm in container"}
        for _, row in ctl.iterrows():
            a = row[addr_col]; t = str(row[type_col])
            rec = by_type.setdefault(t, {"grn": [], "red": []})
            if a in gmap:
                rec["grn"].append(float(gmap[a]))
            if a in rmap:
                rec["red"].append(float(rmap[a]))
        return by_type
    except Exception as e:
        return {"_error": str(e)}


def _extract_n_beads(grn_idat) -> dict:
    """Per-address bead count, if the IdatDataset exposes it. Feeds Stage 0.6.
    VALIDATE-IN-CONTAINER: attribute name (n_beads / probe_nbeads / run_info)."""
    for attr in ("n_beads", "probe_nbeads", "nbeads"):
        v = getattr(grn_idat, attr, None)
        if v is not None:
            try:
                return v.iloc[:, 0].to_dict() if hasattr(v, "iloc") else dict(v)
            except Exception:
                pass
    return {}


def _extract_sex(meth: dict, unmeth: dict, array_type: str) -> dict:
    """minfi getSex logic on total intensity (M+U) over chrX / chrY probes.
    predicted sex F if (median log2 total chrY) - (median log2 total chrX) < -2.
    VALIDATE-IN-CONTAINER: manifest CHR column name."""
    try:
        import numpy as np
        from methylprep.files.manifests import Manifest
        from methylprep.models import ArrayType
        at = {"450k": ArrayType.ILLUMINA_450K,
              "epic": ArrayType.ILLUMINA_EPIC,
              "epic+": ArrayType.ILLUMINA_EPIC_PLUS}.get(array_type, ArrayType.ILLUMINA_EPIC)
        man = Manifest(at)
        mdf = man.data_frame
        chr_col = _first_col(mdf, ["CHR", "Chromosome", "chr"])
        if chr_col is None or not meth:
            return {"predicted_sex": "UNKNOWN", "_note": "no CHR column or no M/U"}
        chrom = mdf[chr_col].astype(str)
        xset = set(mdf.index[chrom.isin(["X", "chrX"])])
        yset = set(mdf.index[chrom.isin(["Y", "chrY"])])

        def med_log2_total(idset):
            tot = [meth.get(c, 0) + unmeth.get(c, 0) for c in idset
                   if c in meth and (meth.get(c, 0) + unmeth.get(c, 0)) > 0]
            return float(np.median(np.log2(tot))) if tot else float("nan")

        xmed, ymed = med_log2_total(xset), med_log2_total(yset)
        pred = "UNKNOWN"
        if xmed == xmed and ymed == ymed:        # not NaN
            pred = "F" if (ymed - xmed) < -2 else "M"
        return {"chrX_median_log2": xmed, "chrY_median_log2": ymed, "predicted_sex": pred}
    except Exception as e:
        return {"predicted_sex": "UNKNOWN", "_error": str(e)}


if __name__ == "__main__":
    # Container validation entry point. Run in the pinned env against a folder
    # holding ONE decompressed IDAT pair:
    #   python idat_decoder.py /path/to/idat_dir
    import sys
    if len(sys.argv) < 2:
        print("usage: python idat_decoder.py <dir_with_one_Grn+Red_idat_pair>")
        sys.exit(1)
    d = sys.argv[1]
    grn = glob.glob(os.path.join(d, "*_Grn.idat"))[0]
    red = glob.glob(os.path.join(d, "*_Red.idat"))[0]
    s = decode_idat_pair(grn, red)
    print("array_type :", s.array_type, "| barcode:", s.barcode, "| addresses:", s.n_addresses)
    print("beta       :", len(s.beta), "cpgs")
    print("meth/unmeth:", len(s.meth), "/", len(s.unmeth))
    print("detection_p:", len(s.detection_p), "(0 = DEFERRED to Stage 0.5)")
    print("controls   :", list(s.controls.keys())[:8])
    print("n_beads    :", len(s.n_beads))
    print("sex        :", s.sex)
    print("status     :", s.status, "| notes:", s.notes)
