#!/usr/bin/env python3
"""The Stage 0 -> Stage 1 hand-off: decode the IDAT pair far enough to run the four deferred intake QCs.

Stage 0's gates 0.4 (control probes), 0.5 (detection p), 0.6 (bead count), 0.7 (call rate) and 0.8 (sex check)
all need intensities, and `extract_control_probes` raised NotImplementedError so that a caller could never be
handed fabricated data. This module supplies the real values from the same decoder Stage 1 uses (methylprep),
reading the Illumina control addresses out of the array's own manifest.

Everything here is per-sample: the patient's own control probes, out-of-band negatives and bead counts. No
cohort, no reference matrix, no other sample enters - the same discipline Stage 1 calibration follows.

decode_qc_inputs(grn, red, array_type) -> dict with
    control_summary   {bisulfite_conversion_I_median, bisulfite_conversion_II_median, hyb_high_median,
                       hyb_low_median, extension_meth_median, extension_unmeth_median}
    probe_intensities  total intensity per assayed probe
    neg_control_stats  {mu_bg, sigma_bg} from the array's NEGATIVE controls
    bead_counts        beads per probe
    sex_intensities    {log2_x_median, log2_y_median}
"""
import numpy as np

_MANIFEST = {}


def _manifest(array_type):
    """The array's own manifest, cached per array type."""
    key = str(array_type or "").upper()
    if key not in _MANIFEST:
        from methylprep.files import Manifest
        from methylprep.models import ArrayType
        at = {"HM450K": "450k", "450K": "450k", "EPIC_V1": "epic", "EPIC_V2": "epic+"}.get(key, "450k")
        _MANIFEST[key] = Manifest(ArrayType(at))
    return _MANIFEST[key]


def _channel_means(path, channel, want_beads=False):
    """Per-address mean intensity (and bead count) from one IDAT file, gzip or plain."""
    from methylprep.files import IdatDataset
    from methylprep.models import Channel
    ds = IdatDataset(path, channel=Channel.GREEN if channel == "grn" else Channel.RED, nbeads=want_beads)
    means = ds.probe_means
    col = "mean_value" if "mean_value" in means.columns else means.columns[0]
    out = {"mean": means[col].astype("float64")}
    if want_beads and "n_beads" in means.columns:
        out["beads"] = means["n_beads"].astype("float64")
    return out


def _median(series, addresses):
    v = series.reindex(addresses).dropna()
    return float(np.median(v)) if len(v) else None


def decode_qc_inputs(grn_path, red_path, array_type="HM450K"):
    m = _manifest(array_type)
    ctl = m.control_data_frame
    g = _channel_means(grn_path, "grn", want_beads=True)
    r = _channel_means(red_path, "red", want_beads=True)
    gm, rm = g["mean"], r["mean"]

    def ctl_addr(control_type, contains=None, excludes=None):
        sub = ctl[ctl["Control_Type"].astype(str) == control_type]
        if contains is not None:
            et = sub["Extended_Type"].astype(str)
            sub = sub[et.str.replace("-", " ", regex=False).str.contains(contains, case=False)]
        if excludes is not None:
            et = sub["Extended_Type"].astype(str)
            sub = sub[~et.str.replace("-", " ", regex=False).str.contains(excludes, case=False)]
        return list(sub.index)

    # SOP §14. The validator computes bs1 / (bs1 + bs2) and gates at 0.95, which is the bisulfite conversion
    # EFFICIENCY of §14 - converted over converted-plus-unconverted. The BS Conversion I control set carries
    # both: the "-C" probes report converted template, the "-U" probes unconverted. They are read in green.
    # SOP §14 bisulfite conversion efficiency. The BS Conversion I control set is built in matched pairs -
    # C1 with U1, C2 with U2 - so efficiency is computed per pair as C / (C + U) and the median taken, rather
    # than dividing one group median by another, which mixes pairs of different brightness.
    et = ctl["Extended_Type"].astype(str).str.replace("-", " ", regex=False)
    bs = ctl[ctl["Control_Type"].astype(str) == "BISULFITE CONVERSION I"]
    pairs_ce, pairs_ue = {}, {}
    for addr, name in et.reindex(bs.index).items():
        mm = np.array([0])
        t = str(name).upper().replace("BS CONVERSION I", "").strip()
        if t.startswith("C"):
            pairs_ce[t[1:].strip() or "1"] = addr
        elif t.startswith("U"):
            pairs_ue[t[1:].strip() or "1"] = addr
    effs = []
    for k in sorted(set(pairs_ce) & set(pairs_ue)):
        c = gm.get(pairs_ce[k], np.nan)
        u = gm.get(pairs_ue[k], np.nan)
        if np.isfinite(c) and np.isfinite(u) and (c + u) > 0:
            effs.append(c / (c + u))
    bs_c = float(np.median([gm.get(v, np.nan) for v in pairs_ce.values()])) if pairs_ce else None
    bs_u = float(np.median([gm.get(v, np.nan) for v in pairs_ue.values()])) if pairs_ue else None
    if effs:
        # express the matched-pair efficiency in the two keys the validator divides
        eff = float(np.median(effs))
        bs_c, bs_u = eff, 1.0 - eff
    control_summary = {
        "bisulfite_conversion_I_median": bs_c,
        "bisulfite_conversion_II_median": bs_u,
        "hyb_high_median": _median(gm, ctl_addr("HYBRIDIZATION", contains="high")),
        "hyb_low_median":  _median(gm, ctl_addr("HYBRIDIZATION", contains="low")),
        # EXTENSION: C and G extend in green, A and T in red; a balanced array sits near 1
        "extension_meth_median":   _median(gm, ctl_addr("EXTENSION", contains=r"\((C|G)\)")),
        "extension_unmeth_median": _median(rm, ctl_addr("EXTENSION", contains=r"\((A|T)\)")),
        "bs_pairs_used": len(effs),
    }
    neg = ctl_addr("NEGATIVE")
    nv = (gm.reindex(neg).fillna(0) + rm.reindex(neg).fillna(0)).dropna()
    neg_control_stats = {"mu_bg": float(np.mean(nv)), "sigma_bg": float(np.std(nv, ddof=1))}

    # Per-probe total intensity and bead count, by Infinium design - the same construction minfi's
    # detectionP uses. A Type II probe is read at address A in both channels. A Type I probe is read at
    # addresses A and B within its own colour channel; summing the other channel would add background only.
    df = m.data_frame
    design = df["Infinium_Design_Type"].astype(str).str.strip()
    colour = df["Color_Channel"].astype(str).str.strip()
    a = df["AddressA_ID"]
    b = df["AddressB_ID"]

    def _at(series, addr):
        idx = addr.astype("float64")
        vals = np.full(len(idx), np.nan)
        ok = idx.notna().to_numpy()
        vals[ok] = series.reindex(idx[ok].astype("int64")).to_numpy()
        return vals

    ga, ra = _at(gm, a), _at(rm, a)
    gb, rb = _at(gm, b), _at(rm, b)
    is_I = (design == "I").to_numpy()
    grn_I = is_I & (colour == "Grn").to_numpy()
    red_I = is_I & (colour == "Red").to_numpy()
    total = np.where(is_I, 0.0, np.nan_to_num(ga) + np.nan_to_num(ra))
    total = np.where(grn_I, np.nan_to_num(ga) + np.nan_to_num(gb), total)
    total = np.where(red_I, np.nan_to_num(ra) + np.nan_to_num(rb), total)
    found = ~np.isnan(ga) | ~np.isnan(ra)
    probe_intensities = total[found]

    gbe = g.get("beads")
    rbe = r.get("beads")
    if gbe is not None and rbe is not None:
        bead_a = np.fmin(_at(gbe, a), _at(rbe, a))
        bead_b = np.fmin(_at(gbe, b), _at(rbe, b))
        bead = np.where(is_I, np.fmin(np.nan_to_num(bead_a, nan=1e9), np.nan_to_num(bead_b, nan=1e9)), bead_a)
        bead_counts = np.nan_to_num(bead[found], nan=0.0)
    else:
        bead_counts = np.full(probe_intensities.shape, np.nan)

    # Sex check: the median total intensity of the X and Y probes, in log2. predict_sex asks for
    # log2_y - log2_x < -2, so both must be built the same design-aware way as the totals above.
    chrom = df["CHR"].astype(str).str.upper().str.replace("CHR", "", regex=False).to_numpy()
    def sex_median(which):
        sel = (chrom == which) & found
        v = total[sel]
        v = v[v > 0]
        return (float(np.log2(np.median(v))) if len(v) else None), int(len(v))
    x_med, n_x = sex_median("X")
    y_med, n_y = sex_median("Y")
    sex_intensities = {"log2_x_median": x_med, "log2_y_median": y_med, "n_x_probes": n_x, "n_y_probes": n_y}

    return {"control_summary": control_summary, "probe_intensities": probe_intensities,
            "neg_control_stats": neg_control_stats, "bead_counts": bead_counts,
            "sex_intensities": sex_intensities,
            "n_probes": int(len(probe_intensities)), "n_negative_controls": int(len(nv))}
