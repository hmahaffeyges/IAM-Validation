#!/usr/bin/env python3
"""Chain v4 stages (2026-10-17): the commissioned instruments that ran from their own scripts, brought into the one run.

  D21  cell shares      IAM-Deconvolver against IAM-Atlas v2.1 (stage_deconv_v21, version 2), the six-cell blood panel.
                        COMMISSIONED 2026-10-17 for the shares of neutrophils, monocytes, CD4 T and CD8 T cells on arrays
                        (COMMISSIONING_NOTE_DECONV_V21.md); B and NK cells are printed as development; the error bars are information.
  Q2   IAM-A            each sample against its own reference (stage_q2_selfref): count (from a .pat or an aligned BAM), the sample-shape
                        check, then A = held error / what the sample's own loose share predicts, on band v1. COMMISSIONED 2026-10-17 on
                        healthy blood CD8 T cells, whole-genome bisulfite (COMMISSIONING_NOTE_IAMA.md); every other cell type is development.
  SKY  C-Score sky      the C-Score's per-site departures (the same z map the C-Score scores), in genomic order, laid on a HEALPix
                        sphere in NESTED order, where neighboring sites fill compact patches. A picture of the commissioned reading,
                        not a new reading: nothing is scored from it.
Nothing here is fitted at run time; every reference is a committed, stamped file."""
import os, gzip, base64, numpy as np, pandas as pd
import stage_deconv_v21 as D21, stage_q2_selfref as Q2

HERE = os.path.dirname(os.path.abspath(__file__))
MIN_MARKER_SHARE = 0.5   # operational: below half the marker sites measured, the shares are withheld (the tool says it cannot read)
SEQ_READ_MARKERS = 500   # 2026-10-23 (read and flag): on sequencing values, read with at least this many marker sites measured, flagged below MIN_MARKER_SHARE (stated)
CD8_TYPES = ("cd8 t cells", "naive cd8 t cells", "effector memory cd8 t cells", "t effector cell cd8")
_REF = {}

def _ex(check, outcome, measured, limit, source, means, todo, provisional=False):   # the five parts of every stop (author rule 2026-10-20)
    return dict(check=check, outcome=outcome, measured=measured, limit=limit, limit_source=source, provisional=provisional, means=means, what_to_do=todo)


# ---------------------------------------------------------------- D21: cell shares
def atlas_panel():
    """Every cell type of IAM-Atlas v2.1 (its best-value columns), for --cell-panel atlas (development, 2026-10-22)."""
    import pyarrow.parquet as pq
    return [n for k in (1, 2, 3) for n in pq.ParquetFile(os.path.join(D21.ATLAS, f"mu_part{k}.parquet")).schema_arrow.names if n not in ("cpg", "__index_level_0__")]

def stage_d21(beta, panel=None, sequencing=False):
    panel = list(panel or D21.BLOOD6)
    if tuple(panel) not in _REF:
        R = D21.reference(panel); _REF[tuple(panel)] = (R, D21.markers(R))
    R, M = _REF[tuple(panel)]
    b = pd.Series(beta, dtype="float64"); b.index = b.index.astype(str)
    have = int(b.reindex(M).notna().sum()); rec = {"stage": "D21", "reading": "cell shares", "tool": "IAM-Deconvolver (version 2)",
        "reference": "IAM-Atlas v2.1", "panel": panel, "markers": len(M), "markers_measured": have,
        "commissioning": D21.DECONV_V21_COMMISSIONING, "error_bars": "information, not calibrated"}
    if sequencing and SEQ_READ_MARKERS <= have < MIN_MARKER_SHARE * len(M):   # 2026-10-23: shallow sequencing (plasma), read and flagged
        rec["coverage_flag"] = (f"read on {have} of {len(M)} marker sites ({have / len(M):.0%}; below the half an array needs): shallow sequencing, "
                                "each site from a few molecules; shares read and flagged")
    elif have < MIN_MARKER_SHARE * len(M):
        rec["withheld"] = f"only {have} of {len(M)} marker sites measured (at least {int(np.ceil(MIN_MARKER_SHARE * len(M)))} needed): shares withheld"
        rec["explained"] = _ex("cell shares (Stage D21)", "withheld", f"{have} of {len(M)} marker sites measured", f"at least {int(np.ceil(MIN_MARKER_SHARE * len(M)))} (half)",
                               "stage_v4.MIN_MARKER_SHARE (operational, not measured)", "too few of the sites that tell the six blood cells apart were measured to solve the shares",
                               "check Stage 1 detection (probes at background are removed); a complete EPIC v1 array measures nearly all of them", provisional=True)
        return rec
    d = D21.deconvolve(b, R, M); com = set(D21.DECONV_V21_COMMISSIONING["cells"]) if list(panel) == list(D21.BLOOD6) else set()   # commissioned on the six-cell panel only
    rec["cells"] = [{"cell": c, "share": round(d["shares"][c], 4), "interval_95": [round(x, 4) for x in d["interval_95"][c]],
                     "resampled_median": round((d.get("resampled_median") or {}).get(c, float("nan")), 4),
                     "status": "commissioned" if c in com else "development"} for c in panel]
    for x in rec["cells"]:   # 2026-10-22: on a wide panel close relatives can trade a share, and the best fit can then fall outside its own 95 % range
        lo, hi = x["interval_95"]
        if not (lo <= x["share"] <= hi): x["resolved"] = False; x["note"] = "share outside its own 95 % range: not resolved from its close relatives in the atlas (development)"
    rec.update(sites_used=d["sites_used"], raw_sum=round(d["raw_sum"], 4))
    return rec

def stage_d21_provisional(beta, panel, cells, sequencing=False):
    """2026-10-24 (plan 40.18, development): the same deconvolution with provisional columns added beside IAM-Atlas v2.1 (stage_provisional);
    recorded beside the atlas's own shares, never in place of them, and flagged."""
    import stage_provisional as PV
    key = ("prov", tuple(panel), tuple(cells))
    if key not in _REF:
        R0 = D21.reference(panel); R, added = PV.with_provisional(R0, cells); _REF[key] = (R, D21.markers(R), added)
    R, M, added = _REF[key]
    if not added: return {"stage": "D21-provisional", "status": "NOT_RUN", "reason": "no provisional column built for " + ", ".join(cells)}
    b = pd.Series(beta, dtype="float64"); b.index = b.index.astype(str); d = D21.deconvolve(b, R, M)
    return {"stage": "D21-provisional", "flag": PV.FLAG, "columns": list(cells), "added": added, "markers": len(M), "markers_measured": int(b.reindex(M).notna().sum()),
            "cells": [{"cell": c, "share": round(d["shares"][c], 4), "interval_95": [round(x, 4) for x in d["interval_95"][c]]} for c in R["cells"]],
            "sites_used": d["sites_used"]}

# ---------------------------------------------------------------- Q2: IAM-A against the sample's own reference
def count_sequencing(pat=None, bam=None, ref_fasta=None, workers=4, assembly="hg19"):
    if pat:
        op = gzip.open if pat.endswith(".gz") else open
        with op(pat, "rt") as fh: return Q2.count_pat(fh, Q2.hg19_tiles())
    return Q2.count_bam(bam, ref_fasta, workers, assembly)

def counting_error(rec, A, B):
    """IAM-A's counting error, from its own count (finding F38, 2026-10-22; author: "let the instrument tell us...not our assumptions"). The held
    error is a count of holes among held-error opportunities, so its counting error is binomial: sd = sqrt(e (1 - e) / n), n the opportunities.
    A = e / prediction carries the same relative error (the prediction, from millions of tile calls, is counted far more finely). Returns the
    opportunities, A's counting sd and 95 % range, and that range's width as a share of the healthy band's width, so a shallow sample states how
    much of the band its own counting uses. Recorded only: it never refuses a reading and never changes A or its state."""
    n = int(rec.get("held_opp") or 0); e = float(rec.get("held_err") or 0.0)
    if n <= 0 or e <= 0: return {"opportunities": n, "note": "no held errors counted: the counting error cannot be stated"}
    sd = float(np.sqrt(e * (1 - e) / n)) / e * A; w = float(B["hi"] - B["lo"])
    return {"opportunities": n, "holes": int(round(e * n)), "A_sd": round(sd, 5), "A_95": [round(A - 1.96 * sd, 4), round(A + 1.96 * sd, 4)],
            "share_of_band_width": round(2 * 1.96 * sd / w, 4) if w > 0 else None,
            "rule": "binomial counting error of the held error, carried to A (stage_v4.counting_error); recorded, not a gate"}

def stage_q2(rec, cell, chemistry="whole-genome bisulfite", source=None):
    cell = str(cell).strip().lower(); B = Q2.band(1)
    out = {"stage": "Q2", "reading": "IAM-A", "reference": "the sample's own loose share (self-reference line v1)", "cell": cell,
           "chemistry": chemistry, "band": {"lo": B["lo"], "hi": B["hi"]}, "count": {k: rec.get(k) for k in ("loose_share", "held_err", "held_opp", "tiles", "tile_p10", "gain_err", "gain_opp", "mol_n", "mol_low_share", "mol_mid_share")},
           "input": source, "A": None}
    if not rec.get("held_opp") or rec.get("loose_share") is None:
        out["refusal"] = "no qualifying molecules or tiles in this file: IAM-A not read"
        out["explained"] = _ex("IAM-A count (Stage Q2)", "not read", f"held-error opportunities {rec.get('held_opp')}; tiles {rec.get('tiles')}",
                               "at least one held run (6 or more CpG calls, at least 80 % methylated) and one tile of 2,000 CpGs with 200 calls", "stage_q2_selfref (fixed counting rules)",
                               "the file holds no molecules IAM-A can count, so there is neither a held error nor a loose share", "check that the file is whole-genome bisulfite of the right build (hg19 CpG index) and not a small region")
        return out
    s = Q2.shape_check(rec, cell); out["shape_check"] = s
    if not s["read"]:
        out["refusal"] = s["reason"]
        out["explained"] = _ex("sample shape (Stage Q2)", "not read", f"tile 10th percentile {s.get('tile_p10')}" if s.get("tile_p10") is not None else s["reason"],
                               f"at least {s['limit']:.4f} (the lowest healthy {cell} minus the margin {s['margin']:.4f})" if s.get("limit") is not None else "a healthy shape reference for this cell type",
                               "Runtime Matrices/IAM_A_SelfRef/shape_reference_v1.json (healthy samples of GSE186458; the margin is computed from them, never typed)",
                               "the sample's spread of methylation across the genome is outside what healthy samples of its type show, so its own loose share is not a healthy state the line can be read from",
                               "check the cell type given with --seq-cell; a sample outside the healthy shape is reported, not read")
        return out
    r = Q2.reading(rec); A = float(r["A"])
    out.update(A=round(A, 4), predicted=round(r["predicted"], 6), held_err=round(r["held_err"], 6), loose_share=round(r["loose_share"], 4),
               state="Normal" if B["lo"] <= A <= B["hi"] else ("above Normal" if A > B["hi"] else "below Normal"))
    out["counting"] = counting_error(rec, A, B)   # 2026-10-22 (finding F38): the reading's own counting error; recorded, never a gate
    if cell in CD8_TYPES and chemistry == "whole-genome bisulfite": out["commissioning"] = Q2.IAMA_COMMISSIONING
    else: out["development"] = "DEVELOPMENT - not commissioned for this cell type or chemistry (commissioned: healthy blood CD8 T cells, whole-genome bisulfite)"
    return out

# ---------------------------------------------------------------- SKY: the C-Score's departures on the sphere
# The C-Score's own sites, in genomic order, are laid on a HEALPix sphere in NESTED order: consecutive indices fill compact patches, so a run
# of neighboring departures (what the C-Score scores, blocks of consecutive sites) shows as a patch and scattered departures as speckle.
# NSIDE is the smallest power of 2 with at least one pixel per two sites. nest_pix2ang matches healpy.pix2ang(nest=True) to 1e-14 rad at
# NSIDE 1, 2, 16 and 128 (checked 2026-10-17); healpy is not needed to run the chain.
JRLL = np.array([2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]); JPLL = np.array([1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7])
def _compress(v):   # take every other bit (bit 0, 2, 4, ...)
    v = v & 0x5555555555555555; v = (v | (v >> 1)) & 0x3333333333333333; v = (v | (v >> 2)) & 0x0F0F0F0F0F0F0F0F
    v = (v | (v >> 4)) & 0x00FF00FF00FF00FF; v = (v | (v >> 8)) & 0x0000FFFF0000FFFF; v = (v | (v >> 16)) & 0x00000000FFFFFFFF; return v
def nest_pix2ang(nside, ipix):
    """HEALPix NESTED pixel centres -> (theta, phi); the standard nest2xyf / xyf2ang algorithm (Gorski et al. 2005)."""
    ipix = np.asarray(ipix, dtype=np.int64); npface = nside * nside; face = ipix // npface; ipf = ipix % npface
    ix = _compress(ipf); iy = _compress(ipf >> 1); jr = JRLL[face] * nside - ix - iy - 1
    nr = np.where(jr < nside, jr, np.where(jr > 3 * nside, 4 * nside - jr, nside)).astype(float)
    z = np.where(jr < nside, 1 - nr * nr / (3.0 * npface), np.where(jr > 3 * nside, nr * nr / (3.0 * npface) - 1, (2 * nside - jr) * 2.0 / (3 * nside)))
    ks = np.where((jr >= nside) & (jr <= 3 * nside), (jr - nside) & 1, 0)
    jp = (JPLL[face] * nr.astype(np.int64) + ix - iy + 1 + ks) // 2
    jp = np.where(jp > 4 * nside, jp - 4 * nside, jp); jp = np.where(jp < 1, jp + 4 * nside, jp)
    phi = (jp - (ks + 1) * 0.5) * (np.pi / 2 / nr); return np.arccos(np.clip(z, -1, 1)), phi

def cscore_sky(z, out_png, title="", vlim=3.0, readings=""):
    """z: the C-Score's per-site departure map (pd.Series by site, in the C-Score's genomic order). Writes a Mollweide view; returns a summary."""
    if z is None: return {"stage": "SKY", "drawn": False, "reason": "no departure map: the C-Score was not computed on this array (its reason is printed with the C-Score)"}
    zz = pd.Series(z, dtype="float64"); n = len(zz)
    if zz.notna().sum() == 0: return {"stage": "SKY", "drawn": False, "reason": "no measured departures"}
    nside = 1
    while 12 * nside * nside * 2 < n: nside *= 2
    npix = 12 * nside * nside; pix = (np.arange(n) * npix) // n
    # Drawn in units of the sample's own spread (2026-10-18): the C-Score (variance of block means over site variance) is unchanged by
    # shifting or rescaling every departure, so the picture shows exactly what it scores. Unscaled, a whole-blood sample read against pure
    # neutrophils sits at the color limits everywhere and hides the pattern.
    med = float(zz.median()); mad = float((zz - med).abs().median()) * 1.4826 or float(zz.std()) or 1.0
    sky = pd.Series(((zz - med) / mad).values).groupby(pix).mean().reindex(range(npix)).values
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from scipy.spatial import cKDTree
    # Drawn in the style of the sky figures on the public page and in the book (2026-10-18; reference_floors_v1/sky/make_sky_figs.py):
    # Mollweide, RdBu_r from -4 to +4 (departures in units of the sample's own spread), no smoothing, nearest-pixel fill, unmeasured sky gray, a thin outline, the
    # readings printed under the map. The pixel layout (NSIDE, NESTED, genome order) is unchanged.
    th, ph = nest_pix2ang(nside, np.arange(npix)); xyz = np.c_[np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]
    xs, ys = 1600, 800; X, Y = np.meshgrid((np.arange(xs) + 0.5) / xs * 4 - 2, (np.arange(ys) + 0.5) / ys * 2 - 1)
    inside = (X / 2) ** 2 + Y ** 2 < 1; t = np.arcsin(np.clip(Y, -1, 1)); lat = np.arcsin(np.clip((2 * t + np.sin(2 * t)) / np.pi, -1, 1))
    lon = np.pi * X / (2 * np.cos(np.where(inside, t, 0)))
    q = np.c_[(np.cos(lat) * np.cos(-lon)).ravel(), (np.cos(lat) * np.sin(-lon)).ravel(), np.sin(lat).ravel()]
    img = sky[cKDTree(xyz).query(q)[1]].reshape(X.shape); img = np.where(inside, img, np.nan)
    cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad("#d9d9d9")
    fig, ax = plt.subplots(figsize=(7.2, 4.2)); fig.patch.set_facecolor("white")
    im = ax.imshow(np.ma.masked_invalid(np.clip(img, -vlim, vlim)), cmap=cmap, vmin=-vlim, vmax=vlim, origin="lower", interpolation="nearest", extent=(-2, 2, -1, 1))
    e = Ellipse((0, 0), 4, 2, transform=ax.transData, fc="none", ec="#555555", lw=0.5); ax.add_patch(e); im.set_clip_path(e)
    ax.set_xlim(-2.02, 2.02); ax.set_ylim(-1.02, 1.02); ax.set_aspect("equal"); ax.set_axis_off()
    ax.set_title(title or "C-Score departures on the sphere", loc="left", fontsize=9.5)
    if readings: ax.text(0, -1.12, readings, ha="center", va="top", fontsize=9, transform=ax.transData)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.045, pad=0.12 if readings else 0.05, shrink=0.45, ticks=[-vlim, 0, vlim])
    cb.set_label("departure, in units of this sample's own spread", fontsize=9); cb.ax.tick_params(labelsize=8)
    fig.savefig(out_png, dpi=200, bbox_inches="tight"); plt.close(fig)
    return {"stage": "SKY", "drawn": True, "png": os.path.abspath(out_png), "nside": nside, "order": "NESTED", "sites": int(n),
            "sites_measured": int(zz.notna().sum()), "pixels_drawn": int(np.isfinite(sky).sum()), "frac_abs_z_gt3": round(float((zz.abs() > 3).mean()), 4),
            "note": "a picture of the C-Score's departures; nothing is scored from it"}

def png_data_uri(path):
    return "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode() if path and os.path.exists(path) else None
