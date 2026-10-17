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
CD8_TYPES = ("cd8 t cells", "naive cd8 t cells", "effector memory cd8 t cells", "t effector cell cd8")
_REF = {}

# ---------------------------------------------------------------- D21: cell shares
def stage_d21(beta, panel=None):
    panel = list(panel or D21.BLOOD6)
    if tuple(panel) not in _REF:
        R = D21.reference(panel); _REF[tuple(panel)] = (R, D21.markers(R))
    R, M = _REF[tuple(panel)]
    b = pd.Series(beta, dtype="float64"); b.index = b.index.astype(str)
    have = int(b.reindex(M).notna().sum()); rec = {"stage": "D21", "reading": "cell shares", "tool": "IAM-Deconvolver (version 2)",
        "reference": "IAM-Atlas v2.1", "panel": panel, "markers": len(M), "markers_measured": have,
        "commissioning": D21.DECONV_V21_COMMISSIONING, "error_bars": "information, not calibrated"}
    if have < MIN_MARKER_SHARE * len(M):
        rec["withheld"] = f"only {have} of {len(M)} marker sites measured (at least {int(np.ceil(MIN_MARKER_SHARE * len(M)))} needed): shares withheld"
        return rec
    d = D21.deconvolve(b, R, M); com = set(D21.DECONV_V21_COMMISSIONING["cells"])
    rec["cells"] = [{"cell": c, "share": round(d["shares"][c], 4), "interval_95": [round(x, 4) for x in d["interval_95"][c]],
                     "status": "commissioned" if c in com else "development"} for c in panel]
    rec.update(sites_used=d["sites_used"], raw_sum=round(d["raw_sum"], 4))
    return rec

# ---------------------------------------------------------------- Q2: IAM-A against the sample's own reference
def count_sequencing(pat=None, bam=None, ref_fasta=None, workers=4, assembly="hg19"):
    if pat:
        op = gzip.open if pat.endswith(".gz") else open
        with op(pat, "rt") as fh: return Q2.count_pat(fh, Q2.hg19_tiles())
    return Q2.count_bam(bam, ref_fasta, workers, assembly)

def stage_q2(rec, cell, chemistry="whole-genome bisulfite", source=None):
    cell = str(cell).strip().lower(); B = Q2.band(1)
    out = {"stage": "Q2", "reading": "IAM-A", "reference": "the sample's own loose share (self-reference line v1)", "cell": cell,
           "chemistry": chemistry, "band": {"lo": B["lo"], "hi": B["hi"]}, "count": {k: rec.get(k) for k in ("loose_share", "held_err", "held_opp", "tiles", "tile_p10")},
           "input": source, "A": None}
    if not rec.get("held_opp") or rec.get("loose_share") is None:
        out["refusal"] = "no qualifying molecules or tiles in this file: IAM-A not read"; return out
    s = Q2.shape_check(rec, cell); out["shape_check"] = s
    if not s["read"]: out["refusal"] = s["reason"]; return out
    r = Q2.reading(rec); A = float(r["A"])
    out.update(A=round(A, 4), predicted=round(r["predicted"], 6), held_err=round(r["held_err"], 6), loose_share=round(r["loose_share"], 4),
               state="Normal" if B["lo"] <= A <= B["hi"] else ("above Normal" if A > B["hi"] else "below Normal"))
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

def cscore_sky(z, out_png, title="", vlim=3.0):
    """z: the C-Score's per-site departure map (pd.Series by site, in the C-Score's genomic order). Writes a Mollweide view; returns a summary."""
    if z is None: return {"stage": "SKY", "drawn": False, "reason": "no departure map (the C-Score was not computed)"}
    zz = pd.Series(z, dtype="float64"); n = len(zz)
    if zz.notna().sum() == 0: return {"stage": "SKY", "drawn": False, "reason": "no measured departures"}
    nside = 1
    while 12 * nside * nside * 2 < n: nside *= 2
    npix = 12 * nside * nside; pix = (np.arange(n) * npix) // n
    sky = pd.Series(zz.values).groupby(pix).mean().reindex(range(npix)).values
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    th, ph = nest_pix2ang(nside, np.arange(npix)); lat = np.pi / 2 - th; lon = np.where(ph > np.pi, ph - 2 * np.pi, ph); m = np.isfinite(sky)
    fig, ax = plt.subplots(figsize=(10, 5.6), subplot_kw={"projection": "mollweide"}); fig.patch.set_facecolor("white")
    size = 32000.0 / npix
    ax.scatter(lon[~m], lat[~m], c="#d9d9d9", s=size, marker="o", linewidths=0, rasterized=True)
    sc = ax.scatter(lon[m], lat[m], c=np.clip(sky[m], -vlim, vlim), s=size, cmap="RdBu_r", vmin=-vlim, vmax=vlim, marker="o", linewidths=0, rasterized=True)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.grid(False)
    cb = fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.05, pad=0.04); cb.set_label("departure from the healthy cell, z (blue below, red above; grey not measured)")
    ax.set_title(title or "C-Score departures on the sphere", fontsize=11)
    fig.savefig(out_png, dpi=130, bbox_inches="tight"); plt.close(fig)
    return {"stage": "SKY", "drawn": True, "png": os.path.abspath(out_png), "nside": nside, "order": "NESTED", "sites": int(n),
            "sites_measured": int(zz.notna().sum()), "pixels_drawn": int(m.sum()), "frac_abs_z_gt3": round(float((zz.abs() > 3).mean()), 4),
            "note": "a picture of the C-Score's departures; nothing is scored from it"}

def png_data_uri(path):
    return "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode() if path and os.path.exists(path) else None
