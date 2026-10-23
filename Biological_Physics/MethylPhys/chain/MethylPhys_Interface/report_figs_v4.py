"""Figures for the chain v4 report (2026-10-18): the floor-to-ceiling gauges, the C-Score bar and the healthy atlas sky beside a sample's own.

Gauge (author 2026-10-16, for webpage viewers and doctors; researchers get the table): left end the floor H_min (below it the cell cannot hold
its identity; derived), the Normal band around healthy 1 (measured), right end saturation, every identity site fully scrambled (derived); printed as 0, 1 and the saturation value (author 2026-10-18). Right of healthy
is more writing than healthy, toward the full surface. Because healthy sits near the floor on IAM-A's scale, a magnified strip under the bar
shows the same numbers from the floor to a little past the band. Every mark comes from a committed file; nothing is typed here.
The C-Score measures clustering, not tagging: it has no derived floor or ceiling, so its bar runs from scattered to clustered with its band."""
import os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
INK, SUB, NORM, BAR = "#1d2330", "#556070", "#2e8b57", "#eef1f5"

def sci(v):   # 1.8e-08 -> 1.8 x 10^-8 (mathtext)
    m, e = f"{v:.1e}".split("e"); return rf"{m}$\times10^{{{int(e)}}}$"
def _bar(ax, lo_v, hi_v, band, A, fl, cl, y0=0.0, h=1.0, grad=True, ticks=None, mid=1.0):
    span = hi_v - lo_v; x = lambda v: (v - lo_v) / span
    if grad:   # neutral at healthy (mid), bluer toward the floor, redder toward the ceiling
        xs = np.linspace(0, 1, 512); xm = min(max(x(mid), 0.02), 0.98)
        g = np.where(xs < xm, 0.5 * xs / xm, 0.5 + 0.5 * (xs - xm) / (1 - xm))[None, :]
        ax.imshow(g, extent=(0, 1, y0, y0 + h), aspect="auto", cmap="RdBu_r", vmin=-0.15, vmax=1.15, alpha=0.38, zorder=0)
    ax.add_patch(Rectangle((0, y0), 1, h, fc="none", ec="#9aa3b2", lw=0.8, zorder=3))
    if band: ax.add_patch(Rectangle((x(band[0]), y0), x(band[1]) - x(band[0]), h, fc=NORM, alpha=0.35, lw=0, zorder=1))
    if A is not None:
        xa = min(max(x(A), 0), 1); ax.plot([xa, xa], [y0 - 0.12 * h, y0 + 1.12 * h], color=INK, lw=2.2, zorder=4, solid_capstyle="butt")
    for v, lab in (ticks or []):
        if lo_v <= v <= hi_v: ax.plot([x(v)] * 2, [y0 - 0.1 * h, y0], color=SUB, lw=0.7); ax.text(x(v), y0 - 0.16 * h, lab, ha="center", va="top", fontsize=6.8, color=SUB)
    return x

def gauge_png(out, name, A, floor, ceiling, band, subject, state="", unit="A"):
    """A floor-to-ceiling gauge for Met-A or IAM-A; A None draws no needle (a withheld state draws no gauge at all in the HTML)."""
    zoom = (band[1] - floor) / (ceiling - floor) < 0.25
    fig = plt.figure(figsize=(7.2, 2.25 if zoom else 1.55), facecolor="white")
    ax = fig.add_axes([0.04, 0.50 if zoom else 0.30, 0.92, 0.22 if zoom else 0.32]); ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.6, 1.9); ax.axis("off")
    _bar(ax, floor, ceiling, band, A, floor, ceiling)
    ax.text(0, -0.45, "0  floor\nbelow it a cell can no longer hold its identity", ha="left", va="top", fontsize=6.8, color=SUB)   # author 2026-10-18: 0, not 1e-7
    ax.text(1, -0.45, f"{ceiling:.2f}  saturation\nthe most this scale can register", ha="right", va="top", fontsize=6.8, color=SUB)
    if (1 - floor) / (ceiling - floor) > 0.22:   # near the floor (IAM-A) the label goes on the magnified strip instead, so the two never collide
        ax.text((1 - floor) / (ceiling - floor), -0.45, "1  healthy reference", ha="center", va="top", fontsize=6.8, color="#1f6b43")
    if A is not None: ax.text(min(max((A - floor) / (ceiling - floor), 0.06), 0.94), 1.3, f"{name} {A:.3f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=INK)
    fig.text(0.04, 0.95, f"{name}  ·  {subject}" + (f"  ·  {state}" if state else ""), fontsize=8, color=INK, va="top")
    if zoom:
        hi_z = min(np.ceil(max(band[1] * 2.0, (A or 0) * 1.25, 2.0) * 2) / 2, ceiling)
        az = fig.add_axes([0.04, 0.10, 0.92, 0.13]); az.set_xlim(-0.01, 1.01); az.set_ylim(-0.9, 1.6); az.axis("off")
        xz = _bar(az, floor, hi_z, band, A, floor, hi_z, ticks=[(v, f"{v:g}") for v in np.arange(0.5, hi_z + 1e-9, 0.5)])
        az.text(0, -0.25, "floor", ha="left", va="top", fontsize=6.8, color=SUB)
        az.text((xz(band[0]) + xz(band[1])) / 2, 1.15, "Normal band around 1, the healthy reference", ha="center", va="bottom", fontsize=6.8, color=NORM)
        fig.text(0.04, 0.27, f"magnified: from the floor to {hi_z:g}, the same numbers and the same needle", fontsize=6.8, color=SUB, style="italic")
    fig.savefig(out, dpi=170, facecolor="white"); plt.close(fig); return out

def cscore_png(out, C, band_hi, band_lo, subject, state=""):
    """The C-Score bar: scattered on the left, clustered on the right; its Normal band; no derived ends (it is a clustering ratio)."""
    lo_v = 0.5; hi_v = max(3.0, float(np.ceil((C or 0) * 1.2)))   # the scale stretches to hold the reading (2026-10-18)
    step = 0.5 if hi_v <= 3 else (1.0 if hi_v <= 8 else float(np.ceil(hi_v / 8)))
    fig = plt.figure(figsize=(7.2, 1.7), facecolor="white"); ax = fig.add_axes([0.04, 0.34, 0.92, 0.30]); ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.6, 1.9); ax.axis("off")
    _bar(ax, lo_v, hi_v, (band_lo, band_hi), C, lo_v, hi_v, ticks=[(v, f"{v:g}") for v in np.arange(1.0 if step >= 1 else 0.5, hi_v + 1e-9, step)])
    fig.text(0.04, 0.06, "\u2190 scattered", ha="left", fontsize=7, color=SUB); fig.text(0.96, 0.06, "clustered \u2192", ha="right", fontsize=7, color=SUB)
    if C is not None: ax.text(min(max((C - lo_v) / (hi_v - lo_v), 0.08), 0.92), 1.3, f"C-Score {C:.2f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=INK)
    fig.text(0.04, 0.95, f"C-Score  ·  {subject}" + (f"  ·  {state}" if state else ""), fontsize=8, color=INK, va="top")
    fig.savefig(out, dpi=170, facecolor="white"); plt.close(fig); return out

def atlas_sky_png(out, ids, cell, atlas_dir, nest_pix2ang, title=""):
    """The healthy reference of `cell` from IAM-Atlas v2.1 at the same sites, in the same order and on the same pixels as the sample's
    C-Score sky (stage_v4.cscore_sky), so the two pictures line up site for site. A picture; nothing is scored from it."""
    import pyarrow.parquet as pq, pandas as pd
    from scipy.spatial import cKDTree
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Ellipse
    sites = pq.read_table(os.path.join(atlas_dir, "sites.parquet")).column("cpg").to_pylist()
    part = next(k for k in (1, 2, 3) if cell in pq.read_schema(os.path.join(atlas_dir, f"mu_part{k}.parquet")).names)
    u = pq.read_table(os.path.join(atlas_dir, f"mu_part{part}.parquet"), columns=[cell]).column(cell).to_numpy()
    v = u.astype(np.float32) / 1e4; v[u == 65535] = np.nan
    mu = pd.Series(v, index=sites).reindex(list(ids)).to_numpy(); n = len(mu)
    nside = 1
    while 12 * nside * nside * 2 < n: nside *= 2
    npix = 12 * nside * nside; pix = (np.arange(n) * npix) // n; ok = np.isfinite(mu)
    s = np.bincount(pix[ok], mu[ok], npix); k = np.bincount(pix[ok], minlength=npix); sky = np.where(k > 0, s / np.maximum(k, 1), np.nan)
    th, ph = nest_pix2ang(nside, np.arange(npix)); xyz = np.c_[np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]
    X, Y = np.meshgrid((np.arange(1600) + 0.5) / 1600 * 4 - 2, (np.arange(800) + 0.5) / 800 * 2 - 1)
    inside = (X / 2) ** 2 + Y ** 2 < 1; t = np.arcsin(np.clip(Y, -1, 1)); lat = np.arcsin(np.clip((2 * t + np.sin(2 * t)) / np.pi, -1, 1))
    lon = np.pi * X / (2 * np.cos(np.where(inside, t, 0)))
    q = np.c_[(np.cos(lat) * np.cos(-lon)).ravel(), (np.cos(lat) * np.sin(-lon)).ravel(), np.sin(lat).ravel()]
    img = np.where(inside, sky[cKDTree(xyz).query(q)[1]].reshape(X.shape), np.nan)
    rgb = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Planck_Parchment_RGB.txt")   # inside the chain, so a Biological_Physics-only clone draws the same sky (2026-10-19)
    cm = ListedColormap(np.loadtxt(rgb) / 255.0); cm = cm.copy() if hasattr(cm, "copy") else cm; cm.set_bad("#d9d9d9")
    fig, ax = plt.subplots(figsize=(7.2, 4.2)); fig.patch.set_facecolor("white")
    im = ax.imshow(np.ma.masked_invalid(img), cmap=cm, vmin=0, vmax=1, origin="lower", interpolation="nearest", extent=(-2, 2, -1, 1))
    e = Ellipse((0, 0), 4, 2, transform=ax.transData, fc="none", ec="#555555", lw=0.5); ax.add_patch(e); im.set_clip_path(e)
    ax.set_xlim(-2.02, 2.02); ax.set_ylim(-1.02, 1.02); ax.set_aspect("equal"); ax.set_axis_off()
    ax.set_title(title or f"Healthy {cell} in IAM-Atlas v2.1, the same sites", loc="left", fontsize=9.5)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.045, pad=0.05, shrink=0.45, ticks=[0, 0.5, 1]); cb.set_label("methylation (atlas best value)", fontsize=9); cb.ax.tick_params(labelsize=8)
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    hw = None   # v2.1 only: how sure the reference is at exactly these sites (half-width of its 95 % range)
    try:
        lo_ = pq.read_table(os.path.join(atlas_dir, f"mu_lo_part{part}.parquet"), columns=[cell]).column(cell).to_numpy()
        hi_ = pq.read_table(os.path.join(atlas_dir, f"mu_hi_part{part}.parquet"), columns=[cell]).column(cell).to_numpy()
        w = (hi_.astype(np.float32) - lo_.astype(np.float32)) / 2e4; w[(lo_ == 65535) | (hi_ == 65535)] = np.nan
        w = pd.Series(w, index=sites).reindex(list(ids)).to_numpy(); hw = dict(median=round(float(np.nanmedian(w)), 4), p90=round(float(np.nanpercentile(w, 90)), 4))
    except Exception: pass
    return {"drawn": True, "png": os.path.abspath(out), "cell": cell, "sites": n, "sites_in_atlas": int(ok.sum()), "nside": nside, "order": "NESTED",
            "atlas": "IAM-Atlas v2.1", "range_halfwidth": hw}

def changes_sky_png(out, z, cell, atlas_dir, nest_pix2ang, readings="", thr=2.0, title=""):
    """The sample laid over its healthy reference (author 2026-10-18: overlay the two, quiet the matches, show the changes). Same sites, same
    order, same pixels as the C-Score sky. Background: the healthy cell from IAM-Atlas v2.1 in grays (darker = more tagged). Color only where
    the sample's pattern (entropy per site, the Met-A departure) differs from healthy by more than `thr` of its own spread (the C-Score's departures, scaled as on the C-Score sky). Scattered
    single colored pixels are ordinary; colored patches are changes that run along the genome, which is what the C-Score measures."""
    import pyarrow.parquet as pq, pandas as pd
    from scipy.spatial import cKDTree
    from matplotlib.patches import Ellipse, Patch
    zz = pd.Series(z, dtype="float64"); ids = list(zz.index); n = len(zz)
    med = float(zz.median()); mad = float((zz - med).abs().median()) * 1.4826 or float(zz.std()) or 1.0; d = ((zz - med) / mad).to_numpy()
    sites = pq.read_table(os.path.join(atlas_dir, "sites.parquet")).column("cpg").to_pylist()
    part = next(k for k in (1, 2, 3) if cell in pq.read_schema(os.path.join(atlas_dir, f"mu_part{k}.parquet")).names)
    u = pq.read_table(os.path.join(atlas_dir, f"mu_part{part}.parquet"), columns=[cell]).column(cell).to_numpy()
    v = u.astype(np.float32) / 1e4; v[u == 65535] = np.nan; mu = pd.Series(v, index=sites).reindex(ids).to_numpy()
    nside = 1
    while 12 * nside * nside * 2 < n: nside *= 2
    npix = 12 * nside * nside; pix = (np.arange(n) * npix) // n
    def pmean(x):
        ok = np.isfinite(x); s = np.bincount(pix[ok], x[ok], npix); k = np.bincount(pix[ok], minlength=npix); return np.where(k > 0, s / np.maximum(k, 1), np.nan)
    D, H = pmean(d), pmean(mu)
    th, ph = nest_pix2ang(nside, np.arange(npix)); xyz = np.c_[np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]
    X, Y = np.meshgrid((np.arange(1600) + 0.5) / 1600 * 4 - 2, (np.arange(800) + 0.5) / 800 * 2 - 1)
    inside = (X / 2) ** 2 + Y ** 2 < 1; t = np.arcsin(np.clip(Y, -1, 1)); lat = np.arcsin(np.clip((2 * t + np.sin(2 * t)) / np.pi, -1, 1))
    lon = np.pi * X / (2 * np.cos(np.where(inside, t, 0)))
    q = np.c_[(np.cos(lat) * np.cos(-lon)).ravel(), (np.cos(lat) * np.sin(-lon)).ravel(), np.sin(lat).ravel()]
    idx = cKDTree(xyz).query(q)[1].reshape(X.shape); di, hi = D[idx], H[idx]
    g = np.where(np.isfinite(hi), 0.95 - 0.30 * np.nan_to_num(hi), 0.97)          # healthy backdrop: light grays
    rgb = np.dstack([g, g, g]); cm = plt.get_cmap("RdBu_r"); ch = np.isfinite(di) & (np.abs(di) > thr) & inside
    col = cm(np.clip((di + 4) / 8, 0, 1))[..., :3]; rgb[ch] = col[ch]; rgb[~inside] = 1.0
    fig, ax = plt.subplots(figsize=(7.2, 4.3)); fig.patch.set_facecolor("white")
    im = ax.imshow(rgb, origin="lower", interpolation="nearest", extent=(-2, 2, -1, 1))
    e = Ellipse((0, 0), 4, 2, transform=ax.transData, fc="none", ec="#555555", lw=0.5); ax.add_patch(e); im.set_clip_path(e)
    ax.set_xlim(-2.02, 2.02); ax.set_ylim(-1.02, 1.02); ax.set_aspect("equal"); ax.set_axis_off()
    ax.set_title(title or f"What changed at Met-A's {len(z):,} {cell} sites: this sample laid over healthy {cell}", loc="left", fontsize=9.5)
    nch = int((np.isfinite(D) & (np.abs(D) > thr)).sum()); nd = int(np.isfinite(D).sum())
    ax.legend(handles=[Patch(color="#c8c8c8", label="matches healthy (gray shade: the healthy cell's tagging)"),
                       Patch(color=cm(0.92), label=f"more disorder than healthy: pattern lost (beyond +{thr:g} of its own spread)"),
                       Patch(color=cm(0.08), label=f"more order than healthy (beyond \u2212{thr:g})")],
              loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=1, fontsize=7.2, frameon=False)
    if readings: fig.text(0.5, 0.005, readings + f"   \u00b7   {nch} of {nd} pixels changed", ha="center", fontsize=9)
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    return {"drawn": True, "png": os.path.abspath(out), "cell": cell, "nside": nside, "order": "NESTED", "threshold": thr,
            "pixels_changed": nch, "pixels": nd, "atlas": "IAM-Atlas v2.1"}
