"""The IAM-A globe (DEVELOPMENT, 2026-10-22; to-do step 29; not commissioned by our standards). IAM-A read region by region on the atlas's
globe, with no other person in the reading.
  1. Counting: chain Stage Q2's own rule (stage_q2_selfref.count_pat), unchanged, but kept per place: on the .pat route the held error is
     counted on odd lines (molecules of at least 6 calls and at least 80 % methylated; an interior call that is unmethylated between two
     methylated calls is an isolated error), the methylation calls on even lines, exactly as count_pat splits them. Every CpG of the genome
     belongs to its nearest IAM-Atlas v2.1 site along the genome (sites_sequencing_key.parquet, the sites' CpG index in the .pat files),
     so every molecule counts and every count sits on the globe's own layout. The per-site sums add up to count_pat's genome totals exactly.
  2. The sample's own healthy value: the held error its own loose share predicts on the self-reference line (stage_q2_selfref.predict).
  3. The departure: blocks of 10 neighboring sites in the globe's order (the other globes' blocks); in each block, the errors counted against
     the errors predicted, in the block's own counting noise, z = (E - O p) / sqrt(O p (1 - p)); a block has changed beyond the 5 %
     family-wise line (report_globe_v4.FW_LINE), the other globes' rule.
Functions: site_owner(), count_pat_sites(lines), departures(err, opp, p)."""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.join(HERE, "MethylPhys_Interface")]
import stage_q2_selfref as Q2
ATLAS = os.path.join(HERE, "..", "atlas", "v2.1"); ORDER = os.path.join(HERE, "Runtime Matrices", "Atlas_V21", "sky_order_v21.npz"); BLOCK = 10
FLAG = "development: not commissioned by our standards"
_OWN = {}
def site_owner():
    """(sorted CpG indexes of the atlas sites that have one, their atlas row numbers, the number of atlas sites). Row numbers follow sites.parquet."""
    if "o" not in _OWN:
        import pyarrow.parquet as pq, pandas as pd
        ids = pq.read_table(os.path.join(ATLAS, "sites.parquet")).column("cpg").to_pylist()
        K = pq.read_table(os.path.join(ATLAS, "sites_sequencing_key.parquet")).to_pandas().drop_duplicates("probe").set_index("probe")["loyfer_cpg_index"]
        ix = pd.Series(K).reindex(ids).to_numpy(dtype="float64"); ok = np.isfinite(ix); rows = np.flatnonzero(ok); o = np.argsort(ix[ok], kind="stable")
        _OWN["o"] = (ix[ok][o].astype(np.int64), rows[o], len(ids))
    return _OWN["o"]
def _owner_of(cpg, idx, rows):
    j = np.searchsorted(idx, cpg); j = np.clip(j, 1, len(idx) - 1); left = idx[j - 1]; right = idx[j]
    return rows[np.where(cpg - left <= right - cpg, j - 1, j)]
def count_pat_sites(lines):
    """Per atlas site: held errors, held opportunities, methylated calls, calls (count_pat's rule and line split), and the genome totals."""
    idx, rows, n = site_owner(); E = np.zeros(n); O = np.zeros(n); MC = np.zeros(n); C = np.zeros(n); bufE, bufO, bufM, bufC = [], [], [], []
    tot_e = tot_o = 0
    def flush():
        for buf, arr in ((bufE, E), (bufO, O), (bufM, MC), (bufC, C)):
            if buf:
                a = np.array(buf, dtype=np.int64).reshape(-1, 2); np.add.at(arr, _owner_of(a[:, 0], idx, rows), a[:, 1]); buf.clear()
    for k, line in enumerate(lines):
        p = line.rstrip("\n").split("\t")
        if len(p) < 4: continue
        i, pat, w = int(p[1]), p[2], int(p[3]); pos = [i + q for q, x in enumerate(pat) if x != "."]; s = [x for x in pat if x != "."]
        if not s: continue
        if k % 2 == 0:
            for q, x in zip(pos, s): bufC.extend((q, w)); (x == "C") and bufM.extend((q, w))
        elif len(s) >= Q2.QUAL_MIN_CALLS and s.count("C") >= Q2.QUAL_MIN_METH * len(s):
            for q in range(1, len(s) - 1):
                bufO.extend((pos[q], w)); tot_o += w
                if s[q] == "T" and s[q - 1] == "C" and s[q + 1] == "C": bufE.extend((pos[q], w)); tot_e += w
        if len(bufO) > 4_000_000 or len(bufC) > 4_000_000: flush()
    flush(); return dict(err=E, opp=O, meth=MC, calls=C, total_err=tot_e, total_opp=tot_o)
def departures(err, opp, p):
    """Blocks of BLOCK neighboring counted sites in the globe's order: z of observed against predicted errors; changed beyond FW_LINE."""
    from report_globe_v4 import FW_LINE
    order = np.load(ORDER)["order"]; e, o = err[order], opp[order]; mi = np.flatnonzero(o > 0); nb = len(mi) // BLOCK
    Eb = e[mi[:nb * BLOCK]].reshape(nb, BLOCK).sum(1); Ob = o[mi[:nb * BLOCK]].reshape(nb, BLOCK).sum(1)
    z = (Eb - Ob * p) / np.sqrt(np.maximum(Ob * p * (1 - p), 1e-12)); line = FW_LINE(nb)
    return dict(order=order, measured=mi, blocks=nb, z=z, line=line, changed=np.abs(z) > line, opp_per_block=float(np.median(Ob)) if nb else 0.0, flag=FLAG)

# ---------------------------------------------------------------- the full-genome globe (2026-10-22; author: the IAM-A globe's home)
# Layout: hg19's autosomal CpGs in the .pat files' own index order (chr1..chr22, Stage Q2's HG19_CPGS; the X and Y are left off, as IAM-A's
# loose share leaves them off, so a difference of sex is never drawn as a change), cut into regions of REGION
# consecutive CpGs; region i of n at NESTED pixel floor(i * NPIX / n), NSIDE 256: the atlas globe's rule (sky_order_v21), applied to every
# CpG instead of the array sites. Nothing is fitted; the layout needs no manifest.
REGION = 100
N_CPG = sum(n for _, n in Q2.HG19_CPGS)
def count_pat_regions(lines, region=REGION):
    """count_pat's rule and line split, per region of `region` consecutive CpG indexes: held errors and opportunities, methylated calls and calls."""
    n = N_CPG // region + 1; E = np.zeros(n); O = np.zeros(n); MC = np.zeros(n); C = np.zeros(n); tot_e = tot_o = 0
    bE, bO, bM, bC = [], [], [], []
    def flush():
        for b, a in ((bE, E), (bO, O), (bM, MC), (bC, C)):
            if b: x = np.array(b, dtype=np.int64).reshape(-1, 2); x = x[x[:, 0] <= N_CPG]; a += np.bincount((x[:, 0] - 1) // region, x[:, 1], n); b.clear()
    for k, line in enumerate(lines):
        p = line.rstrip("\n").split("\t")
        if len(p) < 4: continue
        i, pat, w = int(p[1]), p[2], int(p[3]); pos = [i + q for q, x in enumerate(pat) if x != "."]; s = [x for x in pat if x != "."]
        if not s: continue
        if k % 2 == 0:
            for q, x in zip(pos, s):
                bC.extend((q, w))
                if x == "C": bM.extend((q, w))
        elif len(s) >= Q2.QUAL_MIN_CALLS and s.count("C") >= Q2.QUAL_MIN_METH * len(s):
            for q in range(1, len(s) - 1):
                bO.extend((pos[q], w)); tot_o += w
                if s[q] == "T" and s[q - 1] == "C" and s[q + 1] == "C": bE.extend((pos[q], w)); tot_e += w
        if len(bO) > 4_000_000 or len(bC) > 4_000_000: flush()
    flush(); return dict(err=E, opp=O, meth=MC, calls=C, total_err=float(E.sum()), total_opp=float(O.sum()), region=region)   # autosomes
MIN_OPP = 200   # a region is drawn when it holds at least this many held-error opportunities
def _z_self(R, p):
    o = R["opp"]; ok = o >= MIN_OPP; z = np.where(ok, (R["err"] - o * p) / np.sqrt(np.maximum(o * p * (1 - p), 1e-12)), np.nan); return z, ok
def _z_pair(A, B):
    ok = (A["opp"] >= MIN_OPP) & (B["opp"] >= MIN_OPP); ea, eb = A["err"] / np.maximum(A["opp"], 1), B["err"] / np.maximum(B["opp"], 1)
    pp = (A["err"] + B["err"]) / np.maximum(A["opp"] + B["opp"], 1)
    z = np.where(ok, (ea - eb) / np.sqrt(np.maximum(pp * (1 - pp) * (1 / np.maximum(A["opp"], 1) + 1 / np.maximum(B["opp"], 1)), 1e-12)), np.nan); return z, ok
def _page_layers(z, ok, err_rate, meth, p, change_name, about_ref):
    from report_globe_v4 import FW_LINE, _to_sky, _to_sky_extreme, _lookup, _png_b64, _jpg_b64, _vmask, key_css, HT, W
    from matplotlib.colors import ListedColormap
    nside = 256; npix = 12 * nside * nside; look = _lookup(nside); nb = int(ok.sum()); line = FW_LINE(nb); ch = ok & (np.abs(z) > line)
    ex = np.where(ok, np.where(ch, z, 0.0), np.nan); dv = _to_sky_extreme(ex, npix)[look].reshape(HT, W)
    a = np.clip(np.abs(dv) / 20, 0, 1); c = np.where(np.isfinite(dv), np.where(dv != 0, 128 + np.sign(dv) * np.round(48 + 79 * a), 128), 0)
    code = np.repeat(np.clip(c, 0, 255).astype(np.uint8)[..., None], 3, -1)
    cm = ListedColormap(np.loadtxt(os.path.join(HERE, "MethylPhys_Interface", "Planck_Parchment_RGB.txt")) / 255.0)
    def rgb(m):
        x = (cm(np.clip(np.nan_to_num(m, nan=0), 0, 1))[..., :3] * 255).astype(np.uint8); x[~np.isfinite(m)] = (42, 45, 51); return x
    hm = _to_sky(np.where(ok, np.clip(err_rate / (2 * p), 0, 1), np.nan), npix)[look].reshape(HT, W); mm = _to_sky(meth, npix)[look].reshape(HT, W)
    J = lambda b: "data:image/jpeg;base64," + b; Hm = J(_jpg_b64(rgb(hm)))
    up, dn = int((ch & (z > 0)).sum()), int((ch & (z < 0)).sum())
    L = [dict(id="genome", name=change_name, kind="change", src="data:image/png;base64," + _png_b64(code, "RGB"), base=Hm, full=10, ref_word=about_ref,
              key_up="Red: more held marks lost here than " + about_ref + " (held order lost)", key_down="Blue: fewer held marks lost here than " + about_ref,
              hint="Slide right to quiet everything that is not a change: the map drains to gray, then to dark, until only the change is left lit.",
              fixed=f"Changed: {int(ch.sum()):,} of {nb:,} regions of {REGION} CpGs ({up:,} more loss, {dn:,} less), each beyond {line:.1f} of its own counting wobbles, the line pure chance crosses less than once in 20 genomes. This count is fixed; the slider never changes it.",
              note="Each region's isolated losses on held molecules, counted against " + about_ref + ". Development: not commissioned by our standards."),
         dict(name="held error, region by region", kind="image", src=Hm, mask=_vmask(hm), vscale=[0, round(2 * p, 4)], vword="held error at least:",
              hint="Slide right to keep only the regions that lose held marks most often; the rest go dark.", key_css=key_css(),
              note=f"Isolated losses on held molecules over the chances to lose one, 0 to {2 * p:.3f} (twice the sample's predicted {p:.4f})."),
         dict(name="methylation, region by region", kind="image", src=J(_jpg_b64(rgb(mm))), mask=_vmask(mm), vscale=[0, 1], vword="at least this methylated:",
              hint="Slide right to keep only the regions at least this methylated; the rest go dark.", key_css=key_css(), note="The share of methylated calls in each region, 0 to 1.")]
    return L, dict(regions_drawn=nb, line=round(line, 3), changed=int(ch.sum()), more_loss=up, less_loss=dn)
ABOUT = ("<b>DEVELOPMENT: not commissioned by our standards.</b> Every CpG of the genome has one fixed place on this globe, set only by where it "
         "sits in the genome (regions of __R__ CpGs, chr1 to chr22 in order; the X and Y are left off, so a difference of sex is never drawn as a change). IAM-A counts, on single DNA molecules that hold their marks, how "
         "often one mark is lost between two held ones. __WHAT__ A region is changed only when the difference is too big to be chance: beyond "
         "the line that pure counting noise crosses less than once in twenty genomes. Slide right and everything that is not a change drains away.")
def self_globe(R, loose_share, gid, title):
    """One sample against itself: each region's held error against the held error the sample's own loose share predicts (the self-reference line)."""
    p = Q2.predict(loose_share); z, ok = _z_self(R, p); er = R["err"] / np.maximum(R["opp"], 1); mt = np.where(R["calls"] > 0, R["meth"] / np.maximum(R["calls"], 1), np.nan)
    L, rec = _page_layers(z, ok, er, mt, p, "the change from its own predicted held error", "this sample's own predicted held error")
    rec.update(predicted=round(p, 5), loose_share=round(float(loose_share), 4), measured=round(R["total_err"] / max(R["total_opp"], 1), 5))
    about = ABOUT.replace("__R__", str(REGION)).replace("__WHAT__", f"Here one sample is read against itself: its own loose share predicts a held error of {p:.4f} (the self-reference line); no other person enters.")
    return dict(id=gid, name=title, about=about, groups=[dict(name="this sample", layers=L)]), rec
def pair_globe(A, B, gid, title, name_b):
    """One sample against another: each region's held error against the other's, in both regions' counting noise."""
    z, ok = _z_pair(A, B); pp = (A["total_err"] + B["total_err"]) / max(A["total_opp"] + B["total_opp"], 1)
    er = A["err"] / np.maximum(A["opp"], 1); mt = np.where(A["calls"] > 0, A["meth"] / np.maximum(A["calls"], 1), np.nan)
    L, rec = _page_layers(z, ok, er, mt, pp, "the change from " + name_b, name_b)
    about = ABOUT.replace("__R__", str(REGION)).replace("__WHAT__", f"Here two samples are compared region by region, and only where they differ is left lit.")
    return dict(id=gid, name=title, about=about, groups=[dict(name="the two samples", layers=L)]), rec
