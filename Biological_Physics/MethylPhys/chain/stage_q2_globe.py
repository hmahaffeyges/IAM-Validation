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
