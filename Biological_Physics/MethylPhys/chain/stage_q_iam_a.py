#!/usr/bin/env python3
"""Stage Q - IAM-A (single-molecule sequencing), development build, NOT commissioned. Scope: neutrophils.
IAM-A = H(eps) / (P_cell * H(eps0)). eps0 (canon eps0_meth = 1/(1+exp(phi*M)) = 0.032) and P_cell are read from
Runtime Matrices/IAM_A_Positions/iama_positions_v1.json (keys "eps0", "cells.<cell>.P", "cells.<cell>.pipeline").
eps = isolated copy errors / opportunities on qualifying molecules (>= 6 CpG calls, >= 80 % methylated): an interior CpG call is an
opportunity; an unmethylated interior call with both neighbouring calls methylated is an isolated error. Input: a per-site table with
columns pos, opp_A, err_A, opp_B, err_B (A/B = the two halves of the run). P is valid only for the read-level pipeline it was measured
on; the caller must name the pipeline that produced the table (no default) and any other pipeline is refused.

Extractor for the pipeline P was measured on (loyfer_pat_v1): pat_site_table(), the per-site form of chain_tests/iama_floor.py's .pat
counting - wgbstools .pat lines (chrom, first CpG index, pattern of C/T/., molecule count); '.' calls dropped before neighbours are
taken; halves A/B = odd/even molecule ordinal over the file, as in iama_floor.py's repeat check. P was measured on the first
60,000,000 bytes of each granulocyte .pat.gz (LOYFER_PAT_V1_HEAD_BYTES)."""
import os, json, gzip, zlib, collections, numpy as np, pandas as pd
import stage_m_met_a as SM
HERE = os.path.dirname(os.path.abspath(__file__)); POS = os.path.join(HERE, "Runtime Matrices", "IAM_A_Positions", "iama_positions_v1.json")
MIN_OPPORTUNITIES = 100_000      # total, both halves
MIN_HALF_OPPORTUNITIES = 50_000  # a half-reading is printed only above this
PAT_PIPELINE = "loyfer_pat_v1"
LOYFER_PAT_V1_HEAD_BYTES = 60_000_000
_P = None
def positions():
    global _P
    if _P is None: _P = json.load(open(POS))
    return _P

def _H(e): return float(-(e * np.log2(e) + (1 - e) * np.log2(1 - e)))

def _pat_lines(path, max_bytes=None):
    """Lines of a .pat or .pat.gz file; a gzip file may be multi-member and may be cut at max_bytes (the tail member is dropped)."""
    with open(path, "rb") as f: data = f.read(max_bytes) if max_bytes else f.read()
    if data[:2] != b"\x1f\x8b":
        out = data
    else:
        out = b""
        while data:
            d = zlib.decompressobj(16 + zlib.MAX_WBITS)
            try: out += d.decompress(data)
            except zlib.error: break
            data = d.unused_data
    for l in out.split(b"\n")[:-1]:   # the last piece may be cut (as iama_floor.py)
        if l: yield l

def pat_site_table(path, max_bytes=None):
    """Per-site table (pos, opp_A, err_A, opp_B, err_B) from one .pat(.gz) file, pipeline loyfer_pat_v1. pos = 'chrom:CpG index'.
    Sum of err / sum of opp over the table equals iama_floor.py's eps for the same file and byte range."""
    acc = collections.defaultdict(lambda: [0, 0, 0, 0]); k = 0; n_lines = n_qual = 0
    for l in _pat_lines(path, max_bytes):
        q = l.split(b"\t")
        if len(q) < 4: continue
        chrom = q[0].decode(); start = int(q[1]); pat = q[2].decode(); n = int(q[3]); n_lines += 1
        calls = [(start + j, x) for j, x in enumerate(pat) if x != "."]
        c = [x for _, x in calls]
        if len(c) < 6 or c.count("C") < 0.8 * len(c):
            k += n; continue
        n_qual += 1
        nA = (k + n + 1) // 2 - (k + 1) // 2; nB = n - nA          # odd molecule ordinals -> half A, even -> half B
        for i in range(1, len(c) - 1):
            r = acc[f"{chrom}:{calls[i][0]}"]
            err = int(c[i] == "T" and c[i - 1] == "C" and c[i + 1] == "C")
            r[0] += nA; r[1] += err * nA; r[2] += nB; r[3] += err * nB
        k += n
    T = pd.DataFrame([(p, *v) for p, v in acc.items()], columns=["pos", "opp_A", "err_A", "opp_B", "err_B"])
    T.attrs.update(pipeline=PAT_PIPELINE, source=os.path.basename(path), max_bytes=max_bytes, n_lines=n_lines, n_qualifying_lines=n_qual, n_molecules=k)
    return T

def read(site_table, cell, pipeline, mask=None):
    """site_table: per-site table (see module doc). cell: e.g. 'neutrophils'. pipeline: the read-level pipeline that produced the table
    (required). mask: positions to drop. Returns the IAM-A record; A is None with a refusal when the reading is not permitted."""
    P = positions(); EPS0 = float(P["eps0"])
    rec = {"stage": "Q", "reading": "IAM-A", "cell": cell, "pipeline": pipeline, "build": "development v3", "A": None, "eps0": EPS0}
    c = P["cells"].get(cell)
    if c is None: rec["refusal"] = f"no frozen IAM-A position for {cell}"; return rec
    if not pipeline: rec["refusal"] = "pipeline not stated: name the read-level pipeline that produced the table"; return rec
    if c["pipeline"] != pipeline: rec["refusal"] = f"position for {cell} was measured on {c['pipeline']}, not {pipeline}: measure P on healthy {cell} with this pipeline first"; return rec
    D = site_table if mask is None else site_table[~site_table.pos.isin(mask)]
    o = {h: float(D[f"opp_{h}"].sum()) for h in "AB"}; e = {h: float(D[f"err_{h}"].sum()) for h in "AB"}
    if o["A"] + o["B"] < MIN_OPPORTUNITIES: rec["refusal"] = f"too few opportunities ({o['A'] + o['B']:.0f} < {MIN_OPPORTUNITIES})"; return rec
    eps = (e["A"] + e["B"]) / (o["A"] + o["B"])
    if not 0 < eps < 1: rec["refusal"] = f"copy error {eps} outside (0, 1): no reading"; return rec
    A = _H(eps) / (c["P"] * _H(EPS0))
    halves = {h: round(_H(e[h] / o[h]) / (c["P"] * _H(EPS0)), 4) for h in "AB" if o[h] > MIN_HALF_OPPORTUNITIES and 0 < e[h] < o[h]}
    rec.update(eps=round(eps, 6), A=round(A, 4), P=c["P"], E_kT=round(float(np.log((1 - eps) / eps)), 4), halves=halves,
               state="Normal" if SM.NORMAL[0] <= A <= SM.NORMAL[1] else ("above Normal" if A > SM.NORMAL[1] else "below Normal"),
               opportunities=int(o["A"] + o["B"]), n_sites=int(len(D)))
    return rec
