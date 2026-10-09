#!/usr/bin/env python3
"""Stage Q - IAM-A (single-molecule sequencing), development build, NOT commissioned. Scope: neutrophils.
IAM-A = H(eps) / (P_cell * H(eps0)). eps0 (canon eps0_meth = 1/(1+exp(phi*M)) = 0.032) and P_cell are read from
Runtime Matrices/IAM_A_Positions/iama_positions_v2.json (keys "eps0", "cells.<cell>.P", "cells.<cell>.pipeline"). v2 (2026-10-08,
DEV-IAMA-P-WHOLE-01): P measured on WHOLE files; a reading of part of a file is refused, because it reads other sites than P was measured on.
eps = isolated copy errors / opportunities on qualifying molecules (>= 6 CpG calls, >= 80 % methylated): an interior CpG call is an
opportunity; an unmethylated interior call with both neighbouring calls methylated is an isolated error. Input: a per-site table with
columns pos, opp_A, err_A, opp_B, err_B (A/B = the two halves of the run). P is valid only for the read-level pipeline it was measured
on; the caller must name the pipeline that produced the table (no default) and any other pipeline is refused.

Extractor for the pipeline P was measured on (loyfer_pat_v1): pat_site_table(), the per-site form of chain_tests/iama_floor.py's .pat
counting - wgbstools .pat lines (chrom, first CpG index, pattern of C/T/., molecule count); '.' calls dropped before neighbours are
taken; halves A/B = odd/even molecule ordinal over the file, as in iama_floor.py's repeat check. P (v2) was measured on the whole
granulocyte .pat.gz files; v1 (first 60,000,000 bytes of each file) is kept as a record.

IAM-A C-score (DEVELOPMENT - not commissioned; doors/DEV_IAMA_CSCORE_01.md, 2026-10-04): one C per A. Sites in genomic order (chromosome, CpG
index) are cut into blocks of CSCORE_BLOCK_SITES; per block o_b opportunities and k_b isolated errors; with the reading's own eps,
C = sum (k_b - eps o_b)^2 / sum eps (1 - eps) o_b. Independent errors at one rate give C = 1 within sqrt(2 / n_blocks) (derived, not taken
from other readings); C > 1 = errors cluster in genomic order. Fewer than CSCORE_MIN_BLOCKS blocks: no C."""
import os, json, gzip, zlib, collections, numpy as np, pandas as pd
import stage_m_met_a as SM
HERE = os.path.dirname(os.path.abspath(__file__)); POS = os.path.join(HERE, "Runtime Matrices", "IAM_A_Positions", "iama_positions_v2.json")
MIN_OPPORTUNITIES = 100_000      # total, both halves
MIN_HALF_OPPORTUNITIES = 50_000  # a half-reading is printed only above this
PAT_PIPELINE = "loyfer_pat_v1"
LOYFER_PAT_V1_HEAD_BYTES = 60_000_000
CSCORE_BLOCK_SITES = 1000
CSCORE_MIN_BLOCKS = 10
_P = None
def positions():
    global _P
    if _P is None: _P = json.load(open(POS))
    return _P

def _H(e): return float(-(e * np.log2(e) + (1 - e) * np.log2(1 - e)))

def _pat_lines(path, max_bytes=None, chunk=1 << 24):
    """Lines of a .pat or .pat.gz file; a gzip file may be multi-member (bgzip) and may be cut at max_bytes. Streamed (2026-10-04,
    DEV-IAMA-REAL-01: the earlier whole-file read joined every bgzip member into one buffer, which does not finish on a whole file).
    Same lines as before: every byte that decompresses is read, a member cut at max_bytes gives what it holds, the last piece is dropped."""
    left = max_bytes if max_bytes else None
    with open(path, "rb") as f:
        head = f.read(2); f.seek(0)
        gz = head == b"\x1f\x8b"; d = zlib.decompressobj(16 + zlib.MAX_WBITS) if gz else None; carry = b""; dead = False
        while True:
            n = chunk if left is None else min(chunk, left)
            data = f.read(n) if n > 0 else b""
            if left is not None: left -= len(data)
            if not data: break
            if gz:
                out = b""
                while data and not dead:
                    try: out += d.decompress(data)
                    except zlib.error: dead = True; break
                    if d.eof:
                        data = d.unused_data; d = zlib.decompressobj(16 + zlib.MAX_WBITS)
                    else: data = b""
                if dead and not out: break
            else: out = data
            buf = carry + out; parts = buf.split(b"\n"); carry = parts.pop()
            for l in parts:
                if l: yield l
            if dead: break
    # the last piece (carry) may be cut, as before: dropped


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

def _genomic_order(T):
    """Rows of a site table in genomic order: pos 'chrom:index' sorted by chromosome then index; plain integer pos sorted numerically."""
    p = T["pos"].astype(str)
    if p.str.contains(":").all():
        ch = p.str.rsplit(":", n=1).str[0].str.replace("chr", "", regex=False); ix = p.str.rsplit(":", n=1).str[1].astype("int64")
        ck = ch.map(lambda c: int(c) if c.isdigit() else {"X": 23, "Y": 24, "M": 25}.get(c, 99))
        return T.assign(_ck=ck.values, _ix=ix.values).sort_values(["_ck", "_ix"], kind="mergesort")
    return T.assign(_ix=pd.to_numeric(T["pos"], errors="coerce")).sort_values("_ix", kind="mergesort")


def cscore(site_table, eps=None, halves=("A", "B")):
    """IAM-A C-score for the pooled table (opp = opp_A + opp_B) and for each half. Returns {C, n_blocks, se_null, halves: {...}}."""
    T = _genomic_order(site_table); out = {"block_sites": CSCORE_BLOCK_SITES, "definition": "sum (k_b - eps o_b)^2 / sum eps (1 - eps) o_b over blocks of consecutive sites in genomic order; independent errors -> 1"}
    def one(o, k):
        nb = len(o) // CSCORE_BLOCK_SITES
        if nb < CSCORE_MIN_BLOCKS: return {"C": None, "n_blocks": int(nb), "reason": f"fewer than {CSCORE_MIN_BLOCKS} blocks of {CSCORE_BLOCK_SITES} sites"}
        ob = o[:nb * CSCORE_BLOCK_SITES].reshape(nb, -1).sum(1); kb = k[:nb * CSCORE_BLOCK_SITES].reshape(nb, -1).sum(1)
        e = float(kb.sum() / ob.sum()) if ob.sum() > 0 else None
        if not e or not 0 < e < 1: return {"C": None, "n_blocks": int(nb), "reason": "no errors or no opportunities"}
        C = float(((kb - e * ob) ** 2).sum() / (e * (1 - e) * ob).sum())
        return {"C": round(C, 4), "n_blocks": int(nb), "se_null": round(float(np.sqrt(2.0 / nb)), 4), "eps": round(e, 6)}
    oA, kA = T["opp_A"].to_numpy(float), T["err_A"].to_numpy(float); oB, kB = T["opp_B"].to_numpy(float), T["err_B"].to_numpy(float)
    out.update(one(oA + oB, kA + kB)); out["halves"] = {"A": one(oA, kA), "B": one(oB, kB)}
    return out


def read(site_table, cell, pipeline, mask=None, allow_partial=False):
    """site_table: per-site table (see module doc). cell: e.g. 'neutrophils'. pipeline: the read-level pipeline that produced the table
    (required). mask: positions to drop. allow_partial: development only - read a table cut at max_bytes (to reproduce v1).
    Returns the IAM-A record; A is None with a refusal when the reading is not permitted."""
    P = positions(); EPS0 = float(P["eps0"])
    rec = {"stage": "Q", "reading": "IAM-A", "cell": cell, "pipeline": pipeline, "build": "development v3", "A": None, "eps0": EPS0}
    c = P["cells"].get(cell)
    if c is None: rec["refusal"] = f"no frozen IAM-A position for {cell}"; return rec
    if not pipeline: rec["refusal"] = "pipeline not stated: name the read-level pipeline that produced the table"; return rec
    if c["pipeline"] != pipeline: rec["refusal"] = f"position for {cell} was measured on {c['pipeline']}, not {pipeline}: measure P on healthy {cell} with this pipeline first"; return rec
    mb = getattr(site_table, "attrs", {}).get("max_bytes")
    if mb and not allow_partial:
        rec["refusal"] = (f"partial file (first {mb:,} bytes): P for {cell} was measured on whole files; "
                          "a part of a file covers other sites, so read the whole file"); return rec
    rec["position_version"] = P["version"]; rec["coverage"] = "partial (development)" if mb else c.get("coverage", "whole file")
    D = site_table if mask is None else site_table[~site_table.pos.isin(mask)]
    o = {h: float(D[f"opp_{h}"].sum()) for h in "AB"}; e = {h: float(D[f"err_{h}"].sum()) for h in "AB"}
    if o["A"] + o["B"] < MIN_OPPORTUNITIES: rec["refusal"] = f"too few opportunities ({o['A'] + o['B']:.0f} < {MIN_OPPORTUNITIES})"; return rec
    eps = (e["A"] + e["B"]) / (o["A"] + o["B"])
    if not 0 < eps < 1: rec["refusal"] = f"copy error {eps} outside (0, 1): no reading"; return rec
    A = _H(eps) / (c["P"] * _H(EPS0))
    halves = {h: round(_H(e[h] / o[h]) / (c["P"] * _H(EPS0)), 4) for h in "AB" if o[h] > MIN_HALF_OPPORTUNITIES and 0 < e[h] < o[h]}
    rec.update(eps=round(eps, 6), errors=int(e["A"] + e["B"]), A=round(A, 4), P=c["P"], E_kT=round(float(np.log((1 - eps) / eps)), 4), halves=halves,
               state="Normal" if SM.NORMAL[0] <= A <= SM.NORMAL[1] else ("above Normal" if A > SM.NORMAL[1] else "below Normal"),
               opportunities=int(o["A"] + o["B"]), n_sites=int(len(D)))
    rec["cscore"] = {"label": "DEVELOPMENT - not commissioned", **cscore(D)}   # one C per A (DEV-IAMA-CSCORE-01); band not set
    return rec

MIN_REFS = 3   # same rule as Met-A's median tare (conductor_v3.stage_t_tare)


def tare(rec, refs, sample_id=None):
    """IAM-A same-run tare (author decision 2026-10-09, option a; DEVELOPMENT - not commissioned; doors/DEV_IAMA_KIT_01.md).
    A_rel = A / median(A of >= MIN_REFS healthy references of the same cell read the same way: same laboratory, same library kit, same
    pipeline). refs: plain A values, or records {A, id[, pipeline]}; a record whose id equals sample_id is excluded; a record with a different
    pipeline is refused. Removes the laboratory-and-kit offset (Swift vs TruSeq 0.12 on the same cells), as the median tare does for Met-A."""
    out = {"stage": "QT", "method": "IAM-A median tare (same-run healthy references)", "A_rel": None}
    if rec.get("A") is None: out["reason"] = "no IAM-A to tare"; return out
    vals, n_self = [], 0
    for r in (refs or []):
        if isinstance(r, dict):
            if sample_id is not None and str(r.get("id", "")) == str(sample_id): n_self += 1; continue
            if r.get("pipeline") and r["pipeline"] != rec.get("pipeline"):
                out["reason"] = f"reference {r.get('id')} read with {r['pipeline']}, not {rec.get('pipeline')}"; return out
            if r.get("A") is not None: vals.append(float(r["A"]))
        elif r is not None: vals.append(float(r))
    out["n_self_excluded"] = n_self
    if len(vals) < MIN_REFS:
        out["reason"] = f"untared: {len(vals)} same-run healthy references (>= {MIN_REFS} required); read A against P only"; return out
    v = np.array(vals); m = float(np.median(v)); Ar = rec["A"] / m
    out.update(n_refs=len(vals), reference_median=round(m, 4), A_rel=round(Ar, 4), reference_spread_sd=round(float(np.std(v / m, ddof=1)), 4),
               state="Normal" if SM.NORMAL[0] <= Ar <= SM.NORMAL[1] else ("above Normal" if Ar > SM.NORMAL[1] else "below Normal"))
    return out
