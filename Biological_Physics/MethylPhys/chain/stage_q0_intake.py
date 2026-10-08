#!/usr/bin/env python3
"""Stage Q0 - intake for single-molecule files before Stage Q (IAM-A). DEVELOPMENT - not commissioned (doors/DEV_IAMA_INTAKE_01.md).

An array cannot reach Met-A without Stage 0; a single-molecule file now cannot reach IAM-A without Stage Q0. Each check stops the file
with a named reason, as Stage 0 does for arrays, and every measured value is recorded whether or not it stops the file.

Q0.1 readable    the .pat(.gz) decompresses to its end; every line has 4 fields (chrom, CpG index, pattern of C/T/., count) -> QUARANTINE_UNREADABLE_PAT
Q0.2 build       every line's CpG index lies inside its chromosome's index range in the genome dictionary the position P was measured on
                 (hg19, wgbstools 0.1.0: Runtime Matrices/IAM_A_Positions/hg19_cpg_chrom_ranges.json) -> GENOME_BUILD_MISMATCH
Q0.3 conversion  bisulfite conversion rate from the alignment record (lambda spike-in or non-CpG C), when one is given -> QUARANTINE_CONVERSION
Q0.4 coverage    whole genome: molecules on every autosome -> QUARANTINE_NOT_WHOLE_GENOME (opportunities recorded; Stage Q keeps its minimum)
Q0.5 read length share of molecules with >= 6 CpG calls (Stage Q's qualifying length) -> QUARANTINE_SHORT_READS
Q0.6 duplicates  duplicate fraction from the alignment record (marked by Sambamba 0.6.5, as Loyfer 2023), when one is given -> QUARANTINE_DUPLICATES
Q0.7 specimen    purified cells of a type with a frozen position P (IAM-A reads purified cells only) -> SPECIMEN_REFUSED
Q0.8 gate        proceed, or stop at the first refusal; every value recorded.

Limits for Q0.3-Q0.6 are set from healthy files (the edge of their own range), written in a dated note before any test file is read
(Q0_LIMITS below). Until a limit is set its check records the value with "limit not set" and does not stop the file. Q0.1, Q0.2, the
autosome rule of Q0.4 and Q0.7 are rules, not limits, and stop the file now. Nothing is fitted."""
import os, json, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
RANGES = os.path.join(HERE, "Runtime Matrices", "IAM_A_Positions", "hg19_cpg_chrom_ranges.json")
AUTOSOMES = [f"chr{i}" for i in range(1, 23)]
MIN_CALLS = 6                      # Stage Q's qualifying molecule length (>= 6 CpG calls)
Q0_LIMITS = {                      # set from healthy files in a dated note before any test file is read; None = not set
    "conversion_min": None, "min_opportunities": 100_000, "share_ge6_calls_min": None, "duplicate_fraction_max": None,
    "_note": "development: conversion, read-length share and duplicate limits not set (DEV-IAMA-INTAKE-01: set on healthy files first)"}
ACCEPTED = {"isolated neutrophils": "neutrophils", "purified neutrophils": "neutrophils", "sorted neutrophils": "neutrophils",
            "neutrophils": "neutrophils", "blood granulocytes": "neutrophils", "granulocytes": "neutrophils"}


def _ranges():
    if not os.path.exists(RANGES): return None
    R = json.load(open(RANGES)); return {c: tuple(v) for c, v in R["ranges"].items()}


def scan_pat(path):
    """One pass over a .pat(.gz): integrity, per-chromosome molecules, index-range breaches, call-length counts, opportunities."""
    out = {"file": os.path.basename(path), "complete": False, "bad_lines": 0, "lines": 0, "molecules": 0, "molecules_ge6": 0,
           "calls": 0, "opportunities": 0, "per_chrom_molecules": {}, "out_of_range_lines": 0, "first_bad": None}
    R = _ranges(); per = out["per_chrom_molecules"]
    def line(l):
        q = l.split(b"\t")
        if len(q) != 4 or not q[1].isdigit() or not q[3].strip().isdigit() or q[2].strip(b"CT.") != b"":
            out["bad_lines"] += 1; out["first_bad"] = out["first_bad"] or l[:80].decode(errors="replace"); return
        ch = q[0].decode(); i = int(q[1]); p = q[2]; n = int(q[3]); out["lines"] += 1; out["molecules"] += n
        per[ch] = per.get(ch, 0) + n
        k = len(p) - p.count(b"."); out["calls"] += k * n
        if k >= MIN_CALLS: out["molecules_ge6"] += n; out["opportunities"] += (k - 2) * n
        if R is not None:
            r = R.get(ch)
            if r is None or i < r[0] or i + len(p) - 1 > r[1]: out["out_of_range_lines"] += 1
    with open(path, "rb") as f:
        gz = f.read(2) == b"\x1f\x8b"; f.seek(0)
        d = zlib.decompressobj(16 + zlib.MAX_WBITS) if gz else None; carry = b""; ended = not gz
        while True:
            data = f.read(1 << 24)
            if not data: break
            if gz:
                buf = b""
                while data:
                    try: buf += d.decompress(data)
                    except zlib.error: out["error"] = "gzip stream corrupt"; return out
                    if d.eof: data = d.unused_data; d = zlib.decompressobj(16 + zlib.MAX_WBITS); ended = True
                    else: data = b""; ended = False
            else: buf = data
            parts = (carry + buf).split(b"\n"); carry = parts.pop()
            for l in parts:
                if l: line(l)
        if carry.strip(): line(carry)
    out["complete"] = bool(ended); out["range_check"] = "hg19 chromosome index ranges" if R is not None else "not run: hg19_cpg_chrom_ranges.json missing"
    return out


def intake(path, specimen, cell="neutrophils", alignment_qc=None):
    """Stage Q0 on one .pat(.gz). alignment_qc: dict (or JSON path) from the alignment step with 'conversion_rate' and/or
    'duplicate_fraction'; None = not measured here. Returns {stage, checks:[...], decision, refusal_code?, refusal?, measured}."""
    rec = {"stage": "Q0", "status": "development - not commissioned", "limits": Q0_LIMITS, "checks": []}
    def chk(step, ok, value, rule, code=None, stop=True):
        rec["checks"].append({"step": step, "value": value, "rule": rule, "result": ("pass" if ok else ("REFUSED" if stop else "recorded"))})
        if not ok and stop and "refusal_code" not in rec: rec["refusal_code"] = code; rec["refusal"] = f"{step}: {value} ({rule})"
    sp = ACCEPTED.get(str(specimen or "").strip().lower())
    chk("Q0.7 specimen", sp == cell, specimen, f"purified {cell} only (a position P exists for {cell})", "SPECIMEN_REFUSED")
    S = scan_pat(path); rec["measured"] = {k: v for k, v in S.items() if k != "per_chrom_molecules"}
    rec["measured"]["per_chrom_molecules"] = S["per_chrom_molecules"]
    chk("Q0.1 readable", S.get("complete") and not S.get("error") and S["bad_lines"] == 0 and S["lines"] > 0,
        S.get("error") or (f"{S['bad_lines']} malformed lines" if S["bad_lines"] else ("truncated gzip" if not S["complete"] else f"{S['lines']:,} lines")),
        "decompresses to its end; 4 fields per line", "QUARANTINE_UNREADABLE_PAT")
    if S["range_check"].startswith("not run"):
        chk("Q0.2 build", False, S["range_check"], "index ranges of the build P was measured on", "GENOME_BUILD_UNCHECKED")
    else:
        chk("Q0.2 build", S["out_of_range_lines"] == 0, f"{S['out_of_range_lines']:,} lines outside their chromosome's hg19 index range",
            "every line inside hg19 (wgbstools 0.1.0) ranges", "GENOME_BUILD_MISMATCH")
    miss = [c for c in AUTOSOMES if not S["per_chrom_molecules"].get(c)]
    chk("Q0.4 whole genome", not miss, ("all 22 autosomes" if not miss else f"no molecules on {', '.join(miss)}"),
        "P is measured on whole files: molecules on every autosome", "QUARANTINE_NOT_WHOLE_GENOME")
    chk("Q0.4 coverage", True, f"{S['opportunities']:,} interior calls on molecules with >= {MIN_CALLS} calls",
        f"recorded; Stage Q itself requires >= {Q0_LIMITS['min_opportunities']:,} opportunities on qualifying molecules", stop=False)
    rec["checks"][-1]["result"] = "recorded"
    share = S["molecules_ge6"] / S["molecules"] if S["molecules"] else 0.0; rec["measured"]["share_ge6_calls"] = round(share, 4)
    lim = Q0_LIMITS["share_ge6_calls_min"]
    chk("Q0.5 read length", lim is None or share >= lim, f"{share:.4f} of molecules with >= {MIN_CALLS} calls",
        "limit not set" if lim is None else f">= {lim}", "QUARANTINE_SHORT_READS", stop=lim is not None)
    aq = json.load(open(alignment_qc)) if isinstance(alignment_qc, str) else (alignment_qc or {})
    for step, key, lk, code, cmp in (("Q0.3 conversion", "conversion_rate", "conversion_min", "QUARANTINE_CONVERSION", lambda v, l: v >= l),
                                     ("Q0.6 duplicates", "duplicate_fraction", "duplicate_fraction_max", "QUARANTINE_DUPLICATES", lambda v, l: v <= l)):
        v = aq.get(key); l = Q0_LIMITS[lk]
        if v is None: chk(step, True, "not measured here (no alignment record)", "recorded only", stop=False); rec["checks"][-1]["result"] = "not measured"
        else: chk(step, l is None or cmp(v, l), v, "limit not set" if l is None else f"limit {l}", code, stop=l is not None)
    for c in rec["checks"]:
        if c["result"] == "pass" and c["rule"] == "limit not set": c["result"] = "recorded (limit not set)"
    rec["decision"] = "stop" if "refusal_code" in rec else "proceed to Stage Q"
    return rec
