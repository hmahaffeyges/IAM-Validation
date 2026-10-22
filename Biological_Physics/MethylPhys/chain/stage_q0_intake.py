#!/usr/bin/env python3
"""Stage Q0 - intake for a .pat file before IAM-A (Stage Q2; the retired position reading, Stage Q, also reads it). Its rule checks stop a
file; its limit checks record until a limit is set from healthy files (doors/DEV_IAMA_INTAKE_01.md). The Chain Decoder, section 10.

An array cannot reach Met-A without Stage 0; a single-molecule file now cannot reach IAM-A without Stage Q0. Each check stops the file
with a named reason, as Stage 0 does for arrays, and every measured value is recorded whether or not it stops the file.

Q0.1 readable    the .pat(.gz) decompresses to its end; every line has 4 fields (chrom, CpG index, pattern of C/T/., count) -> QUARANTINE_UNREADABLE_PAT
Q0.2 build       every line's CpG index lies inside its chromosome's index range in the genome dictionary the position P was measured on
                 (hg19, wgbstools 0.1.0: Runtime Matrices/IAM_A_Positions/hg19_cpg_chrom_ranges.json) -> GENOME_BUILD_MISMATCH
Q0.3 conversion  bisulfite conversion rate from the alignment record (lambda spike-in or non-CpG C), when one is given -> QUARANTINE_CONVERSION
Q0.4 coverage    whole genome: molecules on every autosome -> QUARANTINE_NOT_WHOLE_GENOME (opportunities recorded; Stage Q keeps its minimum)
Q0.5 read length share of molecules with >= 6 CpG calls (Stage Q's qualifying length) -> QUARANTINE_SHORT_READS
Q0.6 duplicates  duplicate fraction from the alignment record (marked by Sambamba 0.6.5, as in the healthy atlas GSE186458), when one is given -> QUARANTINE_DUPLICATES
Q0.7 specimen    purified cells of a type with a healthy shape reference (Stage Q2; a frozen position P for the retired Stage Q) -> SPECIMEN_REFUSED
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
    "conversion_min": 0.98,          # ENCODE WGBS data standard: C to T conversion >= 98 % (encodeproject.org/data-standards/wgbs)
    "min_opportunities": 100_000, "share_ge6_calls_min": None, "duplicate_fraction_max": None,
    "_note": "development: conversion limit from the ENCODE WGBS standard; read-length share and duplicate limits not set (DEV-IAMA-INTAKE-01: set on healthy files first)"}
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


GEO_BUILD = "hg19"   # the build Stage Q2's counter and IAM-A's references read (wgbstools 0.1.0 hg19 CpG index); see the Decoder, genome builds
OTHER_BUILDS = (".hg38.", ".grch38.", "_hg38", "-hg38", ".hg18.", ".mm10.", ".mm9.", ".grcm38.", ".grcm39.")

def build_named(path):
    """The genome build a file's name declares, when it names one other than hg19 (e.g. GSM..._x.hg38.pat.gz), else None. Read from the
    name only, before any line is read, so a wrong-build file is named for what it is instead of only failing the index ranges (2026-10-22)."""
    n = os.path.basename(str(path)).lower()
    return next((b.strip("._-") for b in OTHER_BUILDS if b in n), None)

def geo_pat(gsm, dest, build=GEO_BUILD, retries=5):
    """A sequencing sample's .pat file from GEO, on the build the chain reads (2026-10-22, lesson C34: a download that took the first file name
    in alphabetical order got the hg38 file, which sorts before hg19). Lists the sample's supplementary files and keeps the .pat.gz names that
    declare no other build; if more than one remain, the one naming hg19. Refuses, with what GEO holds, when none or several remain.
    Returns {path, name, url, build, other_files, sha256, size}."""
    import re, time, hashlib, urllib.request
    base = f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{gsm[:-3]}nnn/{gsm}/suppl/"
    for t in range(retries):
        try: names = sorted(set(re.findall(rf"{gsm}_[^\"<>/]*\.pat\.gz", urllib.request.urlopen(base, timeout=120).read().decode()))); break
        except Exception:
            if t == retries - 1: raise
            time.sleep(30 * (t + 1))
    keep = [n for n in names if build_named(n) is None]
    if len(keep) > 1: keep = [n for n in keep if build in n.lower()] or keep
    if len(keep) != 1:
        raise RuntimeError(f"{gsm}: GEO holds {names or 'no .pat files'}; the chain reads {build} and could not pick exactly one {build} file. "
                           f"Give the file with --pat after checking its build")
    n = keep[0]; u = base + n; f = os.path.join(dest, n); h = hashlib.sha256()
    for t in range(retries):
        try:
            with urllib.request.urlopen(u, timeout=900) as r, open(f, "wb") as o:
                for c in iter(lambda: r.read(1 << 20), b""): o.write(c); h.update(c)
            break
        except Exception:
            if t == retries - 1: raise
            h = hashlib.sha256(); time.sleep(60 * (t + 1))
    return dict(path=f, name=n, url=u, build=build, other_files=[x for x in names if x != n], sha256=h.hexdigest(), size=os.path.getsize(f))

def intake(path, specimen, cell="neutrophils", alignment_qc=None, selfref_cells=None):
    """Stage Q0 on one .pat(.gz). alignment_qc: dict (or JSON path) from the alignment step with 'conversion_rate' and/or
    'duplicate_fraction'; None = not measured here. Returns {stage, checks:[...], decision, refusal_code?, refusal?, measured}."""
    rec = {"stage": "Q0", "status": "intake: rule checks stop the file; limit checks record only until their limits are set", "limits": Q0_LIMITS, "checks": []}
    _nb = build_named(path)
    if _nb:   # 2026-10-22 (lesson C34): the file's name says it is another build; stopped before reading, and named for what it is
        rec["checks"].append(dict(step="Q0.2 build", ok=False, value=f"the file name declares {_nb}", rule=f"every file on {GEO_BUILD}, the build the counter and references read"))
        rec["refusal_code"] = "GENOME_BUILD_MISMATCH"; rec["decision"] = "stopped"
        rec["refusal"] = (f"Q0.2 build: {os.path.basename(path)} is named as a {_nb} file; the chain reads {GEO_BUILD} (the same CpG index points at a different CpG on "
                          f"another build, so every count would land on the wrong site). Use the {GEO_BUILD} file of the same sample, or run_sample.py --geo-pat GSM...")
        rec["explained"] = dict(check="sequencing intake (Q0.2 build)", outcome="stopped", measured=f"file name declares {_nb}", limit=f"{GEO_BUILD} only",
                                limit_source="stage_q0_intake (fixed rule)", provisional=False, means="the file was made on another genome build than the references",
                                what_to_do=f"use the {GEO_BUILD} .pat file of the same sample (GEO usually holds both), or let the chain fetch it: run_sample.py --geo-pat GSM...")
        return rec
    def chk(step, ok, value, rule, code=None, stop=True):
        rec["checks"].append({"step": step, "value": value, "rule": rule, "result": ("pass" if ok else ("REFUSED" if stop else "recorded"))})
        if not ok and stop and "refusal_code" not in rec:
            rec["refusal_code"] = code; rec["refusal"] = f"{step}: {value} ({rule})"
            rec["explained"] = dict(check=f"sequencing intake ({step})", outcome="stopped", measured=str(value), limit=rule, limit_source="stage_q0_intake (fixed rule)" if step != "Q0.3 conversion" else "the ENCODE whole-genome bisulfite standard",
                                    provisional=False, means={"Q0.1 readable": "the file is damaged or not a .pat file", "Q0.2 build": "the file was made on another genome build than the references",
                                    "Q0.4 whole genome": "the file does not cover the whole genome, so its loose share is not the cell's", "Q0.7 specimen": "IAM-A has no healthy shape reference for this cell type",
                                    "Q0.3 conversion": "too much DNA was left unconverted, so unmethylated sites read as methylated"}.get(step, "the file failed an intake rule"),
                                    what_to_do={"Q0.1 readable": "download the file again", "Q0.2 build": "use the hg19 .pat file of the same sample: GEO usually holds both, named .hg19. and .hg38.; or let the chain fetch the right one: run_sample.py --geo-pat GSM... (Decoder: genome builds)", "Q0.4 whole genome": "give the whole-genome file",
                                    "Q0.7 specimen": "check --seq-cell", "Q0.3 conversion": "re-run the bisulfite conversion"}.get(step, "read the rule and correct the input"))
    if selfref_cells is not None:   # chain v4, Stage Q2: the self-reference reads any purified cell type with a healthy shape reference
        chk("Q0.7 specimen", str(cell).strip().lower() in selfref_cells, cell, "a purified cell type with a healthy shape reference (Stage Q2)", "SPECIMEN_REFUSED")
    else:
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
