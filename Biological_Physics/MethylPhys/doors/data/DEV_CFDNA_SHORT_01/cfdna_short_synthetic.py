"""DEV-CFDNA-SHORT-01 (TIS-053), synthetic only, 2026-10-17. No real read is opened.

The instrument's job: on cell-free DNA-length molecules, IAM-A's counter returns the held error that was planted.

Planted human-like molecules (Bismark XM tags): open-sea stretches at 1 % and 3 % CpG density, methylated, with a planted per-site held
error, a planted sequencer error and conversion failure; unmethylated islands at 10 %. Two arms with the SAME planted rates:
  long   150-letter reads (the whole-genome reads IAM-A's line was built on)
  cfDNA  fragment lengths of plasma cfDNA (mode 167, 10-letter periodicity, a short shoulder near 140)
Counted by the chain's counter (stage_q2_selfref.count_bam_region), put through the counter's own response curve (stage_temperature.invert:
calls per molecule -> per-site rate), and meter-corrected (minus the sequencer error).
Bar (one): B1 the cfDNA arm's per-site held error, meter-corrected, is within 3 % of the planted value.
Recorded, not scored: qualifying molecules per million fragments in each arm, and the raw counter output in each arm (if the raw outputs
differ for the same planted rate, a line built on long reads cannot be read directly on cfDNA; the per-site form or a cfDNA line is needed)."""
import os, sys, json, tempfile, numpy as np, pysam
HERE = os.path.dirname(os.path.abspath(__file__)); CH = os.path.normpath(os.path.join(HERE, "..", "..", "..", "chain")); sys.path.insert(0, CH)
import stage_temperature as TL
rng = np.random.default_rng(23); W = tempfile.mkdtemp()
def seq(n, cg):
    s = list(rng.choice(list("AT"), n))
    for i in rng.choice(n - 1, int(cg * n), replace=False): s[i], s[i + 1] = "C", "G"
    return "".join(s)
BLOCKS = [("sea1", 0.01, True)] * 7 + [("sea3", 0.03, True)] * 2 + [("island", 0.10, False)]
REF = "".join(seq(200_000, cg) for _, cg, _ in BLOCKS); KIND = np.repeat(np.arange(len(BLOCKS)), 200_000)
EPS, SUB, CF = 0.045, 0.004, 0.0   # human runs check conversion on non-CpG C, not on a plastid; none planted here
def lengths(n, arm):
    if arm == "long": return np.full(n, 150)
    L = np.where(rng.random(n) < 0.8, rng.normal(167, 9, n), rng.normal(140, 10, n)); L = 10 * np.round(L / 10) + rng.integers(-1, 2, n)
    return np.clip(L, 90, 220).astype(int)
def build(arm, n):
    hdr = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": "chr1", "LN": len(REF)}]}; reads = []; truth = [0, 0]
    Ls = lengths(n, arm); starts = rng.integers(5, len(REF) - 230, n); o = np.argsort(starts)
    for k in o:
        st, L = int(starts[k]), int(Ls[k]); r_ = REF[st:st + L]; methyl_block = BLOCKS[KIND[st]][2]; q = []; xm = []; calls = []
        for j, b in enumerate(r_):
            if b == "C" and j + 1 < L and r_[j + 1] == "G":
                if methyl_block:
                    m = rng.random() >= EPS; calls.append(m)
                    if not m and rng.random() < CF: m = True
                else: m = rng.random() < CF
                if m and rng.random() < SUB: m = False
                q.append("C" if m else "T"); xm.append("Z" if m else "z")
            elif b == "C": q.append("T"); xm.append("h")
            else:
                x = b
                if rng.random() < SUB: x = "T" if b == "A" else "A"
                q.append(x); xm.append(".")
        nn = len(calls)
        if nn >= 6 and sum(calls) >= 0.8 * nn: truth[1] += nn - 2; truth[0] += sum(1 for i in range(1, nn - 1) if not calls[i] and calls[i - 1] and calls[i + 1])
        a = pysam.AlignedSegment(); a.query_name = f"{arm}_{k}"; a.query_sequence = "".join(q); a.flag = 0; a.reference_id = 0; a.reference_start = st
        a.mapping_quality = 40; a.cigarstring = f"{L}M"; a.query_qualities = pysam.qualitystring_to_array("I" * L); a.set_tags([("XM", "".join(xm)), ("XG", "CT"), ("XR", "CT")]); reads.append(a)
    p = f"{W}/{arm}.bam"
    with pysam.AlignmentFile(p, "wb", header=hdr) as B:
        for a in reads: B.write(a)
    pysam.index(p); return p, truth
open(f"{W}/ref.fa", "w").write(">chr1\n" + REF + "\n"); pysam.faidx(f"{W}/ref.fa")
OUT = {"planted": dict(held_error=EPS, sequencer_error=SUB, conversion_failure=CF)}
for arm, n in (("long", 250_000), ("cfDNA", 250_000)):
    bam, tr = build(arm, n); P = TL.Q2.count_bam_region((bam, f"{W}/ref.fa", "chr1", 0, len(REF)))
    eps = P["e_iso"] / max(P["e_opp"], 1); sub = P["at_mm"] / max(P["at_n"], 1); nh = {int(a): b for a, b in P["n_hist"].items()}   # qualifying molecules only (counter fixed 2026-10-18, LESSONS C23)
    ps = TL.invert(eps, nh); c = dict(held_err=eps, sub_err=sub, eps_corr=(ps - sub) / (1 - sub) if ps is not None else float("nan")); q = P["qualifying"]
    OUT[arm] = dict(fragments=n, qualifying_molecules=int(q), per_million_fragments=round(1e6 * q / n), raw_held_err=round(c["held_err"], 5),
                    sub_err=round(c["sub_err"], 5), per_site_corrected=round(c["eps_corr"], 5), within=round(abs(c["eps_corr"] - EPS) / EPS, 4))
OUT["B1_met"] = bool(OUT["cfDNA"]["within"] <= 0.03)
OUT["raw_outputs_differ_by"] = round(OUT["cfDNA"]["raw_held_err"] / OUT["long"]["raw_held_err"] - 1, 4)
TL.Q2.write_json(OUT, os.path.join(HERE, "cfdna_short_synthetic.json"), manifest=["TIS-053"]); print(json.dumps(OUT, indent=1))
