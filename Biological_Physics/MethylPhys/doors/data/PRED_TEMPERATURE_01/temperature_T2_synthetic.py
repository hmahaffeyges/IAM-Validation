"""PRED-TEMPERATURE-01 stage T2 synthetic (2026-10-17), through chain stage TL (chain/stage_temperature.py). No real read is opened.
A. Counter: planted Arabidopsis-like molecules (Bismark XM tags, nuclear Chr1 and an unmethylated chloroplast) with a planted held error,
   a planted sequencer error and a planted conversion failure; the chain stage must return the rule on the planted molecules after the meter
   correction and the counter's response curve: the planted per-site held error within 2 %, and the conversion failure within 0.001.
B. Design: 111 accessions at 10 C and 16 C (GSE54292), counts drawn as the counter would see them, the two temperatures sequenced with different
   sequencer error. Bars as in chain stage TL: B1 the paired median ratio within a quarter of the law's shift (1.0159-1.0265), B2 its 95 % interval excludes 1.
   Required: law true -> met in at least 95 % of repeats; no temperature effect -> not met in at least 95 %."""
import os, sys, json, tempfile, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); CH = os.path.normpath(os.path.join(HERE, "..", "..", "..", "chain")); sys.path.insert(0, CH)
import stage_temperature as TL
OUT = {}
# ---------------- A. the counter on planted molecules
import pysam
rng = np.random.default_rng(17); W = tempfile.mkdtemp()
def seq(n, cg):   # random sequence with CpGs at a planted density
    s = list(rng.choice(list("AT"), n, p=[0.5, 0.5]))
    for i in rng.choice(n - 1, int(cg * n), replace=False): s[i], s[i + 1] = "C", "G"
    return "".join(s)
REF = {"Chr1": seq(2_000_000, 0.05), "ChrC": seq(60_000, 0.03)}
open(f"{W}/ref.fa", "w").write("".join(f">{k}\n{v}\n" for k, v in REF.items())); pysam.faidx(f"{W}/ref.fa")
EPS, SUB, CF, L = 0.030, 0.004, 0.006, 150
truth = [0, 0]
hdr = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": k, "LN": len(v)} for k, v in REF.items()]}
reads = []
for chrom, n in (("Chr1", 300_000), ("ChrC", 8_000)):
    R = REF[chrom]
    for st in np.sort(rng.integers(5, len(R) - L - 5, n)):
        r_ = R[st:st + L]; q = []; xm = []; calls = []
        for j, b in enumerate(r_):
            cpg = b == "C" and j + 1 < L and r_[j + 1] == "G"
            if cpg:
                if chrom == "ChrC": meth = rng.random() < CF                      # unmethylated plastid; a conversion failure reads methylated
                else:
                    meth = rng.random() >= EPS; calls.append(meth)                # held molecule: each site lost with the planted held error
                    if not meth and rng.random() < CF: meth = True                # a lost site whose conversion failed reads methylated
                if chrom != "ChrC" and meth and rng.random() < SUB: meth = False   # the sequencer misreads a methylated C as T
                q.append("C" if meth else "T"); xm.append("Z" if meth else "z")
            elif b == "C": q.append("T"); xm.append("h")
            else:
                x = b
                if rng.random() < SUB: x = "T" if b == "A" else "A"               # the same sequencer error on A/T bases, which the meter measures
                q.append(x); xm.append(".")
        if calls:
            nn = len(calls)
            if nn >= 6 and sum(calls) >= 0.8 * nn: truth[1] += nn - 2; truth[0] += sum(1 for i in range(1, nn - 1) if not calls[i] and calls[i - 1] and calls[i + 1])
        a = pysam.AlignedSegment(); a.query_name = f"{chrom}_{st}_{len(reads)}"; a.query_sequence = "".join(q); a.flag = 0
        a.reference_id = 0 if chrom == "Chr1" else 1; a.reference_start = int(st); a.mapping_quality = 40; a.cigarstring = f"{L}M"
        a.query_qualities = pysam.qualitystring_to_array("I" * L); a.set_tags([("XM", "".join(xm)), ("XG", "CT"), ("XR", "CT")]); reads.append(a)
with pysam.AlignmentFile(f"{W}/s.bam", "wb", header=hdr) as B:
    for a in reads: B.write(a)
pysam.index(f"{W}/s.bam")
c = TL.count_plant(f"{W}/s.bam", f"{W}/ref.fa", workers=1, chroms=["Chr1"])
tr = truth[0] / truth[1]   # every molecule, while the counter keeps odd molecules for the held error: compare within its own sampling error
se = (tr / max(c["held_opp"], 1)) ** 0.5
OUT["A_counter"] = dict(planted_held_error=EPS, rule_on_planted=round(tr, 5), counter_held_err=round(c["held_err"], 5), sub_err=round(c["sub_err"], 5),
                        planted_sub=SUB, eps_corr=round(c["eps_corr"], 5), conv_fail=round(c["conv_fail"], 5), planted_conv_fail=CF,
                        p_site=round(c["p_site"], 5), corrected_within=round(abs(c["eps_corr"] - EPS) / EPS, 4),
                        met=bool(abs(c["eps_corr"] - EPS) <= 0.02 * EPS and abs(c["conv_fail"] - CF) <= 0.001))
NH = c["n_hist"]   # the planted molecules' calls-per-molecule mix, used for the design below
# ---------------- B. the design
_GRID, _G = TL.response(NH)
def g_of(p): return float(np.interp(p, _GRID, _G))
def sim(ratio_true, n_pairs=111, mol=(50_000, 300_000), sub=(0.002, 0.005), drift=0.0, gxe=0.01, correct=True, rep=0):
    r = np.random.default_rng(1000 + rep); pairs = []
    for _ in range(n_pairs):
        lo_w = TL.logodds(0.03) * np.exp(r.normal(0, 0.05))                      # accession's own warm (16 C) log-odds
        lo_c = lo_w * ratio_true * np.exp(r.normal(0, gxe))                     # the cold one, by the law, with genotype-by-temperature scatter
        row = []
        for lo, s, d in ((lo_c, sub[0], 0.0), (lo_w, sub[1], drift)):
            e = 1 / (1 + np.exp(lo)) * (1 + d); opp = int(r.integers(*mol)) * 5
            p = e + s * (1 - e); obs = r.binomial(opp, g_of(p)) / opp        # what the counter reports: its response to the per-site rate
            sm = r.binomial(10_000_000, s) / 10_000_000
            pi = float(np.interp(obs, _G, _GRID))
            row.append((pi - sm) / (1 - sm) if correct else pi)
        pairs.append(tuple(row))
    return TL.score(pairs, 10, 16, boot=2000, seed=rep)
REP = 40
for name, kw in (("law_true", dict(ratio_true=TL.predicted_ratio(10, 16))), ("no_temperature_effect", dict(ratio_true=1.0)),
                 ("law_true_20k_molecules", dict(ratio_true=TL.predicted_ratio(10, 16), mol=(20_000, 21_000))),
                 ("law_true_no_meter_correction", dict(ratio_true=TL.predicted_ratio(10, 16), correct=False)),
                 ("law_true_batch_drift_3pct", dict(ratio_true=TL.predicted_ratio(10, 16), drift=0.03)),
                 ("law_true_batch_drift_7pct", dict(ratio_true=TL.predicted_ratio(10, 16), drift=0.07))):
    S = [sim(rep=k, **kw) for k in range(REP)]
    OUT[name] = dict(repeats=REP, met=sum(s["met"] for s in S), median_ratio=round(float(np.median([s["median_ratio"] for s in S])), 4),
                     ci_width=round(float(np.median([s["ci95"][1] - s["ci95"][0] for s in S])), 4))
OUT["verdict"] = dict(A=OUT["A_counter"]["met"], B_law=OUT["law_true"]["met"] >= 0.95 * REP, B_null=OUT["no_temperature_effect"]["met"] <= 0.05 * REP)
OUT["met"] = all(OUT["verdict"].values())
TL.Q2.write_json(OUT, os.path.join(HERE, "temperature_T2_synthetic.json"), manifest=["PHY-057"])   # chain-stamped
print(json.dumps(OUT, indent=1))
