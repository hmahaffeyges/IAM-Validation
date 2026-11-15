#!/usr/bin/env python3
"""PHY-091 amendment 3 synthetic test (runs on GitHub explore.yml; written before any library is read, 2026-11-15). The chain's lambda meter
(stage_q2_selfref.lambda_meter, called by run_sample.py --dev-count-only with --lambda-bam/--lambda-fasta) must recover planted conversion failure
on an unmethylated spike-in genome. A constructed 48,502-base genome named NC_001416.1 (random, GC 0.5, fixed seed); bwa-meth style reads
(top strand: C read as T unless the conversion fails; bottom strand: G read as A; YD:Z:f / YD:Z:r, MAPQ 60); planted failure rates 0, 0.003, 0.01
and 0.03 at every cytosine (CpG and non-CpG alike, since lambda carries no methylation). Checks:
  (1) each planted rate recovered within 10 % relative (or below 2e-4 at 0);
  (2) reads shorter than 50 aligned bases are excluded (a planted block of 40-base reads with failure 0.5 must not move the reading);
  (3) a reference with no lambda contig returns None.
Writes xout/lambda_synthetic_summary.txt; exit 1 if any check fails."""
import os, sys, random, subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pysam", "numpy"], check=False)
import pysam
H = os.path.dirname(os.path.abspath(__file__)); CH = os.path.abspath(os.path.join(H, "..", "..", "..", "..", "chain")); sys.path.insert(0, CH)
import stage_q2_selfref as Q2
OUT = os.path.abspath("xout"); W = os.path.join(OUT, "lam_syn"); os.makedirs(W, exist_ok=True)
rng = random.Random(20261115); G = "".join(rng.choice("ACGT") for _ in range(48502)); CTG = "NC_001416.1"
fa = os.path.join(W, "lambda.fa"); open(fa, "w").write(f">{CTG}\n" + "\n".join(G[i:i + 80] for i in range(0, len(G), 80)) + "\n"); pysam.faidx(fa)
fa0 = os.path.join(W, "other.fa"); open(fa0, "w").write(">chrX\n" + G[:5000] + "\n"); pysam.faidx(fa0)
comp = {"A": "T", "C": "G", "G": "C", "T": "A"}

def make_bam(path, f, n=6000, L=100, short_block=0):
    hdr = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": CTG, "LN": len(G)}]}
    reads = []
    for i in range(n + short_block):
        Lr = L if i < n else 40; fr = f if i < n else 0.5
        s = rng.randrange(2, len(G) - Lr - 2); top = rng.random() < 0.5; ref = G[s:s + Lr]
        if top: q = "".join(("C" if rng.random() < fr else "T") if b == "C" else b for b in ref)
        else: q = "".join(("G" if rng.random() < fr else "A") if b == "G" else b for b in ref)
        reads.append((s, top, q, i))
    reads.sort()
    with pysam.AlignmentFile(path, "wb", header=hdr) as B:
        for s, top, q, i in reads:
            a = pysam.AlignedSegment(B.header); a.query_name = f"r{i}"; a.query_sequence = q; a.flag = 0 if top else 16
            a.reference_id = 0; a.reference_start = s; a.mapping_quality = 60; a.cigar = [(0, len(q))]
            a.query_qualities = pysam.qualitystring_to_array("I" * len(q)); a.set_tag("YD", "f" if top else "r"); B.write(a)
    pysam.index(path)

L = ["PHY-091 amendment 3 synthetic: the chain's lambda meter on planted conversion failure"]; bad = 0
for f in (0.0, 0.003, 0.01, 0.03):
    p = os.path.join(W, f"lam_{f}.bam"); make_bam(p, f); r = Q2.lambda_meter(p, fa); got = r["conv_fail_lambda"]
    ok = (got < 2e-4) if f == 0 else abs(got - f) / f < 0.10; bad += not ok
    L.append(f"  planted {f:.4f}  recovered {got:.5f}  reads {r['reads']}  cytosines {r['cytosines']}  {'PASS' if ok else 'FAIL'}")
p = os.path.join(W, "lam_short.bam"); make_bam(p, 0.003, short_block=3000); r = Q2.lambda_meter(p, fa); got = r["conv_fail_lambda"]
ok = abs(got - 0.003) / 0.003 < 0.10; bad += not ok
L.append(f"  planted 0.0030 plus 3,000 reads of 40 bases at 0.5: recovered {got:.5f} ({'PASS: short reads excluded' if ok else 'FAIL'})")
r = Q2.lambda_meter(os.path.join(W, "lam_0.003.bam"), fa0); ok = r is None; bad += not ok
L.append(f"  reference with no lambda contig: {r} ({'PASS' if ok else 'FAIL'})")
L.append("SYNTHETIC MET" if not bad else f"SYNTHETIC NOT MET ({bad} checks failed)")
open(os.path.join(OUT, "lambda_synthetic_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L)); sys.exit(1 if bad else 0)
