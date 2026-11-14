"""PHY-091 synthetic test (runs on GitHub, explore.yml; written after design/PHY-091.json, before any library is read). Builds a small random
genome (chr1, chr2) and a bwa-meth style BAM (YD:Z:f top / YD:Z:r bottom, paired mates on the same strand) with a PLANTED conversion failure at
non-CpG cytosines and a planted CpG methylation level, then runs the one command, chain/MethylPhys_Interface/run_sample.py --bam --dev-count-only
--assembly mm10, and checks the bundle's conv_fail against the planted value (within 4 counting sigma) at four planted rates. Writes
xout/conv_synthetic_summary.txt; exit 0 = all recovered."""
import os, sys, json, random, subprocess, math
import pysam
H = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(H, "..", "..", "..", "..", "..", ".."))
RS = os.path.join(ROOT, "Biological_Physics", "MethylPhys", "chain", "MethylPhys_Interface", "run_sample.py")
OUT = os.path.abspath("xout"); os.makedirs(OUT, exist_ok=True)
def convert(ref, top, p_fail, p_meth, rnd):
    q = []
    for k in range(1, len(ref) - 1):
        b = ref[k]
        if top and b == "C": q.append(("C" if rnd.random() < p_meth else "T") if ref[k + 1] == "G" else ("C" if rnd.random() < p_fail else "T"))
        elif (not top) and b == "G": q.append(("G" if rnd.random() < p_meth else "A") if ref[k - 1] == "C" else ("G" if rnd.random() < p_fail else "A"))
        else: q.append(b)
    return "".join(q)
def one(p_fail, p_meth=0.8, n_pairs=6000, L=100, seed=11):
    rnd = random.Random(seed); d = os.path.join(OUT, f"syn_{p_fail}"); os.makedirs(d, exist_ok=True)
    G = {c: "".join(rnd.choice("ACGT") for _ in range(400_000)) for c in ("chr1", "chr2")}
    fa = os.path.join(d, "syn.fa"); open(fa, "w").write("".join(f">{c}\n{s}\n" for c, s in G.items())); pysam.faidx(fa)
    hdr = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": c, "LN": len(s)} for c, s in G.items()]}
    reads = []
    for i in range(n_pairs):
        c = rnd.choice(list(G)); top = rnd.random() < 0.5; s1 = rnd.randrange(10, len(G[c]) - 400); s2 = s1 + rnd.randrange(150, 250)
        for j, s0 in enumerate((s1, s2)):
            a = pysam.AlignedSegment(); a.query_name = f"p{i}"; a.reference_id = list(G).index(c); a.reference_start = s0
            a.query_sequence = convert(G[c][s0 - 1:s0 + L + 1], top, p_fail, p_meth, rnd); a.cigarstring = f"{L}M"; a.mapping_quality = 60
            a.flag = 1 | 2 | (64 if j == 0 else 128) | (32 if j == 0 else 16); a.next_reference_id = a.reference_id; a.next_reference_start = s2 if j == 0 else s1
            a.query_qualities = pysam.qualitystring_to_array("I" * L); a.set_tag("YD", "f" if top else "r"); reads.append(a)
    reads.sort(key=lambda a: (a.reference_id, a.reference_start))
    bam = os.path.join(d, "syn.bam")
    with pysam.AlignmentFile(bam, "wb", header=hdr) as f:
        for a in reads: f.write(a)
    pysam.index(bam)
    b = os.path.join(d, "bundle.json")
    r = subprocess.run([sys.executable, RS, "--bam", bam, "--ref-fasta", fa, "--assembly", "mm10", "--specimen", "cultured cells", "--seq-cell", "SYNTH",
                        "--id", f"SYN{p_fail}", "--dev-count-only", "--workers", "2", "--out", os.path.join(d, "r.html"), "--bundle", b],
                       cwd=os.path.dirname(RS), capture_output=True, text=True)
    if r.returncode: print(r.stdout[-1500:], r.stderr[-1500:]); return None
    c = json.load(open(b))["iam_a_count_only"]["count"]; return c
L = []; bad = False
for pf in (0.0, 0.003, 0.01, 0.03):
    c = one(pf)
    if c is None: L.append(f"planted {pf}: the one command FAILED"); bad = True; continue
    n = 6000 * 2 * 100 * 0.19   # rough non-CpG C count, for the counting sigma
    se = math.sqrt(max(pf, 1e-5) * (1 - pf) / n); ok = abs(c["conv_fail"] - pf) <= 4 * se + 2e-4 and c["reads"] > 0
    bad |= not ok
    L.append(f"planted conversion failure {pf:.3f}: chain conv_fail {c['conv_fail']:.5f} (reads {c['reads']}, molecules {c['molecules']}, held_err {c['held_err']}): {'ok' if ok else 'WRONG'}")
L.append("ALL RECOVERED" if not bad else "FAILED")
open(os.path.join(OUT, "conv_synthetic_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L)); sys.exit(1 if bad else 0)
