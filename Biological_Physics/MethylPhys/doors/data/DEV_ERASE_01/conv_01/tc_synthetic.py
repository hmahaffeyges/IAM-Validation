"""PHY-091 amendment 2 synthetic test (runs on GitHub, explore.yml; written after the design sheet's amendment_2, before any library is re-read).
Same constructed genome and bwa-meth style BAM as conv_synthetic.py, with a fixed conversion failure (0.003) and a PLANTED sequencing substitution
of the kind that fakes a mark: T read as C at reference T on top-strand reads, A read as G at reference A on bottom-strand reads, at high base
quality (40). A further 1 % of the same substitution is planted at LOW base quality (5) and must be excluded. Runs the one command and checks the
bundle's tc_err against the planted rate (within 4 counting sigma) at three planted rates, and conv_fail unchanged. Writes
xout/tc_synthetic_summary.txt; exit 0 = all recovered."""
import os, sys, json, random, subprocess, math
import pysam
H = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(H, "..", "..", "..", "..", "..", ".."))
RS = os.path.join(ROOT, "Biological_Physics", "MethylPhys", "chain", "MethylPhys_Interface", "run_sample.py")
OUT = os.path.abspath("xout"); os.makedirs(OUT, exist_ok=True)
def convert(ref, top, p_fail, p_meth, rnd, p_tc=0.0, p_low=0.01):
    q = []; ql = []
    for k in range(1, len(ref) - 1):
        b = ref[k]; qq = 40
        if top and b == "C": q.append(("C" if rnd.random() < p_meth else "T") if ref[k + 1] == "G" else ("C" if rnd.random() < p_fail else "T"))
        elif (not top) and b == "G": q.append(("G" if rnd.random() < p_meth else "A") if ref[k - 1] == "C" else ("G" if rnd.random() < p_fail else "A"))
        elif (top and b == "T") or ((not top) and b == "A"):
            x = rnd.random(); e = "C" if top else "G"
            if x < p_tc: q.append(e)
            elif x < p_tc + p_low: q.append(e); qq = 5
            else: q.append(b)
        else: q.append(b)
        ql.append(qq)
    return "".join(q), ql
def one(p_tc, p_fail=0.003, p_meth=0.8, n_pairs=30000, L=100, seed=11):
    rnd = random.Random(seed); d = os.path.join(OUT, f"tc_{p_tc}"); os.makedirs(d, exist_ok=True)
    G = {c: "".join(rnd.choice("ACGT") for _ in range(400_000)) for c in ("chr1", "chr2")}
    fa = os.path.join(d, "syn.fa"); open(fa, "w").write("".join(f">{c}\n{s}\n" for c, s in G.items())); pysam.faidx(fa)
    hdr = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": c, "LN": len(s)} for c, s in G.items()]}
    reads = []
    for i in range(n_pairs):
        c = rnd.choice(list(G)); top = rnd.random() < 0.5; s1 = rnd.randrange(10, len(G[c]) - 400); s2 = s1 + rnd.randrange(150, 250)
        for j, s0 in enumerate((s1, s2)):
            a = pysam.AlignedSegment(); a.query_name = f"p{i}"; a.reference_id = list(G).index(c); a.reference_start = s0
            sq, ql = convert(G[c][s0 - 1:s0 + L + 1], top, p_fail, p_meth, rnd, p_tc); a.query_sequence = sq; a.cigarstring = f"{L}M"; a.mapping_quality = 60
            a.flag = 1 | 2 | (64 if j == 0 else 128) | (32 if j == 0 else 16); a.next_reference_id = a.reference_id; a.next_reference_start = s2 if j == 0 else s1
            a.query_qualities = pysam.qualitystring_to_array("".join(chr(x + 33) for x in ql)); a.set_tag("YD", "f" if top else "r"); reads.append(a)
    reads.sort(key=lambda a: (a.reference_id, a.reference_start))
    bam = os.path.join(d, "syn.bam")
    with pysam.AlignmentFile(bam, "wb", header=hdr) as f:
        for a in reads: f.write(a)
    pysam.index(bam)
    b = os.path.join(d, "bundle.json")
    r = subprocess.run([sys.executable, RS, "--bam", bam, "--ref-fasta", fa, "--assembly", "mm10", "--specimen", "cultured cells", "--seq-cell", "SYNTH",
                        "--id", f"TC{p_tc}", "--dev-count-only", "--workers", "2", "--out", os.path.join(d, "r.html"), "--bundle", b],
                       cwd=os.path.dirname(RS), capture_output=True, text=True)
    if r.returncode: print(r.stdout[-1500:], r.stderr[-1500:]); return None
    c = json.load(open(b))["iam_a_count_only"]["count"]; return c
L = []; bad = False
for pt in (0.0, 0.002, 0.01):
    c = one(pt)
    if c is None: L.append(f"planted {pt}: the one command FAILED"); bad = True; continue
    n = 60000 / 20 * 94 * 0.25   # sampled reads x positions read x share at reference T (top) / A (bottom), for the counting sigma
    te = c.get("tc_err")
    ok = te is not None and abs(te - pt) <= 4 * math.sqrt(max(pt, 2e-4) / n) + 2e-4 and abs(c["conv_fail"] - 0.003) < 0.001
    bad |= not ok
    L.append(f"planted T->C {pt:.3f} (plus 1 % at quality 5, to be excluded): chain tc_err {te}, conv_fail {c['conv_fail']:.5f}, sub_err {c['sub_err']:.5f}: {'ok' if ok else 'WRONG'}")
L.append("ALL RECOVERED" if not bad else "FAILED")
open(os.path.join(OUT, "tc_synthetic_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L)); sys.exit(1 if bad else 0)
