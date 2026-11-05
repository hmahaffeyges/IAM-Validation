"""DEV-SEQ-INTAKE-01, the genome-coverage intake check's synthetic (written and committed before it runs). One spent healthy input of the IAM
Sequencing Atlas (a .beta file: two uint8 per hg19 CpG, methylated and total calls) is written four ways and each is read by the one command
(run_sample.py --dev-seq-beta FILE --dev-seq-values 1 --specimen "sorted b cells"); the bundle's status is read back:
  A  the whole file                                   -> must pass intake (bundle not GENOME_COVERAGE_REFUSED)
  B  calls kept on chromosomes 1 to 8 only            -> must be refused (the GSE98837 case)
  C  every CpG kept with probability 0.05 (seed fixed) -> must pass (a thin library, plasma's case)
  D  chromosome 17 removed                            -> must be refused
Usage: python3 genome_coverage_synthetic.py BETA CPG_BED WORKDIR OUT.txt   (CPG_BED: wgbs_tools' hg19 CpG.bed.gz, 28,217,448 rows)"""
import sys, os, gzip, json, subprocess, numpy as np
beta, bed, W, out = sys.argv[1:5]; os.makedirs(W, exist_ok=True)
X = np.fromfile(beta, dtype=np.uint8).reshape(-1, 2); n = len(X); assert n == 28217448, n
ch = np.empty(n, dtype=object); i = 0
with gzip.open(bed, "rt") as f:
    for l in f: ch[i] = l.split("\t", 1)[0]; i += 1
assert i == n, i
first8 = np.isin(ch, [f"chr{k}" for k in range(1, 9)]); c17 = ch == "chr17"
rng = np.random.default_rng(20261104); keep5 = rng.random(n) < 0.05
cases = dict(A=np.ones(n, bool), B=first8, C=keep5, D=~c17); want = dict(A="pass", B="refused", C="pass", D="refused"); L = []; ok = True
# run 2 (after run 1, disclosed): a refused case must also name exactly the chromosomes it lacks
short_want = dict(B=[f"chr{k}" for k in range(9, 23)], D=["chr17"])
MI = os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 3, "chain", "MethylPhys_Interface")
for k, m in cases.items():
    Y = X.copy(); Y[~m] = 0; f = os.path.join(W, f"case_{k}.beta"); Y.tofile(f)
    b = os.path.join(W, f"case_{k}_bundle.json")
    subprocess.run(["python3", "run_sample.py", "--dev-seq-beta", f, "--dev-seq-values", "1", "--specimen", "sorted b cells", "--id", f"synthetic_{k}",
                    "--out", os.path.join(W, f"case_{k}.html"), "--bundle", b], cwd=MI, capture_output=True, text=True)
    B = json.load(open(b)) if os.path.exists(b) else {}
    got = "refused" if B.get("status") == "GENOME_COVERAGE_REFUSED" else ("pass" if B else "no bundle")
    gc = B.get("sequencing_values", {}).get("genome_coverage", {}) if B.get("status") == "GENOME_COVERAGE_REFUSED" else (B.get("sequencing_values") or {}).get("genome_coverage", {})
    ok &= got == want[k] and (k not in short_want or gc.get('short') == short_want[k])
    L.append(f"case {k}: {want[k]} wanted, {got} | median autosome share {gc.get('median_share')} | short {gc.get('short')}")
L.append(f"synthetic: {'met' if ok else 'not met'} (all four cases as wanted, the refused ones naming exactly the chromosomes they lack)")
open(out, "w").write("\n".join(L) + "\n"); print("\n".join(L))
