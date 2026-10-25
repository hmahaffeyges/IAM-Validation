"""DEV-ONE-DRAW-01, megakaryocyte ploidy check 1 (written and committed 2026-10-24 before it runs; plan 40.18; development, GSE87197 donor 1
only, spent; donor 3 unread). After provisional reference tests 1 and 2 were not met on donor 2, the sample titles showed each cell's ploidy:
megakaryocytes copy their DNA as they mature (2N young, 8N to 32N mature, the platelet-making cells that shed into plasma). Donor 2's set
leans young (8 of 19 at 2N, none at 16N) against donor 1's (8 of 33 at 2N, every ploidy). The question: does a young megakaryocyte's DNA land
on the marrow progenitors (MEP, erythrocyte progenitors) and a mature one's elsewhere? Donor 1's 2N cells and its 8N-and-up cells are pooled
separately (absorption test 1's code, the sample list as an argument; nothing planted) and each pool read once through the one command, the
atlas only. Development, nothing scored; it decides whether a provisional column should be built from mature cells only.
Usage: python3 mk_ploidy_01.py PLASMA_PAT(unused, kept for the shared code) OUTDIR SAMPLES.csv TAG
Absorption test 1's code follows:
"""
import os, sys, gzip, json, hashlib, subprocess, urllib.request, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); MP = os.path.join(H, "..", "..", "..", ".."); CH = os.path.join(MP, "chain"); sys.path.insert(0, CH)
import stage_seq_values as SV
RS = os.path.join(CH, "MethylPhys_Interface", "run_sample.py"); pat, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
ANN = "https://github.com/zhou-lab/InfiniumAnnotationV1/raw/main/Anno/EPIC/EPIC.hg38.manifest.tsv.gz"
L = [f"DEV-ONE-DRAW-01 megakaryocyte absorption test 1 (GSE87197 donor 1; hg38 annotation {ANN})"]
af = os.path.join(out, "EPIC.hg38.manifest.tsv.gz")
if not os.path.exists(af): urllib.request.urlretrieve(ANN, af)
L.append("annotation sha256 " + hashlib.sha256(open(af, "rb").read()).hexdigest())
K = pd.read_parquet(os.path.join(MP, "atlas", "v2.1", "sites_sequencing_key.parquet")).drop_duplicates("probe")
A = pd.read_csv(af, sep="\t", usecols=["CpG_chrm", "CpG_beg", "Probe_ID"]).dropna().rename(columns={"Probe_ID": "probe"})
A = A.merge(K, on="probe"); A["pos"] = A.CpG_beg.astype(np.int64) + 1; A = A.set_index(["CpG_chrm", "pos"])
m = pd.Series(0, index=A.index, dtype=np.int64); n = m.copy()
S = pd.read_csv(os.path.join(H, sys.argv[3]))
for g, f in zip(S.gsm, S.file):
    p = os.path.join(out, f)
    for a in range(4):
        try: urllib.request.urlretrieve(f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{g[:7]}nnn/{g}/suppl/{f}", p); break
        except Exception: import time; time.sleep(30)
    M = pd.read_csv(p, sep="\t", header=None, names=["chr", "pos", "m", "n"]).groupby(["chr", "pos"]).sum()   # a position listed twice is summed
    R = M.reindex(A.index); m += R["m"].fillna(0).to_numpy(np.int64); n += R["n"].fillna(0).to_numpy(np.int64); os.remove(p)
ok = n > 0; L.append(f"pooled {len(S)} samples: {int(ok.sum())} of {len(A)} atlas sites with reads; median reads at those {int(n[ok].median())}")
idx = A["loyfer_cpg_index"].to_numpy(np.int64); NMAX = SV.HG19_CPGS   # the whole hg19 numbering, as values_from_beta requires
def lbeta(path, mm, nn):
    a = np.zeros((NMAX, 2), np.uint16); a[idx - 1, 0] = np.clip(mm, 0, 65535); a[idx - 1, 1] = np.clip(nn, 0, 65535); a.tofile(path)
def read(tag, path):
    b = os.path.join(out, tag + ".json")
    p = subprocess.run([sys.executable, RS, "--dev-seq-beta", path, "--dev-seq-values", "1", "--specimen", "plasma", "--id", tag,
                        "--out", os.path.join(out, tag + ".html"), "--bundle", b], capture_output=True, text=True)
    B = json.load(open(b)) if os.path.exists(b) else {}; c = B.get("cell_shares") or {}
    if not c.get("cells"): return f"{tag}: no shares ({c.get('withheld') or c.get('reason') or p.stdout[-300:]})"
    top = sorted(c["cells"], key=lambda x: -x["share"])[:8]
    return f"{tag:20} markers {c.get('markers_measured')} | " + ", ".join(f"{x['cell']} {x['share']:.3f}" for x in top)
pm = os.path.join(out, sys.argv[4] + ".lbeta"); lbeta(pm, m.to_numpy(), n.to_numpy()); L.append(read(sys.argv[4], pm)); print(L[-1], flush=True)
open(os.path.join(out, sys.argv[4] + "_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
