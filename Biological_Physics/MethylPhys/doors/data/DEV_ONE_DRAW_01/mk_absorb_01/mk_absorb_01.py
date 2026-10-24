"""DEV-ONE-DRAW-01, megakaryocyte absorption test 1 (written and committed 2026-10-24 before it runs; plan 40.13; author: 'figure out where those
megakaryocytes are actually being placed with the deconvolution of plasma'; spends GSE87197 donor 1 only, donors 2 and 3 stay unread).
IAM-Atlas v2.1 holds no megakaryocytes. Where does their DNA land when the IAM-Deconvolver reads it?
  1. Donor 1's 33 megakaryocyte samples (GEO GSE87197; bone marrow; per-CpG methylated and total reads on GRCh38) pooled, each atlas site
     taken at its hg38 CpG (the Zhou lab's public EPIC hg38 annotation, CpG_beg + 1 = the file's coordinate), written as a .lbeta at the
     site's CpG index (atlas/v2.1/sites_sequencing_key.parquet) and read by run_sample.py --dev-seq-beta --specimen plasma.
  2. The same megakaryocyte levels planted into one spent healthy plasma (SRR23279758, plasma_seq_01) at 10, 25 and 40 % of its reads:
     at each site with megakaryocyte reads, Poisson(c x f / (1 - f)) added reads, each methylated with the pooled megakaryocyte level; read the
     same way. Seed 20261024.
Expected, written before it runs: the pure megakaryocyte pool lands mostly on one or a few atlas cells; as the planted share grows, those cells'
shares rise with it. Nothing scored; the shares say which atlas cells stand in for megakaryocytes until the cell enters the atlas.
Usage: python3 mk_absorb_01.py PLASMA_PAT OUTDIR"""
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
S = pd.read_csv(os.path.join(H, "mk_donor1_samples.csv"))
for g, f in zip(S.gsm, S.file):
    p = os.path.join(out, f)
    for a in range(4):
        try: urllib.request.urlretrieve(f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{g[:7]}nnn/{g}/suppl/{f}", p); break
        except Exception: import time; time.sleep(30)
    M = pd.read_csv(p, sep="\t", header=None, names=["chr", "pos", "m", "n"]).set_index(["chr", "pos"])
    j = M.index.intersection(A.index); m.loc[j] += M.loc[j, "m"].to_numpy(); n.loc[j] += M.loc[j, "n"].to_numpy(); os.remove(p)
ok = n > 0; L.append(f"pooled {len(S)} samples: {int(ok.sum())} of {len(A)} atlas sites with reads; median reads at those {int(n[ok].median())}")
idx = A["loyfer_cpg_index"].to_numpy(np.int64); NMAX = int(K.loyfer_cpg_index.max())
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
pm = os.path.join(out, "mk_pool.lbeta"); lbeta(pm, m.to_numpy(), n.to_numpy()); L.append(read("megakaryocytes", pm)); print(L[-1], flush=True)
vb, vc, _ = SV.values_from_pat(pat, 1)   # the spent plasma's own reads at each atlas site
P = pd.DataFrame({"b": vb, "c": vc}).reindex(A.probe.to_numpy()); pc = P.c.fillna(0).to_numpy(np.int64); pmeth = np.rint(P.b.fillna(0).to_numpy() * pc).astype(np.int64)
lv = np.where(ok, m / np.maximum(n, 1), np.nan); g = np.random.default_rng(20261024)
for f in (0.0, 0.10, 0.25, 0.40):
    add = np.where(np.isfinite(lv), g.poisson(pc * f / (1 - f)), 0); am = g.binomial(add, np.nan_to_num(lv))
    q = os.path.join(out, f"plasma_mk{int(100 * f):02d}.lbeta"); lbeta(q, pmeth + am, pc + add)
    L.append(read(f"plasma + {int(100 * f)} % megakaryocyte", q)); print(L[-1], flush=True)
open(os.path.join(out, "mk_absorb_01_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
