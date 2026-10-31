"""Met-A on sequencing, the selection-free healthy test (written and committed before any of its samples is read). 35 healthy sequenced
samples of atlas cells that are not atlas inputs and were never read (selfree_samples.csv: GSE107729, 25 prefrontal NeuN+ neuron nuclei of
control brains, hg19 per-CpG tables; GSE223153, 5 freshly sorted pancreatic acinar and 5 duct samples, hg38 Bismark coverage), each written
as .lbeta by the one command (--build seq-table) and read by it with the full chain against the IAM Sequencing Atlas (--dev-seq-values 1
--dev-seq-ref auto --full-chain --dev-metaa-sites, specimen 'sorted CELL'; depth is the median reads at the identity sites). The instrument's job: a healthy sample reads Met-A Normal.
Bars (at most 3, set before reading): 1. the 25 neurons' median Met-A within 0.05 of 1; 2. the 10 pancreas samples' median Met-A within 0.05
of 1. Recorded, not scored: Met-A against depth; the atlas inputs' 1.05 to 1.07 for comparison (selection at the identity sites predicts the
outside samples read nearer 1). Usage: python3 selfree_score.py BUNDLE_DIR OUT.txt"""
import sys, os, glob, json, pandas as pd
from scipy.stats import spearmanr
H = os.path.dirname(os.path.abspath(__file__)); S = pd.read_csv(os.path.join(H, "selfree_samples.csv")); rows = []
for r in S.itertuples():
    f = sorted(glob.glob(os.path.join(sys.argv[1], "**", f"selfree_{r.gsm}.json"), recursive=True))
    if not f: rows.append(dict(gsm=r.gsm, cell=r.cell, A=None, reason="no bundle")); continue
    B = json.load(open(f[-1])); m = (B.get("development") or {}).get("metaa_atlas") or {}
    t = sorted(glob.glob(os.path.join(sys.argv[1], "**", f"selfree_{r.gsm}_metaa_sites.csv.gz"), recursive=True))
    d = float(pd.read_csv(t[-1]).reads.median()) if t else None   # depth: median reads at the identity sites (--dev-metaa-sites)
    rows.append(dict(gsm=r.gsm, cell=r.cell, A=m.get("A"), depth=d, reading=m.get("reading"), reason=m.get("reason"), n_sites=m.get("n_sites")))
D = pd.DataFrame(rows); D.to_csv(os.path.join(sys.argv[1], "selfree_readings.csv"), index=False); L = [f"read {int(D.A.notna().sum())} of {len(D)}"]; ok = []
for name, sel in (("neurons", D.cell == "cortical neurons"), ("pancreas", D.cell.isin(["pancreatic acinar", "pancreatic duct"]))):
    x = pd.to_numeric(D.A[sel], errors="coerce").dropna(); med = float(x.median()) if len(x) else float("nan"); g = len(x) > 0 and abs(med - 1) <= 0.05; ok.append(g)
    L.append(f"bar {len(ok)} {name}: median Met-A {med:.4f} over {len(x)} (within 0.05 of 1): {'met' if g else 'not met'}")
x = D[D.A.notna() & D.depth.notna()]
if len(x) > 2: L.append(f"recorded: Met-A against depth rho {spearmanr(x.A, x.depth).correlation:+.3f} over {len(x)}")
L.append("not read: " + (", ".join(f"{r.gsm} ({r.reason})" for r in D[D.A.isna()].itertuples()) or "none"))
L.append(f"test: {'met' if all(ok) else 'not met'}"); open(sys.argv[2], "w").write("\n".join(L) + "\n"); print("\n".join(L))
