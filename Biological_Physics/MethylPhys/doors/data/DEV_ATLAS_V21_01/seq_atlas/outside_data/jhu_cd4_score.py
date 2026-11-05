"""The Johns Hopkins healthy CD4 T-cell test (GSE86340; written and committed before any of its samples is read). The instrument's job: a healthy
sample from a laboratory the IAM Sequencing Atlas has never seen, tared on that laboratory's own healthy samples of the same cell, reads Normal.
Six healthy donors' CD4 T cells (aged 18 to 86), one laboratory, read at full depth by the one command. Each donor's own-cell Met-A is tared on
the other five (A_rel = A / median of the other five, stage_metaa_atlas.tare's rule). Bars: 1. the tared own-cell Met-A median within 0.95 to
1.05; 2. the whole-sample Met-A median within 0.95 to 1.05. Recorded, not scored: the untared own-cell Met-A (the laboratory's offset), each
donor's tared reading, the islands, the domains, the C-Score, coupling-A, IAM-A and the CD4 share. A sample the chain refuses (the genome-coverage
check or Met-A's 90 % identity-site rule) is listed with its reason, and the bars are read on the rest if at least 4 are read.
Usage: python3 jhu_cd4_score.py BUNDLE_DIR OUT.csv OUT.txt"""
import os, sys, glob, json, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); S = pd.read_csv(os.path.join(H, "jhu_cd4_samples.csv")).drop_duplicates("gsm"); rows = []
for r in S.itertuples():
    f = sorted(glob.glob(os.path.join(sys.argv[1], "**", r.gsm + "_bundle.json"), recursive=True)); o = dict(gsm=r.gsm, age=r.age)
    if f:
        B = json.load(open(f[-1])); D = B.get("development") or {}; T = (B.get("territories") or {}).get("sets") or {}
        sh = {x["cell"]: x["share"] for x in ((B.get("cell_shares") or {}).get("cells") or [])}
        o.update({"status": B.get("status", "read"), "Met-A own cell": (D.get("metaa_atlas") or {}).get("A"), "Met-A whole sample": (B.get("met_a_whole") or {}).get("A"),
                  "islands": (T.get("islands") or {}).get("A"), "domains": (T.get("domains") or {}).get("A"),
                  "C-Score": ((B.get("cscore_each_cell") or {}).get("whole_genome") or {}).get("C"), "coupling-A": (B.get("coupling_a") or {}).get("A"),
                  "IAM-A": (B.get("iam_a") or {}).get("A"), "cd4 share": sh.get("cd4 t cells"), "not read": (D.get("metaa_atlas") or {}).get("reason") or B.get("reason")})
    else: o["not read"] = "no bundle"
    rows.append(o)
d = pd.DataFrame(rows); a = pd.to_numeric(d.get("Met-A own cell"), errors="coerce")
d["Met-A own cell, tared on the other donors"] = [a.iloc[i] / np.nanmedian(a.drop(a.index[i])) if pd.notna(a.iloc[i]) and a.drop(a.index[i]).notna().sum() >= 3 else np.nan for i in range(len(a))]
d.to_csv(sys.argv[2], index=False); t = d["Met-A own cell, tared on the other donors"].dropna(); w = pd.to_numeric(d.get("Met-A whole sample"), errors="coerce").dropna()
ok = len(t) >= 4; b1 = ok and 0.95 <= t.median() <= 1.05; b2 = len(w) >= 4 and 0.95 <= w.median() <= 1.05
L = [f"Johns Hopkins healthy CD4 T cells (GSE86340): {int(a.notna().sum())} of {len(d)} read",
     f"untared own-cell Met-A median {a.median():.4f} (the laboratory's offset; recorded)" if a.notna().any() else "untared own-cell Met-A: none read",
     "each donor: " + "; ".join(f"{r.gsm} age {r.age}: {r['Met-A own cell']}, tared {r['Met-A own cell, tared on the other donors']}" for _, r in d.iterrows()),
     f"bar 1 tared own-cell Met-A median within 0.95 to 1.05: {t.median() if len(t) else None}: {'met' if b1 else 'not met'}",
     f"bar 2 whole-sample Met-A median within 0.95 to 1.05: {w.median() if len(w) else None}: {'met' if b2 else 'not met'}",
     f"test: {'met' if b1 and b2 else 'not met'}"]
open(sys.argv[3], "w").write("\n".join(L) + "\n"); print("\n".join(L))
