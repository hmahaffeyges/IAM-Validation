"""Healthy samples from another laboratory, read against the IAM Sequencing Atlas (written and committed before any is read).
The instrument's job: a healthy sample sequenced and processed by another laboratory reads Normal (0.95 to 1.05 on the gauge).
Samples: blueprint_healthy_test_samples.csv (BLUEPRINT's public healthy blood-cell bisulfite libraries, GRCh38 call and coverage bigWigs,
none read before), each read by the one command (run_sample.py --dev-seq-beta CALL.bw,COV.bw --specimen "sorted CELL" and the rsflags
column) against every donor of the atlas. Scored on the adult venous-blood libraries (the cord-blood libraries are newborn cells read
against an adult atlas: recorded). Bars: 1. Met-A in the sample's own cell, median within 0.95 to 1.05; 2. at least 75 % of those readings
within 0.95 to 1.05; 3. Met-A for the whole sample, median within 0.95 to 1.05. Recorded, not scored: the islands, the domains, the C-Score,
coupling-A, the cell shares (own cell), and each reading by cell and by group. IAM-A cannot read a bigWig (no molecules).
Usage: python3 blueprint_healthy_score.py BUNDLE_DIR OUT.csv OUT.txt"""
import os, sys, glob, json, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); S = pd.read_csv(os.path.join(H, "blueprint_healthy_test_samples.csv")); rows = []
for r in S.itertuples():
    f = sorted(glob.glob(os.path.join(sys.argv[1], "**", r.gsm + "_bundle.json"), recursive=True)); o = dict(gsm=r.gsm, cell=r.atlas_cell, group=r.group)
    if f:
        B = json.load(open(f[-1])); D = B.get("development") or {}; T = (B.get("territories") or {}).get("sets") or {}
        sh = {x["cell"]: x["share"] for x in ((B.get("cell_shares") or {}).get("cells") or [])}
        o.update({"Met-A own cell": (D.get("metaa_atlas") or {}).get("A"), "Met-A whole sample": (B.get("met_a_whole") or {}).get("A"),
                  "islands": (T.get("islands") or {}).get("A"), "domains": (T.get("domains") or {}).get("A"),
                  "C-Score": ((B.get("cscore_each_cell") or {}).get("whole_genome") or {}).get("C"), "coupling-A": (B.get("coupling_a") or {}).get("A"),
                  "own-cell share": sh.get(r.atlas_cell), "not read": (D.get("metaa_atlas") or {}).get("reason")})
    else: o["not read"] = "no bundle"
    rows.append(o)
d = pd.DataFrame(rows); d.to_csv(sys.argv[2], index=False); A = d[d.group.str.startswith("adult")]
inb = lambda x: x.between(0.95, 1.05)
m1 = A["Met-A own cell"].dropna() if "Met-A own cell" in A else pd.Series(dtype=float); m3 = A["Met-A whole sample"].dropna() if "Met-A whole sample" in A else pd.Series(dtype=float)
b1 = len(m1) > 0 and 0.95 <= m1.median() <= 1.05; b2 = len(m1) > 0 and inb(m1).mean() >= 0.75; b3 = len(m3) > 0 and 0.95 <= m3.median() <= 1.05
L = [f"healthy samples from another laboratory: {int(d['Met-A own cell'].notna().sum()) if 'Met-A own cell' in d else 0} of {len(d)} read ({len(A)} adult scored)"]
for g, x in d.groupby("group"):
    L.append(f"  {g} ({len(x)}): " + "; ".join(f"{c} median {pd.to_numeric(x[c], errors='coerce').median():.4f}" for c in ("Met-A own cell", "Met-A whole sample", "islands", "domains", "C-Score", "coupling-A", "own-cell share") if c in x and pd.to_numeric(x[c], errors="coerce").notna().any()))
for c_, x in A.groupby("cell"): L.append(f"    {c_} ({len(x)}): Met-A own cell " + ", ".join(f"{v:.4f}" for v in x["Met-A own cell"].dropna()) if "Met-A own cell" in x else "")
L += [f"bar 1 own-cell Met-A median within 0.95 to 1.05: {m1.median() if len(m1) else None}: {'met' if b1 else 'not met'}",
      f"bar 2 at least 75 % within 0.95 to 1.05: {inb(m1).mean() if len(m1) else None}: {'met' if b2 else 'not met'}",
      f"bar 3 whole-sample Met-A median within 0.95 to 1.05: {m3.median() if len(m3) else None}: {'met' if b3 else 'not met'}",
      f"another laboratory's healthy samples: {'met' if b1 and b2 and b3 else 'not met'}"]
open(sys.argv[3], "w").write("\n".join(L) + "\n"); print("\n".join(L))
