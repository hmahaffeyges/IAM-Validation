"""The cross-laboratory rehearsal on spent data with the class floor (DEV_ATLAS_V21_01.md; written and committed before any of its readings
exists). The 89 outside healthy samples already read (BLUEPRINT 59, GSE245832 CD8 12, GSE98837 CD8 8, Johns Hopkins CD4 6, GSE134664 B 4),
each read again by the one command with --dev-floor-class (xl_class_job.sh). Reads only the bundles the one command wrote. Reference: the IAM
Sequencing Atlas only, each sample against the atlas in the mix the IAM-Deconvolver measures in it; no tare.
Per person: the islands reading (territories, islands, counted: A), inside Normal 0.95 to 1.05.
Bars: 1. at least 80 of the 89 samples each read the islands inside Normal; 2. in each of the five laboratories at least 75 % of its samples do.
Recorded, not scored: the whole-sample and domains readings, each sample's floor at the islands, the cell the Deconvolver found most of.
A bundle whose islands floor was not counted at the library-floor sites inside the islands is refused (the flag did not take effect).
Usage: python3 xl_class_score.py FOLDER_OF_BUNDLES OUT.csv OUT.txt"""
import sys, os, glob, json
import pandas as pd
land, oc, ot = sys.argv[1:4]
H = os.path.dirname(os.path.abspath(__file__))
LAB = {}
for f, lab in (("xl_bp_samples.csv", "BLUEPRINT"), ("xl_lab2_samples.csv", "GSE245832 CD8"), ("xl_jhu_samples.csv", "JHU CD4")):
    for g in pd.read_csv(os.path.join(H, f)).gsm.unique(): LAB[g] = lab
for _, r in pd.read_csv(os.path.join(H, "xl_bt3_samples.csv")).drop_duplicates("gsm").iterrows(): LAB[r.gsm] = "GSE98837 CD8" if r.gse == "GSE98837" else "GSE134664 B"
rows, refused = [], []
for f in glob.glob(os.path.join(land, "**", "*_bundle.json"), recursive=True):
    g = os.path.basename(f)[: -len("_bundle.json")]
    if g not in LAB: continue
    B = json.load(open(f)); isl = (((B.get("territories") or {}).get("sets") or {}).get("islands") or {}).get("counted") or {}
    fl = isl.get("floor") or {}
    if "library-floor sites inside the islands" not in str(fl.get("measured_at", "")): refused.append(g); continue
    dom = (((B.get("territories") or {}).get("sets") or {}).get("domains") or {}).get("counted") or {}
    rows.append(dict(gsm=g, lab=LAB[g], islands=isl.get("A"), domains=dom.get("A"), whole=(B.get("met_a_whole") or {}).get("A"),
                     floor_sample=fl.get("sample"), floor_reference=fl.get("reference"), floor_sites=fl.get("sites"),
                     chain_cell=(B.get("label_check") or {}).get("chain_cell")))
X = pd.DataFrame(rows, columns=["gsm", "lab", "islands", "domains", "whole", "floor_sample", "floor_reference", "floor_sites", "chain_cell"]).drop_duplicates("gsm"); X["islands"] = pd.to_numeric(X.islands, errors="coerce"); X["normal"] = (X.islands - 1).abs() <= 0.05
X.to_csv(oc, index=False)
L = [f"Cross-laboratory rehearsal with the class floor: {len(X)} of {len(LAB)} outside healthy samples read (reference the IAM Sequencing Atlas only, no tare)"]
if refused: L.append(f"refused (class floor not counted at the floor sites inside the islands): {len(refused)}: {', '.join(sorted(refused)[:10])}")
per = {}
for lab, g in X.groupby("lab"):
    per[lab] = (int(g.normal.sum()), len(g), sum(1 for v in LAB.values() if v == lab))
    L.append(f"  {lab}: islands inside Normal {int(g.normal.sum())} of {len(g)} read ({per[lab][2]} in the set) | islands median {g.islands.median():.4f}, range {g.islands.min():.3f} to {g.islands.max():.3f}")
for c in ("whole", "domains"):
    v = pd.to_numeric(X[c], errors="coerce"); L.append(f"  recorded: {c} median {v.median():.4f}, inside 0.95 to 1.05: {int(((v - 1).abs() <= 0.05).sum())} of {int(v.notna().sum())}")
b1 = int(X.normal.sum()) >= 80 and len(X) == len(LAB)
b2 = len(per) == 5 and all(n >= 0.75 * tot and m == tot for n, m, tot in per.values())
L += [f"bar 1 (at least 80 of the 89 inside Normal): {'met' if b1 else 'not met'} ({int(X.normal.sum())} of {len(X)} read)",
      f"bar 2 (at least 75 % in each of the five laboratories): {'met' if b2 else 'not met'}", "rehearsal: " + ("met" if b1 and b2 else "not met")]
open(ot, "w").write("\n".join(L) + "\n"); print("\n".join(L))
