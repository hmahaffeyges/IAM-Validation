"""The study read with the full chain (development; reported, not scored): every reading of each of the 36 samples, commissioned and in
development, by genotype, with which development readings ran or could not. Usage: python3 hopkins_fullchain_table.py LAND_DIR OUT_CSV OUT_TXT"""
import sys, os, glob, json, collections, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); S = pd.read_csv(os.path.join(H, "hopkins_fullchain_samples.csv")); rows = []; on = collections.Counter(); nr = collections.Counter()
for r in S.itertuples():
    f = sorted(glob.glob(os.path.join(sys.argv[1], "hf_" + r.gsm, "**", r.gsm + "_bundle.json"), recursive=True))
    if not f: rows.append(dict(gsm=r.gsm, genotype=r.genotype, read=False)); continue
    B = json.load(open(f[-1])); D = B.get("development") or {}; fc = B.get("full_chain") or {}
    for k in fc.get("on", []): on[k.split(" (")[0]] += 1
    for k in (fc.get("not_run") or {}): nr[k] += 1
    rows.append(dict(gsm=r.gsm, genotype=r.genotype, read=True, iam_a=(B.get("iam_a") or {}).get("A"), coupling_a=(B.get("coupling_a") or {}).get("A"),
                     metaa_sorted_v21=(D.get("metaa_atlas") or {}).get("A") if "IAM-Atlas v2.1" in str((D.get("metaa_atlas") or {}).get("reading")) else None,
                     metaa_sorted=(D.get("metaa_atlas") or {}).get("A"), library_floor=(D.get("library_floor") or {}).get("rate"),
                     islands=(((B.get("territories") or {}).get("sets") or {}).get("islands") or {}).get("A"), globe=bool((D.get("genome_globe") or {}))))
d = pd.DataFrame(rows); d.to_csv(sys.argv[2], index=False); L = [f"The study read with the full chain: {int(d.read.sum())} of {len(d)} read", "median by genotype:"]
for g, x in d[d.read].groupby("genotype"):
    L.append(f"{g} ({len(x)}): " + ", ".join(f"{c} {pd.to_numeric(x[c], errors='coerce').median():.4f}" for c in ("iam_a", "coupling_a", "metaa_sorted", "library_floor", "islands")))
L += ["development readings run (samples): " + ", ".join(f"{k} {v}" for k, v in sorted(on.items())), "not run (samples): " + ", ".join(f"{k} {v}" for k, v in sorted(nr.items()))]
open(sys.argv[3], "w").write("\n".join(L) + "\n"); print("\n".join(L))
