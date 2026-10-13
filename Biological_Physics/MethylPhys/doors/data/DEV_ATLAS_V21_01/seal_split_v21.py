"""Seal the atlas v2.1 build / held-out split (readiness L3), before any v2.1 array is downloaded.
Records the SHA-256 of the build list and of the commissioning (held-out and truth-set) list, the commit they are sealed at, and the
counts. A later change to either file breaks the seal (CANON/v21_readiness.py checks the checksums); a change must be a new, dated seal
with its reason, the old one kept. Usage: python3 seal_split_v21.py REASON"""
import os, sys, json, hashlib, subprocess, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, *[".."] * 5)); rel = lambda f: os.path.relpath(os.path.join(HERE, f), ROOT)
sha = lambda f: hashlib.sha256(open(os.path.join(HERE, f), "rb").read()).hexdigest()
out = os.path.join(HERE, "sealed_split_v21.json")
if os.path.exists(out): sys.exit("sealed_split_v21.json exists: a new seal must be a new dated file, the old one kept")
B = pd.read_csv(os.path.join(HERE, "atlas_v21_build_samples.csv")); H = pd.read_csv(os.path.join(HERE, "commissioning_sets_v21.csv"))
import pandas as _pd
_R = _pd.read_csv(os.path.join(HERE, "atlas_v21_roster.csv"))
_held = {x for x in ";".join(_R[_R.role != "build"].samples.astype(str)).split(";") if x and x != "nan"}
_shared = set(B[B.state != "OUT"]["sample"]) & _held
assert not _shared, f"{len(_shared)} samples are both build and held-out: {sorted(_shared)[:5]}"
same_lab = sorted(set(B[B.state != "OUT"].series) & set(H.series))
head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
J = {"sealed": "2026-10-13", "reason": sys.argv[1], "commit_at_seal": head,
     "sha256": {rel("atlas_v21_build_samples.csv"): sha("atlas_v21_build_samples.csv"), rel("commissioning_sets_v21.csv"): sha("commissioning_sets_v21.csv"),
                rel("atlas_v21_roster.csv"): sha("atlas_v21_roster.csv"), rel("same_source_v21.json"): sha("same_source_v21.json")},
     "counts": {"build_arrays": B.state.value_counts().to_dict(), "commissioning_rows": len(H), "commissioning_series": int(H.series.nunique())},
     "same_series_split_by_donor": {"series": same_lab, "rule": ("these series build from some arrays and hold out others; no sample is on both sides "
         "(checked above). Donor IDs are not in their metadata, so donor separation is checked at Stage 1 from the SNP probes: a held-out array whose "
         "genotype matches any build array is excluded from testing, by rule. Their held-out rows test new donors in the same laboratory and never "
         "count as an independent laboratory (commissioning_sets_v21.csv, counts_as_independent_laboratory).")},
     "rule": "Build arrays may be downloaded and calibrated; held-out and truth-set arrays are not scored until the atlas they judge is built and its commissioning bars are sealed (L15)."}
json.dump(J, open(out, "w"), indent=1); print(json.dumps(J["counts"]), head[:12])
