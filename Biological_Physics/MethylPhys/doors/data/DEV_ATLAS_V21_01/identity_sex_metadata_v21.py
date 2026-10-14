"""Atlas v2.1: called sex (identity_check_v21_sex.csv) against metadata sex, the comparison identity_check_v21.py's rule c names but did not
compute (added 2026-10-13 after its run; rule c only flags, so nothing it finds changes any build or held-out decision).
Metadata sex: atlas_v21_build_samples.csv 'sex' for build arrays; the GEO snapshot's sex/gender characteristic for held-out arrays.
Writes identity_check_v21_sex_vs_metadata.csv. Usage: python3 identity_sex_metadata_v21.py"""
import os, re, gzip, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
S = pd.read_csv(os.path.join(HERE, "identity_check_v21_sex.csv"))
B = pd.read_csv(os.path.join(HERE, "atlas_v21_build_samples.csv")).set_index("sample")
def norm(x):
    x = str(x).strip().lower()
    return "M" if x in ("m", "male", "man") else "F" if x in ("f", "female", "woman") else ""
meta = {k: norm(v) for k, v in B.sex.items()}
for g in S[S.set == "heldout"].series.unique():
    p = os.path.join(HERE, "geo_snapshots", f"{g}.txt.gz")
    if not os.path.exists(p): continue
    for b in gzip.open(p, "rt", encoding="utf-8").read().replace("\r", "").split("^SAMPLE = ")[1:]:
        gsm = b.split("\n")[0].strip()
        ch = [l.split("=", 1)[1].strip() for l in b.split("\n") if l.startswith("!Sample_characteristics_ch1")]
        sx = [c.split(":", 1)[1] for c in ch if re.match(r"\s*(sex|gender)\s*:", c, re.I)]
        if sx: meta[gsm] = norm(sx[0])
S["sex_metadata"] = S["sample"].map(meta).fillna("")
S["agrees"] = [("" if not m or c == "UNRESOLVED" else str(m == c)) for m, c in zip(S.sex_metadata, S.sex_called)]
S.to_csv(os.path.join(HERE, "identity_check_v21_sex_vs_metadata.csv"), index=False)
k = S[S.agrees != ""]
print(f"arrays {len(S)} | with metadata sex and a call {len(k)} | agree {int((k.agrees == 'True').sum())} | disagree {int((k.agrees == 'False').sum())}")
print(S[S.agrees == "False"].groupby(["set", "series"]).size().to_dict())
