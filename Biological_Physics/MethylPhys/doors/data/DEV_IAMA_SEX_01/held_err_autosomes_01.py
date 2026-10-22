"""DEV-IAMA-SEX-01, exploratory (2026-10-21; not sealed, no bar). Chain Decoder finding F37: on the .pat route, IAM-A's held error is counted
on every chromosome (X and Y included) while its loose share is counted on autosome tiles only; the BAM route counts both on autosomes only.
Question: how much does the X and Y move the held error, and so IAM-A, on the .pat route?
For each healthy sample named below (the committed line's own samples, every CD8 sample in it plus three other blood types): its hg19 .pat
from GEO, counted twice by chain Stage Q2's count_pat, unchanged: once on every line (the committed route) and once with every non-autosome
line blanked (the line's place kept, so count_pat's alternate-line split puts every other line exactly where it did). IAM-A by Q2.reading
on each. Writes OUT/held_err_autosomes_01_rows.csv and _summary.txt. Usage: python3 held_err_autosomes_01.py OUT"""
import os, sys, csv, gzip, urllib.request, re, time
H = os.path.dirname(os.path.abspath(__file__)); MP = os.path.abspath(os.path.join(H, "..", "..", ".."))
sys.path.insert(0, os.path.join(MP, "chain")); import stage_q2_selfref as Q2
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
CELLS = ("cd8 t cells", "t effector cell cd8", "effector memory cd8 t cells", "naive cd8 t cells", "monocytes", "b cells", "nk cells")
S = [r for r in csv.DictReader(open(os.path.join(MP, "doors", "data", "DEV_CHAIN_Q2_01", "loyfer_samples.csv"))) if r["cell"] in CELLS]
AUTO = set(Q2.AUTOSOMES); TILES = Q2.hg19_tiles(); assert not any(c in TILES for c in ("chrX", "chrY")), "loose-share tiles hold the X or Y"
rows = []
for r in S:
    g = r["sample"]; base = f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{g[:7]}nnn/{g}/suppl/"
    for a in range(5):
        try:
            name = sorted(set(re.findall(rf"{g}_[^\"<>]*\.pat\.gz", urllib.request.urlopen(base, timeout=120).read().decode())))[0]
            f = os.path.join(OUT, name); urllib.request.urlretrieve(base + name, f); break
        except Exception as e: print(g, "retry", type(e).__name__, flush=True); time.sleep(60 * (a + 1))
    else: rows.append(dict(sample=g, cell=r["cell"], error="download failed")); continue
    allr = Q2.count_pat(gzip.open(f, "rt"), TILES)
    auto = Q2.count_pat((l if l.split("\t", 1)[0] in AUTO else "\n" for l in gzip.open(f, "rt")), TILES)
    os.remove(f)
    x = dict(sample=g, cell=r["cell"], loose_all=allr["loose_share"], loose_auto=auto["loose_share"], held_all=allr["held_err"], held_auto=auto["held_err"],
             opp_all=allr["held_opp"], opp_auto=auto["held_opp"], A_all=Q2.reading(allr).get("A") if isinstance(Q2.reading(allr), dict) else Q2.reading(allr),
             A_auto=Q2.reading(auto).get("A") if isinstance(Q2.reading(auto), dict) else Q2.reading(auto))
    rows.append(x); print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in x.items()}, flush=True)
with open(os.path.join(OUT, "held_err_autosomes_01_rows.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=sorted({k for x in rows for k in x})); w.writeheader(); w.writerows(rows)
ok = [x for x in rows if "error" not in x]
import numpy as np
d = np.array([x["held_auto"] / x["held_all"] - 1 for x in ok]); a = np.array([x["A_auto"] / x["A_all"] - 1 for x in ok if x["A_all"] and x["A_auto"]])
L = [f"samples read: {len(ok)} of {len(rows)}; loose share identical on both counts: {all(x['loose_all'] == x['loose_auto'] for x in ok)}",
     f"held error, autosomes only against every chromosome: mean {100 * d.mean():+.2f} %, range {100 * d.min():+.2f} to {100 * d.max():+.2f} %",
     f"IAM-A, autosomes only against every chromosome: mean {100 * a.mean():+.2f} %, range {100 * a.min():+.2f} to {100 * a.max():+.2f} %",
     f"held-error opportunities outside the autosomes: mean {100 * np.mean([1 - x['opp_auto'] / x['opp_all'] for x in ok]):.2f} %"]
open(os.path.join(OUT, "held_err_autosomes_01_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
