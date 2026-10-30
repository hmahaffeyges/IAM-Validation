"""R12, every cell the chain read (added after R12 was compared, disclosed; reference only, nothing scored). The replay's scorer compared only
five immune cells; the one command read Met-A for all 79 atlas cells on every sample. For each cell: later breast cancer against no cancer
(medians, Mann-Whitney p), within slides (median slide difference, Wilcoxon p over slides holding both), and among later breast cancers the
Spearman rho of Met-A with years to diagnosis. Reads the chain's rows table. Usage: python3 r12_every_cell.py ROWS.csv OUT.csv OUT.txt"""
import os, sys, pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon, spearmanr
H = os.path.dirname(os.path.abspath(__file__)); L = pd.read_csv(os.path.join(H, "r12_italy_labels.csv"), dtype=str).set_index("gsm")
R = pd.read_csv(sys.argv[1], dtype={"gsm": str}); R = R[R.reading == "Met-A each cell"]
X = R.pivot_table(index="gsm", columns="place", values="A", aggfunc="first").join(L[["slide", "status", "years_to_diagnosis"]], how="inner")
X = X[X.status.isin(["later breast cancer", "no cancer"])]; X["yrs"] = pd.to_numeric(X.years_to_diagnosis, errors="coerce"); cells = [c for c in R.place.unique() if c in X]
rows = []
for c in cells:
    a, b = X[X.status == "later breast cancer"][c].dropna(), X[X.status == "no cancer"][c].dropna()
    if len(a) < 10 or len(b) < 10: continue
    d = [g[g.status == "later breast cancer"][c].median() - g[g.status == "no cancer"][c].median() for _, g in X.groupby("slide") if (g.status == "later breast cancer").any() and (g.status == "no cancer").any()]
    d = [v for v in d if pd.notna(v)]; k = X[(X.status == "later breast cancer") & X.yrs.notna()]
    rows.append(dict(cell=c, case=round(a.median(), 4), control=round(b.median(), 4), p=mannwhitneyu(a, b).pvalue, slide_diff=round(pd.Series(d).median(), 4),
                     slide_p=wilcoxon(d).pvalue if len(d) > 5 else None, slides=len(d), rho_years=round(spearmanr(k[c], k.yrs, nan_policy="omit").correlation, 3),
                     p_years=spearmanr(k[c], k.yrs, nan_policy="omit").pvalue))
T = pd.DataFrame(rows).sort_values("p"); T.to_csv(sys.argv[2], index=False)
F = lambda r: f"{r.cell} | {r.case} vs {r.control} (p {r.p:.2g}) | within slides {r.slide_diff:+.4f} (p {r.slide_p:.2g}) | with years to diagnosis rho {r.rho_years:+.3f} (p {r.p_years:.2g})"
out = [f"R12, every cell: {len(T)} cells read on {len(X)} arrays; later breast cancer vs no cancer | within slides | among the cancers, against years to diagnosis",
       "breast cells:"] + [F(r) for r in T[T.cell.str.contains("breast")].itertuples()] + ["strongest ten by group p:"] + [F(r) for r in T.head(10).itertuples()]
open(sys.argv[3], "w").write("\n".join(out) + "\n"); print("\n".join(out))
