"""DEV-WRITER-02 enzyme side: DNMT1 HM/UM discrimination per NNCGNN context from Adam et al. 2023 (Nucleic Acids Res 51:6622,
doi 10.1093/nar/gkad465), Data Set 1 (relative methylation rates of HM, OH, UM substrates per context; supplied by the author from the journal's
supplementary files). Writes enzyme_D_256.csv: ctx, HM, UM, D = HM/UM, rc (reverse complement), D_rc, pred1 = mean over the two strands of
1/(1+D). Checks against the paper's printed values: ACCGGA ~29, GGCGAC > 300, average ~87. Usage: python3 enzyme_table.py"""
import os, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
t = pd.read_excel(os.path.join(HERE, "Adam2023_Data_Set_1.xlsx"), header=None).iloc[2:, :7]
t.columns = ["ctx", "HM", "OH", "UM", "sem_HM", "sem_OH", "sem_UM"]
t["ctx"] = t.ctx.astype(str).str.replace(" ", "", regex=False); t = t[t.ctx.str.match(r"^[ACGT]{2}CG[ACGT]{2}$")].copy()
for c in ("HM", "OH", "UM", "sem_HM", "sem_OH", "sem_UM"): t[c] = t[c].astype(float)
t["D"] = t.HM / t.UM
comp = lambda s: s[::-1].translate(str.maketrans("ACGT", "TGCA")); t["rc"] = t.ctx.apply(comp)
Dm = t.set_index("ctx").D; t["D_rc"] = t.rc.map(Dm); t["pred1"] = 0.5 * (1 / (1 + t.D) + 1 / (1 + t.D_rc))
assert len(t) == 256 and abs(Dm["ACCGGA"] - 29) < 1 and Dm["GGCGAC"] > 300 and abs(t.D.mean() - 87) < 1
t.to_csv(os.path.join(HERE, "enzyme_D_256.csv"), index=False)
print(len(t), "contexts | D", round(t.D.min(), 1), "-", round(t.D.max(), 1), "mean", round(t.D.mean(), 1), "| strand pairs", t.ctx.ne(t.rc).sum() // 2, "+ palindromes", int((t.ctx == t.rc).sum()))
