"""Every commissioning set for atlas v2.1 (held-out tests and truth sets from atlas_v21_roster.csv), with its readiness computed,
never typed: spent status (CANON/data_spent.csv), raw IDATs present, platform the chain reads (450K / EPIC v1), and what it still
needs. Writes commissioning_sets_v21.csv. Refuses (exit 1) if any commissioning set is spent.
Usage: python3 commissioning_sets_v21.py"""
import os, csv, sys, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, *[".."] * 5))
R = pd.read_csv(os.path.join(HERE, "atlas_v21_roster.csv"))
SP = {r["dataset"]: r["spent_status"] for r in csv.DictReader(open(os.path.join(ROOT, "CANON", "data_spent.csv"), encoding="utf-8"))}
H = R[R.role.str.startswith("held-out") | (R.role == "truth set")].copy()
H["spent_status_now"] = H.series.map(lambda s: SP.get(s, "not listed"))
H["spent"] = ~H.spent_status_now.isin(["unspent", "not listed"])
H["raw_idat"] = H.raw_files.astype(str).str.contains("IDAT pair", case=False)
H["sequencing"] = H.platform.astype(str).str.contains("bisulfite|RRBS|seq", case=False)
H["epic_v2"] = H.platform.astype(str).str.contains("EPIC v2", case=False)
def need(r):
    if r.spent: return "SPENT - cannot test"
    if r.sequencing: return "alignment on the box (sequencing), a separate job; not ready right after the build"
    if r.epic_v2: return "Stage 1 EPIC v2 read check first"
    if not r.raw_idat: return "raw files to be located"
    return "download and Stage 1 after the split is sealed; then ready"
H["needs"] = H.apply(need, axis=1)
H["ready_after_build"] = H.needs.str.startswith("download")
cols = ["cell_type", "series", "role", "use", "arrays_or_libraries", "platform", "laboratory_independence", "prior_reading", "spent_status_now", "raw_idat", "sequencing", "needs", "ready_after_build"]
H[cols].to_csv(os.path.join(HERE, "commissioning_sets_v21.csv"), index=False)
print(f"commissioning sets: {len(H)} rows, {H.series.nunique()} series | ready after build {int(H.ready_after_build.sum())} | sequencing {int(H.sequencing.sum())} | spent {int(H.spent.sum())}")
sys.exit(1 if H.spent.any() else 0)
