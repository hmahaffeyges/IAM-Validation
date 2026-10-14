"""Atlas v2.1: the final build sample table (2026-10-14). Metadata only; no methylation value is read. Every rule is an outcome already
recorded elsewhere, applied here in one place so the terms job and the fit read the same table:
 1. v2's admitted samples, exactly as the twin/terms job builds them (roster_samples + normal GSE63409 HSCs; qc; v2 cells IN).
 2. v2.1 rows (v21_samples_all.csv, twin run 2's table; build ids via the Stage 1 manifest) enter if qc, and their build id is IN on atlas_v21_build_samples.csv, or CONDITIONAL in a
    series whose condition (conditions_v21.json) resolved IN.
 3. A new cell that is a sample-level twin of an admitted cell is HELD (v2's rule; twin_sample_level_v21.csv).
 4. A new cell enters only with at least 2 admitted samples (v2's n >= 2 rule).
Usage: python3 final_table_v21.py <dir with the inputs>  -> final_samples_v21.csv, final_samples_v21_summary.txt"""
import json, os, sys, pandas as pd
def build(W):
    j = lambda f: os.path.join(W, f)
    S2 = pd.read_csv(j("roster_samples.csv")); RO = pd.read_csv(j("roster.csv"), index_col=0)
    H = pd.read_csv(j("hsc_manifest.csv")); H = H[H.subject_status == "normal"]
    S12 = pd.concat([S2, pd.DataFrame({"cell": H.label.str.replace(" of normal bone marrow", "", regex=False) + " (bone marrow)", "source": "GSE63409", "platform": "array", "sample": H.gsm, "qc": H.call_rate >= 0.93})])
    adm = RO.index[RO.v2_status.astype(str).str.startswith("IN")].tolist(); S12 = S12[S12.qc & S12.cell.isin(adm)].copy(); S12["path"] = ""; S12["why"] = "v2"
    S21 = pd.read_csv(j("v21_samples_all.csv"), keep_default_na=False, dtype={"path": str}); add = S21[S21.path != ""].copy()
    add["qc"] = add.qc.astype(str).eq("True")
    # crosswalk: v2.1 rows are named by array base (chip_position or GSM_chip_position); the build list by its own sample id (GSM, or the
    # ArrayExpress donor code such as 12SC). The Stage 1 manifest holds both, one row per array (base is unique).
    M1 = pd.read_csv(j("stage1_v21_manifest.csv")); assert M1.base.is_unique
    add["gsm"] = add["sample"].map(dict(zip(M1.base, M1["sample"]))); assert add.gsm.notna().all(), "a v2.1 row has no Stage 1 manifest entry"
    BL = pd.read_csv(j("atlas_v21_build_samples.csv")); st = dict(zip(BL["sample"], BL.state))
    C = json.load(open(j("conditions_v21.json"), encoding="utf-8"))
    cond_in = {k.split("_", 1)[1] for k, v in C.items() if isinstance(v, dict) and str(v.get("outcome", "")).startswith("IN")}
    add["state"] = add.gsm.map(st)
    add["why"] = ["v2.1 IN" if s == "IN" else (f"v2.1 conditional IN ({src})" if s == "CONDITIONAL" and src in cond_in else "") for s, src in zip(add.state, add.source)]
    out = add[(add.why == "") | ~add.qc]
    keep = add[(add.why != "") & add.qc].copy()
    T = pd.read_csv(j("twin_sample_level_v21.csv")); tw = T[T.twin.astype(str) == "True"]
    held = sorted({b for a, b in zip(tw.a, tw.b) if a in adm and b not in adm} | {a for a, b in zip(tw.a, tw.b) if b in adm and a not in adm})
    keep = keep[~keep.cell.isin(held)]
    n = keep.groupby("cell").size(); new = [c for c in n.index if c not in adm]
    thin = [c for c in new if n[c] < 2]; keep = keep[~keep.cell.isin(thin)]
    F = pd.concat([S12[["cell", "source", "platform", "sample", "qc", "path", "why"]], keep[["cell", "source", "platform", "sample", "qc", "path", "why"]]], ignore_index=True)
    L = [f"final build table: {len(F)} samples, {F.cell.nunique()} cells ({len(S12)} v2 samples in {S12.cell.nunique()} cells; {len(keep)} v2.1 samples)",
         f"conditional series resolved IN: {sorted(cond_in)}",
         f"v2.1 rows not entering: {len(out)} (not IN, an OUT condition, or failed qc); by source: {out.groupby('source').size().to_dict()}",
         f"held as sample-level twins: {held}; new cells with fewer than 2 samples: {thin}",
         f"new cells in v2.1: {sorted(set(F.cell) - set(adm))}"]
    return F, L
if __name__ == "__main__":
    W = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    F, L = build(W); F.to_csv(os.path.join(W, "final_samples_v21.csv"), index=False)
    open(os.path.join(W, "final_samples_v21_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
