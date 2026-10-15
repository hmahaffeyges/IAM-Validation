"""DEV-ARCH-01 rerun at a chosen cut (2026-10-15): identical to arch_box.py except that the loose-share cut is CUT (env, default 0.70).
The physics candidate is 2/3: a stretch whose marks are held by at least one Landauer bit has methylated:unmethylated odds >= 2.
Original header: DEV-ARCH-01 (exploration, 2026-10-14): three architecture numbers per Loyfer cell type, from its healthy array-site betas, for a first look
beside the held error. Sites inside the 399 windows the held error was counted on (PROC-CHANNEL-01) are left out when their hg19 position is known.
A1 loosely held share: of 100 kb hg19 bins with >= 5 open-sea sites, the share whose median beta is below 0.70.
A2 fully methylated share: sites with cell mean beta >= 0.80.  A3 identity sites: sites where |cell mean - median of the other cell types| >= 0.5."""
import os, sys, json, subprocess, numpy as np, pandas as pd
CUT = float(os.environ.get("CUT", "0.70"))
sys.path.insert(0, os.getcwd())
# Run 1 (c9efb60a, commit 8be68b2d) published its code from the job's command line; from run 2 the script publishes itself.
if False:   # publishing is done by the job wrapper for this rerun
    from publish_code import publish_code
    publish_code(f"results/DEV_ARCH_01/code/run{os.environ['RUN']}/", ["arch_box.py", "roster_samples.csv", "windows_hg19_cpgidx.bed", "publish_code.py"], job="DEV-ARCH-01", commit=os.environ.get("REPO_COMMIT"))
B = pd.read_parquet("/home/ubuntu/data/atlas_sources/loyfer2023/array_beta.parquet").astype("float32")
S = pd.read_csv("roster_samples.csv"); S = S[S.source.str.contains("oyfer") & S["sample"].isin(B.columns)]
M = pd.concat({c: B[g["sample"]].mean(axis=1) for c, g in S.groupby("cell")}, axis=1); print("cells", M.shape, flush=True)
man = pd.read_csv("/home/ubuntu/.methylprep_manifest_files/HumanMethylationEPIC_manifest_v2.csv.gz", low_memory=False)
cols = {c.lower(): c for c in man.columns}; print([c for c in man.columns][:30], flush=True)
idc = cols.get("ilmnid") or cols.get("illumina_id") or list(man.columns)[0]
chc = next(cols[k] for k in ("chr", "chromosome", "chr_hg19") if k in cols); poc = next(cols[k] for k in ("mapinfo", "pos", "position") if k in cols)
isl = next((cols[k] for k in ("relation_to_ucsc_cpg_island", "relation_to_island") if k in cols), None)
man = man[[idc, chc, poc] + ([isl] if isl else [])].dropna(subset=[poc]); man.columns = ["id", "chr", "pos"] + (["island"] if isl else [])
man["id"] = man["id"].astype(str).str.split("_").str[0]; man = man.drop_duplicates("id").set_index("id"); man["chr"] = "chr" + man.chr.astype(str).str.replace("chr", "")
man = man.reindex(M.index); print("sites with hg19 position", int(man.pos.notna().sum()), "of", len(man), flush=True)
excl = pd.Series(False, index=M.index); note = "windows not excluded"
try:
    w = subprocess.run(["wgbstools", "convert", "-L", "windows_hg19_cpgidx.bed", "--genome", "hg19"], capture_output=True, text=True, timeout=600).stdout
    W = pd.read_csv(pd.io.common.StringIO(w), sep="\t", header=None).iloc[:, :3]; W.columns = ["chr", "start", "end"]
    for ch, g in W.groupby("chr"):
        k = man.chr.eq(ch).values; p = man.pos.values
        for s, e in zip(g.start, g.end): excl.values[k & (p >= s) & (p <= e)] = True
    note = f"excluded {int(excl.sum())} sites inside the {len(W)} windows"
except Exception as e: note = f"windows not excluded ({type(e).__name__}: {str(e)[:80]})"
print(note, flush=True)
keep = ~excl & man.pos.notna(); Mk = M[keep]; mk = man[keep]
os_ = mk.island.astype(str).str.lower().isin(["nan", "", "opensea", "open sea"]) if "island" in mk else pd.Series(True, index=mk.index)
bins = (mk.chr + ":" + (mk.pos // 100000).astype(int).astype(str))[os_]
grp = Mk[os_].groupby(bins.values); med = grp.median(); n = grp.size(); med = med[n >= 5]
A1 = (med < CUT).mean(); A2 = (Mk >= 0.80).mean()
others = {c: Mk.drop(columns=c).median(axis=1) for c in Mk.columns}; A3 = pd.Series({c: int(((Mk[c] - others[c]).abs() >= 0.5).sum()) for c in Mk.columns})
O = pd.DataFrame({"loose_share": A1, "fully_meth_share": A2, "identity_sites": A3, "samples": S.groupby("cell").size()}).round(5)
O.to_csv(f"arch_01_cut{CUT:.4f}.csv"); open("arch_01_note.txt", "w").write(f"{note}; bins {len(med)}; open-sea sites {int(os_.sum())}; island column {'yes' if 'island' in mk else 'no'}\n"); print(O.head(), flush=True)
