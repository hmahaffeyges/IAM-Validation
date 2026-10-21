"""DEV-SELFTARE-03, exploration (2026-10-21, not sealed): a self-tare that keeps a real loss of pattern.
Why. Stage T step 1 (self-tare II, DEV-SELFTARE-02) maps each probe design onto the reference arrays' scale by this array's own fixed-site
anchors. Its derivation assumes the fixed sites hold the same true state on every array; a cell losing its pattern breaks that, and on
synthetic bloods a 1 % neutrophil loss at every site was mapped away (DEV-CELLS-01, selftare_uniform_loss_01_summary.txt). A loss toward
0.5 and a technical squeeze are the same affine form on one array, so no anchor map can tell them apart; what differs is that the
technical offsets differ between the two probe designs while a biological loss hits both alike.
The variants, each followed by the chain's median tare (Stage T step 2), read on the SAME calibrated betas:
  absolute  the chain as it is: each design mapped onto the reference arrays' scale, beta' = T_d(beta), T_d(x) = Lr_d + (x - L_d)(Ur_d - Lr_d)/(U_d - L_d)
  relative  type II put on this array's own type I scale, beta'_II = T_I^-1(T_II(beta_II)); type I unchanged. The squeeze both designs share
            (the laboratory's and any biological loss) is left for the median tare against same-run healthy arrays, which share the
            laboratory's squeeze but not the person's loss. On the reference arrays T_I and T_II are close to the identity, so the map is too.
  none      no step 1
Sets: replicates (GSE250556, the 63 replicate whole bloods of DEV-SELFTARE-02); otherlab (the purified neutrophils of GSE122244, GSE247193,
GSE247195 of DEV-SELFTARE-02); groupa (the first N healthy whole bloods of GSE318173 by slide), read healthy, with a 1 % loss of the
neutrophils' pattern at every site (IAM-Atlas v2.1 neutrophil means, this array's own Stage A share) and at the identity sites only.
Usage: python3 selftare_relative_explore.py SET OUTDIR [N]. Writes OUTDIR/selftare03_SET_rows.csv."""
import os, sys, json, urllib.request, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); MP = os.path.abspath(os.path.join(H, "..", "..", "..")); CH = os.path.join(MP, "chain")
sys.path[:0] = [CH, os.path.join(MP, "doors", "data", "GROUPA_COMMISSIONING")]
import conductor_v3 as C, dev_stages as DV, stage_1_idat_calibration as S1
from read_groupa import fetch
SET, out = sys.argv[1], sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 100; os.makedirs(out, exist_ok=True)
ST = DV._json("dev_selftare_typeII_EPIC_v1.json"); DES = pd.Series(ST["design"]); RA = ST["ref_anchors"]

def files(series):
    sub = series[:-3] + "nnn"; t = urllib.request.urlopen(f"https://ftp.ncbi.nlm.nih.gov/geo/series/{sub}/{series}/suppl/filelist.txt", timeout=120).read().decode()
    F = {}
    for l in t.strip().split("\n")[1:]:
        n = l.split("\t")[1]
        if n.endswith("_Grn.idat.gz"): F.setdefault(n.split("_")[0], {})["grn"] = n
        if n.endswith("_Red.idat.gz"): F.setdefault(n.split("_")[0], {})["red"] = n
    return F

def relative(beta):
    an = DV.anchors(beta, ST); b2 = beta.copy(); g = {}
    for d in ("I", "II"):
        L, U = an[f"{d}_low"]["mean"], an[f"{d}_high"]["mean"]
        if L is None or U is None or U - L <= 0.1: return beta, {"step": "1 relative", "status": "NOT_RUN", "reason": f"design {d} anchors missing"}
        g[d] = (L, U, RA[f"{d}_low"], RA[f"{d}_high"])
    (LI, UI, LrI, UrI), (LII, UII, LrII, UrII) = g["I"], g["II"]
    s = DES.index[DES.values == "II"].intersection(b2.dropna().index)
    y = LrII + (b2.loc[s] - LII) * (UrII - LrII) / (UII - LII)          # T_II
    b2.loc[s] = (LI + (y - LrI) * (UI - LI) / (UrI - LrI)).clip(1e-6, 1 - 1e-6)   # T_I^-1
    return b2, {"step": "1 relative", "status": "OK", "slope_II_to_I": round((UrII - LrII) / (UII - LII) * (UI - LI) / (UrI - LrI), 5)}

ORIG = C.stage_t_selftare_ii; NONE = lambda b: (b, {"step": "1 none", "status": "OFF"}); VAR = {"absolute": ORIG, "relative": relative, "none": NONE}
def read(beta, spec):
    res = {}
    for v, fn in VAR.items():
        C.stage_t_selftare_ii = fn
        try: o = C.run_neutrophil(beta, specimen=spec); res[v] = ((o.get("met_a") or {}).get("A"), (o.get("met_a_cscore") or {}).get("C"), ((o.get("composition") or {}).get("fractions") or {}).get("NEU"))
        finally: C.stage_t_selftare_ii = ORIG
    return res

if SET == "groupa":
    S = pd.read_csv(os.path.join(MP, "doors", "data", "GROUPA_COMMISSIONING", "groupa_samples.csv")).sort_values(["slide", "position"]).head(N)
    S["series"] = "GSE318173"; S["specimen"] = "whole blood"; S["person"] = ""
    import stage_deconv_v21 as D
    MU = D.reference(["neutrophils"])["mu"]["neutrophils"]
    NS = pd.Index(json.load(open(os.path.join(CH, "Runtime Matrices", "Met_A_Floors", "metA_floors_v1_3.json")))["platforms"]["EPIC"]["neutrophils"]["sites"])
else:
    S = pd.read_csv(os.path.join(MP, "doors", "data", "DEV_SELFTARE_02", f"selftare02_{SET}.csv"))
    if "specimen" not in S: S["specimen"] = "isolated neutrophils"
    if "person" not in S: S["person"] = ""
    F = {}
    for se in S.series.unique(): F.update(files(se))
    S["grn"] = [F.get(g, {}).get("grn") for g in S.gsm]; S["red"] = [F.get(g, {}).get("red") for g in S.gsm]
rows = []
for r in S.itertuples():
    try:
        if not r.grn or not r.red: raise RuntimeError("no IDAT pair on GEO")
        g = fetch(r.gsm, r.grn, os.path.join(out, r.grn)); d = fetch(r.gsm, r.red, os.path.join(out, r.red))
        beta, _ = S1.calibrate_idat_to_beta(g, d, verbose=False); os.remove(g); os.remove(d)
        beta = pd.Series(beta, dtype="float64"); beta.index = beta.index.astype(str)
        cases = {"healthy": beta}
        if SET == "groupa":
            f = read(beta, r.specimen)["absolute"][2] or 0.0
            m = MU.reindex(beta.index); plant = (f * 0.01 * (0.5 - m)).fillna(0.0)
            cases["loss_every_site"] = (beta + plant).clip(1e-6, 1 - 1e-6)
            ids = beta.index.isin(NS); cases["loss_identity_sites"] = (beta + plant.where(ids, 0.0)).clip(1e-6, 1 - 1e-6)
        for case, b in cases.items():
            for v, (A, Cs, fn) in read(b, r.specimen).items():
                rows.append(dict(set=SET, series=r.series, gsm=r.gsm, slide=str(r.slide), person=r.person, specimen=r.specimen, case=case, variant=v, A=A, C=Cs, f_neu=fn))
    except Exception as e:
        rows.append(dict(set=SET, series=r.series, gsm=r.gsm, slide=str(r.slide), person=r.person, specimen=r.specimen, case="ERROR", variant=str(e)[:200]))
    print(r.gsm, rows[-1].get("case"), flush=True)
pd.DataFrame(rows).to_csv(os.path.join(out, f"selftare03_{SET}_rows.csv"), index=False); print("rows", len(rows))
