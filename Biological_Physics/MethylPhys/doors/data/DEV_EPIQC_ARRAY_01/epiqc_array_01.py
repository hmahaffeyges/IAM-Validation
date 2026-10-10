"""DEV-EPIQC-ARRAY-01 (written 2026-10-10 before any GSE230132 array is calibrated): Met-A and C-score instrument readings on IDENTICAL DNA at three
laboratories (SEQC2 EpiQC; GIAB lymphoblastoid lines HG001-HG007; lab A all 7 x 2 replicates, lab B all 7 x 1, lab C HG005-HG007 x 3).
Per array (chain Stage 1, then conductor_v3 Stage T self-tare II and the isolated-specimen readers, which compute the instrument quantities the
commissioned path uses; these are not neutrophils, so no reading is a health state):
  A = mean H(beta) at the 6,000 identity sites / the neutrophil floor;  C = the chain's C-score of the residual against the neutrophil reference.
Same-run tare, identical in every laboratory: for line X in {HG005, HG006, HG007}, A_rel = A / median A of the OTHER two lines in that laboratory
(2 references; the chain requires 3, so this tests the tare rule, not the chain's tare). Same for C_rel.
Steps: 'calib' (download + Stage 1), 'read'. Usage: epiqc_array_01.py WORKDIR calib|read"""
import os, sys, re, subprocess, requests, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); CH = os.path.abspath(os.path.join(HERE, "../../../chain")); sys.path.insert(0, CH)
W, STEP = sys.argv[1], sys.argv[2]; os.makedirs(os.path.join(W, "betas"), exist_ok=True)
def samples():
    t = requests.get("https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi", params={"acc": "GSE230132", "targ": "gsm", "form": "text", "view": "brief"}, timeout=120).text.replace("\r", "")
    rows = []
    for b in t.split("^SAMPLE = ")[1:]:
        get = lambda k: [l.split("=", 1)[1].strip() for l in b.split("\n") if l.startswith(k)]
        m = re.search(r"of (HG00\d) at lab ([ABC]), replicate (\d)", get("!Sample_title")[0])
        rows.append(dict(gsm=b.split("\n")[0].strip(), line=m.group(1), lab=m.group(2), rep=int(m.group(3)), files=[x for x in get("!Sample_supplementary_file") if "idat" in x.lower()]))
    return pd.DataFrame(rows)
if STEP == "calib":
    from stage_1_idat_calibration import calibrate_idat_to_beta
    S = samples(); S.drop(columns="files").to_csv(os.path.join(W, "samples.csv"), index=False)
    for _, r in S.iterrows():
        out = os.path.join(W, "betas", r.gsm + ".parquet")
        if os.path.exists(out): continue
        loc = []
        for u in r.files:
            f = os.path.join(W, u.rsplit("/", 1)[-1]); subprocess.run(["curl", "-s", "-o", f, u.replace("ftp://", "https://")], check=True); loc.append(f)
        o = calibrate_idat_to_beta([x for x in loc if "_Grn" in x][0], [x for x in loc if "_Red" in x][0], verbose=False); b = o[0] if isinstance(o, tuple) else o
        pd.DataFrame({"beta": b}).to_parquet(out); [os.remove(x) for x in loc]; print(r.gsm, r.line, r.lab, r.rep, "ok", flush=True)
    print("CALIB_DONE")
else:
    import conductor_v3 as C3
    S = pd.read_csv(os.path.join(W, "samples.csv")); rows = []
    for _, r in S.iterrows():
        b = pd.read_parquet(os.path.join(W, "betas", r.gsm + ".parquet")).beta; b.index = b.index.astype(str)
        bs, _ = C3.stage_t_selftare_ii(b); m, z = C3.stage_m_isolated(bs); cs = C3.stage_mc_cscore(z)
        rows.append(dict(gsm=r.gsm, line=r.line, lab=r.lab, rep=r.rep, A=m.get("A"), C=cs.get("C"), reason=m.get("reason")))
    R = pd.DataFrame(rows); T = R[R.line.isin(["HG005", "HG006", "HG007"])].copy()
    for q in ("A", "C"):
        T[q + "_rel"] = [r[q] / np.median(T[(T.lab == r.lab) & (T.line != r.line)].groupby("line")[q].mean()) for _, r in T.iterrows()]
    R.to_csv(os.path.join(HERE, "epiqc_array_01_rows.csv"), index=False); T.to_csv(os.path.join(HERE, "epiqc_array_01_tared.csv"), index=False)
    pd.set_option("display.width", 200); print(R.pivot_table(index="line", columns="lab", values=["A", "C"], aggfunc="mean").round(4).to_string())
    rep = R.groupby(["line", "lab"]).filter(lambda g: len(g) > 1).groupby(["line", "lab"]).agg(dA=("A", lambda s: s.max() - s.min()), dC=("C", lambda s: s.max() - s.min()))
    print("replicates:", rep.round(4).to_dict("index"))
    L = T.groupby(["line", "lab"])[["A", "A_rel", "C", "C_rel"]].mean()
    for q in ("A", "A_rel", "C", "C_rel"):
        sp = L[q].unstack().max(1) - L[q].unstack().min(1); print(f"{q}: across-lab spread per line", sp.round(4).to_dict())
    bar1 = (rep.dA <= 0.012).mean(); bar2 = ((L.A_rel.unstack().max(1) - L.A_rel.unstack().min(1)) <= 0.02).mean()
    bar3 = (rep.dC <= 0.10).mean(); bar4 = ((L.C_rel.unstack().max(1) - L.C_rel.unstack().min(1)) <= 0.10).mean()
    print(f"BARS 1 A repeat <= 0.012: {bar1:.2f} | 2 A_rel across labs <= 0.02: {bar2:.2f} | 3 C repeat <= 0.10: {bar3:.2f} | 4 C_rel across labs <= 0.10: {bar4:.2f} (each met if >= 0.80)")
