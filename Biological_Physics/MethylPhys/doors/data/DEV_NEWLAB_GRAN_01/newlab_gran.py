"""DEV-NEWLAB-GRAN-01: GSE226298 granulocytes (EPIC v1) through Stage 1 and conductor_v3, with the Stage T same-run tare on healthy controls.
Run on the box. Writes gran_rows.csv and gran_summary.json in the work dir."""
import os, sys, glob, json, warnings, subprocess
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
W = sys.argv[1]; CH = sys.argv[2]; sys.path.insert(0, CH); os.chdir(CH)
LO, HI = 0.751, 1.409


def meta():
    t = open(os.path.join(W, "gsm.txt")).read(); out = []; cur = {}
    for l in t.split("\n"):
        l = l.rstrip("\r")
        if l.startswith("^SAMPLE"):
            if cur: out.append(cur)
            cur = {"gsm": l.split("=")[1].strip()}
        elif l.startswith("!Sample_source_name_ch1"): cur["src"] = l.split("=", 1)[1].strip()
        elif l.startswith("!Sample_characteristics_ch1"):
            k, _, v = l.split("=", 1)[1].strip().partition(": "); cur[k] = v
    if cur: out.append(cur)
    return pd.DataFrame(out)


def read_one(args):
    gsm, grn, red = args
    import stage_1_idat_calibration as S1, conductor_v3 as C
    try:
        b, m = S1.calibrate_idat_to_beta(grn, red, verbose=False)
        b.to_frame("beta").to_parquet(os.path.join(W, "betas", gsm + ".parquet"))
        o = C.run_neutrophil(b, specimen="isolated neutrophils", sample_id=gsm)
        a = o.get("met_a") or {}; cs = o.get("met_a_cscore") or {}
        return dict(gsm=gsm, slide=os.path.basename(grn).split("_")[1], A=a.get("A"), N=a.get("noise_index"), gate=a.get("noise_gate"),
                    n_sites=a.get("n_sites"), C=cs.get("C"), status="ok")
    except Exception as e:
        return dict(gsm=gsm, status=f"error: {type(e).__name__}: {str(e)[:150]}")


if __name__ == "__main__":
    md = meta(); md = md[md.src == "Granulocytes"]
    os.makedirs(os.path.join(W, "betas"), exist_ok=True)
    jobs = []
    for g in md.gsm:
        gr = glob.glob(os.path.join(W, "idat", f"{g}_*_Grn.idat*")); rd = glob.glob(os.path.join(W, "idat", f"{g}_*_Red.idat*"))
        if gr and rd:
            for p in gr + rd:
                if p.endswith(".gz"): subprocess.run(["gunzip", "-f", p])
            jobs.append((g, gr[0].replace(".gz", ""), rd[0].replace(".gz", "")))
    print("arrays", len(jobs), flush=True)
    with ProcessPoolExecutor(int(sys.argv[3]) if len(sys.argv) > 3 else 16) as ex: rows = list(ex.map(read_one, jobs))
    r = pd.DataFrame(rows).merge(md[["gsm", "disease state", "loy status", "donor id"]], on="gsm", how="left")
    r["healthy"] = r["disease state"].eq("control")
    hh = r[r.healthy & r.A.notna()]
    for col, out in (("A", "A_tared"), ("C", "C_rel")):
        r[out] = np.nan; r[out + "_scope"] = None
        for i, x in r.iterrows():
            if pd.isna(x[col]): continue
            for scope in ("slide", "series"):
                ref = hh[(hh.gsm != x.gsm) & ((hh.slide == x.slide) if scope == "slide" else True)][col].dropna()
                if len(ref) >= 3: r.at[i, out] = x[col] / ref.median(); r.at[i, out + "_scope"] = scope; break
    r.to_csv(os.path.join(W, "gran_rows.csv"), index=False)
    H = r[r.healthy]; P = r[~r.healthy]
    s = dict(n_healthy=int(len(H)), n_patient=int(len(P)), errors=int((r.status != "ok").sum()),
             healthy_A_tared_n=int(H.A_tared.notna().sum()), healthy_normal=int(H.A_tared.between(0.95, 1.05).sum()),
             healthy_A_tared_median=float(H.A_tared.median()), healthy_A_tared_q=[float(x) for x in H.A_tared.quantile([.025, .975])],
             healthy_C_rel_n=int(H.C_rel.notna().sum()), healthy_C_in_band=int(H.C_rel.between(LO, HI).sum()),
             healthy_C_rel_q=[float(x) for x in H.C_rel.quantile([.025, .5, .975])], healthy_C_untared_in_band=int(H.C.between(LO, HI).sum()),
             healthy_gate_withheld=int((H.N > 0.149).sum()),
             patient_A_tared_median=float(P.A_tared.median()), patient_normal=int(P.A_tared.between(0.95, 1.05).sum()), patient_n_tared=int(P.A_tared.notna().sum()))
    json.dump(s, open(os.path.join(W, "gran_summary.json"), "w"), indent=1); print(json.dumps(s))
