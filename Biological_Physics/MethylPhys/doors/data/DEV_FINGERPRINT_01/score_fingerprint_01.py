"""DEV-FINGERPRINT-01 scoring (written 2026-10-10 before any LNCaP run or array is read). LNCaP (cancer) against PrEC (normal prostate epithelium),
GSE86833, one laboratory. Both instruments on the same cells.
IAM-A (sequencing; Box Run 8 .pat files, pinned pipeline): per run ε by Stage Q (pat_site_table, whole file); each cell's runs pooled
(errors / opportunities). IAM-A_rel = H(ε_LNCaP) / H(ε_PrEC). share_read = (read / >= 6-call molecules), LNCaP ÷ PrEC (Stage Q's rule).
Met-A (EPIC arrays GSM2309170-73, chain Stage 1 betas, no self-tare: both cells in the same series): prostate identity sites from the atlas v2
prostate epithelium posterior by the canon site rule (posterior mean 0.75-0.95 or 0.05-0.25, posterior SD <= 0.05, up to 3,000 per channel,
smallest SD), sites measured on all four arrays. Met-A_rel = mean over LNCaP arrays of mean H(beta) ÷ the same over PrEC arrays.
ARM CHOICE: share_read >= 0.70 -> Arm A, else Arm B (DEV-FINGERPRINT-01).
Arm A: curve B (DEV-LINK-IAMA-METAA-01 table, simple-form IAM-A) re-expressed in Stage Q's IAM-A by PrEC's own measured response
(insilico_prec_SRR4238614.csv). z = (Met-A_rel - curveB(IAM-A_rel)) / se, se = sqrt(0.020^2 * 2/nA + slope^2 * 0.009^2 * (1/nW + 1/4)),
nA = 2 arrays, nW = LNCaP runs (fingerprint_power_02). FINGERPRINT if z > 1.645.
Arm B: run-loss reading (DEV_RUNLOSS_01/runloss_01.py functions) with the territory from all four PrEC runs pooled; every PrEC and LNCaP run read
against it. FINGERPRINT if every LNCaP run's excess run loss exceeds the largest PrEC run's.
Steps: sites | calib | score. Usage: score_fingerprint_01.py WORKDIR STEP [ATLAS_PROSTATE_PARQUET]"""
import os, sys, json, math, gzip, subprocess, requests, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); MP = os.path.abspath(os.path.join(HERE, "../../.."))
sys.path.insert(0, os.path.join(MP, "chain")); sys.path.insert(0, os.path.join(HERE, "../DEV_RUNLOSS_01"))
W, STEP = sys.argv[1], sys.argv[2]; os.makedirs(os.path.join(W, "betas"), exist_ok=True)
PREC = ["SRR4238614", "SRR4238615", "SRR4238616", "SRR4238617"]; LNCAP = ["SRR4238609", "SRR4238610", "SRR4238611", "SRR4238612", "SRR4238613"]
ARR = {"GSM2309170": "LNCaP", "GSM2309171": "LNCaP", "GSM2309172": "PrEC", "GSM2309173": "PrEC"}
SITES = os.path.join(HERE, "prostate_identity_sites_v1.json")
def H(e): return -(e * math.log2(e) + (1 - e) * math.log2(1 - e))
def Hb(b): b = np.clip(b, 1e-6, 1 - 1e-6); return -(b * np.log2(b) + (1 - b) * np.log2(1 - b))
if STEP == "sites":
    P = pd.read_parquet(sys.argv[3]).set_index("cpg"); ok = P["sd"] <= 0.05
    hi = P[ok & P["mean"].between(0.75, 0.95)].sort_values("sd").index[:3000]; lo = P[ok & P["mean"].between(0.05, 0.25)].sort_values("sd").index[:3000]
    S = sorted(hi.union(lo)); json.dump({"rule": "atlas v2 prostate epithelium posterior: mean 0.75-0.95 or 0.05-0.25, posterior SD <= 0.05, <= 3,000 per channel, smallest SD",
        "n": len(S), "n_hi": len(hi), "n_lo": len(lo), "sites": S, "posterior_mean": [round(float(P.loc[s, "mean"]), 5) for s in S]}, open(SITES, "w"))
    print("prostate identity sites", len(S), "(methylated", len(hi), ", unmethylated", len(lo), ")")
elif STEP == "calib":
    from stage_1_idat_calibration import calibrate_idat_to_beta
    for g in ARR:
        out = os.path.join(W, "betas", g + ".parquet")
        if os.path.exists(out): continue
        t = requests.get("https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi", params={"acc": g, "targ": "self", "form": "text", "view": "brief"}, timeout=60).text
        loc = []
        for u in [l.split("=", 1)[1].strip() for l in t.split("\n") if l.startswith("!Sample_supplementary_file") and "idat" in l.lower()]:
            f = os.path.join(W, u.rsplit("/", 1)[-1]); subprocess.run(["curl", "-s", "-o", f, u.replace("ftp://", "https://")], check=True); loc.append(f)
        o = calibrate_idat_to_beta([x for x in loc if "_Grn" in x][0], [x for x in loc if "_Red" in x][0], verbose=False); b = o[0] if isinstance(o, tuple) else o
        pd.DataFrame({"beta": b}).to_parquet(out); [os.remove(x) for x in loc]; print(g, ARR[g], "ok", flush=True)
else:
    import stage_q_iam_a as Q, runloss_01 as RL
    def counts(p):
        m6 = rd = 0
        with gzip.open(p, "rt") as f:
            for l in f:
                q = l.split("\t"); n = int(q[3]); c = [x for x in q[2] if x != "."]
                if len(c) < 6: continue
                m6 += n; rd += n * (c.count("C") >= 0.8 * len(c))
        return m6, rd
    rows = []
    for r in PREC + LNCAP:
        p = os.path.join(W, "pat", r + ".pat.gz"); T = Q.pat_site_table(p); m6, rd = counts(p)
        rows.append(dict(run=r, cell="PrEC" if r in PREC else "LNCaP", errors=float(T.err_A.sum() + T.err_B.sum()), opportunities=float(T.opp_A.sum() + T.opp_B.sum()), m6=m6, read=rd))
        print(rows[-1], flush=True)
    R = pd.DataFrame(rows); R["eps"] = R.errors / R.opportunities; g = R.groupby("cell")[["errors", "opportunities", "m6", "read"]].sum()
    eP, eL = g.loc["PrEC", "errors"] / g.loc["PrEC", "opportunities"], g.loc["LNCaP", "errors"] / g.loc["LNCaP", "opportunities"]
    IA = H(eL) / H(eP); share = (g.loc["LNCaP", "read"] / g.loc["LNCaP", "m6"]) / (g.loc["PrEC", "read"] / g.loc["PrEC", "m6"])
    S = json.load(open(SITES))["sites"]; B = {k: pd.read_parquet(os.path.join(W, "betas", k + ".parquet")).beta for k in ARR}
    for k in B: B[k].index = B[k].index.astype(str)
    M = pd.DataFrame({k: B[k].reindex(S) for k in ARR}).dropna(); mh = {k: float(Hb(M[k]).mean()) for k in ARR}
    MA = np.mean([mh[k] for k in ARR if ARR[k] == "LNCaP"]) / np.mean([mh[k] for k in ARR if ARR[k] == "PrEC"])
    RS = pd.read_csv(os.path.join(HERE, "insilico_prec_SRR4238614.csv"))
    XS = np.array([1.00, 1.02, 1.05, 1.10, 1.16]); YB = np.array([1.000, 1.023, 1.056, 1.115, 1.188]); XQ = np.interp(XS, RS.IAMA_rel_simple, RS.IAMA_rel)
    SLOPE = np.polyfit(XQ, YB, 1)[0]; cB = float(np.interp(IA, XQ, YB, right=YB[-1] + SLOPE * (IA - XQ[-1])))
    se = math.sqrt(0.020 ** 2 * 2 / 2 + SLOPE ** 2 * 0.009 ** 2 * (1 / len(LNCAP) + 1 / 4)); z = (MA - cB) / se
    V = [RL.load(os.path.join(W, "pat", r + ".pat.gz")) for r in PREC]; bv = RL.beta_v(V); v0 = RL.read(V[0], bv)
    ex = {}
    for r in PREC + LNCAP:
        f = V[PREC.index(r)] if r in PREC else RL.load(os.path.join(W, "pat", r + ".pat.gz")); x = RL.read(f, bv); de = 1 - x["beta_T"] / v0["beta_T"]
        ls = RL.read(V[0], bv, RL.scat(V[0], max(de, 0), 7))["L"]; ex[r] = x["L"] - ls; print(r, {k: round(v, 4) for k, v in x.items()}, f"excess {ex[r]:+.4f}", flush=True)
    R["excess_run_loss"] = R.run.map(ex); R.to_csv(os.path.join(HERE, "fingerprint_01_rows.csv"), index=False)
    armB = min(ex[r] for r in LNCAP) > max(ex[r] for r in PREC)
    print(f"IAM-A_rel {IA:.4f} (eps PrEC {eP:.5f}, LNCaP {eL:.5f}) | share_read {share:.3f} | Met-A_rel {MA:.4f} on {len(M)} sites (arrays {json.dumps({k: round(v, 5) for k, v in mh.items()})})")
    print(f"ARM {'A' if share >= 0.70 else 'B'} decides | Arm A: curve B at IAM-A_rel {cB:.4f}, excess {MA - cB:+.4f}, se {se:.4f}, z {z:.2f} -> {'FINGERPRINT' if z > 1.645 else 'no fingerprint'}"
          f" | Arm B: LNCaP min excess {min(ex[r] for r in LNCAP):+.4f} vs PrEC max {max(ex[r] for r in PREC):+.4f} -> {'FINGERPRINT' if armB else 'no fingerprint'}")
