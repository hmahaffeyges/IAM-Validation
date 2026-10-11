"""DEV-FINGERPRINT-02 scorer (written 2026-10-10 before any cancer file or cancer array is read). Same test and rule as DEV-FINGERPRINT-01
(score_fingerprint_01.py), per pair, with the differences stated in DEV_FINGERPRINT_02.md (RRBS 36 bp; one array per side; 450K or EPIC within a pair).
Two steps, in this order:
  seal PAIR   NORMAL side only. Writes sealed_<PAIR>.json: normal files (sha256), Stage Q copy error and readability, the arm-A curve B in Stage Q
              units from the normal's own planted-loss response (normal_<cell>/insilico_*.csv), the IAM-A repeat SD, the normal arrays' Met-A,
              the arm-B territory bar (largest normal-file excess run loss), and the sha256 of this script. Commit it before scoring.
  score PAIR  Refuses unless sealed_<PAIR>.json is committed unchanged in HEAD and this script's sha256 equals the sealed one. Then reads the cancer
              files and arrays and applies the sealed rule.
RULE (as DEV-FINGERPRINT-01). IAM-A_rel = H(eps_cancer)/H(eps_normal), each side's files pooled (errors/opportunities). share_read = cancer read share
of >= 6-call molecules / normal's. share_read >= 0.70 -> Arm A decides, else Arm B.
Arm A: z = (Met-A_rel - curveB(IAM-A_rel)) / se, se = sqrt(0.020^2 * (1/nA_c + 1/nA_n) + slope^2 * sd_rep^2 * (1/nW_c + 1/nW_n)); FINGERPRINT if z > 1.645.
  sd_rep (sealed) = max(0.009, the pair's own normal spread: SD of H(eps_file)/H(eps_pooled) over the normal files); curve B beyond its table is
  extended linearly at its slope and the verdict then says so.
Arm B: territory and reference from the normal files (DEV_RUNLOSS_01 functions); FINGERPRINT if every cancer file's excess run loss exceeds the
  largest normal file's. Met-A_rel = mean over cancer arrays of mean H(beta) at the normal cell's identity sites (measured on every array of the
  pair) / the same over normal arrays.
Usage: score_fingerprint_02.py seal|score PAIR PATDIR BETADIR"""
import os, sys, json, math, gzip, hashlib, subprocess, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); MP = os.path.abspath(os.path.join(HERE, "../../.."))
sys.path.insert(0, os.path.join(MP, "chain")); sys.path.insert(0, os.path.join(HERE, "../DEV_RUNLOSS_01"))
import stage_q_iam_a as Q, runloss_01 as RL
PAIRS = {  # pair: normal RRBS cell, cancer RRBS cell, identity-site cell, normal arrays, cancer arrays (same platform within a pair)
    "prostate":      ("epithelial cell of prostate", "LNCaP clone FGC", "prostate_epithelium", ["ENCSR000ACD"], ["ENCSR000ADA"]),
    "liver":         ("hepatocyte", "HepG2", "hepatocyte", ["ENCSR955LKF"], ["ENCSR291NYD"]),
    "lung_alveolar": ("epithelial cell of alveolus of lung", "A549", "lung_alveolar_epithelium", ["ENCSR422EPB"], ["ENCSR408IKV"]),
    "lung_bronchial":("bronchial epithelial cell", "A549", "lung_bronchus_epithelium", ["ENCSR000ACA", "ENCSR000ACY"], ["ENCSR000ABU"]),
    "breast_hmec":   ("mammary epithelial cell", "MCF-7", "breast_basal_epithelium", ["ENCSR148KKY", "ENCSR583ILE"], ["ENCSR889ZLL"]),
    "breast_mcf10a": ("MCF 10A", "MCF-7", "breast_basal_epithelium", ["ENCSR329HJR"], ["ENCSR889ZLL"])}
STEP, PAIR, PATD, BETD = sys.argv[1:5]; NC, CC, SC, AN, AC = PAIRS[PAIR]; SEAL = os.path.join(HERE, f"sealed_{PAIR}.json")
R9 = os.path.join(MP, "boxruns/run9_fp2")
def H(e): return -(e * math.log2(e) + (1 - e) * math.log2(1 - e))
def Hb(b): b = np.clip(b, 1e-6, 1 - 1e-6); return -(b * np.log2(b) + (1 - b) * np.log2(1 - b))
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def files(kind, cell): d = pd.read_csv(os.path.join(R9, f"{kind}_rrbs_files.csv")); return sorted(d[d.cell == cell].file)
def stageq(fs):
    rows = []
    for f in fs:
        p = os.path.join(PATD, f + ".pat.gz"); T = Q.pat_site_table(p); m6 = rd = 0
        with gzip.open(p, "rt") as g:
            for l in g:
                q = l.split("\t"); n = int(q[3]); c = [x for x in q[2] if x != "."]
                if len(c) >= 6: m6 += n; rd += n * (c.count("C") >= 0.8 * len(c))
        rows.append(dict(file=f, sha256=sha(p), errors=float(T.err_A.sum() + T.err_B.sum()), opportunities=float(T.opp_A.sum() + T.opp_B.sum()), m6=m6, read=rd))
    R = pd.DataFrame(rows); R["eps"] = R.errors / R.opportunities; return R
def meta(arrs, sites):
    B = {}
    for k in arrs:
        b = pd.read_parquet(os.path.join(BETD, k + ".parquet")).beta; b.index = b.index.astype(str); B[k] = b.reindex(sites)
    return pd.DataFrame(B)
SITES = json.load(open(os.path.join(HERE, f"identity_sites_{SC}.json")))["sites"]
if STEP == "seal":
    if os.path.exists(SEAL): sys.exit(f"{SEAL} exists: a seal is written once")
    nf = files("normal", NC); assert len(nf) >= 2, nf; RN = stageq(nf); eN = RN.errors.sum() / RN.opportunities.sum()
    spread = float(np.std([H(e) / H(eN) for e in RN.eps], ddof=1)); sd_rep = max(0.009, spread)
    tag = NC.replace(" ", "_"); ins = sorted(x for x in os.listdir(os.path.join(HERE, f"normal_{tag}")) if x.startswith("insilico_"))[0]
    RS = pd.read_csv(os.path.join(HERE, f"normal_{tag}", ins))
    XS = np.array([1.00, 1.02, 1.05, 1.10, 1.16]); YB = np.array([1.000, 1.023, 1.056, 1.115, 1.188]); XQ = np.interp(XS, RS.IAMA_rel_simple, RS.IAMA_rel)
    slope = float(np.polyfit(XQ, YB, 1)[0])
    M = meta(AN, SITES)
    V = [RL.load(os.path.join(PATD, f + ".pat.gz")) for f in nf]; bv = RL.beta_v(V); v0 = RL.read(V[0], bv); exN = {}
    for f, x in zip(nf, V):
        r = RL.read(x, bv); de = 1 - r["beta_T"] / v0["beta_T"]; exN[f] = float(r["L"] - RL.read(V[0], bv, RL.scat(V[0], max(de, 0), 7))["L"])
    S = dict(pair=PAIR, normal_cell=NC, cancer_cell=CC, identity_cell=SC, normal_files=RN.to_dict("records"), eps_normal=float(eN),
             share_read_normal=float(RN.read.sum() / RN.m6.sum()), sd_rep=sd_rep, sd_rep_own_spread=spread,
             response_file=f"normal_{tag}/{ins}", response_sha256=sha(os.path.join(HERE, f"normal_{tag}", ins)),
             curveB_XQ=[float(x) for x in XQ], curveB_YB=[float(y) for y in YB], slope=slope,
             normal_arrays=AN, cancer_arrays=AC, normal_arrays_metA_H={k: float(Hb(M[k].dropna()).mean()) for k in AN}, normal_arrays_sites={k: int(M[k].notna().sum()) for k in AN},
             armB_normal_excess=exN, armB_bar=max(exN.values()), scorer_sha256=sha(os.path.abspath(__file__)),
             rule="FINGERPRINT-01 rule; arm by share_read >= 0.70; Arm A z > 1.645; Arm B every cancer file's excess > armB_bar")
    json.dump(S, open(SEAL, "w"), indent=1); print(json.dumps({k: v for k, v in S.items() if k not in ("normal_files",)}, indent=1)[:2500])
elif STEP == "score":
    rel = os.path.relpath(SEAL, subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, cwd=HERE).stdout.strip())
    top = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    committed = subprocess.run(["git", "show", f"HEAD:{rel}"], capture_output=True, cwd=top)
    if committed.returncode != 0 or committed.stdout != open(SEAL, "rb").read(): sys.exit(f"REFUSED: {rel} is not committed unchanged in HEAD")
    S = json.load(open(SEAL))
    if S["scorer_sha256"] != sha(os.path.abspath(__file__)): sys.exit("REFUSED: this scorer differs from the one sealed")
    nf = [r["file"] for r in S["normal_files"]]; cf = files("cancer", CC); assert cf, CC
    for r in S["normal_files"]: assert sha(os.path.join(PATD, r["file"] + ".pat.gz")) == r["sha256"], r["file"]
    RC = stageq(cf); eC = RC.errors.sum() / RC.opportunities.sum(); IA = H(eC) / H(S["eps_normal"])
    share = (RC.read.sum() / RC.m6.sum()) / S["share_read_normal"]
    M = meta(AN + AC, SITES).dropna(); mh = {k: float(Hb(M[k]).mean()) for k in AN + AC}
    MA = np.mean([mh[k] for k in AC]) / np.mean([mh[k] for k in AN])
    XQ, YB, sl = np.array(S["curveB_XQ"]), np.array(S["curveB_YB"]), S["slope"]; ext = IA > XQ[-1]
    cB = float(np.interp(IA, XQ, YB, right=YB[-1] + sl * (IA - XQ[-1])))
    se = math.sqrt(0.020 ** 2 * (1 / len(AC) + 1 / len(AN)) + sl ** 2 * S["sd_rep"] ** 2 * (1 / len(cf) + 1 / len(nf))); z = (MA - cB) / se
    V = [RL.load(os.path.join(PATD, f + ".pat.gz")) for f in nf]; bv = RL.beta_v(V); v0 = RL.read(V[0], bv); exC = {}
    for f in cf:
        r = RL.read(RL.load(os.path.join(PATD, f + ".pat.gz")), bv); de = 1 - r["beta_T"] / v0["beta_T"]
        exC[f] = float(r["L"] - RL.read(V[0], bv, RL.scat(V[0], max(de, 0), 7))["L"])
    armB = min(exC.values()) > S["armB_bar"]; arm = "A" if share >= 0.70 else "B"
    vA = "FINGERPRINT" if z > 1.645 else "no fingerprint"; vB = "FINGERPRINT" if armB else "no fingerprint"
    RC["excess_run_loss"] = RC.file.map(exC); RC.to_csv(os.path.join(HERE, f"scored_{PAIR}_cancer_rows.csv"), index=False)
    out = dict(pair=PAIR, IAMA_rel=IA, eps_cancer=float(eC), eps_normal=S["eps_normal"], share_read=share, MetA_rel=float(MA), sites_all_arrays=len(M),
               arrays_H=mh, curveB=cB, curveB_extended=bool(ext), se=se, z=z, armA=vA, cancer_excess=exC, armB_bar=S["armB_bar"], armB=vB, deciding_arm=arm,
               VERDICT=vA if arm == "A" else vB)
    json.dump(out, open(os.path.join(HERE, f"scored_{PAIR}.json"), "w"), indent=1); print(json.dumps(out, indent=1))
