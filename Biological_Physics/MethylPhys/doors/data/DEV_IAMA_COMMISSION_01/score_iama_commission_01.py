"""DEV-IAMA-COMMISSION-01 scorer (written 2026-10-11, before any EpiQC read). Input: one row per library (LIBS.csv, written by the box run):
kit, lab, line, rep, eps (Stage Q copy error), opp, conversion (human chr1 CHH, alignment_qc.py), q0 (Stage Q0 state: PROCEED or a refusal
code), eps_planted (Stage Q copy error after 1 % of methylated calls are lost, insilico_loss_02.py rule), c1_iama (IAM-A at planted conversion
0.98 relative to none, conversion_05.py rule; one library per line, blank elsewhere).
A library = kit x laboratory x line x replicate; sequencing runs of one library (lanes, sequencers) are pooled before reading.
Rules, all fixed here:
  H(e) = binary entropy; A = H(eps).
  Evaluable: q0 == PROCEED and conversion >= 0.98 (Stage Q0's limit). Others are excluded and counted.
  C1 holds when every c1_iama >= 0.99 (bias at the limit within 0.01, the bound of synthetic test 5). If C1 fails, C2-C5 are not scored.
  Run = kit x lab x rep. Tare: A_rel = A / median(A of the other evaluable lines of its run), at least 3 references (stage_q_iam_a.MIN_REFS).
  TOL = 0.03. "k of n" rule: pass when n >= 4 and at least n - 1 items are within TOL; n < 4 is not assessed.
  C2 per kit x lab: IAM-A rep2 / rep1 (raw, same run conditions) for each line with both replicates evaluable.
  C3 kits: replicate 1, laboratory LAB01, tared A_rel; per line the median over kits; per kit, each line within TOL of that median.
  C3 labs: EMSeq replicate 1, tared A_rel, LAB01 against LAB02, per line.
  C4: not assessed on these lines (lymphoblastoid cell lines, not neutrophils).
  C5: every evaluable library: H(eps_planted) / H(eps) - 1 > TOL.
  Scope: a kit is inside the commissioned scope only when it meets C2 and C3 kits (EMSeq also C3 labs), with C1 and C5 met.
seal: writes sealed_iama_commission_01.json (rules, this file's sha256, the 78 library list). score LIBS.csv: refuses unless the seal is committed
unchanged and this scorer matches its sha256; writes scored_iama_commission_01.json. selftest: runs planted_scorer_test (no data).
Usage: python3 score_iama_commission_01.py seal LIBLIST.csv | score LIBS.csv | evaluate LIBS.csv (no seal check; planted tests only)"""
import os, sys, json, hashlib, subprocess, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); SEAL = os.path.join(HERE, "sealed_iama_commission_01.json")
TOL, CONV_MIN, C1_MIN, MIN_REFS = 0.03, 0.98, 0.99, 3
H = lambda e: -(e * np.log2(e) + (1 - e) * np.log2(1 - e))

def k_of_n(vals):
    v = [abs(x) <= TOL for x in vals if x is not None and not np.isnan(x)]; n = len(v)
    if n < 4: return dict(n=n, within=sum(v), verdict="NOT ASSESSED")
    return dict(n=n, within=int(sum(v)), verdict="MET" if sum(v) >= n - 1 else "NOT MET")

def evaluate(L):
    L = L.copy(); L["evaluable"] = (L.q0 == "PROCEED") & (L.conversion >= CONV_MIN); L["A"] = H(L.eps)
    out = dict(libraries=len(L), excluded=L[~L.evaluable][["kit", "lab", "line", "rep", "q0", "conversion"]].to_dict("records"))
    c1 = L.c1_iama.dropna(); out["C1"] = dict(values=[round(float(x), 4) for x in c1], verdict="MET" if len(c1) and bool((c1 >= C1_MIN).all()) else "NOT MET")
    if out["C1"]["verdict"] != "MET":
        out.update(C2="NOT SCORED (C1 not met)", C3="NOT SCORED (C1 not met)", C5="NOT SCORED (C1 not met)", scope=[]); return out
    E = L[L.evaluable].copy(); E["A_rel"] = np.nan
    for (k, lab, r), g in E.groupby(["kit", "lab", "rep"]):
        for i in g.index:
            ref = g.drop(i).A
            if len(ref) >= MIN_REFS: E.loc[i, "A_rel"] = E.loc[i, "A"] / float(np.median(ref))
    c2 = {}
    for (k, lab), g in E.groupby(["kit", "lab"]):
        p = g.pivot_table(index="line", columns="rep", values="A")
        d = (p["REP02"] / p["REP01"] - 1).dropna().tolist() if {"REP01", "REP02"} <= set(p.columns) else []
        c2[f"{k} {lab}"] = dict(k_of_n(d), values=[round(x, 4) for x in d])
    out["C2"] = c2
    R1 = E[(E.rep == "REP01") & (E.lab == "LAB01")].pivot_table(index="line", columns="kit", values="A_rel")
    med = R1.median(axis=1); c3k = {k: dict(k_of_n((R1[k] - med).tolist()), values=[round(x, 4) for x in (R1[k] - med).dropna()]) for k in R1.columns}
    out["C3_kits"] = c3k
    em = E[(E.kit == "EMSeq") & (E.rep == "REP01")].pivot_table(index="line", columns="lab", values="A_rel")
    dl = (em["LAB02"] - em["LAB01"]).dropna().tolist() if {"LAB01", "LAB02"} <= set(em.columns) else []
    out["C3_labs"] = dict(k_of_n(dl), values=[round(x, 4) for x in dl])
    mv = (H(E.eps_planted) / E.A - 1); out["C5"] = dict(n=int(mv.notna().sum()), min_move=round(float(mv.min()), 4), verdict="MET" if bool((mv > TOL).all()) else "NOT MET")
    out["C4"] = "NOT ASSESSED (lymphoblastoid cell lines, not neutrophils; stopping rule)"
    scope = []
    if out["C5"]["verdict"] == "MET":
        for k in sorted(E.kit.unique()):
            ok = c3k.get(k, {}).get("verdict") == "MET" and all(v["verdict"] == "MET" for kl, v in c2.items() if kl.split()[0] == k)
            if k == "EMSeq": ok = ok and out["C3_labs"]["verdict"] == "MET"
            if ok: scope.append(k)
    out["scope"] = scope; return out

def committed(path):
    return os.path.exists(path) and subprocess.run(["git", "-C", os.path.dirname(path), "ls-files", "--error-unmatch", os.path.basename(path)], capture_output=True).returncode == 0 \
        and subprocess.run(["git", "-C", os.path.dirname(path), "diff", "--quiet", "HEAD", "--", os.path.basename(path)]).returncode == 0

if __name__ == "__main__":
    mode = sys.argv[1]; sha = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
    if mode == "seal":
        LL = pd.read_csv(sys.argv[2]); import datetime
        json.dump(dict(test="DEV-IAMA-COMMISSION-01", sealed_utc=datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z", scorer_sha256=sha,
                       TOL=TOL, CONV_MIN=CONV_MIN, C1_MIN=C1_MIN, MIN_REFS=MIN_REFS, libraries=LL[["kit", "lab", "line", "rep"]].to_dict("records")), open(SEAL, "w"), indent=1)
        print("sealed", len(LL), "libraries")
    elif mode == "score":
        if not committed(SEAL): sys.exit("REFUSED: seal not committed unchanged")
        S = json.load(open(SEAL))
        if S["scorer_sha256"] != sha: sys.exit("REFUSED: scorer changed since the seal")
        O = evaluate(pd.read_csv(sys.argv[2])); json.dump(O, open(os.path.join(HERE, "scored_iama_commission_01.json"), "w"), indent=1)
        print(json.dumps({k: (v.get("verdict") if isinstance(v, dict) and "verdict" in v else v) for k, v in O.items() if k != "excluded"}, default=str)[:2000])
    elif mode == "evaluate":
        print(json.dumps(evaluate(pd.read_csv(sys.argv[2])), default=str)[:3000])
