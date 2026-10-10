"""DEV-CSCORE-MOSS-01 step 1 (2026-10-10, before any GSE122126 mix array is downloaded): the chain's own readings on the 9 Moss 2018 in vitro mixes,
built in silico from the series' PURE component arrays (EPIC, GPL21145): leukocytes (1), hepatocytes, lung epithelial, colon epithelial, cortical
neurons. Mix makeup from the GEO sample descriptions (= the paper's Supplementary Data 1, Table 6; GEO mix_9..17 = paper Mix1..9).
A mix is formed in beta space, beta = (1 - sum f) * beta_leuk + sum_k f_k * beta_k (DNA mass fractions; ignores per-array intensity differences),
once for every combination of the component replicates, and read by conductor_v3.run_neutrophil(specimen="constructed DNA mixture"):
Met-A, its composition, and the C-score (untared: only one leukocyte array, so no same-run tare). The pure leukocyte array is read the same way.
Steps: 'calib' downloads and calibrates the components through chain Stage 1; 'read' builds and reads the mixes. Usage: insilico_moss_01.py WORKDIR calib|read|readmix"""
import os, sys, re, json, itertools, gzip, shutil, subprocess, requests, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); CH = os.path.abspath(os.path.join(HERE, "../../../chain")); sys.path.insert(0, CH)
W, STEP = sys.argv[1], sys.argv[2]; os.makedirs(os.path.join(W, "betas"), exist_ok=True)
COMP = {"leukocytes": "Leuk", "hepatocytes": "Hep", "lung epithelial cells": "Lung", "colon epithelial cells": "Colon", "cortical neurons": "Neuron"}
MIX = {"Mix9": {"Hep": .10, "Lung": .035}, "Mix10": {"Hep": .05, "Neuron": .10}, "Mix11": {"Hep": .035, "Colon": .05}, "Mix12": {"Colon": .10, "Neuron": .05},
       "Mix13": {"Lung": .05, "Colon": .035}, "Mix14": {"Lung": .10, "Neuron": .035}, "Mix15": {"Hep": .04}, "Mix16": {"Lung": .08}, "Mix17": {"Colon": .06}}
def components():
    t = requests.get("https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi", params={"acc": "GSE122126", "targ": "gsm", "form": "text", "view": "brief"}, timeout=120).text.replace("\r", "")
    rows = []
    for b in t.split("^SAMPLE = ")[1:]:
        get = lambda k: [l.split("=", 1)[1].strip() for l in b.split("\n") if l.startswith(k)]
        src = get("!Sample_source_name_ch1")[0]
        if get("!Sample_platform_id")[0] == "GPL21145" and src in COMP:
            rows.append(dict(gsm=b.split("\n")[0].strip(), comp=COMP[src], files=[x for x in get("!Sample_supplementary_file") if "idat" in x.lower()]))
    return pd.DataFrame(rows)
if STEP == "readmix":   # the 9 real mix arrays (sealed prediction in doors/DEV_CSCORE_MOSS_01.md)
    from stage_1_idat_calibration import calibrate_idat_to_beta
    import conductor_v3 as C3
    t = requests.get("https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi", params={"acc": "GSE122126", "targ": "gsm", "form": "text", "view": "brief"}, timeout=120).text.replace("\r", "")
    rows = []
    for b in t.split("^SAMPLE = ")[1:]:
        get = lambda k: [l.split("=", 1)[1].strip() for l in b.split("\n") if l.startswith(k)]
        ti = get("!Sample_title")[0]
        if not ti.startswith("in_vitro_mix"): continue
        g = b.split("\n")[0].strip(); out = os.path.join(W, "betas", g + ".parquet")
        if not os.path.exists(out):
            loc = []
            for u in [x for x in get("!Sample_supplementary_file") if "idat" in x.lower()]:
                f = os.path.join(W, u.rsplit("/", 1)[-1]); subprocess.run(["curl", "-s", "-o", f, u.replace("ftp://", "https://")], check=True); loc.append(f)
            o = calibrate_idat_to_beta([x for x in loc if "_Grn" in x][0], [x for x in loc if "_Red" in x][0], verbose=False); bb = o[0] if isinstance(o, tuple) else o
            pd.DataFrame({"beta": bb}).to_parquet(out); [os.remove(x) for x in loc]
        beta = pd.read_parquet(out).beta; beta.index = beta.index.astype(str)
        o = C3.run_neutrophil(beta, specimen="constructed DNA mixture"); m, cs = o.get("met_a", {}), o.get("met_a_cscore", {})
        rows.append(dict(gsm=g, mix="Mix" + ti.rsplit("_", 1)[-1], A=m.get("A"), C=cs.get("C"), reason=m.get("reason"), f_neu=((o.get("composition") or {}).get("fractions") or {}).get("NEU")))
        print(rows[-1], flush=True)
    R = pd.DataFrame(rows).sort_values("mix"); R.to_csv(os.path.join(HERE, "moss_mixes_01_rows.csv"), index=False)
    L = pd.read_parquet(os.path.join(W, "betas", "GSM3455862.parquet")).beta; L.index = L.index.astype(str); AL = C3.run_neutrophil(L, specimen="constructed DNA mixture")["met_a"]["A"]
    ok1 = R.A.between(0.95, 1.05) & ((R.A - AL).abs() <= 0.03); ok2 = R.C.between(0.90, 1.10)
    print(f"leukocytes A {AL:.4f} | bar 1 met {int(ok1.sum())}/{len(R)} | bar 2 met {int(ok2.sum())}/{len(R)}")
    sys.exit(0)
if STEP == "calib":
    from stage_1_idat_calibration import calibrate_idat_to_beta
    C = components(); C.drop(columns="files").to_csv(os.path.join(W, "components.csv"), index=False); print(C.groupby("comp").size().to_dict(), flush=True)
    for _, r in C.iterrows():
        out = os.path.join(W, "betas", r.gsm + ".parquet")
        if os.path.exists(out): continue
        loc = []
        for u in r.files:
            f = os.path.join(W, u.rsplit("/", 1)[-1]); subprocess.run(["curl", "-s", "-o", f, u.replace("ftp://", "https://")], check=True); loc.append(f)
        g = [x for x in loc if "_Grn" in x][0]; rd = [x for x in loc if "_Red" in x][0]
        o = calibrate_idat_to_beta(g, rd, verbose=False); b = o[0] if isinstance(o, tuple) else o
        pd.DataFrame({"beta": b}).to_parquet(out); [os.remove(x) for x in loc]; print(r.gsm, r.comp, "ok", len(b), flush=True)
    print("CALIB_DONE")
else:
    import conductor_v3 as C3
    C = pd.read_csv(os.path.join(W, "components.csv")); Bt = {g: pd.read_parquet(os.path.join(W, "betas", g + ".parquet")).beta for g in C.gsm}
    by = {k: list(C[C.comp == k].gsm) for k in C.comp.unique()}; L = Bt[by["Leuk"][0]]
    # A component array loses probes at background (Stage 1); a real mix array is measured as one array. Where a component replicate is missing a
    # site, it takes the mean of the other replicates of that component that measured it; otherwise the site stays missing (2026-10-10, before reading).
    for k, gs in by.items():
        if len(gs) > 1:
            M_ = pd.concat([Bt[g].reindex(L.index) for g in gs], axis=1)
            for i, g in enumerate(gs): Bt[g] = Bt[g].reindex(L.index).fillna(M_.drop(columns=M_.columns[i]).mean(1))
    def read(beta, lab):
        o = C3.run_neutrophil(beta, specimen="constructed DNA mixture"); m, cs, a = o.get("met_a", {}), o.get("met_a_cscore", {}), o.get("composition", {})
        return dict(mix=lab, A=m.get("A"), f_neu=(a.get("fractions") or {}).get("NEU"), C=cs.get("C"), clustering=cs.get("clustering"), refusal=o.get("refusal"))
    rows = [dict(read(L, "leukocytes alone"), reps="")]
    for mx, fr in MIX.items():
        for reps in itertools.product(*[by[k] for k in fr]):
            b = (1 - sum(fr.values())) * L
            for k, g in zip(fr, reps): b = b + fr[k] * Bt[g].reindex(L.index).fillna(L)   # still missing: the leukocyte value (no tissue difference assumed there; understates the change)
            rows.append(dict(read(b, mx), reps="+".join(reps), tissue=sum(fr.values()), makeup=json.dumps(fr)))
    R = pd.DataFrame(rows); R.to_csv(os.path.join(HERE, "insilico_moss_01_rows.csv"), index=False)
    print(R.groupby("mix", sort=False).agg(n=("A", "size"), tissue=("tissue", "first"), A=("A", "median"), A_min=("A", "min"), A_max=("A", "max"),
          C=("C", "median"), C_min=("C", "min"), C_max=("C", "max"), f_neu=("f_neu", "median")).round(4).to_string())
