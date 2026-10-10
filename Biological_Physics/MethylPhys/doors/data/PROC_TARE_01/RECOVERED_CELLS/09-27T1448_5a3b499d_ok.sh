set -e; W="$(pwd)"; MP="$W/iamrepo/Biological_Physics/MethylPhys"
cat > "$MP/doors/PROC_STAGE2D_02_OUTCOME.md" <<'EOF'
# PROC-STAGE2D-02 — outcome: NOT ADOPTED. The detector's design, not its lines, was the defect.

Scored 2026-09-27 against [`PROC_STAGE2D_02_PREREG.md`](PROC_STAGE2D_02_PREREG.md).

| bar | measured | |
|---|---|---|
| B1 held-out FP, leave-one-chip-out, 732 arrays | 8 detectable templates at 1.1–1.2 %; ≥10-fire arrays 0 | MET |
| B2 cross-laboratory (23 Karolinska + UCLA arrays) | 0 fires | MET |
| B3 detection kept (real spikes) | **0 of 8 at 5 % for Breast, Bladder, Kidney, Cortical neurons** | **FAILED** |
| B4 composition unchanged | structurally unchanged (Stage 2d writes no composition field); the shard comparison used a different Stage 1 input and is void | not scored |
| B5 thin-source templates NOT DETECTABLE on every array | 13 of 13 on 47 arrays | MET |
| B6 kit test | not written — superseded | — |
| B7 (added after the author caught a Glia BREACH on a healthy reference array) no non-blood cell scored unless detected | 0 undetected non-blood cells scored on 60 arrays | MET |

## What B3 found, in order
1. My first spike construction dropped every locus where the cell is undefined and was void.
2. **PROC-MF-03's spike was circular**: `v(1−f) + template·f` injected the panel's own filled template and detected it. That is
   where the 0.5–1 % detection limits came from.
3. With honest spikes (the cell's atlas profile at the loci where it is measured; host β elsewhere) the raw detector's
   response to a 5 % spike is **2–13 % of f**; at 20 % it is ~50 %. The blood-only NNLS is fitted *after* the foreign material
   is present and absorbs it. Real detection limit of v1/v2: ~15–20 %.
4. At 20 % a Breast spike moves the uterus and endothelial templates more than Breast: v1/v2 cannot name the epithelium.
5. Common-mode removal (this procedure's change) subtracts a real spike too, because every template rises with any foreign
   material (epithelial–epithelial r = 0.56, epithelial–neural 0.68 on healthy blood). Group-wise estimators do not rescue it.
6. **A joint fit — blood columns and all 21 templates in one NNLS — does not have the defect.** Healthy null median 0 (q99
   0.000–0.021), response 0.6–1.0 of f, 6/6 detected at 5 % for Kidney, Colon, Glia, β-cells with the largest coefficient on the
   right cell; Breast needs 10 %. Glia and Thyroid carry a standing 0.2–0.8 % on healthy blood (the Glia BREACH's origin).

## Correction to the finding of this morning
FINDING_DETECTION_PANEL_HELDOUT said the biased templates were the thin-source family. Coverage does not sort them: 19 of 21
templates are on 1.3–4.5 % of the array; only cortical neurons and glia are full-coverage; the gastric families at 79 % were
among the biased. All 1,506 panel markers are defined for every template. The split was real; the cause I gave was wrong.

## Decision
Not adopted. v2's noise floor and NOT-DETECTABLE list are correct for a detector that should not be used; the B7 gate stays
(a non-blood cell in whole blood is scored only when detected). The detector is rebuilt on the joint fit under PROC-STAGE2D-03.
EOF
cat > "$MP/doors/PROC_STAGE2D_03_PREREG.md" <<'EOF'
# PROC-STAGE2D-03 — pre-registration: foreign-cell detection as one joint fit

**Written 2026-09-27 before the bars are scored.** The exploration that led here is recorded in PROC_STAGE2D_02_OUTCOME.md
(6 quiet hosts, 60 healthy arrays); every bar below is scored on arrays and hosts not used there where the design allows.

## The detector (fixed)
On the 1,506 panel markers (mapped β): one NNLS of the specimen on [the blood reference columns | all 21 foreign templates].
f̂_cell = coefficient_cell / Σ coefficients. No residual projection, no common mode, no centre. Author's decision 2026-09-27:
the line is the instrument's **noise floor**, the 0.99 quantile of f̂_cell over the Uppsala arrays the intake gate admits, stated
on the page with N. A template whose f̂ median on healthy blood exceeds 0.002 carries a standing bias: its floor is still the
0.99 quantile, and the page says "standing bias b on blood" beside it. Gate: composition check verified blood-like.
UNSPECIFIC if ≥ 3 templates fire together. The B7 gate (a non-blood cell in whole blood is scored only when detected) reads this.

## Bars
- **B1** leave-one-chip-out over all admitted Uppsala arrays (732 minus intake refusals; none refused on the first 100): per-template
  FP ≤ 1.5 %; UNSPECIFIC arrays ≤ 1 %.
- **B2** Karolinska + UCLA admitted panel arrays (23) under the Uppsala floors: ≤ 2 fires per template.
- **B3** honest spikes into 12 quiet hosts NOT among the 6 used in exploration, all 21 templates, f = 0.02 / 0.05 / 0.10: at 0.05,
  ≥ 90 % detected for full-coverage templates (neurons, glia) and ≥ 75 % across the thin templates; at 0.10, ≥ 90 % for all.
- **B4** naming: at 0.10 the largest foreign coefficient is the spiked cell in ≥ 80 % of spikes; at 0.05 in ≥ 60 %. Where it is not,
  the page names the group ("epithelial material, cell not resolved") not a cell.
- **B5** composition unchanged: class fractions and the composition check identical to 1e-9 on 12 arrays with the old and new
  stage_2d (the stage writes no composition field; this checks it).
- **B6** the reference report's Glia row: NOT DETECTED, not scored; no non-blood cell scored undetected on 100 healthy arrays.
- **B7** kit test `test_stage2d_panels.py` rewritten to this contract and passing.

## Decision rule
All met → adopted; detection_panel_v3.json replaces v2; v1 and v2 retired with the record. B3 or B4 failed for the thin templates
only → adopted for neurons/glia, thin templates print the measured limit and "cell not resolved". Anything else → not adopted.
EOF
cd "$W" && cat > stage2d03.py <<'PYEOF'
#!/usr/bin/env python3
# INSTRUMENT-TEST: PROC-STAGE2D-03 - joint-fit foreign-cell detector. Bar results only. Own output dir; polls STOP.
import os, sys, json, glob, time, numpy as np, pandas as pd
from scipy.optimize import nnls
W=os.path.dirname(os.path.abspath(__file__)); CH=os.path.join(W,"iamrepo/Biological_Physics/MethylPhys/chain"); sys.path.insert(0,CH); sys.path.insert(0,CH+"/Synthetic_Patient_Generator")
import cpg_conductor as C, synthetic_patient_generator as SPG
OUT=os.path.join(W,"results/stage2d03"); os.makedirs(OUT,exist_ok=True); open(os.path.join(OUT,"PID"),"w").write(str(os.getpid()))
P=json.load(open(C._find("detection_panel_v2.json"))); M=P["markers"]; Ab0=np.array([P["blood_ref"][c] for c in P["blood_columns"]]).T
cells=list(P["foreign_ref"]); T0=np.array([P["foreign_ref"][c] for c in cells]).T
pf=pd.read_parquet(os.path.join(W,"stage1_betas_GSE87571_FULL.parquet")); pf.index=pf.index.map(str)
def fit(beta):
    mapped,_=C.stage_1s_scale_map(beta.to_dict(),"stage1_noob_450K"); v=np.array([mapped.get(m,np.nan) for m in M]); ok=~np.isnan(v)
    X=np.column_stack([Ab0[ok],T0[ok]]); coef,_=nnls(X,v[ok]); return pd.Series(coef[Ab0.shape[1]:]/max(coef.sum(),1e-9),index=cells)
# ---- null on all 732 (shard per array)
rows={}
for k,g in enumerate(pf.columns):
    if os.path.exists(os.path.join(OUT,"STOP")): print("STOP",flush=True); break
    sh=os.path.join(OUT,f"null_{g}.json")
    if os.path.exists(sh): rows[g]=pd.Series(json.load(open(sh))); continue
    s=fit(pf[g].dropna()); json.dump(s.to_dict(),open(sh+".tmp","w")); os.replace(sh+".tmp",sh); rows[g]=s
    if k%100==99: print(f"null {k+1}/{len(pf.columns)}",flush=True)
N=pd.DataFrame(rows).T; N.to_csv(os.path.join(OUT,"null_732.csv"))
tare=pd.read_parquet(os.path.join(W,"results/tare01/PROC_TARE_01_per_array.parquet")); tare=tare[tare.lab=="GSE87571"].set_index("gsm"); chip=tare.reindex(N.index)["chip"]
fp={c:0 for c in cells}; unsp=0; n=0
for ch in chip.dropna().unique():
    te=chip.index[chip==ch]; tr=chip.index[(chip!=ch)&chip.notna()]; L=N.loc[tr].quantile(.99); h=(N.loc[te]>L)
    for c in cells: fp[c]+=int(h[c].sum())
    unsp+=int((h.sum(axis=1)>=3).sum()); n+=len(te)
b1=max(fp.values())/n<=0.015 and unsp/n<=0.01
print("B1 leave-one-chip-out FP %:",{c:round(fp[c]/n*100,2) for c in cells},f"| unspecific {unsp}/{n} ->","MET" if b1 else "FAILED",flush=True)
floors=N.quantile(.99); bias={c:float(N[c].median()) for c in cells}
# ---- B3/B4 spikes: 12 quiet hosts not among the exploration 6
per=pd.read_csv(os.path.join(W,"handoff/heldout2d_per_array.csv")).set_index("gsm"); quiet=[g for g in per[per.ndet==0].index if g in pf.columns]
hosts=quiet[6:18]; mu=SPG._cell_means(os.path.join(W,"atlas_work/IAMAtlasREBUILD.csv")); mu.index=mu.index.map(str)
FULL=["Cortical_neurons","Glia"]; spk=[]
for c in cells:
    if os.path.exists(os.path.join(OUT,"STOP")): break
    members=c[7:].split("+") if c.startswith("family:") else [c]; col=mu[[m for m in members if m in mu.columns]].mean(axis=1).dropna()
    for f in (0.02,0.05,0.10):
        for h in hosts:
            b=pf[h].dropna(); loci=b.index.intersection(col.index); s=b.copy(); s.loc[loci]=(1-f)*b.loc[loci]+f*col.loc[loci].values
            r=fit(s); fired=[x for x in cells if r[x]>floors[x]]
            spk.append(dict(cell=c,f=f,host=h,amp=float(r[c]),detected=bool(r[c]>floors[c]),named=(r.idxmax()==c),n_fired=len(fired)))
    d=pd.DataFrame([x for x in spk if x["cell"]==c]); print(f"  {c:<48} det@.02 {d[d.f==.02].detected.mean():.2f} @.05 {d[d.f==.05].detected.mean():.2f} @.10 {d[d.f==.10].detected.mean():.2f} | named@.05 {d[d.f==.05].named.mean():.2f} @.10 {d[d.f==.10].named.mean():.2f}",flush=True)
S=pd.DataFrame(spk); S.to_csv(os.path.join(OUT,"spikes.csv"),index=False)
full=S[S.cell.isin(FULL)]; thin=S[~S.cell.isin(FULL)]
b3=(full[full.f==.05].detected.mean()>=.9) and (thin[thin.f==.05].detected.mean()>=.75) and (S[S.f==.10].detected.mean()>=.9)
b4=(S[S.f==.10].named.mean()>=.8) and (S[S.f==.05].named.mean()>=.6)
print(f"B3 detection: full@.05 {full[full.f==.05].detected.mean():.2f} thin@.05 {thin[thin.f==.05].detected.mean():.2f} all@.10 {S[S.f==.10].detected.mean():.2f} ->","MET" if b3 else "FAILED")
print(f"B4 naming: @.10 {S[S.f==.10].named.mean():.2f} @.05 {S[S.f==.05].named.mean():.2f} ->","MET" if b4 else "FAILED")
json.dump({"B1":{"met":b1,"fp":{c:fp[c]/n for c in cells},"unspecific":unsp,"n":n},"B3":{"met":bool(b3)},"B4":{"met":bool(b4)},"floors":floors.to_dict(),"bias":bias,"hosts":hosts},open(os.path.join(W,"handoff/stage2d03_results.json"),"w"),indent=1)
print("RESULTS WRITTEN",flush=True)
PYEOF
python3 -c "import ast; ast.parse(open('stage2d03.py').read()); print('parses')"
cd "$W/iamrepo" && git add -A Biological_Physics && PYTHON="$(which python3)" IAM_WEBDRAFTS="$W/webdrafts" sh Biological_Physics/MethylPhys/chain/guarded_push.sh "PROC-STAGE2D-02 NOT ADOPTED (B3 failed; outcome records the investigation): PROC-MF-03's 0.5 % detection limits came from a circular spike (the panel's own template injected and detected); with honest spikes the v1/v2 detector responds at 2-13 % of a 5 % spike because the blood-only NNLS is fitted after the foreign material is present and absorbs it - real limit ~15-20 %, and it cannot name the epithelium. Common-mode removal subtracts real signal too. A JOINT fit (blood + all 21 templates in one NNLS) has none of this: null median 0, response 0.6-1.0 of f, 6/6 at 5 % with the right cell named for Kidney/Colon/Glia/beta. Correction to this morning's finding: coverage does not sort biased from clean templates. PROC-STAGE2D-03 pre-registered on the joint fit (7 bars, decision rule). Conductor: B7 gate - a non-blood cell in a whole-blood specimen is scored only when Stage 2d detects it (the author caught a Glia BREACH at 1.03 % on a healthy reference array); minority cells (< 5 %) flagged. stage_2d reads detection_panel_v2 (noise floor + not-detectable) until 03 lands." 2>&1 | grep -vE "protected ref|^remote: *$|Bypassed" | tail -3; echo "origin/main: $(git fetch -q origin main && git rev-parse --short origin/main)"
grep -E "FAIL" .build_all_last_run.txt | head -3 | cut -c1-300 || true
cd "$W" && HOME="$W/stage1/mp_home" nohup python3 stage2d03.py > stage2d03.log 2>&1 & 
sleep 120; grep -vE "Deprecat|^INFO|Scanning|Selecting|class markers|class columns|coverage:|cell-type markers|exclusive markers|twins:" "$W/stage2d03.log" | tail -3