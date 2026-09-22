import os, sys, importlib.util as iu, pickle, json
import numpy as np, pandas as pd
os.environ["CPG_ROOT"]="/home/claude/CPG_CMB_v4"; sys.path.insert(0,"/home/claude/CPG_CMB_v4")
_sp=iu.spec_from_file_location("pdshim","/home/claude/pdshim.py"); _m=iu.module_from_spec(_sp); _sp.loader.exec_module(_m)
import walther_clinical as wc, stage_5_second_chain as s5
cfg=wc.DEFAULT_CONFIG
CJ=json.load(open("/home/claude/strawman_data_v2.json"))            # crown jewel
mapping=json.load(open(cfg["matrix_mapping_json"]))["mapping"]
BAND=0.07   # healthy band edge: A in [1-BAND, 1+BAND] = green (1.07 tier line)

def load_beta(p):
    b=pickle.load(open(p,"rb"))
    return b if isinstance(b,pd.Series) else (b.iloc[:,0] if isinstance(b,pd.DataFrame) else pd.Series(b))

def build(bundle_path, beta_path, substrate, patient_id, age, sex):
    b=pickle.load(open(bundle_path,"rb"))
    s4=b.get("stage4_output") or next((v for v in b.values() if isinstance(v,dict) and "celltype_ascores" in v),None)
    ct=s4["celltype_ascores"]
    # per matrix-column patient profile: mean A, dep, CI over atlas cells mapping to the column.
    # PRESENT CELLS ONLY -- same gate as _build_patient_departure_profile: a cell must be above
    # its H_min floor (not below_floor) AND have a real deconvolver fraction (>= MIN_FRACTION).
    # Zero-fraction duplicate panel labels / absent cell types read background and must not appear.
    MIN_FRACTION=0.001
    def _present(rec):
        if not (isinstance(rec,dict) and rec.get("A") is not None and not rec.get("below_floor")):
            return False
        frac=rec.get("celltype_fraction")
        return True if frac is None else float(frac)>=MIN_FRACTION
    bycol={}
    for cell,rec in ct.items():
        if not _present(rec): continue
        col=mapping.get(cell)
        if not col: continue
        bycol.setdefault(col,[]).append((float(rec["A"]),
                                         rec.get("A_ci_lo"), rec.get("A_ci_hi"), cell))
    patient_cells={}
    for col,vals in bycol.items():
        A=float(np.mean([v[0] for v in vals])); dep=A-1.0
        cis=[(v[1],v[2]) for v in vals if v[1] is not None and v[2] is not None]
        ci=[float(np.mean([c[0] for c in cis])), float(np.mean([c[1] for c in cis]))] if cis else None
        # five gauge tiers (tier_breakpoints v1.3, the exact bands cpg_gauge.py draws)
        if A < 0.95: tier="SUPPRESSED"
        elif A < 1.04: tier="NORMAL"
        elif A < 1.07: tier="ELEVATED"
        elif A < 1.10: tier="SIGNIFICANTLY_ELEVATED"
        else: tier="BREACH"
        # confident departure from NORMAL when the 95% CI does not overlap [0.95, 1.04]
        confident = bool(ci and (ci[0] >= 1.04 or ci[1] <= 0.95))
        patient_cells[col]={"A":round(A,3),"dep":round(dep,3),"ci":[round(c,3) for c in ci] if ci else None,
                            "tier":tier,"confident":confident,"atlas":[v[3] for v in vals]}
    # run the chain to get flagged diseases
    s8=wc.stage_8_dual_matching(s4,{},None,patient_meta={"substrate":substrate,"age":age,"sex":sex},config=cfg)
    bundle={"stage4":s4,"stage8":s8,"context":{"substrate":substrate,"age":age,"sex":sex},
            "systemic_stress":wc.detect_systemic_stress_pattern(s8.patient_departure)}
    beta=load_beta(beta_path)
    s5out=s5.run_second_chain(bundle, beta, cfg)
    # flagged diseases shown on the wall = CONFIRMED findings only. The matched-filter sweep is
    # the detector; an unconfirmed per-cell resemblance on a sparse present-cell profile (the
    # generic stress pattern resembling many myeloid conditions) is NOT pulled here -- it would
    # put lung/infection rows beside a breast patient. Only a per-cell flag the second chain
    # actually confirmed is added.
    flagged=[]
    if s5out:
        trig=s5out.get("trigger",{})
        for d in (trig.get("residual_sweep_fired") or []):
            flagged.append({"disease":{"breast_cancer":"breast_cancer","alzheimers_disease":"alzheimers_disease",
                                       "immune_universal_alarm":"immune"}.get(d,d),"via":"matched filter (residual sweep)"})
        if trig.get("flagged_confirmed") and trig.get("flagged_disease"):
            flagged.append({"disease":trig["flagged_disease"],"via":"per-cell matcher (confirmed)"})
    # pull the matching crown-jewel rows for each flagged disease
    flagged_ids={f["disease"] for f in flagged}
    cj_rows=[r for r in CJ["disease_rows"] if r["disease"] in flagged_ids]
    return {"patient_id":patient_id,"substrate":substrate,"age":age,"sex":sex,
            "patient_cells":patient_cells,"stress":bundle["systemic_stress"],
            "flagged":flagged,"flagged_rows":cj_rows,
            "verdict":(s5out or {}).get("overall_verdict",""),
            "sec_cols":CJ["sec_cols"],"sections_used":CJ["sections_used"]}

data=build("/home/claude/bundle_breast.pkl","/home/claude/breast/GSM1235926_beta.pkl",
           "whole_blood","GSM1235926",58,"F")
json.dump(data,open("/home/claude/patient_wall_data.json","w"))
print("patient scorable cells:", len(data["patient_cells"]))
import collections
tc=collections.Counter(c["tier"] for c in data["patient_cells"].values())
print("  tiers:", dict(tc))
print("  confident departures:", sum(1 for c in data['patient_cells'].values() if c['confident']))
print("stress:", data["stress"]["level"], "n=",data["stress"]["n_axis_cells"], "mag=",data["stress"]["mean_magnitude"])
print("flagged:", data["flagged"])
print("crown-jewel rows pulled for comparison:", len(data["flagged_rows"]),
      "->", [(r['disease'],r['phase']) for r in data['flagged_rows']])
