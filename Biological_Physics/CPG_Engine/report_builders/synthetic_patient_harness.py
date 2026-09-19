import os, sys, importlib.util as iu, pickle, json, copy, random
import numpy as np
os.environ["CPG_ROOT"]="/home/claude/doctor_pkg/CPG_CMB_v4"; sys.path.insert(0,"/home/claude/doctor_pkg/CPG_CMB_v4")
_sp=iu.spec_from_file_location("pdshim","/home/claude/pdshim.py"); _m=iu.module_from_spec(_sp); _sp.loader.exec_module(_m)
import walther_clinical as wc
cfg=wc.DEFAULT_CONFIG
CJ=json.load(open("/home/claude/strawman_data_v2.json"))
mapping=json.load(open(cfg["matrix_mapping_json"]))["mapping"]   # atlas_cell -> matrix_col
# reverse: matrix_col -> [atlas_cells]
rev={}
for ac,col in mapping.items(): rev.setdefault(col,[]).append(ac)

tmpl=pickle.load(open("/home/claude/bundle_breast.pkl","rb"))
tmpl_s4=tmpl.get("stage4_output") or next((v for v in tmpl.values() if isinstance(v,dict) and "celltype_ascores" in v),None)
c2c=tmpl_s4["celltype_to_class"]

# a realistic resolved whole-blood panel (the cells a blood deconvolution actually returns)
BLOOD_PANEL=["Neutrophils_reinius","CD4_T-cells","CD8_T-cells","CD56_NK-cells","CD14_monocytes",
             "B-cells","Eosinophils_reinius","GMP","CMP","MPP"]

def make_synthetic(present_A, fractions=None, ci_half=0.025):
    """present_A: {atlas_cell: A}. Everything else is absent (fraction 0, below_floor).
    Returns a valid stage4 dict the chain accepts."""
    s4=copy.deepcopy(tmpl_s4)
    ct=s4["celltype_ascores"]
    fr=fractions or {}
    # default fractions: split evenly across present cells
    if not fr:
        f=round(1.0/max(len(present_A),1),4); fr={c:f for c in present_A}
    for cell,rec in ct.items():
        if not isinstance(rec,dict): continue
        if cell in present_A:
            A=float(present_A[cell])
            rec.update(A=A, celltype_fraction=float(fr.get(cell,0.05)), below_floor=False,
                       assessable=True, status="OK", A_ci_lo=A-ci_half, A_ci_hi=A+ci_half)
        else:
            rec.update(A=1.0, celltype_fraction=0.0, below_floor=True, assessable=True)
    # class_ascores = mean A over present cells in each class
    by_cls={}
    for cell,A in present_A.items():
        cls=c2c.get(cell); 
        if cls: by_cls.setdefault(cls,[]).append(A)
    for cls,rec in s4["class_ascores"].items():
        if cls in by_cls:
            mA=float(np.mean(by_cls[cls])); rec.update(A=mA, assessable=True, status="OK", below_floor=False)
        else:
            rec.update(A=None, assessable=False, status="NOT_ASSESSABLE_IN_SUBSTRATE")
    return s4

def signature_to_present(disease_row, dep_mag=0.10, healthy_base=True):
    """Build present_A from a crown-jewel disease row: signature cells depart in their d-direction;
    a healthy blood baseline fills the rest of the panel near A=1.0."""
    present={}
    if healthy_base:
        for ac in BLOOD_PANEL: present[ac]=1.0+random.uniform(-0.015,0.015)
    for col,v in disease_row["cells"].items():
        d=v.get("d"); arr=v.get("arr")
        sign=(1 if (d is not None and d>=0) or arr=="up" else -1)
        A=1.0+sign*dep_mag
        for ac in rev.get(col,[]):
            present[ac]=A
    return present

def run(present_A, sub="whole_blood", age=58, sex="F"):
    s4=make_synthetic(present_A)
    s8=wc.stage_8_dual_matching(s4,{},None,patient_meta={"substrate":sub,"age":age,"sex":sex},config=cfg)
    stress=wc.detect_systemic_stress_pattern(s8.patient_departure)
    concern=[m for m in (s8.route_B_concordance or [])
             if m["specificity"]=="SPECIFIC" and m["resemblance"] in ("STRONG_RESEMBLANCE","MODERATE_RESEMBLANCE") and m["cosine"]>=0.60]
    top=[(m["disease"],m["phase"],m["specificity"][:4],m["resemblance"].split("_")[0],round(m["cosine"],2)) for m in (s8.route_B_concordance or [])[:4]]
    return dict(n_present=len(s8.patient_departure), stress=stress["level"],
                concern=[(m["disease"],m["phase"]) for m in concern], top=top)

# ---- TEST BATCH ----
random.seed(7)
print("="*70)
print("TEST 1 — HEALTHY (all present blood cells at A~1.0)")
h={ac:1.0+random.uniform(-0.02,0.02) for ac in BLOOD_PANEL}
print("  ",run(h))

print("\nTEST 2 — NOISE (random departures, no coherent disease pattern)")
noise={ac:1.0+random.uniform(-0.18,0.18) for ac in BLOOD_PANEL}
print("  ",run(noise))

# pick a few diseases with clear per-cell signatures from the crown jewel
rows={ (r["disease"],r["phase"]):r for r in CJ["disease_rows"] }
def find(dis, phase=None):
    for (d,p),r in rows.items():
        if d==dis and (phase is None or p==phase): return r
    return None
for dis,phase in [("leukemia_aml","active"),("alzheimers_disease",None),("recent_infection_bacterial","context")]:
    r=find(dis,phase)
    if r:
        print(f"\nTEST — inject {dis} ({r['phase']}) signature  [expect {dis} to surface]")
        res=run(signature_to_present(r))
        print("  ", res)
        print("   -> target in concern:", any(d==dis for d,_ in res["concern"]))

print("\n"+"="*70)
print("SPECIFICITY SWEEP — inject each disease, record what CONCERN-flags fire")
print("="*70)
test_diseases=[("leukemia_aml","active"),("leukemia_cml","active"),("recent_infection_bacterial","context"),
               ("inflammaging","chronic"),("lung_cancer","mid_late_pre_dx"),("crohns_disease","active"),
               ("multiple_myeloma","active"),("mds","active")]
seen=set()
for dis,phase in test_diseases:
    r=find(dis,phase)
    if not r: 
        print(f"  {dis}: (not in crown jewel)"); continue
    res=run(signature_to_present(r))
    flags=[d for d,_ in res["concern"]]
    hit = "SELF" if dis in flags else "MISS"
    extras=[d for d in dict.fromkeys(flags) if d!=dis]
    print(f"  inject {dis:28s} stress={res['stress']:8s} self={hit:4s} | also flags: {extras if extras else '(none)'}")
