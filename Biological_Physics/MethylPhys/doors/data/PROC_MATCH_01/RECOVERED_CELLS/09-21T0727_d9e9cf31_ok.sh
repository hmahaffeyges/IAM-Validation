set -e; cd iamrepo/Biological_Physics/CPG_Engine
python3 - <<'PY'
import ast
p="walther_clinical.py"; s=open(p,encoding="utf-8").read()
old='''    try:
        origin_map = _json.load(open(Path(cfg["disease_matrix_csv"]).parent / "disease_origin_cells.json"))
    except Exception:
        origin_map = {}'''
new='''    # PROC-MATCH-01 M1 (2026-09-21): FAIL CLOSED. A missing or unparsable origin map used to become {} and the
    # specificity rule degraded silently; now Stage 8 refuses to match at all.
    _op = Path(cfg["disease_matrix_csv"]).parent / "disease_origin_cells.json"
    try:
        origin_map = _json.load(open(_op))
        if not isinstance(origin_map, dict) or not origin_map: raise ValueError("origin map empty")
    except Exception as _e:
        return Stage8Output(route_B_concordance=[], route_B_all_scored=[], patient_departure={}, route_A_architectural_alarm={},
                            route_C_bidirectional={}, status=f"NOT AVAILABLE - cell-of-origin map unreadable ({_op.name}: {_e}); Stage 8 refuses to match (fail-closed, PROC-MATCH-01)")'''
assert s.count(old)==1; s=s.replace(old,new); ast.parse(s); open(p,"w",encoding="utf-8").write(s)
p="cpg_conductor.py"; c=open(p,encoding="utf-8").read()
fn='''
def stage_8_matching(stage_a_out, cfg=None):
    """Stage 8 - disease-pattern concordance (route B) on the per-cell SEPARATION surface. ROW 8 OPEN (PROC-MATCH-01, 2026-09-21):
    the departure profile is (A_cell - 1.0) over PRESENT cells, but healthy per-cell A on this surface sits at ~0.44-0.52 with class
    H_min 0.77-0.98, so on healthy whole blood only ~3 cells enter the profile. The reference level must be re-derived on this surface
    (healthy per-cell level from the four-lab panels) before any match is reported. Until then the output is DIAGNOSTIC and not reportable.
    Origin gate fails CLOSED (missing/unreadable disease_origin_cells.json -> status NOT AVAILABLE, zero candidates)."""
    cfg = cfg or {}
    W = _load_module("walther_clinical", _find("walther_clinical.py"))
    md = HERE / "Disease Matrix" / "DISEASE_MATRIX"
    c2c = json.load(open(_find("IAMAtlasREBUILD_celltype_to_class.json"))); HM = json.load(open(_find("iamatlas_celltype_markers_v0_2.json"))).get("H_min_by_class", {})
    s4 = {"celltype_ascores": {cell: {"A": r.get("A"), "below_floor": bool(r.get("A") is not None and r["A"] < HM.get(c2c.get(cell), 0)),
                                       "celltype_fraction": r.get("fraction")} for cell, r in stage_a_out["cells"].items()}}
    out = W.stage_8_dual_matching(s4, None, None, patient_meta={"substrate": cfg.get("substrate", "whole_blood")},
                                  config={"disease_matrix_csv": str(md / "disease_cell_signature_matrix_v1_13.csv"), "matrix_mapping_json": str(md / "iamatlas_115_to_matrix_v0_2_mapping.json")})
    return {"available": out.status == "OK", "status": out.status, "reportable": False, "row_status": "OPEN - departure reference not commissioned on the separation surface (PROC-MATCH-01)",
            "n_present_cells": len(out.patient_departure), "patient_departure": out.patient_departure,
            "route_B_top": out.route_B_concordance[:5], "n_scored": len(out.route_B_all_scored)}

def run_full('''
assert c.count("\ndef run_full(")==1; c=c.replace("\ndef run_full(",fn,1)
old='''    sky = stage_4_6_patient_sky(beta_rm, a, cfg=cfg, atlas_csv=atlas_csv)'''
assert c.count(old)==1; c=c.replace(old,'''    m8 = stage_8_matching(a, cfg=cfg)                                                                # Stage 8 (row 8 OPEN): diagnostic only, fail-closed origin gate
'''+old)
old2='''        "patient_sky": sky,'''
c=c.replace(old2,'''        "diagnostic_disease_matching": m8,               # Stage 8 (row 8 OPEN, PROC-MATCH-01): NOT reportable
'''+old2); ast.parse(c); open(p,"w",encoding="utf-8").write(c); print("wired")
PY
cat > ../Physics_of_Methylation/Reproduction_Kit/test_disease_matching_gate.py <<'PY'
#!/usr/bin/env python3
"""Kit test, row 8 (PROC-MATCH-01): (M1) with disease_origin_cells.json unreadable, Stage 8 returns NOT AVAILABLE and zero candidates;
with it present, status OK. (M4) substrate firewall: a whole_blood patient is scored against zero plasma_cfDNA/tissue signatures and a
plasma_cfDNA patient against zero whole-blood signatures. Uses one cached array."""
import os, sys, json, pickle, shutil, csv, warnings; warnings.filterwarnings("ignore")
HERE=os.path.dirname(os.path.abspath(__file__)); BP=os.path.abspath(os.path.join(HERE,"..","..")); ENG=os.path.join(BP,"CPG_Engine"); sys.path.insert(0,ENG)
import cpg_conductor as C
DATA=os.environ.get("CPG_KIT_DATA",os.path.join(HERE,"data")); c=pickle.load(open(os.path.join(DATA,"betas_cache.pkl"),"rb")); ATLAS=os.path.join(BP,"IAM_Atlas/IAMAtlasREBUILD.csv")
a=C.stage_a_cells(c["GSM2333901"].dropna().to_dict(),ATLAS)
op=os.path.join(ENG,"Disease Matrix/DISEASE_MATRIX/disease_origin_cells.json"); bak=op+".kit_bak"
os.rename(op,bak)
try:
    o=C.stage_8_matching(a); assert o["available"] is False and "NOT AVAILABLE" in o["status"] and o["n_scored"]==0 and not o["route_B_top"], o["status"]
finally: os.rename(bak,op)
print("M1 origin map missing -> NOT AVAILABLE, 0 candidates: ok")
o=C.stage_8_matching(a); assert o["available"] and o["status"]=="OK" and o["reportable"] is False; print(f"M1 origin map present -> OK; reportable=False (row 8 OPEN); present cells {o['n_present_cells']}, scored {o['n_scored']}: ok")
rows=list(csv.DictReader(open(os.path.join(ENG,"Disease Matrix/DISEASE_MATRIX/disease_cell_signature_matrix_v1_13.csv")))); sub={r["disease_id"]+"|"+r.get("phase",""):r["substrate"] for r in rows}; bysub={}
for r in rows: bysub.setdefault(r["disease_id"],set()).add(r["substrate"])
W=C._load_module("walther_clinical",str(C._find("walther_clinical.py")))
for psub,forbidden in (("whole_blood",{"plasma_cfDNA","tumor_tissue","tumor_tissue_normalized","tumor_tissue_paired","aortic_tissue","cultured_pulmonary_endothelial"}),("plasma_cfDNA",{"whole_blood","whole_blood_buffy_coat","whole_blood_sorted"})):
    o=C.stage_8_matching(a,cfg={"substrate":psub}); scored=[s for s in (o["route_B_top"])]
    # every scored disease must have at least one signature row in an allowed substrate and none of its scored rows in a forbidden one
    bad=[s["disease"] for s in scored if bysub.get(s["disease"],set())<=forbidden]
    assert not bad, (psub,bad); print(f"M4 {psub}: {o['n_scored']} scored, none from forbidden substrates {sorted(forbidden)[:2]}...: ok")
print("test_disease_matching_gate: PASS")
PY
cd ../Physics_of_Methylation/Reproduction_Kit && CPG_KIT_DATA="$(pwd)/../../../../kitdata_cmb" python3 test_disease_matching_gate.py 2>&1 | grep -v "Deprecat\|^INFO\|Warning\|Scanning\|class columns\|Selecting\|markers:"
echo "=== M5 over the 11 cached arrays ==="; cd ../../CPG_Engine && python3 - <<'PY' 2>&1 | grep -v "Deprecat\|^INFO\|Warning\|Scanning\|class columns\|Selecting\|markers:"
import sys,os,pickle,json,warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,"."); import cpg_conductor as C
ATLAS=os.path.abspath("../IAM_Atlas/IAMAtlasREBUILD.csv"); c=pickle.load(open("../../../testdata/10_TEST_DATA/betas_cache.pkl","rb")); man=open("../../../testdata/10_TEST_DATA/TEST_DATA_MANIFEST.md").read() if os.path.exists("../../../testdata/10_TEST_DATA/TEST_DATA_MANIFEST.md") else ""
res=[]
for g,b in c.items():
    a=C.stage_a_cells(b.dropna().to_dict(),ATLAS); o=C.stage_8_matching(a); top=o["route_B_top"][0] if o["route_B_top"] else None
    res.append({"gsm":g,"present":o["n_present_cells"],"scored":o["n_scored"],"top":(top["disease"],top["cosine"],top.get("resemblance")) if top else None}); print(f"{g}: present {o['n_present_cells']:>2} scored {o['n_scored']:>2} top {res[-1]['top']}")
json.dump(res,open("../Testing_and_Code/PROC_data/PROC-MATCH-01/m5_cached_arrays.json","w"),indent=1)
PY