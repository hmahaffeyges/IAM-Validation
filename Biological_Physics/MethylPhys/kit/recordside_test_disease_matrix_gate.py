#!/usr/bin/env python3
"""Kit test, row 8 (PROC-MATCH-01): (M1) with disease_origin_cells.json unreadable, Stage 8 returns NOT AVAILABLE and zero candidates;
with it present, status OK. (M4) substrate firewall: a whole_blood patient is scored against zero plasma_cfDNA/tissue signatures and a
plasma_cfDNA patient against zero whole-blood signatures. Uses one cached array."""
import os, sys, json, pickle, shutil, csv, warnings; warnings.filterwarnings("ignore")
HERE=os.path.dirname(os.path.abspath(__file__)); BP=os.path.abspath(os.path.join(HERE,"..","..")); ENG=os.path.join(BP,"MethylPhys/chain"); sys.path.insert(0,ENG)
import cpg_conductor as C
DATA=os.environ.get("CPG_KIT_DATA",os.path.join(HERE,"data")); c=pickle.load(open(os.path.join(DATA,"betas_cache.pkl"),"rb")); ATLAS=os.path.join(BP,"MethylPhys/atlas/IAMAtlasREBUILD.csv")
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
