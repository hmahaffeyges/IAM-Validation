import sys, pickle, os, json, shutil
sys.path.insert(0,".")
import importlib.util as _iu
_sp=_iu.spec_from_file_location("pdshim","/home/claude/pdshim.py"); _m=_iu.module_from_spec(_sp); _sp.loader.exec_module(_m)
betas=pickle.load(open("/home/claude/betas_cache.pkl","rb"))
import stage_1_idat_calibration as s1
def _p(grn,red,**k):
    for g,b in betas.items():
        if g in str(grn): return b,{"array":"450k","cached":True}
    raise RuntimeError("no beta")
s1.calibrate_idat_to_beta=_p
import walther_clinical as wc
def setup(pid,visit,gsm,age,sex):
    d=f"/home/claude/demo/{pid}/{visit}"; os.makedirs(d,exist_ok=True)
    shutil.copy(f"/home/claude/geo_test/{gsm}_Grn.idat.gz",d); shutil.copy(f"/home/claude/geo_test/{gsm}_Red.idat.gz",d)
    json.dump({"patient_id":pid,"age":age,"sex":sex,"substrate":"whole_blood"},open(f"{d}/questionnaire.json","w"))
    return d
# 1. clean healthy single report
setup("SAMPLE_01","2026-06-22","GSM2333901",58,"M"); wc.run_from_folder("/home/claude/demo/SAMPLE_01/2026-06-22","/home/claude/demo/SAMPLE_01/2026-06-22")
# 2. flag + confirmation single report
setup("SAMPLE_02","2026-06-22","GSM2333905",67,"F"); wc.run_from_folder("/home/claude/demo/SAMPLE_02/2026-06-22","/home/claude/demo/SAMPLE_02/2026-06-22")
# 3. trajectory pair (two draws, same patient)
setup("SAMPLE_03","2026-06-22","GSM2333901",58,"M"); wc.run_from_folder("/home/claude/demo/SAMPLE_03/2026-06-22","/home/claude/demo/SAMPLE_03/2026-06-22")
setup("SAMPLE_03","2027-01-12","GSM2333905",58,"M"); wc.run_from_folder("/home/claude/demo/SAMPLE_03/2027-01-12","/home/claude/demo/SAMPLE_03/2027-01-12")
print("DEMOS DONE")
