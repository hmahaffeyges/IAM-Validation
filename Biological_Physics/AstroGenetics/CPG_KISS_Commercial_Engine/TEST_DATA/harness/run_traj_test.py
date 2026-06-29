import sys, pickle
sys.path.insert(0,".")
import importlib.util as _iu
_sp=_iu.spec_from_file_location("pdshim","/home/claude/pdshim.py"); _m=_iu.module_from_spec(_sp); _sp.loader.exec_module(_m)
betas=pickle.load(open("/home/claude/betas_cache.pkl","rb"))
import stage_1_idat_calibration as s1
def _patched(grn, red, **kw):
    for gsm,b in betas.items():
        if gsm in str(grn): print(f"      (cached calibration for {gsm})"); return b, {"array":"450k","cached":True}
    raise RuntimeError("no cached beta")
s1.calibrate_idat_to_beta=_patched
import walther_clinical as wc
for visit in ["2026-06-22","2027-01-10"]:
    f=f"/home/claude/work/patients/ANON_0001/{visit}"
    print(f"=== visit {visit} ==="); wc.run_from_folder(f, f)
