import sys, pickle
sys.path.insert(0,".")
import importlib.util as _iu
_sp=_iu.spec_from_file_location("pdshim","/home/claude/pdshim.py"); _m=_iu.module_from_spec(_sp); _sp.loader.exec_module(_m)
import stage_1_idat_calibration as s1
cache=pickle.load(open("/home/claude/betas_cache.pkl","rb"))
for gsm in ["GSM5065990","GSM5065985"]:
    grn=f"/home/claude/geo_test/{gsm}_Grn.idat.gz"; red=grn.replace("_Grn","_Red")
    try:
        beta,meta=s1.calibrate_idat_to_beta(grn,red,verbose=False)
        cache[gsm]=beta; print(f"  {gsm}: calibrated {len(beta)} CpGs")
    except Exception as e:
        import traceback; traceback.print_exc()
pickle.dump(cache,open("/home/claude/betas_cache.pkl","wb"))
print("done")
