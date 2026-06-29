import sys, pickle
sys.path.insert(0,".")
import importlib.util as _iu
_sp=_iu.spec_from_file_location("pdshim","/home/claude/pdshim.py"); _m=_iu.module_from_spec(_sp); _sp.loader.exec_module(_m)
import stage_1_idat_calibration as s1
cache=pickle.load(open("/home/claude/betas_cache.pkl","rb"))
import gzip
for gsm in ["GSM8772491","GSM8772492"]:
    grn=f"/home/claude/geo_test/{gsm}_Grn.idat.gz"; red=grn.replace("_Grn","_Red")
    with gzip.open(grn,'rb') as f: magic=f.read(4)
    print(f"{gsm}: magic={magic}")
    try:
        beta,meta=s1.calibrate_idat_to_beta(grn,red,verbose=False)
        cache[gsm]=beta
        print(f"  CALIBRATED {len(beta)} CpGs (array={meta.get('array','?')})")
    except Exception as e:
        import traceback; traceback.print_exc()
pickle.dump(cache,open("/home/claude/betas_cache.pkl","wb"))
# now run deconvolution on one to see if secretory resolves
import walther_clinical as wc
beta=cache["GSM8772491"]
ctx=wc.PatientContext(age=60,sex="F",family_history=None,substrate="whole_blood")
b=wc.run_pipeline(beta,context=ctx,patient_id="CRC_TISSUE")
s2=b["stage2"]; s4=b["stage4"]
fr=s2.get("class_fractions",{})
print("\n=== colon adenoma tissue deconvolution (does secretory resolve?) ===")
for cls in sorted(fr, key=lambda c:-fr[c]):
    a=s4["class_ascores"].get(cls,{}).get("A")
    print(f"  {cls:12} fraction={fr[cls]*100:5.1f}%  A={('%.3f'%a) if a is not None else 'not scored'}")
