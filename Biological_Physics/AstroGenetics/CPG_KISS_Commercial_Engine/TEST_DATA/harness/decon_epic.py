import sys, pickle
sys.path.insert(0,".")
import importlib.util as _iu
_sp=_iu.spec_from_file_location("pdshim","/home/claude/pdshim.py"); _m=_iu.module_from_spec(_sp); _sp.loader.exec_module(_m)
cache=pickle.load(open("/home/claude/betas_cache.pkl","rb"))
import walther_clinical as wc
beta=cache["GSM8772491"]  # high-grade colon adenoma, EPIC
ctx=wc.PatientContext(age=60,sex="F",family_history=None,substrate="whole_blood")
b=wc.run_pipeline(beta,context=ctx,patient_id="COLON_ADENOMA")
s2=b["stage2"]; s4=b["stage4"]
fr=s2.get("class_fractions",{})
print("=== high-grade colon adenoma (EPIC) deconvolution — does secretory resolve? ===")
for cls in sorted(fr, key=lambda c:-fr[c]):
    a=s4["class_ascores"].get(cls,{}).get("A")
    print(f"  {cls:12} fraction={fr[cls]*100:5.1f}%  A={('%.3f'%a) if a is not None else 'not scored'}")
