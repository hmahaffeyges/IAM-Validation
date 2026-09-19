import sys, pickle
sys.path.insert(0,".")
import importlib.util as _iu
_sp=_iu.spec_from_file_location("pdshim","/home/claude/pdshim.py"); _m=_iu.module_from_spec(_sp); _sp.loader.exec_module(_m)
cache=pickle.load(open("/home/claude/betas_cache.pkl","rb"))
import walther_clinical as wc
for gsm,lab in [("GSM5065990","CRC carcinoma stage 1"),("GSM5065985","CRC carcinoma stage 4")]:
    b=wc.run_pipeline(cache[gsm],context=wc.PatientContext(age=55,sex="M",family_history=None,substrate="whole_blood"),patient_id=gsm)
    s2=b["stage2"]; s4=b["stage4"]; fr=s2.get("class_fractions",{})
    print(f"\n=== {lab} ({gsm}, EPIC) ===")
    for cls in sorted(fr, key=lambda c:-fr[c]):
        a=s4["class_ascores"].get(cls,{}).get("A")
        if fr[cls]>0.001: print(f"  {cls:12} fraction={fr[cls]*100:5.1f}%  A={('%.3f'%a) if a is not None else 'not scored'}")
