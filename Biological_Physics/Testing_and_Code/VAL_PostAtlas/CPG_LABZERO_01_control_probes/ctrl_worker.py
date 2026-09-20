import sys, json, glob, os, numpy as np, pandas as pd, logging; logging.disable(logging.CRITICAL)
from methylprep.files import IdatDataset, Manifest
from methylprep.models import Channel, ArrayType
man=Manifest(ArrayType.ILLUMINA_450K); cdf=man.control_data_frame
ctypes=sorted(cdf.Control_Type.unique()); addr=cdf.index.values
rows={}
for g in json.load(open(sys.argv[1])):
    gsm=os.path.basename(g).split("_")[0]; r=g.replace("_Grn","_Red")
    try:
        G=IdatDataset(g,channel=Channel.GREEN).probe_means; R=IdatDataset(r,channel=Channel.RED).probe_means
        f={}
        for ch,pm in (("G",G),("R",R)):
            v=pm.reindex(addr); 
            for ct in ctypes:
                m=(cdf.Control_Type==ct).values; x=v.values[m]; x=x[np.isfinite(x)&(x>0)]
                f[f"{ct}_{ch}"]=float(np.log2(x).mean()) if len(x) else np.nan
            f[f"all_{ch}_med"]=float(np.log2(pm[pm>0]).median())
        f["GR_ratio"]=f["all_G_med"]-f["all_R_med"]; rows[gsm]=f
    except Exception as e: rows[gsm]={"error":str(e)[:80]}
pd.DataFrame(rows).T.to_pickle(sys.argv[2])
