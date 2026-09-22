#!/usr/bin/env python3
"""run_sample.py - one command, IDAT pair (or a beta table) to a MethylPhys report.

This exists because the Run tab used to advertise a command that did not: `build_methylphys.py` only ever
took a pre-computed bundle. Written 2026-09-22.

    # an Illumina IDAT pair (needs methylprep for Stage 1)
    python3 run_sample.py --grn SAMPLE_Grn.idat.gz --red SAMPLE_Red.idat.gz --age 58 --lab MYLAB --out report.html

    # or a beta table you have already calibrated: a two-column CSV, cpg_id,beta (no pipeline map is applied
    # unless you name one with --pipeline, and without a map the class gauge is NOT REPORTABLE by design)
    python3 run_sample.py --betas mysample.csv --age 58 --lab MYLAB --out report.html

A laboratory the chain has not commissioned has no zero and no sky scale, so the report prints
NOT REPORTABLE with the reason rather than a number. To commission one: 40 healthy arrays of that
laboratory through Stage 1, then CPG_Engine/lab_zero.py - the procedure is in the RUNBOOK.
"""
import os, sys, argparse, pickle, json
HERE=os.path.dirname(os.path.abspath(__file__)); ENG=os.path.dirname(HERE); BIO=os.path.dirname(ENG)
for d in (ENG, HERE, os.path.join(ENG,"Walther_iam_deconvolver")): sys.path.insert(0,d)

def _atlas():
    csv=os.path.join(BIO,"IAM_Atlas","IAMAtlasREBUILD.csv"); xz=csv+".xz"
    if not os.path.exists(csv):
        if not os.path.exists(xz): sys.exit(f"atlas not found: {xz}")
        import lzma, shutil
        print(f"decompressing the atlas once -> {csv} (605 MB)", flush=True)
        with lzma.open(xz,"rb") as f, open(csv,"wb") as g: shutil.copyfileobj(f,g,1<<24)
    return csv

def main():
    ap=argparse.ArgumentParser(description="IDAT pair or beta table -> MethylPhys report")
    ap.add_argument("--grn"); ap.add_argument("--red"); ap.add_argument("--betas")
    ap.add_argument("--age", type=int, help="declared age in years; without it the age term cannot be removed")
    ap.add_argument("--lab", help="laboratory / pipeline identity, e.g. GSE87571 for a commissioned one")
    ap.add_argument("--pipeline", default="stage1_noob_450K", help="pipeline map name from beta_scale_maps_v1.json")
    ap.add_argument("--specimen", default="whole blood")
    ap.add_argument("--lab-zero", type=float, default=None, help="this laboratory's measured zero; omit to read UNSET")
    ap.add_argument("--out", default="methylphys_report.html"); ap.add_argument("--id", default=None)
    a=ap.parse_args()
    if not (a.betas or (a.grn and a.red)): ap.error("give either --betas or both --grn and --red")

    if a.betas:
        import pandas as pd
        d=pd.read_csv(a.betas, index_col=0).iloc[:,0].dropna(); beta=d.to_dict()
        sid=a.id or os.path.basename(a.betas).split(".")[0]
    else:
        try:
            from stage_1_idat_calibration import calibrate_idat_to_beta
        except ImportError:
            for c in (os.path.join(BIO,"CPG_Engine"), os.path.join(BIO,"Physics_of_Methylation","Reproduction_Kit")):
                sys.path.insert(0,c)
            from stage_1_idat_calibration import calibrate_idat_to_beta
        print("Stage 1: calibrating the IDAT pair (methylprep noob) - about 25 s", flush=True)
        b,meta=calibrate_idat_to_beta(a.grn,a.red)
        b=b.iloc[:,0] if hasattr(b,"columns") else b
        beta=b.dropna().to_dict(); sid=a.id or os.path.basename(a.grn).split("_")[0]

    import cpg_conductor as C, build_methylphys as B
    cfg={"age":a.age,"pipeline":a.pipeline,"lab_zero":a.lab_zero,"substrate":a.specimen}
    if a.lab: cfg["lab"]=a.lab
    print(f"running the chain on {len(beta):,} CpGs", flush=True)
    o=C.run_full(beta, _atlas(), cfg=cfg)
    r=B.build(o, a.out, sid)
    imm=o["classes"].get("immune",{})
    print(f"\n{sid}: immune A'' {imm.get('A_abs')}  placement {imm.get('placement')}  tier {imm.get('tier')}")
    print(f"refusals: {len(r['refusals'])}" + (f" -> {r['refusals'][:3]}" if r["refusals"] else ""))
    print(f"report: {r['out']}  ({r['bytes']//1000} KB)")

if __name__=="__main__": main()
