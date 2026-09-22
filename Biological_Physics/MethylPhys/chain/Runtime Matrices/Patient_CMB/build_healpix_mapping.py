#!/usr/bin/env python3
"""Build iamatlas_cpg_to_healpix_nside128.npz — the CpG -> HEALPix pixel mapping for the patient's sky (Stage 4.6).
Deterministic: atlas CpGs are ordered by (chromosome, position) from the combined manifest and assigned sequentially to
the 12*NSIDE^2 = 196,608 pixels in RING order, so genomic order runs across the sky exactly as Plate 1 (April 2026) did.
Multiple CpGs share a pixel (~2.4 per pixel); CpGs without a manifest position are excluded and counted.
Usage: python3 build_healpix_mapping.py [atlas_csv] [manifest_csv] [out_npz]"""
import sys, os, hashlib, numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); BP=os.path.abspath(os.path.join(HERE,"..","..",".."))
atlas=sys.argv[1] if len(sys.argv)>1 else os.path.join(BP,"IAM_Atlas/IAMAtlasREBUILD.csv")
man=sys.argv[2] if len(sys.argv)>2 else os.path.join(BP,"IAM_Atlas/external_manifests/EPIC_plus_HM450_combined_manifest_normalized.csv")
out=sys.argv[3] if len(sys.argv)>3 else os.path.join(HERE,"iamatlas_cpg_to_healpix_nside128.npz")
NSIDE=128; NPIX=12*NSIDE*NSIDE
cpgs=pd.read_csv(atlas,usecols=["cpg_id"])["cpg_id"].astype(str)
m=pd.read_csv(man,low_memory=False); cols={c.lower():c for c in m.columns}
idc=cols.get("ilmnid") or cols.get("name") or cols.get("cpg_id") or cols.get("probe_id"); chc=cols.get("chr") or cols.get("chromosome"); poc=cols.get("mapinfo") or cols.get("position") or cols.get("pos")
m=m[[idc,chc,poc]].rename(columns={idc:"cpg",chc:"chr",poc:"pos"}).dropna(); m["cpg"]=m["cpg"].astype(str)
def chrkey(c):
    c=str(c).replace("chr",""); return {"X":23,"Y":24,"M":25,"MT":25}.get(c, int(c) if c.isdigit() else 99)
m["ck"]=m["chr"].map(chrkey); m=m[m.ck<99].drop_duplicates("cpg")
j=pd.DataFrame({"cpg":cpgs}).merge(m,on="cpg",how="left")
has=j["pos"].notna(); n_missing=int((~has).sum())
order=j[has].sort_values(["ck","pos","cpg"],kind="mergesort")
pix=np.full(len(j),-1,dtype=np.int32)
pix[order.index.values]=(np.arange(len(order))*NPIX//len(order)).astype(np.int32)   # sequential, genomic order, RING index
np.savez_compressed(out,cpg_id=j["cpg"].to_numpy().astype("U16"),pixel=pix,nside=NSIDE,n_missing=n_missing,chr_key=j["ck"].fillna(-1).to_numpy().astype(np.int8))
h=hashlib.sha256(open(out,"rb").read()).hexdigest()
print(f"mapped {int(has.sum()):,} of {len(j):,} atlas CpGs to {NPIX:,} pixels ({has.sum()/NPIX:.2f} CpG/pixel); no manifest position: {n_missing:,}; sha256 {h[:16]}")
# sha of CONTENT (npz compression is deterministic with fixed inputs, but record the array digest too)
print("content sha256", hashlib.sha256(pix.tobytes()+j["cpg"].to_numpy().astype("U16").tobytes()).hexdigest()[:16])
