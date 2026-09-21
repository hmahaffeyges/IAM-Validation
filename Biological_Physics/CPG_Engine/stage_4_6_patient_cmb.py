#!/usr/bin/env python3
"""Stage 4.6 — the patient's sky (PROC-CMB-01, 2026-09-21). Replaces the retired patient_brightness_comparison.py.

What changed and why (PROC-CMB-01 C1): the retired module divided (beta_patient - mu_class) by the atlas POSTERIOR SD OF THE
CLASS MEAN and compared whole blood to a PURE-class mean; a healthy whole-blood array read ~40% of CpGs at |z|>2. Here:
    expectation  E_i = sum_c f_c * mu_{c,i}      f = the sample's own Stage 2 class fractions (PROC-SWITCH-02: beta is the mixture)
    residual     r_i = beta_i - E_i               beta on the mapped scale
    scale        s_i = per-CpG SD of r_i across the laboratory's healthy panel, floored and shrunk to a beta-binned pooled scale
    zero         m_i = per-CpG mean of r_i across the same panel (the atlas-as-fifth-laboratory constant, per CpG; PROC-CMB-01 C2)
    z_i          = (r_i - m_i) / s_i
    gating       a class panel is ASSESSABLE only if f_c >= presence_min (default 0.02); otherwise masked, NOT ASSESSABLE
    sky          z projected to HEALPix NSIDE 128 (RING) in genomic order via Runtime Matrices/Patient_CMB/iamatlas_cpg_to_healpix_nside128.npz
The scale, like the lab zero, belongs to the laboratory (one processing pipeline): build it from the same 40-array healthy panel.
"""
from __future__ import annotations
import os, json, math, numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); RT=os.path.join(HERE,"Runtime Matrices")
CLASSES=["stem_pluri","stem_adult","progenitor","terminal","cycling","immune","secretory","stromal"]
NSIDE=128; NPIX=12*NSIDE*NSIDE

def load_atlas_means(atlas_csv, cpgs=None):
    cols=["cpg_id"]+[f"{c}_mean" for c in CLASSES]
    at=pd.read_csv(atlas_csv,usecols=cols,index_col="cpg_id"); at.columns=[c[:-5] for c in at.columns]
    return at if cpgs is None else at.reindex(cpgs)

def load_mapping(path=None):
    z=np.load(path or os.path.join(RT,"Patient_CMB","iamatlas_cpg_to_healpix_nside128.npz"))
    return pd.Series(z["pixel"],index=z["cpg_id"].astype(str))

def expectation(atlas_means: pd.DataFrame, fractions: dict) -> pd.Series:
    """E_i = sum_c f_c mu_ci over classes with a mean at i; renormalised by the present mass so a missing class mean does not bias."""
    f=pd.Series({c:float(fractions.get(c,0.0)) for c in CLASSES}); M=atlas_means[CLASSES]
    w=M.notna().astype(float).mul(f,axis=1); num=(M.fillna(0.0)*w).sum(axis=1); den=w.sum(axis=1)
    E=num/den.replace(0,np.nan); E[den<0.5]=np.nan   # need at least half the composition mass covered at this CpG
    return E

def residual(beta_mapped: pd.Series, atlas_means, fractions):
    E=expectation(atlas_means.reindex(beta_mapped.index),fractions); return (beta_mapped-E), E

def build_residual_scale(panel: pd.DataFrame, panel_fractions: dict, atlas_means, floor=0.005, k_shrink=10, nbins=20):
    """panel: cpg x sample mapped betas (one laboratory, healthy). Returns the per-CpG panel ZERO m (mean residual) and SPREAD s.
    PROC-CMB-02: m is subtracted at patient time (z = (r - m)/s); the pooled binned scale is computed on CENTRED residuals."""
    R=pd.DataFrame({g:residual(panel[g].dropna(),atlas_means,panel_fractions[g])[0] for g in panel.columns})
    n=R.notna().sum(axis=1); m=R.mean(axis=1); sd=R.std(axis=1,ddof=1); Rc=R.sub(m,axis=0)
    E=pd.concat([expectation(atlas_means.reindex(panel.index),panel_fractions[g]) for g in panel.columns],axis=1).mean(axis=1)
    bins=pd.cut(E,np.linspace(0,1,nbins+1),include_lowest=True); pooled=sd.groupby(bins,observed=True).median()   # PROC-CMB-04: bin MEDIAN of per-CpG SD (RMS inflated the scale ~18%)
    s_bin=bins.map(pooled).astype(float)
    w=(n-1)/((n-1)+k_shrink); s=np.sqrt(w*sd.pow(2).fillna(0)+(1-w)*s_bin.pow(2)).clip(lower=floor)
    keep=n>=max(5,int(0.5*panel.shape[1])); m=m[keep]; s=s[keep]; n=n[keep]
    return {"m":m,"s":s,"n":n,"s_bin":s_bin,"pooled_by_bin":pooled.to_dict(),"n_panel":int(panel.shape[1]),"floor":floor,"k_shrink":k_shrink}

BLOOD_LINEAGE=("immune","progenitor","stem_adult")
def presence_floors(panel_fractions_all: dict, q=0.99, minimum=0.02):
    """Per-class presence floor (PROC-CMB-03). For the five classes healthy blood does NOT carry: max(minimum, q-quantile of the
    class fraction across all healthy panel arrays) - the deconvolver's noise floor, measured. For blood-lineage classes: minimum."""
    df=pd.DataFrame(list(panel_fractions_all.values())).reindex(columns=CLASSES).fillna(0.0)
    return {c:(minimum if c in BLOOD_LINEAGE else float(max(minimum,df[c].quantile(q)))) for c in CLASSES}

def save_scale(scale, path, lab, meta=None):
    idx=scale["s"].index.to_numpy().astype("U16")
    np.savez_compressed(path,cpg_id=idx,m=scale["m"].to_numpy(),s=scale["s"].to_numpy(),n=scale["n"].to_numpy(),lab=lab,n_panel=scale["n_panel"],meta=json.dumps(meta or {}))
def load_scale(path):
    z=np.load(path,allow_pickle=False); ix=z["cpg_id"].astype(str)
    return {"m":pd.Series(z["m"],index=ix),"s":pd.Series(z["s"],index=ix),"n":pd.Series(z["n"],index=ix),"lab":str(z["lab"]),"n_panel":int(z["n_panel"])}

def patient_sky(beta_mapped: pd.Series, fractions: dict, atlas_means, scale, identity_loci: dict|None=None, mapping: pd.Series|None=None, presence_min=0.02, presence_floors_by_class: dict|None=None):
    """Returns {'z': Series, 'E': Series, 'classes': {cls: {assessable, fraction, n, frac_abs_z_gt2, median_z, pixels?}}, 'all': {...}}"""
    r,E=residual(beta_mapped,atlas_means,fractions); s=scale["s"].reindex(r.index); m=scale["m"].reindex(r.index); z=((r-m)/s).dropna()
    mapping=mapping if mapping is not None else load_mapping()
    def summ(zz):
        return {"n":int(len(zz)),"frac_abs_z_gt2":float((zz.abs()>2).mean()) if len(zz) else None,"median_z":float(zz.median()) if len(zz) else None,"mean_abs_z":float(zz.abs().mean()) if len(zz) else None}
    def pixels(zz):
        px=mapping.reindex(zz.index).dropna().astype(int); sums=np.bincount(px.values,weights=zz.loc[px.index].values,minlength=NPIX); cnt=np.bincount(px.values,minlength=NPIX)
        out=np.full(NPIX,np.nan); m=cnt>0; out[m]=sums[m]/cnt[m]; return out
    out={"z":z,"E":E,"all":summ(z),"all_pixels":pixels(z),"classes":{}}
    for c in CLASSES:
        f=float(fractions.get(c,0.0)); thr=float((presence_floors_by_class or {}).get(c,presence_min)); ok=f>=thr
        d={"assessable":bool(ok),"fraction":f,"presence_floor":thr,"status":"ASSESSABLE" if ok else f"NOT ASSESSABLE · f={f:.3f} < {thr:.3f}"}
        if ok:
            loci=identity_loci.get(c) if identity_loci else None; zz=z.reindex([l for l in loci if l in z.index]).dropna() if loci else z
            d.update(summ(zz)); d["pixels"]=pixels(zz)
        out["classes"][c]=d
    return out

# ---- HEALPix RING pix2ang (no healpy; Gorski et al. 2005 RING scheme) ----
def ring_pix2ang(nside, ipix):
    ipix=np.asarray(ipix,dtype=np.int64); npix=12*nside*nside; ncap=2*nside*(nside-1); nl4=4*nside
    z=np.empty(len(ipix)); phi=np.empty(len(ipix))
    a=ipix<ncap                                   # north polar cap
    ip=ipix[a]+1; iring=(np.floor(np.sqrt(ip/2.0-np.sqrt(np.floor(ip/2.0))))+1).astype(np.int64); iphi=ip-2*iring*(iring-1)
    z[a]=1.0-iring*iring/(3.0*nside*nside); phi[a]=(iphi-0.5)*np.pi/(2.0*iring)
    b=(~a)&(ipix<npix-ncap)                       # equatorial belt
    ip=ipix[b]-ncap; iring=ip//nl4+nside; iphi=ip%nl4+1; fodd=0.5*(1+((iring+nside)&1))
    z[b]=(2*nside-iring)*2.0/(3.0*nside); phi[b]=(iphi-fodd)*np.pi/(2.0*nside)
    c=~(a|b)                                      # south polar cap
    ip=npix-ipix[c]; iring=(np.floor(np.sqrt(ip/2.0-np.sqrt(np.floor(ip/2.0))))+1).astype(np.int64); iphi=4*iring+1-(ip-2*iring*(iring-1))
    z[c]=-1.0+iring*iring/(3.0*nside*nside); phi[c]=(iphi-0.5)*np.pi/(2.0*iring)
    return np.arccos(np.clip(z,-1,1)), np.mod(phi,2*np.pi)

def render_plate(sky, out_png, title="", nside=NSIDE, vlim=3.0, dpi=110):
    """Nine panels: all loci + eight classes (masked panels black with NOT ASSESSABLE). Returns the stacked pixel array used (for C6)."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    theta,phi=ring_pix2ang(nside,np.arange(NPIX)); lat=np.pi/2-theta; lon=np.where(phi>np.pi,phi-2*np.pi,phi)
    fig,axes=plt.subplots(3,3,figsize=(15,10),subplot_kw={"projection":"mollweide"}); fig.patch.set_facecolor("black")
    panels=[("ALL LOCI",sky["all_pixels"],sky["all"],True)]+[(c.upper(),sky["classes"][c].get("pixels"),sky["classes"][c],sky["classes"][c]["assessable"]) for c in CLASSES]
    stack=[]
    for ax,(name,px,summ,ok) in zip(axes.ravel(),panels):
        ax.set_facecolor("black"); ax.grid(False); ax.set_xticklabels([]); ax.set_yticklabels([])
        if ok and px is not None:
            m=np.isfinite(px); ax.scatter(lon[m],lat[m],c=np.clip(px[m],-vlim,vlim),s=1.6,cmap="RdBu_r",vmin=-vlim,vmax=vlim,marker=".",linewidths=0,rasterized=True)
            ax.set_title(f"{name}  ·  f={summ.get('fraction',1.0):.2f}  ·  |z|>2: {100*summ['frac_abs_z_gt2']:.1f}%  ·  median z {summ['median_z']:+.2f}",color="white",fontsize=9); stack.append(np.nan_to_num(px,nan=-99.0))
        else:
            ax.set_title(f"{name}  ·  {summ.get('status','')}",color="#888",fontsize=9); stack.append(np.full(NPIX,-99.0))
    fig.suptitle(title,color="white",fontsize=13); fig.text(0.5,0.01,"z = (beta - sum_c f_c mu_c - m_lab) / s_lab   ·   red = above the composition expectation, blue = below   ·   black = not assessable (Stage 2 fraction below the measured healthy presence floor)",color="#aaa",ha="center",fontsize=9)
    fig.savefig(out_png,dpi=dpi,facecolor="black",bbox_inches="tight"); plt.close(fig); return np.vstack(stack)
