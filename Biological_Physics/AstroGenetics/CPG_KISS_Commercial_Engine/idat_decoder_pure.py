"""Pure-Python IDAT -> beta. No methylprep/minfi at runtime. Needs only:
   - idat_parse.read_idat (stdlib+numpy)
   - a static array manifest CSV (Name, AddressA_ID, AddressB_ID, Infinium_Design_Type, Color_Channel)
Infinium convention: II  -> M=Grn[A], U=Red[A]
                     I/Grn-> M=Grn[B], U=Grn[A]
                     I/Red-> M=Red[B], U=Red[A]   ; beta = M/(M+U+offset)."""
import numpy as np, pandas as pd
from idat_parse import read_idat

def decode_to_beta(grn_path, red_path, manifest_path, offset=100):
    grn = read_idat(grn_path); red = read_idat(red_path)
    gmap = pd.Series(grn["mean"], index=grn["ids"].astype(np.int64))
    rmap = pd.Series(red["mean"], index=red["ids"].astype(np.int64))
    gmap = gmap[~gmap.index.duplicated()]; rmap = rmap[~rmap.index.duplicated()]
    man = pd.read_csv(manifest_path,
                      usecols=["Name","AddressA_ID","AddressB_ID","Infinium_Design_Type","Color_Channel"],
                      dtype=str)
    man = man[man["Infinium_Design_Type"].isin(["I","II"])].copy()
    man = man[man["AddressA_ID"].notna() & (man["AddressA_ID"]!="")]
    A  = man["AddressA_ID"].astype(np.int64).values
    Bv = man["AddressB_ID"].fillna("0").replace("", "0").astype(np.int64).values
    typ = man["Infinium_Design_Type"].values
    col = man["Color_Channel"].fillna("").values
    gA = gmap.reindex(A).to_numpy(float); rA = rmap.reindex(A).to_numpy(float)
    gB = gmap.reindex(Bv).to_numpy(float); rB = rmap.reindex(Bv).to_numpy(float)
    M = np.full(len(man), np.nan); U = np.full(len(man), np.nan)
    is_II = typ == "II"; is_Ig = (typ == "I") & (col == "Grn"); is_Ir = (typ == "I") & (col == "Red")
    M[is_II] = gA[is_II]; U[is_II] = rA[is_II]
    M[is_Ig] = gB[is_Ig]; U[is_Ig] = gA[is_Ig]
    M[is_Ir] = rB[is_Ir]; U[is_Ir] = rA[is_Ir]
    beta = M / (M + U + offset)
    s = pd.Series(beta, index=man["Name"].values, name="beta")
    return s[~s.index.duplicated()], {"barcode": grn["barcode"], "chip": grn["chiptype"],
                                       "n_addr": grn["n"], "n_probes": int(s.notna().sum())}

if __name__ == "__main__":
    MAN = "/root/.methylprep_manifest_files/HumanMethylationEPIC_manifest_v2.csv.gz"
    beta, meta = decode_to_beta("/home/claude/idat_run/GSM3228562_R01C01_Grn.idat",
                                "/home/claude/idat_run/GSM3228562_R01C01_Red.idat", MAN)
    print("decoded:", meta)
    # validate against the trusted methylprep RAW beta
    ref = pd.read_csv("/home/claude/idat_run/GSM3228562_beta.csv")
    ref = pd.to_numeric(ref.set_index(ref.columns[0])[ref.columns[1]], errors="coerce").dropna()
    common = beta.dropna().index.intersection(ref.index)
    a = beta.reindex(common); b = ref.reindex(common)
    d = (a - b).abs()
    print(f"\nVALIDATION vs methylprep raw beta  ({len(common):,} common probes)")
    print(f"  mean |diff| = {d.mean():.5f}   median = {d.median():.5f}   max = {d.max():.5f}")
    print(f"  within 0.01: {100*(d<0.01).mean():.2f}%   within 0.001: {100*(d<0.001).mean():.2f}%")
    print(f"  correlation r = {np.corrcoef(a.values, b.values)[0,1]:.6f}")
