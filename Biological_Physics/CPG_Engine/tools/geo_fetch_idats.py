#!/usr/bin/env python3
"""geo_fetch_idats.py — fetch ONLY the IDAT pairs you need from GEO, in parallel (~10x a single-stream RAW.tar).
Usage: python3 geo_fetch_idats.py GSExxxxx OUTDIR [--field 'diagnosis' --value 'control'] [--workers 8] [--limit N]
Reads the series-matrix header (range request, no betas), selects samples by a characteristics field, then downloads
each GSM's *_Grn/_Red.idat(.gz) from geo/samples/GSMnnn/GSM/suppl/ with a thread pool. Idempotent (skips complete files).
Writes OUTDIR/selected.json (gsm -> characteristics) and OUTDIR/idats/. Added 2026-09-20 (band_v2 test): the RAW.tar
of GSE125105 (5.7 GB, 699 samples) ran at 0.7 MB/s; the 210 controls this way took 4 minutes at 7 MB/s."""
import sys, os, re, json, time, argparse, zlib, urllib.request
from concurrent.futures import ThreadPoolExecutor
def header(gse):
    stub=re.sub(r"\d{3}$","nnn",gse); url=f"https://ftp.ncbi.nlm.nih.gov/geo/series/{stub}/{gse}/matrix/{gse}_series_matrix.txt.gz"
    raw=urllib.request.urlopen(urllib.request.Request(url,headers={"Range":"bytes=0-1500000"}),timeout=120).read()
    return zlib.decompressobj(16+zlib.MAX_WBITS).decompress(raw).decode("utf-8","replace")
def select(gse, field=None, value=None):
    t=header(gse); gsms=None; ch={}
    for l in t.split("\n"):
        if l.startswith("!Sample_geo_accession"): gsms=[v.strip('"') for v in l.split("\t")[1:]]
        if l.startswith("!Sample_characteristics_ch1"):
            v=[x.strip('"') for x in l.split("\t")[1:]]; k=v[0].split(":")[0].strip().lower()
            ch[k]=[x.split(":",1)[-1].strip() for x in v]
    rows={g:{k:ch[k][i] for k in ch if i<len(ch[k])} for i,g in enumerate(gsms)}
    if field: rows={g:r for g,r in rows.items() if str(r.get(field.lower(),"")).lower()==value.lower()}
    return rows
def sample_files(g):
    stub=re.sub(r"\d{3}$","nnn",g); base=f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{stub}/{g}/suppl/"
    for _ in range(3):
        try:
            html=urllib.request.urlopen(base,timeout=60).read().decode(); return [(base+f,f) for f in sorted(set(re.findall(r'href="(GSM[^"]+\.idat(?:\.gz)?)"',html)))]
        except Exception: time.sleep(2)
    return []
def fetch(u,d):
    for _ in range(3):
        try:
            if os.path.exists(d) and os.path.getsize(d)>1e5: return os.path.getsize(d)
            urllib.request.urlretrieve(u,d+".part"); os.replace(d+".part",d); return os.path.getsize(d)
        except Exception: time.sleep(2)
    return 0
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("gse"); ap.add_argument("out"); ap.add_argument("--field"); ap.add_argument("--value"); ap.add_argument("--workers",type=int,default=8); ap.add_argument("--limit",type=int)
    a=ap.parse_args(); os.makedirs(os.path.join(a.out,"idats"),exist_ok=True)
    rows=select(a.gse,a.field,a.value); gs=list(rows)[:a.limit] if a.limit else list(rows)
    json.dump({g:rows[g] for g in gs},open(os.path.join(a.out,"selected.json"),"w"),indent=1); print(f"{a.gse}: selected {len(gs)} samples", flush=True)
    t0=time.time()
    with ThreadPoolExecutor(a.workers) as ex: todo=[(u,os.path.join(a.out,"idats",f)) for lst in ex.map(sample_files,gs) for (u,f) in lst]
    with ThreadPoolExecutor(a.workers) as ex: sizes=list(ex.map(lambda p: fetch(*p), todo))
    mb=sum(sizes)/1048576; dt=time.time()-t0; miss=[g for g in gs if not any(g in d for _,d in todo)]
    print(f"{len(todo)} files, {mb:.0f} MB, {dt:.0f} s, {mb/max(dt,1):.1f} MB/s; samples with no IDATs listed: {len(miss)}", flush=True)
if __name__=="__main__": main()
