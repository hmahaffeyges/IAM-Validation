"""DEV-ATLAS-V21-01 culture check inputs: GSE197512 'Baseline profiling' arrays (182, EPIC v1; seven Coriell lines followed to replicative
senescence, population doublings per array), listed in culture_gse197512_baseline.csv from GEO metadata (2026-10-12). Reference data for the
division-drift CpG set, not a test. Fetches each GSM's two IDATs from GEO (8 at a time; retries). Usage: python3 fetch_culture_check.py OUT_DIR"""
import os, sys, gzip, shutil, time, re, urllib.request, pandas as pd
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
L = pd.read_csv(os.path.join(HERE, "culture_gse197512_baseline.csv"))
def urls(g):
    t = urllib.request.urlopen(f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={g}&targ=self&form=text&view=brief", timeout=120).read().decode()
    return [l.split("=", 1)[1].strip().replace("ftp://", "https://") for l in t.split("\n") if l.startswith("!Sample_supplementary_file") and ".idat" in l.lower()]
def get(g):
    for i in range(5):
        try:
            n = 0
            for u in urls(g):
                p = os.path.join(OUT, os.path.basename(u)[:-3] if u.endswith(".gz") else os.path.basename(u))
                if os.path.exists(p) and os.path.getsize(p) > 0: n += 1; continue
                urllib.request.urlretrieve(u, p + ".part.gz")
                with gzip.open(p + ".part.gz") as a, open(p + ".part", "wb") as b: shutil.copyfileobj(a, b)
                os.replace(p + ".part", p); os.remove(p + ".part.gz"); n += 1
            return n
        except Exception: time.sleep(10 * (i + 1))
    return -1
with ThreadPoolExecutor(8) as ex: res = list(ex.map(get, L.gsm))
print("samples", len(res), "with 2 IDATs", sum(r == 2 for r in res), "failed", sum(r < 0 for r in res))
