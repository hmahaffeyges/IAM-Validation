"""DEV-TISSUE-LUNG-04 inputs (same as DEV-TISSUE-LUNG-03 v3): the healthy, untreated purified lung stromal arrays (no tumor involved). GSE146376 airway smooth muscle,
Vehicle arrays of donors without asthma (54, EPIC v1); GSE107226 lung fibroblasts from normal lungs (4, 450K). List in lung_stroma_arrays.json
(GEO metadata, 2026-10-12). Usage: python3 fetch_lung_stroma.py OUT_DIR"""
import os, sys, json, gzip, shutil, time, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
for r in json.load(open(os.path.join(HERE, "lung_stroma_arrays.json"))):
    for u in r["idat"]:
        p = os.path.join(OUT, f"{r['cell']}__{os.path.basename(u)[:-3]}")
        if os.path.exists(p) and os.path.getsize(p) > 0: continue
        for i in range(5):
            try:
                urllib.request.urlretrieve(u.replace("ftp://", "https://"), p + ".gz")
                with gzip.open(p + ".gz") as a, open(p, "wb") as b: shutil.copyfileobj(a, b)
                os.remove(p + ".gz"); break
            except Exception as e: print(u, "retry", i + 1, e, flush=True); time.sleep(10 * (i + 1))
print("idat files", len([f for f in os.listdir(OUT) if f.endswith(".idat")]))
