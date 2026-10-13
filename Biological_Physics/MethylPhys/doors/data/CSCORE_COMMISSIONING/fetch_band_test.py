"""C-score test (a), second laboratory: fetch the healthy granulocyte and whole-blood IDATs of GSE184269 (NIA-IRP, GESTALT
healthy donors, EPIC v1), listed in band_test_GSE184269_samples.csv. Usage: python3 fetch_band_test.py OUT_DIR"""
import os, re, sys, time, requests, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
S = pd.read_csv(os.path.join(HERE, "band_test_GSE184269_samples.csv"))
def urls(gsm):
    t = requests.get("https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi", params={"acc": gsm, "targ": "self", "form": "text", "view": "brief"}, timeout=120).text
    return [l.split("=", 1)[1].strip() for l in t.replace("\r", "").split("\n") if l.startswith("!Sample_supplementary_file") and ".idat" in l.lower()]
n = 0
for g in S.gsm:
    for u in urls(g):
        p = os.path.join(out, f"{g}_{os.path.basename(u)}")
        if os.path.exists(p) and os.path.getsize(p) > 1000: continue
        for i in range(5):
            try:
                r = requests.get(u.replace("ftp://", "https://"), timeout=600); r.raise_for_status()
                open(p, "wb").write(r.content); n += 1; break
            except Exception:
                if i == 4: print("FAILED", g, u, flush=True)
                time.sleep(5 * (i + 1))
    time.sleep(0.2)
print("files fetched", n, "| total", len([f for f in os.listdir(out) if f.lower().endswith((".idat", ".idat.gz"))]))
