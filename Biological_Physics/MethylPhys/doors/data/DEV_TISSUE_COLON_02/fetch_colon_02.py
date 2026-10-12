"""DEV-TISSUE-COLON-02: fetch GSE199057 IDATs from GEO per sample. Only non-cancer arrays (patients' normal mucosa and non-CRC normals) until
sealed_colon_02.json is committed; a tumour GSM is refused by the lock below before any request is made. Usage:
python3 fetch_colon_02.py OUT_DIR normal | tumour"""
import os, sys, subprocess, time, requests, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); S = pd.read_csv(os.path.join(HERE, "gse199057_samples.csv"))
TUMOUR = set(S[S.kind.str.startswith("colon tumor")].gsm)
def sealed():
    p = os.path.join(HERE, "sealed_colon_02.json")
    if not os.path.exists(p): return False
    r = subprocess.run(["git", "-C", HERE, "status", "--porcelain", p], capture_output=True, text=True)
    t = subprocess.run(["git", "-C", HERE, "ls-files", "--error-unmatch", p], capture_output=True, text=True)
    return t.returncode == 0 and r.stdout.strip() == ""
def lock(g):
    if g in TUMOUR and not sealed(): raise SystemExit(f"LOCKED: {g} is a tumour array and DEV-TISSUE-COLON-02 is not sealed")
def files(g):
    t = requests.get("https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi", params={"acc": g, "targ": "self", "form": "text", "view": "brief"}, timeout=60).text
    return [l.split("=", 1)[1].strip().replace("ftp://", "https://") for l in t.split("\n") if l.startswith("!Sample_supplementary_file") and ".idat" in l.lower()]
if __name__ == "__main__":
    out, which = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
    G = sorted(TUMOUR) if which == "tumour" else sorted(set(S.gsm) - TUMOUR)
    for g in G:
        lock(g)
        for u in files(g):
            p = os.path.join(out, u.split("/")[-1])
            if os.path.exists(p) and os.path.getsize(p) > 0: continue
            for i in range(6):
                try: r = requests.get(u, timeout=300); r.raise_for_status(); open(p, "wb").write(r.content); break
                except Exception:
                    if i == 5: raise
                    time.sleep(5 * (i + 1))
        print(g, flush=True)
