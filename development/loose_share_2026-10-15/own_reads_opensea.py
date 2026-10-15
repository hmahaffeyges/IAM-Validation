"""PHY-049 open sea (2026-10-15): the loose share from each sample's own molecules, counted the way the arrays count it. KISS.
DEV-ARCH-01's array loose share uses open-sea probes only (arch_box.py line 33: islands, shores and shelves left out). The whole-genome own-read
count (own_reads_genome.py) used every CpG, so its loose share mixed island territory in. Here ONLY the loose share changes: a window's
methylation is the mean of its molecules' calls at OPEN-SEA CpGs (more than 4 kb from any UCSC hg19 CpG island, cpgIslandExt; the map is
opensea_hg19_cpgidx.npz, built from our own hg19 copy and checked against the read files' own numbering on all 22 chromosomes), a window counts
when it has >= 200 open-sea calls, and is loose below the cut (0.70 and 2/3). The held error is counted exactly as before (odd lines, every
molecule; own_reads_loose.one). Tiles are 2,000 consecutive CpGs inside each chromosome's TRUE range (pat_bounds_hg19.json; own_reads_genome.py's
size table was wrong for chr14-22, so 534 of its 13,225 tiles held no reads: erratum 2026-10-15).
Usage: python own_reads_opensea.py synthetic | real <samples.txt> <workers>"""
import os, sys, gzip, json, re, urllib.request, multiprocessing as mp, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
import own_reads_loose as OR
CUTS = OR.CUTS; W = 2000; MINCALLS = 200
def load_mask(path=os.path.join(H, "opensea_hg19_cpgidx.npz")):
    Z = np.load(path); return np.unpackbits(Z["open_sea"])[:int(Z["n"])].astype(bool)
def true_tiles(bounds):
    T = {}
    for c in [f"chr{i}" for i in range(1, 23)]:
        lo, hi = bounds[c]; s = np.arange(lo, hi - W + 2, W); T[c] = np.stack([s, s + W - 1], 1)
    return T
def count(lines, T, OS):
    calls = {}; e_iso = e_opp = 0
    for k, line in enumerate(lines):
        p = line.rstrip("\n").split("\t")
        if len(p) < 4: continue
        c, i, pat, n = p[0], int(p[1]), p[2], int(p[3])
        if k % 2 == 0:
            w = T.get(c)
            if w is None: continue
            j = np.searchsorted(w[:, 0], i, side="right") - 1
            if j < 0 or i > w[j, 1]: continue
            m = u = 0
            for q, x in enumerate(pat):
                if x == "." or not OS[i + q - 1]: continue
                u += 1; m += x == "C"
            if u: a = calls.setdefault((c, j), [0, 0]); a[0] += n * m; a[1] += n * u
        else:
            s = [x for x in pat if x != "."]
            if len(s) >= 6 and s.count("C") >= 0.8 * len(s):
                e_opp += n * (len(s) - 2); e_iso += n * sum(1 for q in range(1, len(s) - 1) if s[q] == "T" and s[q-1] == "C" and s[q+1] == "C")
    b = np.array([a[0] / a[1] for a in calls.values() if a[1] >= MINCALLS])
    out = {f"loose_{cut:.4f}": float((b < cut).mean()) if len(b) else np.nan for cut in CUTS}
    out.update(windows=int(len(b)), held_err=e_iso / max(e_opp, 1), held_opp=int(e_opp)); return out
def fetch(g):
    t = urllib.request.urlopen(f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{g[:7]}nnn/{g}/suppl/", timeout=120).read().decode()
    return [x for x in re.findall(r'href="([^"]+\.pat\.gz)"', t) if ".hg38." not in x][0]
def run_one(g):
    try:
        name = fetch(g); u = f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{g[:7]}nnn/{g}/suppl/{name}"
        with urllib.request.urlopen(u, timeout=900) as r, gzip.open(r, "rt") as fh: o = count(fh, TT, MASK)
        return dict(sample=g, file=name, **o)
    except Exception as e: return dict(sample=g, error=f"{type(e).__name__}: {str(e)[:120]}")
if sys.argv[1] == "synthetic":
    rng = np.random.default_rng(49); L = []
    nW = 120; idx0 = 1; OSm = np.zeros(nW * W + 10, bool); bounds = {f"chr{i}": (1, 0) for i in range(1, 23)}; bounds["chr1"] = (1, nW * W)
    for w in range(nW): OSm[w * W + 0: w * W + 1400] = True          # each window: 1,400 open-sea CpGs, then 600 island-territory CpGs
    T = true_tiles(bounds)
    for share in (0.2, 0.5, 0.8):
        loose = set(rng.choice(nW, int(share * nW), replace=False)); lines = []
        for w in range(nW):
            for r_ in range(60):
                st = w * W + 1 + int(rng.integers(0, W - 20)); L_ = 10
                pat = "".join(("T" if (rng.random() < (0.55 if w in loose else 0.15)) else "C") if OSm[st + q - 1] else ("T" if rng.random() < 0.95 else "C") for q in range(L_))
                for rep in range(2): lines.append(f"chr1\t{st}\t{pat}\t1")
        o = count(lines, T, OSm); o_all = count(lines, T, np.ones_like(OSm))
        L.append(f"planted open-sea loose share {share}: open-sea count {o['loose_0.7000']:.3f} at 0.70 ({o['windows']} windows); every-CpG count {o_all['loose_0.7000']:.3f}")
    ok = all(abs(float(l.split('open-sea count ')[1].split()[0]) - float(l.split('share ')[1].split(':')[0])) <= 0.03 for l in L) and \
         all(float(l.split('every-CpG count ')[1]) > float(l.split('share ')[1].split(':')[0]) + 0.1 for l in L)
    L.append(f"SYNTHETIC: {'MET' if ok else 'NOT MET'} (the open-sea count recovers the planted share within 0.03; counting every CpG inflates it by island territory)")
    open(os.path.join(H, "own_reads_opensea_synthetic.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
else:
    # resumable (spot machines are reclaimed): results go to S3 as JSON lines every 10 samples; a restart skips finished samples; GEO refusals
    # (HTTP 503) are retried in up to three later rounds with fewer workers
    import time, boto3
    S3 = boto3.client("s3"); BK = "methylphys-data-945451304272-us-west-2-an"; KEY = "results/LOOSE_SHARE_2026-10-15/opensea/own_reads_opensea.jsonl"
    MASK = load_mask(); TT = true_tiles(json.load(open(os.path.join(H, "pat_bounds_hg19.json")))); G = [l.strip() for l in open(sys.argv[2]) if l.strip()]
    try: R = [json.loads(l) for l in S3.get_object(Bucket=BK, Key=KEY)["Body"].read().decode().splitlines() if l.strip()]
    except Exception: R = []
    def save(): S3.put_object(Bucket=BK, Key=KEY, Body="\n".join(json.dumps(x) for x in R).encode())
    for rnd, nw in enumerate((int(sys.argv[3]), 6, 3, 2)):
        done = {x["sample"] for x in R if "error" not in x}; R = [x for x in R if "error" not in x]; todo = [g for g in G if g not in done]
        if not todo: break
        if rnd: time.sleep(300)
        with mp.Pool(nw) as P:
            for k, o in enumerate(P.imap_unordered(run_one, todo)):
                R.append(o); print(o, flush=True)
                if k % 10 == 9: save()
        save()
    pd.DataFrame(R).to_csv("own_reads_opensea.csv", index=False); S3.upload_file("own_reads_opensea.csv", BK, KEY.replace(".jsonl", ".csv"))
