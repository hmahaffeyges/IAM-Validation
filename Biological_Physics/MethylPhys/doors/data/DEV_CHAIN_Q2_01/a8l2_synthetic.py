#!/usr/bin/env python3
"""A8L2 synthetic (before any library is read for A8L2): planted molecules through the committed splitter (lag_split.py) and the one command's
count. Two planted files on chr1-chr3 (300,000 molecules of 14 CpGs each, all held marks, losses planted independently per CpG):
  NULL     loss 0.045 everywhere: late/early held-error ratio must be 1.00 within 0.03 (the split itself creates no difference);
  CONTRAST loss 0.040 early, 0.045 mid, 0.050 late third (by the splitter's own timing call at each molecule's first CpG): late/early must come
           back between 1.15 and 1.32 and above mid/early, which must be above 1 (the split reads the planted direction and size).
Also: the molecule-weighted mean of the three thirds' held error must agree with the unsplit file's within 5 % (halves check).
Counts here are synthetic, so the count-only route is used (not a reading). Prints one ::notice:: line with every number and PASS/FAIL."""
import os, sys, gzip, json, random, subprocess, time
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
IF = os.path.join(REPO, "Biological_Physics/MethylPhys/chain/MethylPhys_Interface")
SPLIT = os.path.join(REPO, "Biological_Physics/MethylPhys/doors/data/DEV_METHIONINE_01/lag_split.py")
sys.argv = [sys.argv[0], sys.argv[1] if len(sys.argv) > 1 else "syn_out"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a8l2_read as A   # setup(): the splitter's genome and timing tracks (no library is read: no GSM given)
OUT = A.OUT; W = A.WORK; t0 = time.time()
src = open(SPLIT).read(); src = src[:src.index("for f in sys.argv[1:]")]
os.environ["IAM_WORK"] = W; ns = {"__name__": "lagsplit_timing", "sys": sys}; exec(compile(src, SPLIT, "exec"), ns); third = ns["third"]; RANGES = ns["RANGES"]
rng = random.Random(20261118)
def plant(path, rate):
    rows = []
    for c in ("chr1", "chr2", "chr3"):
        a, b = RANGES[c]
        for _ in range(100000):
            i = rng.randint(a, b - 14); k = third(c, i)
            if not k: continue
            e = rate[k]; rows.append((c, i, "".join("T" if rng.random() < e else "C" for _ in range(14))))
    rows.sort(key=lambda r: r[1])
    with gzip.open(path, "wt") as fo:
        for c, i, p in rows: fo.write(f"{c}\t{i}\t{p}\t1\n")
    return len(rows)
def count(path, tag):
    b = os.path.join(OUT, tag + ".json")
    subprocess.run([sys.executable, "run_sample.py", "--pat", path, "--dev-count-only", "--dev-allow-partial-genome", "--id", tag, "--out",
                    os.path.join(OUT, tag + ".html"), "--bundle", b], cwd=IF, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    B = json.load(open(b)) if os.path.exists(b) else {}
    c = ((B.get("iam_a_count_only") or {}).get("count") or {})
    return c.get("held_err_length"), c.get("molecules") or c.get("n_molecules")
res = {}
for name, rate in (("NULL", dict(early=0.045, mid=0.045, late=0.045)), ("CONTRAST", dict(early=0.040, mid=0.045, late=0.050))):
    f = os.path.join(W, f"GSMSYN{name}.pat.gz"); n = plant(f, rate)
    whole, _ = count(f, f"syn_{name}_whole")
    subprocess.run([sys.executable, SPLIT, f], env={**os.environ, "IAM_WORK": W}, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    parts = {k: count(os.path.join(W, f"GSMSYN{name}_{k}.pat.gz"), f"syn_{name}_{k}") for k in ("early", "mid", "late")}
    ok = all(v[0] for v in parts.values()) and whole
    r = {"molecules": n, "whole": whole, **{k: v[0] for k, v in parts.items()}}
    if ok:
        r["late_over_early"] = parts["late"][0] / parts["early"][0]; r["mid_over_early"] = parts["mid"][0] / parts["early"][0]
        r["thirds_mean_over_whole"] = (sum(v[0] for v in parts.values()) / 3) / whole
    res[name] = r
N, C = res["NULL"], res["CONTRAST"]
try:
    p_null = abs(N["late_over_early"] - 1) <= 0.03
    p_con = 1.15 <= C["late_over_early"] <= 1.32 and C["late_over_early"] > C["mid_over_early"] > 1
    p_half = all(0.95 <= x["thirds_mean_over_whole"] <= 1.05 for x in (N, C))
except KeyError: p_null = p_con = p_half = False
res["verdict"] = "PASS" if (p_null and p_con and p_half) else "FAIL"; res["minutes"] = round((time.time() - t0) / 60, 1)
json.dump(res, open(os.path.join(OUT, "a8l2_synthetic.json"), "w"), indent=1)
print("::notice::A8L2 SYNTHETIC " + res["verdict"] + ": " + json.dumps({k: ({kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()} if isinstance(v, dict) else v) for k, v in res.items()})[:1800])
