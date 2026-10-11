"""DEV-FINGERPRINT-02 Arm G (rule 02c, written 2026-10-11 before any MCF-7 file is read). Arm G reads GAIN: molecules of normally unmethylated territory
(healthy beta <= 0.2 pooled from the pair's normal files, >= 6 calls) that read >= 80 % methylated (why/both_directions.read2; runloss_01 loader).
Found post hoc on HepG2/A549/LNCaP (why/both_directions_*.csv); tested here only on the unread MCF-7 pairs (breast_mcf10a, breast_hmec).
BAR = max(every normal file of the pair read against its other normal files; the bronchial donor null both ways; every normal file under the two
non-gain disturbances LOSS05 and SCAT). Version 1 set BAR from the first two only and failed its planted test (why/plant_gain_*_v1_FAILED.json:
scattered de novo error lifted the bar-setting file 0.0002 over a zero-margin bar); revised before any MCF-7 file was read. VERDICT: FINGERPRINT if every MCF-7 file listed in the seal reads gain > BAR; no fingerprint if every file <= BAR; else UNDECIDED.
Steps:  donor PATD | plant PATD PAIR | seal PATD PAIR | score PATD PAIR
plant (normal data only): each normal file, leave-one-out reference: NULL; GAIN05 (5 % of its unmethylated-territory molecules made fully methylated,
expect above BAR); LOSS05 (5 % of methylated-territory molecules lose every methylated call, expect at or below BAR: Arm G must not read loss);
SCAT (each unmethylated call of unmethylated-territory molecules gained with p 0.04, scattered de novo error, expect at or below BAR);
sensitivity: planted gain 0.25/0.5/1/2 % reported. PASS on every normal file: NULL <= BAR; GAIN05 > BAR; planted 0.5 % gain > BAR;
specificity: |LOSS05 - NULL| and |SCAT - NULL| each < 2 % of (GAIN05 - NULL)."""
import os, sys, json, hashlib, subprocess, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "why")); sys.path.insert(0, os.path.join(HERE, "../DEV_RUNLOSS_01"))
import runloss_01 as R
from both_directions import read2
STEP, PATD = sys.argv[1], sys.argv[2]; PAIR = sys.argv[3] if len(sys.argv) > 3 else None
L = lambda x: R.load(os.path.join(PATD, x + ".pat.gz"))
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
def seal_of(P): return json.load(open(os.path.join(HERE, f"sealed_{P}.json")))
def loo(nf, NL): return {x: read2(NL[x], R.beta_v([NL[y] for y in nf if y != x]))["gain_U"] for x in nf}
def plant_gain(f, bv, r, seed, scat=False):
    called = f["C"] | f["T"]; calls = np.bincount(f["mol"], weights=called, minlength=f["N"])
    tb = np.bincount(f["mol"], weights=np.nan_to_num(bv[f["pos"]]) * called, minlength=f["N"])
    with np.errstate(invalid="ignore", divide="ignore"): U = (calls >= 6) & (tb / calls <= 0.2)
    g = np.random.default_rng(seed)
    if scat: hit = U[f["mol"]] & f["T"] & (g.random(len(f["T"])) < r)
    else: hit = (U & (g.random(f["N"]) < r))[f["mol"]] & called
    return dict(f, C=f["C"] | hit, T=f["T"] & ~hit)
def plant_loss(f, bv, r, seed):
    called = f["C"] | f["T"]; calls = np.bincount(f["mol"], weights=called, minlength=f["N"])
    tb = np.bincount(f["mol"], weights=np.nan_to_num(bv[f["pos"]]) * called, minlength=f["N"])
    with np.errstate(invalid="ignore", divide="ignore"): M = (calls >= 6) & (tb / calls >= 0.8)
    hit = (M & (np.random.default_rng(seed).random(f["N"]) < r))[f["mol"]] & f["C"]
    return dict(f, C=f["C"] & ~hit, T=f["T"] | hit)
DONOR = os.path.join(HERE, "why", "gain_donor_null_02c.json")
if STEP == "donor":
    A, B = ["ENCFF000MME", "ENCFF000MMO"], ["ENCFF000MJV", "ENCFF000MJW"]; NL = {x: L(x) for x in A + B}
    out = {f"{x} vs donor {'B' if x in A else 'A'}": read2(NL[x], R.beta_v([NL[y] for y in (B if x in A else A)]))["gain_U"] for x in A + B}
    out["max"] = max(out.values()); json.dump(out, open(DONOR, "w"), indent=1); print(out)
elif STEP == "plant":
    S = seal_of(PAIR); nf = [r["file"] for r in S["normal_files"]]; NL = {x: L(x) for x in nf}; D = json.load(open(DONOR))["max"]
    base = loo(nf, NL); C = {}
    for i, x in enumerate(nf):
        bv = R.beta_v([NL[y] for y in nf if y != x]); f = NL[x]
        c = dict(NULL=read2(f, bv)["gain_U"], GAIN05=read2(plant_gain(f, bv, 0.05, 11 + i), bv)["gain_U"],
                 LOSS05=read2(plant_loss(f, bv, 0.05, 21 + i), bv)["gain_U"], SCAT=read2(plant_gain(f, bv, 0.04, 31 + i, scat=True), bv)["gain_U"])
        sens = {r: read2(plant_gain(f, bv, r, 41 + i), bv)["gain_U"] for r in (0.0025, 0.005, 0.01, 0.02)}
        C[x] = (c, sens)
    bar = max(max(base.values()), D, max(max(c["LOSS05"], c["SCAT"]) for c, _ in C.values())); res = {"bar": bar, "files": {}}; ok = True
    for x, (c, sens) in C.items():
        g = c["GAIN05"] - c["NULL"]; spec = abs(c["LOSS05"] - c["NULL"]) < 0.02 * g and abs(c["SCAT"] - c["NULL"]) < 0.02 * g
        p = (c["NULL"] <= bar) and (c["GAIN05"] > bar) and (sens[0.005] > bar) and spec; ok &= p
        res["files"][x] = dict(c, sensitivity={str(k): v for k, v in sens.items()}, specificity_ok=bool(spec), PASS=bool(p))
        print(x, {k: round(v, 4) for k, v in c.items()}, "| sens", {k: round(v, 4) for k, v in sens.items()}, "| bar", round(bar, 4), "PASS" if p else "FAIL", flush=True)
    res["PASS"] = bool(ok); json.dump(res, open(os.path.join(HERE, f"plant_gain_{PAIR}_output.json"), "w"), indent=1); print("PLANTED TEST", "PASSED" if ok else "FAILED")
elif STEP == "seal":
    out = os.path.join(HERE, f"sealed_gain_{PAIR}.json"); assert not os.path.exists(out), "a seal is written once"
    PL = json.load(open(os.path.join(HERE, f"plant_gain_{PAIR}_output.json"))); assert PL["PASS"], "planted test failed: not sealed"
    S = seal_of(PAIR); nf = [r["file"] for r in S["normal_files"]]
    seal = dict(pair=PAIR, rule="02c Arm G (gain); FINGERPRINT if every listed cancer file > bar", bar=PL["bar"], normal_files=nf,
                normal_sha256={x: sha(os.path.join(PATD, x + ".pat.gz")) for x in nf}, cancer_files_listed=S["cancer_files_listed"],
                donor_null=json.load(open(DONOR)), plant_sha256=sha(os.path.join(HERE, f"plant_gain_{PAIR}_output.json")), scorer_sha256=sha(os.path.abspath(__file__)))
    json.dump(seal, open(out, "w"), indent=1); print(json.dumps({k: seal[k] for k in ("pair", "bar", "cancer_files_listed")}))
elif STEP == "score":
    p = os.path.join(HERE, f"sealed_gain_{PAIR}.json")
    assert subprocess.run(["git", "-C", HERE, "ls-files", "--error-unmatch", p], capture_output=True).returncode == 0, "seal not committed"
    assert subprocess.run(["git", "-C", HERE, "diff", "--quiet", "HEAD", "--", p]).returncode == 0, "seal changed since commit"
    S = json.load(open(p)); assert S["scorer_sha256"] == sha(os.path.abspath(__file__)), "scorer changed since the seal"
    NL = {x: L(x) for x in S["normal_files"]}; bv = R.beta_v(list(NL.values()))
    g = {x: read2(L(x), bv)["gain_U"] for x in S["cancer_files_listed"]}
    v = "FINGERPRINT" if all(x > S["bar"] for x in g.values()) else ("no fingerprint" if all(x <= S["bar"] for x in g.values()) else "UNDECIDED")
    out = dict(pair=PAIR, bar=S["bar"], cancer_gain=g, ratio_min_to_bar=min(g.values()) / S["bar"], VERDICT=v)
    json.dump(out, open(os.path.join(HERE, f"scored_gain_{PAIR}.json"), "w"), indent=1); print(json.dumps(out, indent=1))
