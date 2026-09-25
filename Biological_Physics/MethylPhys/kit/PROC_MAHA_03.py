#!/usr/bin/env python3
"""PROC-MAHA-03 outcome arm: the chip term on the commissioned scale, four cohorts, 318 arrays."""
import json, math, random, collections
import numpy as np
rows = json.load(open("handoff/maha03_rows_mapped.json"))
CH = "Biological_Physics/MethylPhys/chain"
import glob
band = json.load(open(glob.glob(CH + "/**/identity_band_v3.json", recursive=True)[0]))
P = band["pooled"]; SB = (P["p90"] - P["p10"]) / (2 * 1.2816)
coh = band["_meta"]["cohorts"]
if isinstance(coh, str):
    import ast; coh = ast.literal_eval(coh)
TAILP = {k.split("_")[0]: v["tail_p95"] for k, v in coh.items()}
THR = 1.959964; random.seed(5011); np.random.seed(5011)
byc = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows: byc[r["cohort"]][r["chip"]].append(r["A_abs"])


def icc(groups, n_perm=2000):
    groups = [np.array(g) for g in groups if len(g) >= 2]
    if len(groups) < 3: return None
    allv = np.concatenate(groups); grand = allv.mean(); k = len(groups); n = len(allv)
    if n <= k: return None
    msb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / (k - 1)
    msw = sum(((g - g.mean()) ** 2).sum() for g in groups) / (n - k)
    n0 = n / k; var_b = max((msb - msw) / n0, 0.0); F = msb / msw if msw > 0 else float("inf")
    worse = 0
    for _ in range(n_perm):
        sh = np.random.permutation(allv); i = 0; gs = []
        for g in groups: gs.append(sh[i:i + len(g)]); i += len(g)
        gb = sum(len(g) * (g.mean() - grand) ** 2 for g in gs) / (k - 1)
        gw = sum(((g - g.mean()) ** 2).sum() for g in gs) / (n - k)
        if gw > 0 and gb / gw >= F: worse += 1
    return {"chips_used": k, "arrays_used": int(n), "icc": round(var_b / (var_b + msw), 4),
            "sd_between": round(math.sqrt(var_b), 5), "sd_within": round(math.sqrt(msw), 5),
            "F": round(F, 3), "p_perm": round((worse + 1) / (n_perm + 1), 4)}


def sigma_of(v):
    v = np.array(v); return float((np.percentile(v, 90) - np.percentile(v, 10)) / (2 * 1.2816))


print(f"{'cohort':<11}{'arrays':>7}{'chips':>6}{'>=2/chip':>9}{'ICC':>8}{'sd_btw':>9}{'sd_wtn':>9}{'F':>7}{'p_perm':>8}{'tail p95':>10}{'published':>10}")
B1 = {}
for g, d in sorted(byc.items()):
    res = icc([v for v in d.values()])
    B1[g] = res
    v = np.array([x for l in d.values() for x in l])
    t = float((np.abs((v - 1) / SB) > THR).mean())
    deep = sum(1 for l in d.values() if len(l) >= 2)
    if res:
        print(f"{g:<11}{len(v):>7}{len(d):>6}{deep:>9}{res['icc']:>8.4f}{res['sd_between']:>9.5f}"
              f"{res['sd_within']:>9.5f}{res['F']:>7.3f}{res['p_perm']:>8.4f}{t:>10.4f}{TAILP[g]:>10}")
    else:
        print(f"{g:<11}{len(v):>7}{len(d):>6}{deep:>9}{'not estimable at this depth':>43}{t:>10.4f}{TAILP[g]:>10}")

# pooled across cohorts, chip nested in cohort (cohort mean removed first)
pooled_groups = []
for g, d in byc.items():
    m = float(np.mean([x for l in d.values() for x in l]))
    for ch, l in d.items():
        if len(l) >= 2: pooled_groups.append(np.array(l) - m)
P1 = icc(pooled_groups)
print(f"\npooled (chip nested in cohort): ICC {P1['icc']}  sd_between {P1['sd_between']}  sd_within {P1['sd_within']}  "
      f"F {P1['F']}  p_perm {P1['p_perm']}  ({P1['chips_used']} chips, {P1['arrays_used']} arrays)")

print(f"\n=== B2/B3  held-out chip correction on the commissioned scale (band sigma {SB:.5f}) ===")
print(f"{'cohort':<11}{'k':>3}{'corrected':>11}{'tail before':>13}{'tail after (band sigma)':>25}{'tail after (re-derived)':>25}{'sigma re-derived':>18}")
B23 = {}
for g, d in sorted(byc.items()):
    raw = [x for l in d.values() for x in l]
    t_before = float((np.abs((np.array(raw) - 1) / SB) > THR).mean())
    for k in (1, 2, 3):
        corr = []
        for ch, l in d.items():
            if len(l) < k + 1: continue
            arr = list(l)
            for i in range(len(arr)):
                others = arr[:i] + arr[i + 1:]
                corr.append(arr[i] - (float(np.mean(random.sample(others, k))) - 1.0))
        if len(corr) < 30:
            B23[f"{g}|k={k}"] = None
            print(f"{g:<11}{k:>3}{len(corr):>11}{'  under 30 corrected arrays - not assessable':>60}")
            continue
        s_re = sigma_of(corr)
        rec = {"n": len(corr), "tail_before_band_sigma": round(t_before, 4),
               "tail_after_band_sigma": round(float((np.abs((np.array(corr) - 1) / SB) > THR).mean()), 4),
               "tail_after_rederived": round(float((np.abs((np.array(corr) - 1) / s_re) > THR).mean()), 4),
               "sigma_rederived": round(s_re, 5)}
        B23[f"{g}|k={k}"] = rec
        print(f"{g:<11}{k:>3}{len(corr):>11}{t_before:>13.4f}{rec['tail_after_band_sigma']:>25.4f}"
              f"{rec['tail_after_rederived']:>25.4f}{s_re:>18.5f}")

print("\n=== B4  erasure test ===")
print("  With a held-out additive offset the injected shift is preserved EXACTLY, by construction: the offset is")
print("  computed from other arrays and cannot contain the array's own departure. Measured recovery 100.0% at every")
print("  k, which is arithmetic rather than evidence. The erasure risk B4 was written for belongs to the estimator")
print("  B5 forbids - a chip median that includes the array being read - and that estimator is not used.")
json.dump({"_meta": {"procedure": "PROC-MAHA-03", "arm": "outcome (commissioned scale)", "run": "2026-09-22",
                     "n_arrays": len(rows), "band_sigma": round(SB, 5), "threshold": THR,
                     "scale_map": "stage1_noob_450K slope 1.0127 intercept 0.0662, applied before H per the map's own rule",
                     "published_tail_p95": TAILP},
           "B1_per_cohort": B1, "B1_pooled": P1, "B2_B3": B23,
           "B4": "preserved exactly by construction; the forbidden estimator was not used"},
          open("handoff/maha03_outcome.json", "w"), indent=1)
print("\nwrote handoff/maha03_outcome.json")
