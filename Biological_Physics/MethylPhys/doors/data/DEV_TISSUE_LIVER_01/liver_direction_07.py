"""DEV-TISSUE-LIVER-01 synthetic test 7, direction of composition (written and committed 2026-10-12 BEFORE it was run; author decision after v2
independence failed). Question: can a change in tissue mix alone push the sealed COUNT rule to a false FINGERPRINT? Independence (test 3) failed
because Met-A drifts with the hepatocyte share; this test asks whether any realistic mix change can RAISE R enough to fake a call.
Setup: liver_synthetic.py v2 unchanged (sites, bar, CRIT, stand-in pairs), imported by running its code up to the cases.
For every stand-in pair the tested normal j is blended with a mix component at share a, with NO methylation change:
  stroma (mean of fibroblasts, smooth muscle, endothelium), fibroblasts, smooth muscle, endothelium, immune (macrophages, T, B, neutrophils,
  monocytes, NK), hepatocyte, adipocytes, another patient's normal liver from the same plate;
  each at a ~ U(0.3, 0.6) and at a ~ U(0.6, 0.9) (a tumor far richer in that component than its normal).
BAR (fixed here): for EVERY component and both shares, the COUNT rule's false FINGERPRINT call over 2,000 draws of N pairs <= 0.05 and the share of
pairs above the bar <= 0.10. PASS replaces the independence bar for this set (composition cannot fake the call); FAIL: the set is not sealed.
Writes liver_direction_07_output.csv and liver_direction_07_summary.txt. Usage: python3 liver_direction_07.py POSTDIR BETADIR"""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); src = open(os.path.join(HERE, "liver_synthetic.py")).read()
exec(src[:src.index('out.append(f"2-6 cases')]); rg = np.random.default_rng(20261013)
comp = {"stroma": Mu[["fibroblast", "smooth_muscle", "endothelium"]].mean(1).values, "fibroblasts": Mu["fibroblast"].values,
        "smooth muscle": Mu["smooth_muscle"].values, "endothelium": Mu["endothelium"].values, "immune": imm, "hepatocyte": Mu["hepatocyte"].values,
        "adipocytes": Mu["adipocytes"].values, "other normal liver": None}
rows, out = [], [f"bar {bar:.4f} | CRIT {crit} of N {NP} | stand-in pairs {len(pairs)}"]
for nm, tpl in comp.items():
    for lo_a, hi_a in ((.3, .6), (.6, .9)):
        def f(j, k):
            a = rg.uniform(lo_a, hi_a); T = other(j) if tpl is None else tpl
            return H(np.clip((1 - a) * X[j] + a * T, 0, 1)[Sx]).mean() / base[k]
        v = np.array([f(j, k) for j, k in pairs]); above = v > bar
        cc = float(np.mean([above[rg.integers(len(v), size=NP)].sum() >= crit for _ in range(2000)]))
        rows.append(dict(component=nm, share=f"{lo_a}-{hi_a}", mean_R=round(float(v.mean()), 4), share_above_bar=round(float(above.mean()), 3), count_call=round(cc, 3)))
        out.append(f"  {nm} at {lo_a}-{hi_a}: mean R {v.mean():.4f}, share above bar {above.mean():.3f}, COUNT false call {cc:.3f}")
T = pd.DataFrame(rows); ok = bool((T.count_call <= 0.05).all() and (T.share_above_bar <= 0.10).all())
worst = T.sort_values("count_call").iloc[-1]
out.append(f"worst: {worst.component} at {worst.share} (false call {worst.count_call}, share above bar {worst.share_above_bar})")
out.append(f"7 direction (no mix change fakes the COUNT call): {'PASSED' if ok else 'FAILED'}")
T.to_csv(os.path.join(HERE, "liver_direction_07_output.csv"), index=False); open(os.path.join(HERE, "liver_direction_07_summary.txt"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
