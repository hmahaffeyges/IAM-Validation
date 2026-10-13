"""DEV-TERRITORY-01 synthetic tests 1-3 (written 2026-10-13, before they were run): does the territory contrast behave as the balance
equation predicts, does it cancel the nuisances it is built to cancel, and is it blind where it is declared blind?

The contrast is a DIFFERENCE, not a ratio, and that choice is forced by the derivation:
  eps = f/(f+g); a common proportional shortfall g -> g(1-d) gives  eps' = eps / (1 - d(1 - eps)),  so  D eps ~ eps(1-eps) d.
  - The ABSOLUTE response is larger at the fragile (higher-eps) sites, so the difference eps_f - eps_s GROWS with a shortfall:
        D(eps_f - eps_s) ~ [eps_f(1-eps_f) - eps_s(1-eps_s)] d        <- parameter-free slope
  - The RATIO eps_f/eps_s moves the other way and only weakly (the relative rise is larger at low-eps sites), so a ratio would be a
    worse instrument. An earlier draft of the design note called the reading a ratio; that was wrong and is corrected.
  - A plate-style ADDITIVE offset adds the same amount to both classes, so the difference cancels it exactly, which is the whole point.

# NO CHAIN: the quantity here is the held error itself. At a site held methylated the error IS 1 - beta, so this file needs no reading
# stage - it plants eps directly and measures eps. The chain's H() is imported only to report the entropy reading alongside, so the
# numbers can be compared with Met-A's units; the physics under test is in eps.

Bars, all written before running:
  1. the measured response ratio D eps_f / D eps_s is within 10 % of [eps_f(1-eps_f)] / [eps_s(1-eps_s)] at every shortfall 2-20 %.
  2. with NO shortfall, each nuisance separately (additive plate offset +-0.02, composition change, measurement noise) moves the
     contrast by less than a THIRD of what a 2 % shortfall moves it.
  3. a UNIFORM absolute loss added to both territories does NOT move the contrast (the declared blind spot, confirmed not hidden).
Writes territory_synth_01_output.csv and territory_synth_01_summary.txt. Usage: python3 territory_synth_01.py [OUTDIR]
"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = sys.argv[1] if len(sys.argv) > 1 else HERE
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "chain")))
from dev_comp_methods import H                      # entropy reading, for comparability with Met-A's units only
rg = np.random.default_rng(20261013)
NS, NF, NDRAW = 4000, 4000, 200                     # island probes, open-sea probes, people per condition
EPS_S, EPS_F = 0.012, 0.052                         # healthy baselines of the two territories (stable, fragile)

def sample(eps_s, eps_f, d=0.0, offset=0.0, comp=0.0, noise=0.004, uniform=0.0):
    """One person. eps per class after a proportional shortfall d, an additive plate offset, a composition shift and noise."""
    def cls(eps, n):
        e = eps / (1 - d * (1 - eps)) + uniform              # the derived response, plus any uniform absolute loss
        e = e * (1 + comp)                                   # a composition change scales the apparent error in both classes
        b = np.clip(1 - e + rg.normal(0, noise, n) - offset, 1e-4, 1 - 1e-4)
        return b
    return cls(eps_s, NS), cls(eps_f, NF)

def read(bs, bf):
    """The instrument: mean held error per territory, and their difference. Entropy reading reported alongside."""
    es, ef = float(np.mean(1 - bs)), float(np.mean(1 - bf))
    return es, ef, ef - es, float(H(bf).mean() - H(bs).mean())

base = np.array([read(*sample(EPS_S, EPS_F))[2] for _ in range(NDRAW)])
L = ["DEV-TERRITORY-01 synthetic tests 1-3", "",
     f"baselines: stable {EPS_S:.4f}, fragile {EPS_F:.4f} | healthy contrast {base.mean():.5f} (SD {base.std():.5f}) | {NDRAW} people per condition", ""]
rows, ok1 = [], True
pred_ratio = (EPS_F * (1 - EPS_F)) / (EPS_S * (1 - EPS_S))
L.append(f"TEST 1 - predicted response ratio from the baselines alone: {pred_ratio:.3f} (nothing fitted)")
for d in (0.02, 0.05, 0.10, 0.20):
    r = np.array([read(*sample(EPS_S, EPS_F, d=d)) for _ in range(NDRAW)])
    h = np.array([read(*sample(EPS_S, EPS_F)) for _ in range(NDRAW)])
    d_s, d_f = r[:, 0].mean() - h[:, 0].mean(), r[:, 1].mean() - h[:, 1].mean()
    meas = d_f / d_s; rel = abs(meas / pred_ratio - 1); ok1 &= rel <= 0.10
    rows.append(dict(test="1_response_ratio", shortfall=d, measured_ratio=meas, predicted_ratio=pred_ratio, rel_error=rel,
                     contrast_shift=r[:, 2].mean() - h[:, 2].mean()))
    L.append(f"  shortfall {d * 100:4.0f} % | measured ratio {meas:7.3f} | off prediction {rel * 100:5.2f} % "
             f"| contrast moves {(r[:, 2].mean() - h[:, 2].mean()) * 1000:+.3f} x10^-3 {'ok' if rel <= 0.10 else 'FAIL'}")
ref = abs([x for x in rows if x["shortfall"] == 0.02][0]["contrast_shift"])
L += ["", f"TEST 2 - nuisance cancellation, measured against the move a 2 % shortfall makes ({ref * 1000:.3f} x10^-3)"]
ok2 = True
for name, kw in (("plate offset +0.02", dict(offset=0.02)), ("plate offset -0.02", dict(offset=-0.02)),
                 ("composition +10 %", dict(comp=0.10)), ("composition -10 %", dict(comp=-0.10)),
                 ("noise x3", dict(noise=0.012))):
    r = np.array([read(*sample(EPS_S, EPS_F, **kw)) for _ in range(NDRAW)])
    mv = abs(r[:, 2].mean() - base.mean()); hit = mv < ref / 3; ok2 &= hit
    rows.append(dict(test="2_nuisance", shortfall=0.0, nuisance=name, move=mv, third_of_signal=ref / 3, passed=hit))
    L.append(f"  {name:<20} moves the contrast {mv * 1000:7.3f} x10^-3 | a third of the 2 % signal is {ref / 3 * 1000:.3f} "
             f"{'ok' if hit else 'FAIL'}")
r = np.array([read(*sample(EPS_S, EPS_F, uniform=0.01)) for _ in range(NDRAW)])
mv = abs(r[:, 2].mean() - base.mean()); ok3 = mv < ref / 3
rows.append(dict(test="3_blind_spot", shortfall=0.0, nuisance="uniform absolute loss +0.01", move=mv, third_of_signal=ref / 3, passed=ok3))
L += ["", "TEST 3 - the declared blind spot: a uniform absolute loss in BOTH territories",
      f"  moves the contrast {mv * 1000:7.3f} x10^-3 | blind as declared: {'yes' if ok3 else 'NO - it moved'}", "",
      f"TEST 1 {'PASSED' if ok1 else 'FAILED'} | TEST 2 {'PASSED' if ok2 else 'FAILED'} | TEST 3 {'PASSED' if ok3 else 'FAILED'}",
      ("ALL THREE BARS MET: the contrast responds as the balance equation predicts, cancels plate, composition and noise, and is "
       "blind exactly where declared." if (ok1 and ok2 and ok3) else
       "NOT ALL BARS MET: the instrument is not ready for a real array; the failing bar is above.")]
pd.DataFrame(rows).to_csv(os.path.join(OUT, "territory_synth_01_output.csv"), index=False)
open(os.path.join(OUT, "territory_synth_01_summary.txt"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
