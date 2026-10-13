"""DEV-TERRITORY-02 synthetic tests (written 2026-10-13, BEFORE they were run): the territory contrast, version 2.

Why a version 2. v1 (DEV-TERRITORY-01) met its physics bar - the response ratio predicted from healthy baselines alone, 4.158, was
measured 4.12-4.20 - but failed the nuisance bar for two reasons that are physics of the measurement, not bugs:
  (a) the DIFFERENCE  D = eps_f - eps_s  cancels an ADDITIVE offset exactly, but a composition change SCALES both classes, and a
      difference scales with them: +10 % composition moved D by five times what a 2 % shortfall does;
  (b) the beta = 1 CEILING: near fully held sites the noise and a negative offset are clipped on one side, so the class mean is biased.

What changes, and why each change is forced rather than chosen:
  1. A SECOND reading, the log-ratio  L = ln eps_f - ln eps_s.  If composition scales both classes by (1 + c), L is unchanged
     EXACTLY. Under a common proportional shortfall d the balance equation gives  ln eps' = ln eps - ln(1 - d(1 - eps)),  so
         D L  ~  -d (eps_f - eps_s)                                   (parameter-free)
         D D  ~  +d [eps_f(1 - eps_f) - eps_s(1 - eps_s)]             (parameter-free, carried from v1)
     A shortfall therefore moves D UP and L DOWN with a fixed ratio; a composition change moves D and leaves L. The PAIR separates
     the two, which neither number can do alone. That two-number signature is the instrument.
  2. The class level is estimated from the MEDIAN beta, not the mean. Clipping at beta = 1 cannot move a median until half the sites
     are clipped. The ceiling is MEASURED, not assumed: the fraction of sites at the ceiling is reported per person, and above 25 % the
     reading is REFUSED rather than returned. A refused reading is an honest answer; a silently biased one is not.
  3. The test now includes real between-person variation: each person carries a person-level multiplicative factor (SD 6 %, the size
     of the healthy whole-blood spread measured in DEV-BLOOD-SENSITIVITY-01; a stated ASSUMPTION about units, since that spread was
     measured on Met-A, not on eps). This is the problem that blocked per-person blood detection, and L is built to cancel it.

# NO CHAIN: the quantity here is the held error itself. At a site held methylated the error IS 1 - beta, so this file needs no reading
# stage - it plants eps and measures eps. The physics under test is the balance equation's response, not a chain stage.

BARS, all written before running:
  1. PHYSICS (carried from v1, with the median estimator): the measured D-response ratio fragile/stable is within 10 % of
     [eps_f(1-eps_f)] / [eps_s(1-eps_s)] at every shortfall 2-20 %.
  2. COMPOSITION: +-10 % composition moves L by less than a THIRD of what a 2 % shortfall moves L. (D under composition is reported;
     it is expected to fail, which is why L exists.)
  3. ADDITIVE OFFSET: +0.02 moves D by less than a third of what a 2 % shortfall moves D. (L under offset is reported.)
  4. CEILING: an offset of -0.02 and noise x3 must EITHER move each reading by less than a third of the 2 % signal OR be REFUSED by the
     ceiling rule. A wrong number returned without a refusal fails the bar.
  5. SIGNATURE: under shortfalls 5-20 %, the measured ratio  D L / D D  is within 15 % of the parameter-free
     -(eps_f - eps_s) / [eps_f(1-eps_f) - eps_s(1-eps_s)];  and under composition +-10 % the same ratio is within 0.1 of zero.
  6. BLIND SPOT (carried): a uniform absolute loss in both classes moves D by less than a third of the 2 % signal.
  REPORTED, NOT BARRED (it decides what the instrument is for): with person-level variation, the per-person SD of D and of L, and the
  smallest shortfall each could call in ONE person (two per-person SDs over its slope), at 4,000 and at 40,000 sites per class.

Writes territory_synth_02_output.csv and territory_synth_02_summary.txt. Usage: python3 territory_synth_02.py [OUTDIR]
"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = sys.argv[1] if len(sys.argv) > 1 else HERE
rg = np.random.default_rng(20261013)
NDRAW = 200
EPS_S, EPS_F = 0.012, 0.052                   # healthy baselines of the two territories, as in v1
CEIL_REFUSE = 0.25                            # above this fraction of sites at the ceiling, the reading is refused

def person(n, d=0.0, offset=0.0, comp=0.0, noise=0.004, uniform=0.0, pvar=0.0):
    """One person's two classes of sites. Returns (beta_s, beta_f)."""
    pf = 1 + rg.normal(0, pvar) if pvar else 1.0          # a person-level multiplicative factor on both classes
    def cls(eps):
        e = (eps / (1 - d * (1 - eps)) + uniform) * (1 + comp) * pf
        return np.clip(1 - e + rg.normal(0, noise, n) - offset, 1e-4, 1.0)
    return cls(EPS_S), cls(EPS_F)

def read(bs, bf):
    """The two readings from one person, or None if the ceiling rule refuses."""
    ceil = max(np.mean(bs >= 1.0), np.mean(bf >= 1.0))
    if ceil > CEIL_REFUSE: return dict(D=np.nan, L=np.nan, ceil=ceil, refused=True)
    es, ef = 1 - np.median(bs), 1 - np.median(bf)
    return dict(D=ef - es, L=np.log(ef) - np.log(es), ceil=ceil, refused=False)

def cohort(n, **kw):
    R = [read(*person(n, **kw)) for _ in range(NDRAW)]
    T = pd.DataFrame(R)
    return dict(D=T.D.mean(), L=T.L.mean(), sdD=T.D.std(), sdL=T.L.std(), refused=T.refused.mean(), ceil=T.ceil.mean())

rows, L = [], ["DEV-TERRITORY-02: two-number territory contrast", "",
               f"baselines stable {EPS_S}, fragile {EPS_F}; {NDRAW} people per condition; ceiling refusal above {CEIL_REFUSE:.0%}", ""]
N = 4000
h = cohort(N)
pred_ratio = (EPS_F * (1 - EPS_F)) / (EPS_S * (1 - EPS_S))
pred_sig = -(EPS_F - EPS_S) / (EPS_F * (1 - EPS_F) - EPS_S * (1 - EPS_S))
L += [f"healthy: D {h['D']:.5f}  L {h['L']:.4f}  | predicted D-response ratio {pred_ratio:.3f}, predicted signature dL/dD {pred_sig:.3f}", ""]

# bar 1 - physics: response ratio of the two classes
L.append("BAR 1 - physics: the fragile/stable response ratio at each shortfall")
ok1 = True
for d in (0.02, 0.05, 0.10, 0.20):
    rs = [read(*person(N, d=d)) for _ in range(NDRAW)]
    es = np.mean([1 - np.median(person(N, d=d)[0]) for _ in range(NDRAW)]); ef = np.mean([1 - np.median(person(N, d=d)[1]) for _ in range(NDRAW)])
    es0 = np.mean([1 - np.median(person(N)[0]) for _ in range(NDRAW)]); ef0 = np.mean([1 - np.median(person(N)[1]) for _ in range(NDRAW)])
    r = (ef - ef0) / (es - es0); ok = abs(r / pred_ratio - 1) <= 0.10; ok1 &= ok
    rows.append(dict(test="1_physics", condition=f"d={d}", value=r, target=pred_ratio, passed=ok))
    L.append(f"  shortfall {d:>4.0%}: measured {r:.3f} vs predicted {pred_ratio:.3f} -> {'ok' if ok else 'FAIL'}")

s2 = cohort(N, d=0.02); sigD, sigL = s2["D"] - h["D"], s2["L"] - h["L"]
L += ["", f"the 2 % shortfall signal: D moves {sigD*1e3:+.3f} x10^-3, L moves {sigL:+.4f}", ""]

# bars 2, 3, 4, 6 - nuisances, each against a third of the 2 % signal on the reading it is barred on
def nuis(name, bar_on, refuse_ok=False, **kw):
    c = cohort(N, **kw); mD, mL = c["D"] - h["D"], c["L"] - h["L"]
    if refuse_ok and c["refused"] > 0.5:
        ok = True; verdict = f"REFUSED ({c['refused']:.0%} of people, ceiling {c['ceil']:.0%}) - honest"
    else:
        tgt = {"D": abs(sigD), "L": abs(sigL)}[bar_on] / 3; mv = {"D": abs(mD), "L": abs(mL)}[bar_on]
        if refuse_ok:   # bar 4: BOTH readings must be inside a third of their own signal
            ok = abs(mD) < abs(sigD) / 3 and abs(mL) < abs(sigL) / 3
        else: ok = mv < tgt
        verdict = f"{'ok' if ok else 'FAIL'}"
    rows.append(dict(test=name, condition=str(kw), D_move=mD, L_move=mL, refused=c["refused"], passed=ok))
    L.append(f"  {name:<32} D {mD*1e3:+8.3f} x10^-3 ({mD/sigD:+6.2f} of signal) | L {mL:+.4f} ({mL/sigL:+6.2f} of signal) | {verdict}")
    return ok
L.append("BARS 2, 3, 4, 6 - nuisances (moves shown as a fraction of the 2 % shortfall's own signal on that reading)")
ok2 = nuis("2 composition +10 % (bar on L)", "L", comp=0.10) & nuis("2 composition -10 % (bar on L)", "L", comp=-0.10)
ok3 = nuis("3 offset +0.02 (bar on D)", "D", offset=0.02)
ok4 = nuis("4 offset -0.02 (refuse or hold)", "D", refuse_ok=True, offset=-0.02) & nuis("4 noise x3 (refuse or hold)", "D", refuse_ok=True, noise=0.012)
ok6 = nuis("6 uniform loss (bar on D)", "D", uniform=0.01)

# bar 5 - the signature
L += ["", "BAR 5 - the two-number signature, dL/dD"]
ok5 = True
for d in (0.05, 0.10, 0.20):
    c = cohort(N, d=d); s = (c["L"] - h["L"]) / (c["D"] - h["D"]); ok = abs(s / pred_sig - 1) <= 0.15; ok5 &= ok
    rows.append(dict(test="5_signature_shortfall", condition=f"d={d}", value=s, target=pred_sig, passed=ok))
    L.append(f"  shortfall {d:>4.0%}: dL/dD {s:+.3f} vs predicted {pred_sig:+.3f} -> {'ok' if ok else 'FAIL'}")
for cm in (0.10, -0.10):
    c = cohort(N, comp=cm); s = (c["L"] - h["L"]) / (c["D"] - h["D"]); ok = abs(s) <= 0.1; ok5 &= ok
    rows.append(dict(test="5_signature_composition", condition=f"comp={cm}", value=s, target=0.0, passed=ok))
    L.append(f"  composition {cm:+.0%}: dL/dD {s:+.3f} vs 0 -> {'ok' if ok else 'FAIL'}")

# reported: per-person precision with real between-person variation
L += ["", "REPORTED - one person at a time, with person-level variation (SD 6 %, an assumption carried from the blood spread)"]
for n in (4000, 40000):
    hp = cohort(n, pvar=0.06); d2 = cohort(n, d=0.02)
    hn = cohort(n); slD = (d2["D"] - hn["D"]) / 0.02; slL = (d2["L"] - hn["L"]) / 0.02
    limD, limL = 2 * hp["sdD"] / abs(slD), 2 * hp["sdL"] / abs(slL)
    rows.append(dict(test="reported_per_person", condition=f"sites={n}", sdD=hp["sdD"], sdL=hp["sdL"], limD=limD, limL=limL))
    L.append(f"  {n:>6} sites per class: per-person SD  D {hp['sdD']*1e3:.3f} x10^-3, L {hp['sdL']:.4f} | smallest shortfall callable "
             f"in one person: D {limD:.1%}, L {limL:.1%}")

allok = ok1 and ok2 and ok3 and ok4 and ok5 and ok6
L += ["", " | ".join(f"BAR {k} {'MET' if v else 'NOT MET'}" for k, v in ((1, ok1), (2, ok2), (3, ok3), (4, ok4), (5, ok5), (6, ok6))),
      ("ALL SIX BARS MET: the pair separates a maintenance shortfall from a composition change with nothing fitted, the ceiling is "
       "refused rather than misread, and the blind spot stays where it was declared." if allok else
       "NOT ALL BARS MET: the failing bar is above, and v2 is not used on any real array until it is fixed.")]
pd.DataFrame(rows).to_csv(os.path.join(OUT, "territory_synth_02_output.csv"), index=False)
open(os.path.join(OUT, "territory_synth_02_summary.txt"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
