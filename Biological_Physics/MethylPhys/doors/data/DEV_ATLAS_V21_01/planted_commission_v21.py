"""Planted test of the atlas v2.1 commissioning scorers (readiness L15). Written 2026-10-14 before it runs. Synthetic only; seed 20261014.
A synthetic cell: 4,000 sites, truth mu bimodal (Beta(0.5, 0.5) clipped to 0.02-0.98), donor spread 0.02-0.05, posterior draws = truth + N(0, 0.01),
100 draws. Held-out sets of 8 samples each, generated from the model. Each scorer must give the planted answer:
 U1-a correct atlas, new lab offset 0 -> COMMISSIONED          U1-b intervals planted too narrow (donor spread halved in the draws) -> FAILED
 U1-c a new laboratory offset +0.06 -> FAILED                   U1-d intervals far too wide (spread x 3) -> FAILED (above 0.97)
 U1-e sequencing sample (coverage 15, the adopted transfer and scatter) from the model -> COMMISSIONED
 U2-a a panel that truly agrees (every cell within 0.02 of the anchor at blind sites) -> COMMISSIONED
 U2-b one panel cell planted 0.08 off at 30 % of blind sites -> FAILED       U2-c epithelium not assessed -> NOT ASSESSED
 U3-a estimates with error N(0, 0.01) -> met                    U3-b estimates biased +0.04 -> not met
PASSED only if every case gives its planted answer. Writes planted_commission_v21_summary.txt."""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import commission_v21 as CM
rng = np.random.default_rng(20261014); S = 4000
mu = np.clip(rng.beta(0.5, 0.5, S), 0.02, 0.98); sd = rng.uniform(0.02, 0.05, S)
def draws(sd_scale=1.0): return (mu + rng.normal(0, 0.01, (100, S))).clip(0.001, 0.999), np.tile(sd * sd_scale, (100, 1))
def samples(n=8, off=0.0): return [np.clip(mu + off + rng.normal(0, 1, S) * np.sqrt(sd ** 2 + CM.FLOOR), 0, 1) for _ in range(n)]
def u1(sd_scale=1.0, off=0.0):
    d, s = draws(sd_scale); lo, hi = CM.predictive_interval(d, s); c = float(np.mean([CM.u1_coverage(lo, hi, y) for y in samples(off=off)]))
    return c, CM.u1_verdict({"setA": c})
L = []; ok = True
def check(name, got, want, extra=""):
    global ok; good = got == want; ok &= good; L.append(f"{name}: {got} (planted {want}) {'ok' if good else 'WRONG'} {extra}")
for name, a, o, want in (("U1-a", 1.0, 0.0, "COMMISSIONED"), ("U1-b", 0.5, 0.0, "FAILED"), ("U1-c", 1.0, 0.06, "FAILED"), ("U1-d", 3.0, 0.0, "FAILED")):
    c, v = u1(a, o); check(name, v, want, f"coverage {c:.4f}")
T = json.load(open(os.path.join(HERE, "transfer_v21_final.json"))); kx, ky = np.array(T["knots_x"]), np.array(T["knots_y"])
g = lambda x: np.interp(x, kx, ky); sc = lambda x: np.interp(x, np.array(T["scatter_mid"]), np.array(T["scatter_s"]))
d, s = draws(); lo, hi = CM.predictive_interval(d, s, kind="wgbs", g=g, cov=15, scatter=sc)
cs = []
for _ in range(8):
    h = g(mu) + rng.normal(0, 1, S) * sc(mu); slope = (g(mu + 0.01) - g(mu - 0.01)) / 0.02
    m = np.clip(h + slope * rng.normal(0, 1, S) * sd, 0, 1); y = rng.binomial(15, m) / 15 + rng.normal(0, 0.01, S); cs.append(CM.u1_coverage(lo, hi, y))
c = float(np.mean(cs)); check("U1-e", CM.u1_verdict({"seqA": c}), "COMMISSIONED", f"coverage {c:.4f}")
B = 6000; anchor = rng.uniform(0.05, 0.95, B)
def panel(bad=None):
    sh = {}
    for cell in ("epithelium", "fibroblasts", "macrophages", "t cells"):
        v = anchor + rng.uniform(-0.02, 0.02, B)
        if cell == bad: k = rng.random(B) < 0.30; v[k] += 0.08
        sh[cell] = CM.u2_share(np.mean([v + rng.normal(0, 0.03, B) for _ in range(8)], 0), anchor, 0.05)
    return sh
sa = panel(); check("U2-a", CM.u2_verdict(sa, epithelium="epithelium"), "COMMISSIONED", f"min share {min(sa.values()):.4f}")
sb = panel("fibroblasts"); check("U2-b", CM.u2_verdict(sb, epithelium="epithelium"), "FAILED", f"fibroblast share {sb['fibroblasts']:.4f}")
sc_ = {k: v for k, v in panel().items() if k != "epithelium"}; check("U2-c", CM.u2_verdict(sc_, epithelium="epithelium"), "NOT ASSESSED")
tr = rng.uniform(0.4, 0.75, 24)
r = CM.u3_neutrophil(tr + rng.normal(0, 0.01, 24), tr); check("U3-a", r["met"], True, f"MAE {r['mae']:.4f} max {r['max']:.4f}")
r = CM.u3_neutrophil(tr + 0.04, tr); check("U3-b", r["met"], False, f"MAE {r['mae']:.4f}")
L.insert(0, "Atlas v2.1 planted test of the commissioning scorers, run once (planted_commission_v21.py)"); L.append(f"PLANTED COMMISSIONING TEST: {'PASSED' if ok else 'FAILED'}")
open(os.path.join(HERE, "planted_commission_v21_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
