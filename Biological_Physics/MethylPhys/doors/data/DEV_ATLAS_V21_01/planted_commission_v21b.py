"""Planted test of the AMENDED atlas v2.1 commissioning scorers (chain Stage A21, 2026-10-16), written before it runs. Synthetic only.
Every case of the original planted test, plus the amendment's own cases. PASSED only if every case gives its planted answer.
 U1-a correct atlas, new lab offset drawn from the measured lab spread -> COMMISSIONED   U1-b intervals too narrow -> FAILED
 U1-c a laboratory offset +0.06 (about 9 lab SDs) -> FAILED                              U1-d intervals far too wide -> FAILED
 U1-e sequencing sample from the model -> COMMISSIONED
 U1-f 1 odd sample of 24 (offset +0.06) -> COMMISSIONED      U1-g 3 odd of 24 -> FAILED      U1-h 3 odd of 24, all withheld by the chain -> COMMISSIONED
 U2-a panel agrees -> COMMISSIONED   U2-b 1 of 4 panel cells off -> FAILED   U2-c epithelium not assessed -> NOT ASSESSED   U2-d 1 of 20 off -> COMMISSIONED
 U3-a error N(0, 0.01) -> met   U3-b bias +0.04 -> not met   U3-c one of 24 samples off by 0.07 -> met"""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.abspath(os.path.join(HERE, "../../../chain")))
import stage_atlas_commission as CM
rng = np.random.default_rng(20261016); S = 4000; LAB = CM.lab_sd()
mu = np.clip(rng.beta(0.5, 0.5, S), 0.02, 0.98); sd = rng.uniform(0.02, 0.05, S)
def draws(k=1.0): return (mu + rng.normal(0, 0.01, (100, S))).clip(0.001, 0.999), np.tile(sd * k, (100, 1))
def sample(off): return np.clip(mu + off + rng.normal(0, 1, S) * np.sqrt(sd ** 2 + CM.FLOOR), 0, 1)
def u1(k=1.0, offs=None, withheld=()):
    d, s = draws(k); lo, hi = CM.predictive_interval(d, s, lab=LAB); offs = offs if offs is not None else rng.normal(0, LAB, 24)
    return CM.u1_verdict({f"s{i}": CM.u1_coverage(lo, hi, sample(o)) for i, o in enumerate(offs)}, withheld=withheld)
L = [f"lab spread from the fit's offsets: {LAB:.4f}"]; ok = True
def check(name, got, want, extra=""):
    global ok; good = got == want; ok &= good; L.append(f"{name}: {got} (planted {want}) {'ok' if good else 'WRONG'} {extra}")
for name, k, offs, wh, want in (("U1-a", 1.0, None, (), "COMMISSIONED"), ("U1-b", 0.5, None, (), "FAILED"), ("U1-c", 1.0, np.full(24, 0.06), (), "FAILED"),
                                ("U1-d", 3.0, None, (), "FAILED")):
    r = u1(k, offs, wh); check(name, r["verdict"], want, f"share {r['share']:.3f}")
good = list(rng.normal(0, LAB, 24))
r = u1(1.0, good[:23] + [0.06]); check("U1-f", r["verdict"], "COMMISSIONED", f"share {r['share']:.3f}")
r = u1(1.0, good[:21] + [0.06] * 3); check("U1-g", r["verdict"], "FAILED", f"share {r['share']:.3f}")
r = u1(1.0, good[:21] + [0.06] * 3, withheld=("s21", "s22", "s23")); check("U1-h", r["verdict"], "COMMISSIONED", f"share {r['share']:.3f}, withheld {r['withheld']}")
T = json.load(open(CM.TERMS)); kx, ky = np.array(T["knots_x"]), np.array(T["knots_y"])
g = lambda x: np.interp(x, kx, ky); sc = lambda x: np.interp(x, np.array(T["scatter_mid"]), np.array(T["scatter_s"]))
d, s = draws(); lo, hi = CM.predictive_interval(d, s, kind="wgbs", g=g, cov=15, scatter=sc); cs = {}
for i in range(24):
    h = g(mu) + rng.normal(0, 1, S) * sc(mu); slope = (g(mu + 0.01) - g(mu - 0.01)) / 0.02
    m = np.clip(h + slope * rng.normal(0, 1, S) * sd, 0, 1); cs[f"q{i}"] = CM.u1_coverage(lo, hi, rng.binomial(15, m) / 15 + rng.normal(0, 0.01, S))
r = CM.u1_verdict(cs); check("U1-e", r["verdict"], "COMMISSIONED", f"share {r['share']:.3f}")
B = 6000; anchor = rng.uniform(0.05, 0.95, B)
def panel(cells, bad=()):
    sh = {}
    for c in cells:
        v = anchor + rng.uniform(-0.02, 0.02, B)
        if c in bad: k = rng.random(B) < 0.30; v[k] += 0.08
        sh[c] = CM.u2_share(np.mean([v + rng.normal(0, 0.03, B) for _ in range(8)], 0), anchor, 0.05)
    return sh
four = ["epithelium", "fibroblasts", "macrophages", "t cells"]; twenty = four + [f"cell{i}" for i in range(16)]
check("U2-a", CM.u2_verdict(panel(four), epithelium="epithelium")["verdict"], "COMMISSIONED")
check("U2-b", CM.u2_verdict(panel(four, ("fibroblasts",)), epithelium="epithelium")["verdict"], "FAILED")
check("U2-c", CM.u2_verdict({k: v for k, v in panel(four).items() if k != "epithelium"}, epithelium="epithelium")["verdict"], "NOT ASSESSED")
check("U2-d", CM.u2_verdict(panel(twenty, ("cell3",)), epithelium="epithelium")["verdict"], "COMMISSIONED")
tr = rng.uniform(0.4, 0.75, 24)
check("U3-a", CM.u3_neutrophil(tr + rng.normal(0, 0.01, 24), tr)["met"], True)
check("U3-b", CM.u3_neutrophil(tr + 0.04, tr)["met"], False)
e = rng.normal(0, 0.01, 24); e[5] = 0.07; check("U3-c", CM.u3_neutrophil(tr + e, tr)["met"], True)
L.insert(0, "Atlas v2.1 planted test of the AMENDED commissioning scorers (chain Stage A21), run once"); L.append(f"PLANTED COMMISSIONING TEST (amended): {'PASSED' if ok else 'FAILED'}")
open(os.path.join(HERE, "planted_commission_v21b_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
