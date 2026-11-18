#!/usr/bin/env python3
"""Lead's independent recheck (runs ON A GITHUB MACHINE) of the two-step spider's key identities before they go on IAM's web (2026-11-17)."""
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sympy"], check=True)
import sympy as s
G, c, hb, k, M, R, E, H, lam, p, m1, m2, z, bm = s.symbols("G c hbar k M R E H lambda p m1 m2 z beta_m", positive=True)
res = {}
def chk(name, expr):
    ok = s.simplify(expr) == 0; res[name] = ok; print(("PASS " if ok else "FAIL ") + name)
TR = hb * c / (2 * s.pi * k * R)                                  # Unruh / horizon temperature of radius R
Sbek = 2 * s.pi * k * R * E / (hb * c)                            # Bekenstein bound
chk("Bekenstein bound = E/T_R (Landauer form)", Sbek - E / TR)
rs = 2 * G * M / c**2; Sbh = k * c**3 * 4 * s.pi * rs**2 / (4 * G * hb); TH = hb * c**3 / (8 * s.pi * G * M * k)
chk("black hole saturates: S_BH = S_bek(r_s, Mc^2)", Sbh - Sbek.subs({R: rs, E: M * c**2}))
chk("T_R(r_s) = 2 T_H", TR.subs(R, rs) - 2 * TH)
chk("Smarr: T_H S_BH = Mc^2/2", TH * Sbh - M * c**2 / 2)
chk("S_bek/S_BH(E) = R/r_s", Sbek.subs(E, M * c**2) / Sbh - R / rs)
# degree rule: on R = lam E^p, S ~ E^(1+p); (dE/dS) S/E = 1/(1+p)
Sfam = Sbek.subs(R, lam * E**p)
chk("degree rule (dE/dS) S/E = 1/(1+p)", (1 / s.diff(Sfam, E)) * Sfam / E - 1 / (1 + p))
# Hawking merger bound: area ~ M^2 -> M_f >= sqrt(m1^2+m2^2); equal masses released share
share = 1 - s.sqrt(2 * m1**2) / (2 * m1)
chk("equal-mass merger released share <= 1 - 1/sqrt2", share - (1 - 1 / s.sqrt(2)))
# LISA: line-to-line differences sum to zero (Kirchhoff)
a1, a2, a3 = s.symbols("a1 a2 a3"); chk("X + Y + Z = 0", (a2 - a3) + (a3 - a1) + (a1 - a2))
# spin-s pattern on three arms at 120 deg: zero-sequence sum vanishes unless s = 0 mod 3
for sp_ in (0, 1, 2, 3):
    tot = s.nsimplify(sum(s.exp(s.I * sp_ * 2 * s.pi * j / 3) for j in range(3)))
    print(f"spin {sp_}: arm sum = {s.simplify(tot)}"); res[f"spin {sp_} sum"] = str(s.simplify(tot))
# GW flux forms: c^3/(32 pi G) <hdot_ij hdot_ij> with hdot_ij hdot_ij = 2(hp^2 + hx^2)  ->  c^3/(16 pi G)(hp^2 + hx^2)
hp, hx = s.symbols("hp hx")
hij = s.Matrix([[hp, hx, 0], [hx, -hp, 0], [0, 0, 0]])
chk("hdot_ij hdot_ij = 2(hp^2 + hx^2)", sum(hij[i, j]**2 for i in range(3) for j in range(3)) - 2 * (hp**2 + hx**2))
# siren lead: H_m^2 = H^2 + bm e^{-z} H0^2 at z -> 0: d_m/d_gamma -> 1/sqrt(1+bm)
H0 = s.symbols("H0", positive=True)
ratio0 = s.limit(H0 / s.sqrt(H0**2 + bm * s.exp(-z) * H0**2), z, 0)
chk("siren low-z ratio -> 1/sqrt(1+beta_m)", ratio0 - 1 / s.sqrt(1 + bm))
import json as _j; _b = _j.load(open("CANON/iam_canon.json"))["constants"]["beta_m"]["value"]
print(f"::notice::siren low-z ratio at the canon beta_m {_b}: {float((1 / s.sqrt(1 + bm)).subs(bm, _b)):.5f}")
bad = [k_ for k_, v in res.items() if v is False]
print(f"::notice::SPIDER RECHECK: {sum(v is True for v in res.values())} pass, {len(bad)} fail {bad}")
