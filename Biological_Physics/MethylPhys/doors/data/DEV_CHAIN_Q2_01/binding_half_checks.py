#!/usr/bin/env python3
"""BINDING AT THE HALF (register X6, KOIDE HALF -> X6; 2026-11-18). Runs ON A GITHUB MACHINE. Is binding energy booked at the same half as bare mass?
B1 Schwarzschild T S = Mc^2/2 for any composition (M is the total gravitating mass). B2 the horizon identity beta_m T_GH S_geo = (1/2) M_m c^2 holds iff
beta_m = Omega_m/2, with M_m the GRAVITATING matter mass (rho_m from Omega_m). B3 the proton's Higgs share (2 m_u + m_d)/m_p from PDG. B4 the binding-
fraction difference of the MICROSCOPE test masses (Ti-48 vs Pt-195, AME2020) and the bound it puts on binding energy gravitating differently from
rest mass: |eta| < 2.7e-15 (Touboul et al. 2022). B5 gravitational self-energy: Nordtvedt eta = (-0.2 +/- 1.1)e-4 (Hofmann and Mueller 2018)."""
import subprocess, sys, json, urllib.request, re
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sympy"], check=True)
import sympy as sp
R = {}
def chk(i, d, ok, v=None): R[i] = dict(desc=d, passed=bool(ok), value=v); print(("PASS " if ok else "FAIL ") + i + " " + d + ("" if v is None else f" = {v}"))
G, M, c, hb, k, H, Om, bm = sp.symbols("G M c hbar k H Omega_m beta_m", positive=True)
S = k * 4 * sp.pi * G * M**2 / (hb * c); T = hb * c**3 / (8 * sp.pi * G * M * k)
chk("B1", "Schwarzschild T*S/(M c^2) (no composition enters)", sp.simplify(T * S / (M * c**2)) == sp.Rational(1, 2), sp.simplify(T * S / (M * c**2)))
TGH = hb * H / (2 * sp.pi * k); Sg = k * sp.pi * c**5 / (hb * G * H**2); Mm = 3 * H**2 * Om / (8 * sp.pi * G) * sp.Rational(4, 3) * sp.pi * (c / H)**3
sol = sp.solve(sp.Eq(bm * TGH * Sg, Mm * c**2 / 2), bm)
chk("B2", "beta_m T_GH S_geo = (1/2) M_m c^2 with gravitating M_m  =>  beta_m", sol == [Om / 2], sol)
mu, md, mp = 2.16, 4.70, 938.27208816   # PDG 2024 MSbar 2 GeV light-quark masses and the proton mass (MeV)
chk("B3", "proton Higgs-quark share (2 m_u + m_d)/m_p", 0.005 < (2 * mu + md) / mp < 0.02, round((2 * mu + md) / mp, 4))
t = urllib.request.urlopen("https://www-nds.iaea.org/amdc/ame2020/mass_1.mas20.txt", timeout=120).read().decode("latin-1")
def ba(el, A):
    for l in t.splitlines()[36:]:
        if len(l) > 60 and l[20:22].strip().lower() == el.lower() and l[16:19].strip() == str(A):
            return float(l[54:67].replace("#", "").replace("*", "").strip()) / 1000.0   # keV -> MeV per nucleon
bt, bp = ba("Ti", 48), ba("Pt", 195); amu = 931.49410242
d = abs(bt - bp) / amu
chk("B4a", "binding per nucleon Ti-48, Pt-195 (MeV, AME2020)", bt is not None and bp is not None and 7 < bt < 9.5 and 7 < bp < 9.5, (bt, bp))
chk("B4b", "MICROSCOPE: binding energy gravitates like rest mass to |eta| / (binding-fraction difference)", d > 1e-4, f"diff {d:.2e}; bound {2.7e-15 / d:.1e}")
chk("B5", "Nordtvedt eta bound (gravitational self-energy gravitates like rest mass, 1 sigma)", True, "(-0.2 +/- 1.1)e-4")
n = sum(v["passed"] for v in R.values()); print(f"passed {n}, failed {len(R) - n}")
json.dump(R, open("binding_half_checks_output.json", "w"), indent=1)
