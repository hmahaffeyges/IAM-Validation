#!/usr/bin/env python3
"""minicosmos_levers.py: the cosmos lever card (2026-11-15; register R17; exploration). The author's grid method applied to IAM's cosmology:
every lever carries its value, its origin, its source and its VALID RANGE (the operating window a substituted or fitted value was set in);
the core's analyses run on it (simcore: op = power flow, sensitivity = sensitivity factors, contingency = contingency analysis).

Readings (each from minicosmos_loop.py, the book's nine-step chain closed, record closure B = the book's own numerics, feedback on):
  H0_matter   = H0_light sqrt(1 + beta_m E(1))       (book eq:Hm; E(1) = 1)
  growth_z0   = D(1) with the record / D(1) without   (the record's suppression of growth today; book 0.9922)
  c_direct    = ln E(50)/E(1)                        (the final size of the record; canon c = 1 is a fit)
  beta_fit    = the book's fitted coefficient over a = 0.15-2
OBSERVABILITY: for each lever, the 1-sigma range a measurement allows, = (measurement sigma / reading) / |d ln reading / d ln lever|, through
H0_matter (local rates read as matter: SH0ES 73.04 +/- 1.04 and TRGB 70.39 +/- 1.94, inverse-variance 72.45 +/- 0.92; book p2_04) and growth_z0.
A lever no measured reading moves is UNOBSERVABLE with current data (grid: an unmetered bus with no redundant telemetry around it).
Writes xout/minicosmos_levers_summary.txt when run on GitHub."""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import simcore as Sc
import minicosmos_loop as ML

def card():
    P = Sc.Param
    return Sc.ModelCard("cosmos levers", "cosmos", {
        "Omega_m":  P(0.3153, "fraction", "measured", "Planck 2018 (Aghanim et al. 2020, Table 2)", note="valid: all epochs"),
        "beta_m":   P(0.15765, "fraction", "derived", "Omega_m/2 from the virial half (book p2_04); free fit 0.155 +/- 0.029 agrees", origin="IAM",
                      note="valid: today's amplitude; pinned by the local Hubble rates"),
        "n":        P(3.5, "exponent", "derived", "rate exponent for the 1/a shape (book sec:th:exponent); halo n_eff runs 5.3 -> 2.0", origin="IAM",
                      note="valid: matter era; the halo statistics give a running value"),
        "H0_light": P(67.16, "km/s/Mpc", "measured", "Level 2 Planck chain (book p2_04)", note="valid: photon sector"),
    })

def model(cd):
    ML.set_background(om0=cd.v("Omega_m"), beta_m=cd.v("beta_m"))
    out = ML.loop(f"D^{cd.v('n')}", "B", feedback=True)
    D_off, _ = ML.growth(np.zeros_like(ML.A), True)
    r = ML.report(out)
    return {"H0_matter": cd.v("H0_light") * math.sqrt(1 + cd.v("beta_m") * out["E"][ML.I1]),
            "growth_z0": (out["D"][ML.I1] / out["D"][0]) / (D_off[ML.I1] / D_off[0]),
            "c_direct": r["c_direct"], "beta_fit": r["beta"]}

MEAS = {"H0_matter": (72.45, 0.92, "SH0ES + TRGB read as matter-sector rates (book p2_04)")}

if __name__ == "__main__":
    cd = card(); readings = ("H0_matter", "growth_z0", "c_direct", "beta_fit")
    base, S = Sc.sensitivity(model, cd, readings, rel=0.02)
    L = ["COSMOS LEVER CARD (exploration)", "  lever      value      origin       label     valid range / source"]
    for k, p in cd.items(): L.append(f"  {k:10s} {p.value:<10g} {p.origin:12s} {p.label:9s} {p.note} | {p.source}")
    L += ["", "OPERATING POINT (power flow): " + ", ".join(f"{k} {float(v):.4f}" for k, v in base.items())]
    L += ["", "SENSITIVITY FACTORS d ln(reading)/d ln(lever)", "  lever      " + " ".join(f"{r:>11s}" for r in readings)]
    for k, row in S.items(): L.append(f"  {k:10s} " + " ".join(f"{(row[r] if row[r] is not None else float('nan')):11.4f}" for r in readings))
    L += ["", "OBSERVABILITY (1-sigma range on each lever allowed by each measurement; 'unobservable' if the reading does not move)"]
    for m, (val, sig, src) in MEAS.items():
        L.append(f"  through {m} = {val} +/- {sig} ({src}); model gives {float(base[m]):.2f}, pull {(float(base[m]) - val) / sig:+.2f} sigma")
        for k, row in S.items():
            s_ = row[m]
            if s_ is None or abs(s_) < 1e-6: L.append(f"    {k:10s} unobservable through {m}")
            else: L.append(f"    {k:10s} +/- {100 * (sig / val) / abs(s_):.1f} % (+/- {cd.v(k) * (sig / val) / abs(s_):.4g})")
    L.append("  c_direct: no current measurement reads it (the far-future record); it is pinned only by the model, so it stays OPEN in the card.")
    cont = Sc.contingency(model, cd, {"beta_m at the free fit (0.155)": {"beta_m": 0.155}, "n = 3": {"n": 3.0}, "n = 4": {"n": 4.0}})
    L += ["", "CONTINGENCY"] + [f"  {name:32s} " + ", ".join(f"{k} {float(v):.4f}" for k, v in r.items()) for name, r in cont.items()]
    txt = "\n".join(L); print(txt)
    if os.environ.get("GITHUB_ACTIONS"):
        os.makedirs("xout", exist_ok=True); open("xout/minicosmos_levers_summary.txt", "w").write(txt + "\n")
