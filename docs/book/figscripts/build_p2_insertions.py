"""Builds docs/book/figures_insertions_p2.json: the figure and table blocks for Part 2 chapters p2_05, p2_10, p2_12, p2_13, p2_15-p2_22
(TODO 9.2: at least two figures and one table per chapter). Every number in the tables is computed here, with the equations of the
verification scripts named in each block, or read from the chain files with _chains.py. Run from any directory after the fig_p2_* scripts:
    python docs/book/figscripts/build_p2_insertions.py
Each entry: {file, anchor, latex}; 'anchor' is one full line copied from the current chapter file, unique in it; 'latex' goes after it.
"""
import sys, json, pathlib, subprocess, re; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, scipy.constants as C
from scipy.integrate import quad
from scipy.optimize import brentq
import _bookstyle as S
import _cosmo as K
import _chains as Ch

OUT = []
def add(file, anchor, latex):
    OUT.append(dict(file=f"part2/{file}", anchor=anchor, latex=latex.strip("\n")))
def fig(name, width, short, caption, label):
    return (f"\\begin{{figure}}[htbp]\\centering\n\\includegraphics[width={width}]{{figures/part2/{name}.pdf}}\n"
            f"\\caption[{short}]{{{caption}}}\\label{{{label}}}\n\\end{{figure}}")
def pct(x, d=2): return f"{x:+.{d}f}"   # used inside math only

# ---------------------------------------------------------------- p2_05
c, H0, Om, Or = 299792.458, 67.4, 0.315, 9.24e-5; OL = 1 - Om - Or; obs, err, rs = 0.0104110, 3.1e-6, 144.43
Hg = lambda a, b: H0 * np.sqrt(Om * a**-3 + Or * a**-4 + OL + b * np.exp(1 - 1 / a))
theta = lambda b: rs / (quad(lambda z: 1 / Hg(1 / (1 + z), b), 0, 1090, limit=500, epsabs=0, epsrel=1e-12)[0] * c)
t0 = theta(0.0); c0 = ((t0 - obs) / err)**2
bg95 = brentq(lambda b: ((theta(b) - obs) / err)**2 - c0 - 4, 1e-6, 0.05)
bm = 0.3153 / 2; sm = (theta(bm) - t0) / err; pm = 100 * (theta(bm) / t0 - 1)
mu0 = K.mu(1.0) - 1
add("p2_05_dual_sector_note.tex", "on real data with two rulers (Chapter~\\ref{ch:sectortension}).",
    fig("fig_beta_gamma", "\\textwidth", "The photon coupling tested on the CMB acoustic scale",
        f"The photon coupling tested on the CMB acoustic scale. (a) $\\Delta\\chi^2$ of $\\theta_s$ against a photon coupling $\\beta_\\gamma\\ge0$ "
        f"in $H_\\gamma^2=H_0^2[\\Omega_ma^{{-3}}+\\Omega_ra^{{-4}}+\\Omega_\\Lambda+\\beta_\\gamma E(a)]$, sound horizon fixed at 144.43\\,Mpc, "
        f"$\\theta_s=0.0104110\\pm0.0000031$~\\cite{{Planck2018VI}}: $\\beta_\\gamma<{bg95:.4f}$ at 95\\,\\%, so $\\beta_\\gamma/\\beta_m<{bg95/bm:.3f}$. "
        f"(b) The full matter coupling $\\beta_m=\\Omega_m/2$ applied to photon paths moves $\\theta_s$ by ${pm:+.2f}\\,\\%$, ${sm:.0f}\\sigma$ at Planck precision "
        f"(band: $\\pm2\\sigma$). Method of \\texttt{{docs/verification/scripts/\\allowbreak{{}}verify\\_beta\\_gamma.py}}. \\calc", "fig:beta_gamma"))
add("p2_05_dual_sector_note.tex", "Euclid, DESI Year~5 and CMB-S4 test each of these.",
    "\\begin{table}[htbp]\\centering\\small\n\\caption[What would count against the dual-sector structure]{What would count against the dual-sector structure. "
    "Present values are from the chains and the acoustic-scale fit of this chapter; the survey forecast is in Section~\\ref{sec:sp_euclid}.}"
    "\\label{tab:dsnote_falsifiers}\n"
    "\\begin{tabularx}{\\linewidth}{>{\\raggedright\\arraybackslash}p{0.19\\linewidth}>{\\raggedright\\arraybackslash}p{0.2\\linewidth}"
    ">{\\raggedright\\arraybackslash}X>{\\raggedright\\arraybackslash}p{0.17\\linewidth}l}\\toprule\n"
    "Statement & Value & Would count against it & Test & Label\\\\\\midrule\n"
    "Light is not coupled (lensing) & $\\Sigma=1$ at every $z$ & $\\Sigma\\neq1$ detected & Euclid tomography & \\prediction\\\\\n"
    f"Light is not coupled (acoustic scale) & $\\beta_\\gamma=0$; now $\\beta_\\gamma<{bg95:.4f}$ (95\\,\\%) & $\\beta_\\gamma>0$ detected & CMB-S4 & \\calc\\\\\n"
    "Virial amplitude & $\\beta_m/\\Omega_m=1/2$ & $\\Omega_m$ revised while growth still needs the old $\\beta_m$ & DESI Year~5, Euclid & \\prediction\\\\\n"
    f"Growth coupling today & $\\mu_0={mu0:.3f}$ & $\\mu_0$ consistent with 0 and excluding ${mu0:.3f}$ & Euclid, DESI (Section~\\ref{sec:sp_euclid}) & \\prediction\\\\\n"
    "Redshift dependence only & $\\mu(a)$, $\\Sigma(a)$ & scale-dependent $\\mu$ or $\\Sigma$ & Euclid, DESI & \\prediction\\\\\\bottomrule\n"
    "\\end{tabularx}\n\\end{table}")

# ---------------------------------------------------------------- p2_10
add("p2_10_dual_sector_validation.tex", "$H_0=73.04$ is set by the Cepheid calibration of $M$, which is the measurement the dual-sector picture assigns to the matter sector.",
    fig("fig_sn_h0_flat", "\\textwidth", "Pantheon+: the Hubble-flow shape and the flat $H_0$ direction",
        "Pantheon+SH0ES~\\cite{Brout2022,Scolnic2022}. (a) Hubble-flow residuals ($z_{\\rm HD}>0.01$, 1590 supernovae) from $\\Lambda$CDM at "
        "$\\Omega_m=0.315$, the offset $M$ fitted with the full statistical and systematic covariance; points are inverse-variance means of the "
        "diagonal errors in bins of $\\log z$. Solid: $\\beta_m=0.15765$ put into the distances ($\\Delta\\chi^2=+23.6$); dash-dotted: the best "
        "$\\beta=-0.035$; each curve carries its own fitted offset. (b) Minimum $\\chi^2$ at fixed $H_0$ with $M$, $\\Omega_m$ and $\\beta$ free "
        "(diagonal errors, $0.01<z_{\\rm CMB}<2.5$, 1588 supernovae): flat at 721.12, with $M-5\\log_{10}H_0=-28.935$ at every $H_0$. The "
        "magnitudes alone do not choose between 67.4 and 73.04. Method of \\texttt{docs/verification/scripts/\\allowbreak{}verify\\_dual\\_sector\\_validation.py}. \\observed\\ \\calc",
        "fig:sn_h0_flat"))

# ---------------------------------------------------------------- p2_12
hbar, cc_, G = C.hbar, C.c, C.G; Mpc = 3.0857e22
lP = np.sqrt(hbar * G / cc_**3); EP = np.sqrt(hbar * cc_**5 / G)
H0s = 67.4e3 / Mpc; lH = cc_ / H0s; Ob, Om3, OL3 = 0.0493, 0.3153, 0.6846
rvac = EP**4 / (hbar * cc_)**3; rL = OL3 * 3 * H0s**2 / (8 * np.pi * G) * cc_**2; obsr = rL / rvac
base = 2 / np.pi * (lP / lH)**2 * Ob / Om3; corr = base * np.sqrt(OL3); pcl = np.log(obsr / base) / np.log(OL3)
vc = subprocess.run([sys.executable, str(S.REPO / "docs/verification/scripts/verify_cc_and_baryon.py")], cwd=S.REPO, capture_output=True, text=True).stdout
heat = float(re.search(r"accumulated virial heat of baryons / rho_L c\^2 = ([0-9.e+-]+)", vc).group(1))
def man(x, d=3):
    e = int(np.floor(np.log10(abs(x)))); return f"{x/10**e:.{d}f}\\times10^{{{e}}}"
chains_cc = {}
for nm, fl in (("18th chain (CMB only)", [Ch.MG + "iam_baryon_test.1.txt"]), ("$\\Lambda$CDM, Planck", [Ch.MG + "lcdm_baseline.1.txt"]),
               ("$\\Lambda$CDM, Planck + BAO", [Ch.MG + "planck_bao_lcdm_baseline.1.txt"]), ("$\\Lambda$CDM, Planck + Pantheon+", [Ch.MG + "planck_pantheon_lcdm_baseline.1.txt"])):
    X = Ch.load(*fl); w = X.weight.values; ob = X.ombh2.values; h = X.H0.values / 100; OLx = X.omegal.values
    rr = ob / h**2 / (1 - OLx) / (3 / 16 * np.sqrt(OLx))
    chains_cc[nm] = dict(rows=len(X), ob=Ch.wmean_sd(ob, w), r=Ch.wmean_sd(rr, w))
lc = [v["r"][0] for k, v in chains_cc.items() if "CDM" in k]
add("p2_12_lambda.tex", "$0.79\\,\\%$ above the measured value. The exponent of $\\Omega_\\Lambda$ that would close Eq.~\\eqref{eq:base} exactly is $0.521$.",
    fig("fig_cc_factors", "\\textwidth", "The factors of the cosmological-constant expression",
        f"The factors of the cosmological-constant expression. (a) The expression divided by the measured $\\rho_\\Lambda/\\rho_{{\\rm vac}}={man(obsr,3)}$ as each factor "
        f"is applied: $(l_P/l_H)^2$ alone is {(lP/lH)**2/obsr:.2f} times too large, $\\Omega_b/\\Omega_m$ brings it to {(lP/lH)**2*Ob/Om3/obsr:.3f}, $2/\\pi$ to "
        f"{base/obsr:.3f} and $\\sqrt{{\\Omega_\\Lambda}}$ to {corr/obsr:.4f}. The identity factor $3\\Omega_\\Lambda/8\\pi$ closes it exactly. (b) "
        f"$(2/\\pi)(l_P/l_H)^2(\\Omega_b/\\Omega_m)\\Omega_\\Lambda^p$ against $p$: $p={pcl:.3f}$ closes it, $p=1/2$ leaves ${100*(corr/obsr-1):+.2f}\\,\\%$. "
        "Planck 2018 inputs~\\cite{Planck2018VI}; CODATA constants. None of the three factors is derived. \\calc\\ \\openprob", "fig:cc_factors"))
add("p2_12_lambda.tex", "term, which is $\\Lambda$CDM before structure and differs only in late-time growth (Chapters~\\ref{ch:latetime}--\\ref{ch:level2}).",
    "\\begin{table}[htbp]\\centering\\small\n\\caption[The cosmological-constant numbers]{The cosmological-constant numbers of this chapter (Planck 2018: $H_0=67.4$, "
    "$\\Omega_b=0.0493$, $\\Omega_m=0.3153$, $\\Omega_\\Lambda=0.6846$; chains with 30\\,\\% burn-in, weighted). Recomputed with "
    "\\texttt{docs/verification/scripts/\\allowbreak{}verify\\_cc\\_and\\_baryon.py}.}\\label{tab:lambda_numbers}\n"
    "\\begin{tabularx}{\\linewidth}{>{\\raggedright\\arraybackslash}Xll}\\toprule\nQuantity & Value & Label\\\\\\midrule\n"
    f"$\\rho_{{\\rm vac}}=E_P^4/(\\hbar c)^3$ & ${man(rvac)}$\\,J\\,m$^{{-3}}$ & \\calc\\\\\n"
    f"$\\rho_\\Lambda=\\Omega_\\Lambda\\rho_cc^2$ & ${man(rL)}$\\,J\\,m$^{{-3}}$ & \\observed\\\\\n"
    f"$\\rho_\\Lambda/\\rho_{{\\rm vac}}$, and the identity $(3\\Omega_\\Lambda/8\\pi)(l_P/l_H)^2$ & ${man(obsr)}$ (both) & \\derived\\\\\n"
    f"$(2/\\pi)(l_P/l_H)^2\\,\\Omega_b/\\Omega_m$ & ${man(base)}$ ($\\times{base/obsr:.3f}$) & \\calc\\\\\n"
    f"$\\times\\sqrt{{\\Omega_\\Lambda}}$ & ${man(corr)}$ (${pct(100*(corr/obsr-1))}\\,\\%$) & \\calc\\\\\n"
    f"Exponent $p$ of $\\Omega_\\Lambda$ that closes the expression & ${pcl:.3f}$ & \\calc\\\\\n"
    f"$\\Omega_b/\\Omega_m$ against $(3/16)\\sqrt{{\\Omega_\\Lambda}}$, Planck values & ${Ob/Om3:.4f}$ against ${3/16*np.sqrt(OL3):.4f}$ & \\observed\\\\\n"
    f"Ratio of the two sides, 18th chain (CMB only) & ${chains_cc['18th chain (CMB only)']['r'][0]:.4f}\\pm{chains_cc['18th chain (CMB only)']['r'][1]:.4f}$ & \\measured\\\\\n"
    f"Ratio of the two sides, three $\\Lambda$CDM chains & ${min(lc):.4f}$--${max(lc):.4f}$ & \\measured\\\\\n"
    f"Heat radiated by baryonic virialisation, summed to today, over $\\rho_\\Lambda c^2$ & ${man(heat,2)}$ & \\calc\\\\\n"
    "A derivation of $3/16$ from physics & --- & \\openprob\\\\\\bottomrule\n\\end{tabularx}\n\\end{table}")

# ---------------------------------------------------------------- p2_13
omh2 = 0.3153 * 0.6736**2; H67 = 67.36e3 / Mpc; o = 0.6847 * 3 * H67**2 / (8 * np.pi * G) * cc_**2 / rvac
pref = 2 / np.pi * (lP / (cc_ / H67))**2 / omh2
eta3 = 273.9 * o / pref; eta5 = 273.9 * o / (pref * np.sqrt(0.6847))
rngs = {"18th chain (CMB only)": "0.010--0.040"}
rowsT = []
for nm, v in chains_cc.items():
    nrow = f"{v['rows']:,}".replace(",", "{,}")
    rowsT.append(f"{nm} & {rngs.get(nm, '0.020--0.025')} & {nrow} & ${v['ob'][0]:.5f}\\pm{v['ob'][1]:.5f}$ & ${273.9*v['ob'][0]:.3f}\\pm{273.9*v['ob'][1]:.3f}$ & ${v['r'][0]:.4f}\\pm{v['r'][1]:.4f}$ & \\measured\\\\")
add("p2_13_baryon.tex", "relation of Chapter~\\ref{ch:lambda} evaluated on a CMB-only posterior: $0.7\\sigma$.",
    fig("fig_baryon_posterior", "\\textwidth", "The 18th chain against the $\\Lambda$CDM chains",
        f"(a) The $\\Omega_bh^2$ posterior of the 18th chain, sampled over the flat range 0.010--0.040 (light band; the darker band is the "
        f"0.020--0.025 of the other runs), and of the three $\\Lambda$CDM chains; inset: the peaks enlarged. The acoustic peaks fix "
        f"$\\Omega_bh^2={chains_cc['18th chain (CMB only)']['ob'][0]:.5f}\\pm{chains_cc['18th chain (CMB only)']['ob'][1]:.5f}$ whatever the range. "
        f"(b) The posterior of $(\\Omega_b/\\Omega_m)/[(3/16)\\sqrt{{\\Omega_\\Lambda}}]$ on the same chains. Chain files in "
        "\\texttt{mgcamb\\_validation/\\allowbreak{}chains}, 30\\,\\% burn-in, weighted. \\measured", "fig:baryon_posterior"))
add("p2_13_baryon.tex", "The $\\Lambda$CDM chains in the same repository return the same $\\eta$: $6.118$ (Planck), $6.137$ (Planck + BAO), $6.117$ (Planck + Pantheon+).",
    "\\begin{table}[htbp]\\centering\\small\n\\caption[The baryon density on every chain]{The baryon density on every chain, $\\eta=273.9\\times10^{-10}\\,\\Omega_bh^2$ "
    "(30\\,\\% burn-in, weighted mean and standard deviation; rows after burn-in), and the cosmological-constant expressions inverted for "
    "$\\Omega_bh^2$ at the Planck $\\Omega_mh^2=0.1430$.}\\label{tab:baryon_chains}\n"
    "\\begin{tabular}{lcrccll}\\toprule\nSource & $\\Omega_bh^2$ range & rows & $\\Omega_bh^2$ & $10^{10}\\eta$ & ratio to $(3/16)\\sqrt{\\Omega_\\Lambda}$ & Label\\\\\\midrule\n"
    + "\n".join(rowsT) + "\n"
    f"Expression with $2/\\pi$, without $\\sqrt{{\\Omega_\\Lambda}}$ & --- & --- & ${eta3/273.9:.5f}$ & ${eta3:.3f}$ & --- & \\calc\\\\\n"
    f"Expression with $\\sqrt{{\\Omega_\\Lambda}}$ & --- & --- & ${eta5/273.9:.5f}$ & ${eta5:.3f}$ & --- & \\calc\\\\\\bottomrule\n"
    "\\end{tabular}\n\\end{table}")

# ---------------------------------------------------------------- p2_15
al, me = C.alpha, C.m_e; mP = np.sqrt(hbar * cc_ / G)
mfp = lambda H, pf=(2 * np.pi)**-0.1: pf * (hbar * (H * 1e3 / 3.0856775814913673e22) * np.log(2) * mP**1.5 / (al**2.5 * cc_**2))**0.4
dev = {H: 1e6 * (mfp(H) / me - 1) for H in (67.36, 67.4, 73.04)}
mm = np.array([0.51099895, 105.6583755, 1776.86]); s = np.sqrt(mm); x = s.sum() / 3; Q = mm.sum() / s.sum()**2
dl = brentq(lambda t: (1 + np.sqrt(2) * np.cos(t)) - s[2] / x, 0, 1)
rec = sorted((x * (1 + np.sqrt(2) * np.cos(dl + 2 * np.pi * k / 3)))**2 for k in range(3))
d0 = sorted((x * (1 + np.sqrt(2) * np.cos(2 * np.pi * k / 3)))**2 for k in range(3))
dg = np.linspace(0, 2 * np.pi, 200001)
frac = {n: np.mean((1 + np.sqrt(2) * np.cos(dg[:, None] + 2 * np.pi * np.arange(n) / n)).min(1) > 1e-9) for n in (2, 3, 4)}
add("p2_15_particle_masses.tex", "numerically.",
    "\\begin{table}[htbp]\\centering\\small\n\\caption[The electron fixed point and the charged-lepton pattern]{The electron fixed point and the charged-lepton "
    "pattern. CODATA 2018 constants; pole masses from PDG~\\cite{PDG2022}. Recomputed with \\texttt{verify\\_electron\\_mass.py} and "
    "\\texttt{verify\\_koide.py} (\\texttt{docs/verification/scripts/}).}\\label{tab:particle_numbers}\n"
    "\\begin{tabularx}{\\linewidth}{>{\\raggedright\\arraybackslash}Xll}\\toprule\nQuantity & Value & Label\\\\\\midrule\n"
    f"Bracket without prefactor, in units of $m_e$ ($H_0=67.4$) & ${mfp(67.4,1.0)/me:.4f}$ & \\calc\\\\\n"
    f"Prefactor needed; $(2\\pi)^{{-1/10}}$ & ${me/mfp(67.4,1.0):.6f}$; ${(2*np.pi)**-0.1:.6f}$ & \\calc\\\\\n"
    f"Fixed point against $m_e$: $H_0=67.36$, $67.4$, $73.04$ & ${dev[67.36]:+.0f}$\\,ppm, ${dev[67.4]:+.1f}$\\,ppm, ${dev[73.04]/1e4:+.2f}\\,\\%$ & \\calc\\\\\n"
    f"Spread in $m_e$ from $\\sigma(H_0)=0.54$ alone & $\\pm{0.4*0.54/67.36*100:.2f}\\,\\%$ & \\calc\\\\\n"
    f"Koide $Q$ of the pole masses & ${Q:.8f}$ & \\observed\\\\\n"
    f"Scale $x^2$ and offset $\\delta$ & ${x**2:.2f}$\\,MeV, ${dl:.5f}$\\,rad ($2/9={2/9:.5f}$) & \\calc\\\\\n"
    f"Masses from $(x,\\delta)$ & {rec[0]:.3f}, {rec[1]:.2f}, {rec[2]:.2f}\\,MeV & \\calc\\\\\n"
    f"Masses with $\\delta=0$ & {d0[0]:.1f}, {d0[1]:.1f}, {d0[2]:.1f}\\,MeV & \\calc\\\\\n"
    f"Offsets with all masses positive ($y/x=\\sqrt2$): $n=2$, 3, 4 & {frac[2]:.2f}, {frac[3]:.2f}, {frac[4]:.2f} & \\calc\\\\\\bottomrule\n"
    "\\end{tabularx}\n\\end{table}")

# ---------------------------------------------------------------- p2_16
add("p2_16_survey_predictions.tex", "rescaling, and it is gone by $z\\approx2$.",
    fig("fig_survey_ramp", "\\textwidth", "The growth-rate ramp and what light sees",
        f"(a) The coupling $1-\\mu$, the $f\\sigma_8$ deficit and the growth deficit $|\\Delta D/D|$ against redshift, from the linear growth equation with "
        f"the same early amplitude as $\\Lambda$CDM ($\\Omega_m=0.3153$, $\\beta_m=\\Omega_m/2$, $\\Sigma=1$). Dotted lines: 10, 50 and 90\\,\\% of today's "
        f"$1-\\mu$ reached. (b) The $E_G$ change (${pct(100*(K.f(K.LCDM,1/1.3)/K.f(K.IAM,1/1.3)-1),1)}\\,\\%$ at $z=0.3$), the potential change "
        "$\\Delta\\Phi/\\Phi=\\Delta D/D$ and the ISW source $(1-f)D/a$. Equations of \\texttt{docs/verification/scripts/\\allowbreak{}verify\\_obs\\_chapters.py}. \\calc",
        "fig:survey_ramp"))
add("p2_16_survey_predictions.tex", "($73.04\\pm1.04$~\\cite{Riess2022}) is $0.75\\sigma$ from $72.26$.",
    fig("fig_survey_precision", "0.6\\textwidth", "The precision the siren test needs",
        "The precision the siren test needs. "
        f"Separation of the matter-sector rate 72.26 from the photon-sector 67.16 by a siren population against its $\\sigma(H_0)$; "
        f"$3\\sigma$ needs $\\sigma(H_0)\\le{(72.26-67.16)/3:.2f}$. Square: GW170817 alone ($70.0^{{+12.0}}_{{-8.0}}$~\\cite{{Abbott2017Siren}}, mean half-width 10). \\prediction\\ \\calc",
        "fig:survey_precision").replace("=-0.136", "=-0.136"))

# ---------------------------------------------------------------- p2_17
L1m = [Ch.wmean_sd(X.sigma8.values, X.weight.values)[0] for X in (Ch.load(*Ch.L1["Planck"][0]), Ch.load(*Ch.L1["Planck"][1]))]
L2m = [Ch.wmean_sd(X.sigma8.values, X.weight.values)[0] for X in (Ch.load(*Ch.L2["C"]), Ch.load(*Ch.L2["A"]))]
R = lambda z: 1 / K.mu(1 / (1 + z))
add("p2_17_lensing_dynamics.tex", "depends on the cluster's redshift only, not on its mass, radius or dynamical state. \\calc",
    fig("fig_lensdyn_forms", "\\textwidth", "Lensing mass over dynamical mass in the two forms; $\\sigma_8$ in the chains",
        f"(a) $M_{{\\rm lens}}/M_{{\\rm dyn}}$ in the two implementation forms: $1/\\mu(z)$ in the Level~1 effective-coupling form ({R(0):.3f} today; "
        f"{R(0.2):.3f}, {R(0.5):.3f}, {R(1.0):.3f} and {R(2.0):.3f} at the test redshifts) and 1 in the Level~2 form. Shaded: the quasi-static "
        f"$f(R)$ range $3/4\\le M_{{\\rm lens}}/M_{{\\rm dyn}}\\le1$ ($1\\le\\mu\\le4/3$, $\\Sigma=1$)~\\cite{{PogosianSilvestri2016}}. \\calc\\ "
        f"(b) $\\sigma_8$ posteriors: Level~1 (Planck, MGCAMB) {L1m[0]:.4f} for $\\Lambda$CDM and {L1m[1]:.4f} with the informational term "
        f"(${pct(100*(L1m[1]/L1m[0]-1))}\\,\\%$); Level~2 (modified CAMB) {L2m[0]:.4f} and {L2m[1]:.4f} (${pct(100*(L2m[1]/L2m[0]-1))}\\,\\%$). Chain files, "
        "30\\,\\% burn-in, weighted. \\measured", "fig:lensdyn_forms"))

# ---------------------------------------------------------------- p2_18
Cn = lambda z: 1 + 0.20 * (1 + z)**0.2; h = 1e-4; dR = lambda z: (R(z + h) - R(z - h)) / (2 * h)
add("p2_18_three_way_clusters.tex", "slope in a sample measured one way throughout, not the size of the offset, which hydrostatic bias alone can supply. \\calc",
    fig("fig_threeway_slope", "\\textwidth", "The gravitational and non-thermal parts of the lensing-to-hydrostatic ratio",
        f"Level~1 form. (a) The gravitational part $R=1/\\mu$, the non-thermal factor $C_{{\\rm NT}}=1+0.20(1+z)^{{0.2}}$ (illustrative form; normalisation and "
        f"index still to be traced to simulations) and their product, with the four bin centres. (b) Their redshift slopes: $dR/dz={dR(0.3):.3f}$ "
        f"at $z=0.3$ against $dC_{{\\rm NT}}/dz=+{0.04*(1.65)**-0.8:.3f}$ to $+{0.04*(1.15)**-0.8:.3f}$ over the bin centres; the product falls. In the Level~2 form $R=1$. "
        "Equations of \\texttt{docs/verification/scripts/\\allowbreak{}verify\\_obs\\_chapters.py}, section E. \\calc\\ \\openprob", "fig:threeway_slope"))

zb5 = np.array([0.2, 0.5, 0.8, 1.2, 1.8]); Rb5 = R(zb5); s3_17 = np.sqrt(np.sum((Rb5 - Rb5.mean())**2) / 9)
add("p2_17_lensing_dynamics.tex", "with no free parameter, and a two-parameter power law $1+A(1+z)^n$. The discriminant is the shape in redshift, not the existence of an offset.",
    fig("fig_lensdyn_test", "\\textwidth", "What the redshift-shape test needs",
        f"Level~1 form. (a) The slope of $M_{{\\rm lens}}/M_{{\\rm dyn}}=1/\\mu(z)$: ${dR(0.0+1e-4):.2f}$ at $z=0$ and ${dR(0.3):.2f}$ at $z=0.3$, against a "
        f"hydrostatic bias whose slope is zero or positive. (b) The significance with which five redshift bins (centres 0.2, 0.5, 0.8, 1.2 and 1.8, "
        f"one mass method throughout, equal errors) separate the parameter-free curve from the best constant: $3\\sigma$ needs an error of "
        f"{100*s3_17:.1f}\\,\\% per bin on the ratio. \\calc\\ \\prediction", "fig:lensdyn_test"))
zc4 = np.array([0.15, 0.25, 0.40, 0.65]); P4 = R(zc4) * Cn(zc4); sl4 = np.polyfit(zc4, P4, 1)[0]; Sxx4 = np.sum((zc4 - zc4.mean())**2)
s3_18 = abs(sl4) * np.sqrt(Sxx4) / (3 * P4.mean())
add("p2_18_three_way_clusters.tex", "At the redshifts of eROSITA samples ($z\\approx0.2$--$0.4$) the gravitational part of the ratio is $1.07$--$1.11$.",
    fig("fig_threeway_estimators", "\\textwidth", "Three cluster mass estimators and the slope test",
        f"(a) The three estimators over the true mass in the Level~1 form: the X-ray hydrostatic mass follows $\\mu(z)$, the SZ mass follows it through the "
        f"$Y$--$M$ calibration, and the weak-lensing mass follows $\\Sigma=1$; in the Level~2 form all three are 1. \\calc\\ (b) The significance of a "
        f"non-zero slope of $R\\times C_{{\\rm NT}}$ fitted as a straight line over the four bin centres (slope ${sl4:.3f}$), against the fractional error "
        f"per bin (equal errors): $3\\sigma$ needs {100*s3_18:.1f}\\,\\% per bin. $C_{{\\rm NT}}$ is the illustrative form. \\calc\\ \\openprob", "fig:threeway_estimators"))

# ---------------------------------------------------------------- p2_19
add("p2_19_missing_satellites.tex", "the correct source is still to be found. \\openprob",
    fig("fig_sat_mechanisms", "\\textwidth", "The two mechanisms in numbers",
        f"(a) Mechanism~A: the Press--Schechter abundance change at fixed mass, $|\\Delta\\ln n|=|\\nu^2-1|\\,\\epsilon$, for the growth change "
        f"$\\epsilon={K.amp_deficit(0.0):.2f}\\,\\%$ of the exact coupling~\\cite{{PressSchechter1974}}, against the order-of-magnitude deficit "
        "($\\ln10$); satellite halos lie at $\\nu<1$ (shaded). (b) Mechanism~B: the minimum mass $M_{\\min}=4\\Omega_m\\sigma^3/(GH_0)$ and the "
        f"dynamical-time form $\\sqrt{{6/\\pi}}\\,\\sigma^3/(GH_0)$ ($\\Omega_m=0.3153$, $H_0=67.36$); at $4$\\,km\\,s$^{{-1}}$, $10^{{8.44}}\\,M_\\odot$. \\calc",
        "fig:sat_mechanisms"))
add("p2_19_missing_satellites.tex", "dispersion or an upper limit.",
    fig("fig_sat_census", "\\textwidth", "Velocity dispersions of the Milky Way satellites against the dispersal floor",
        "Velocity dispersions of the 54 Milky Way satellites in the Local Volume Database~\\cite{Pace2025LVDB} that have a measured value "
        "(circles, 68\\,\\% intervals) or an upper limit (triangles), ordered by value; 25 lie below the 4\\,km\\,s$^{-1}$ floor of "
        "Mechanism~B (orange). File \\texttt{docs/\\allowbreak{}verification/\\allowbreak{}observations/\\allowbreak{}data/\\allowbreak{}lvdb\\_dwarf\\_mw.csv}. \\observed",
        "fig:sat_census"))

# ---------------------------------------------------------------- p2_20
add("p2_20_wz_far_future.tex", "$E\\to e$, it reaches $0.626\\,\\rho_\\Lambda$.",
    fig("fig_wz_history", "\\textwidth", "The equation of state of the record and its weight through time",
        "(a) $w_{\\rm info}(z)=-1-(1+z)/3$, which tends to $-1$ as $a\\to\\infty$. (b) The weight of the informational density, "
        "$\\rho_{\\rm info}/\\rho_\\Lambda=\\beta_mE(a)/\\Omega_\\Lambda$ and its share of the vacuum-like total, from $a=0.2$ to $a=100$: "
        f"{0.3153/2/0.685*np.exp(1-4):.3f} at $z=3$, {0.3153/2/0.685:.3f} today, {0.3153/2*np.e/0.685:.3f} at saturation. "
        "$H_0=67.4$, $\\Omega_m=0.315$, $\\beta_m=0.3153/2$. \\calc", "fig:wz_history"))
add("p2_20_wz_far_future.tex", "$E(1)=1$ because $1-1/a=0$ at $a=1$; it is today's normalisation, not a choice of reference epoch.",
    fig("fig_wz_cpl_clocks", "\\textwidth", "The CPL image of the record term, and its writing rate in three clocks",
        "(a) The CPL image $(w_0,w_a)=(-\\tfrac43,-\\tfrac13)$ of $w_{\\rm info}$ beside the DESI DR2 fits with the CMB and Pantheon+, Union3 "
        "or DES Y5 supernovae~\\cite{DESI2025} (1$\\sigma$ on each axis) \\observed. The fits use light-ruler distances, on which the term predicts "
        "$w=-1$; the comparison is not a test. (b) The writing rate of $E(a)$ per unit $a$ (peak at $a=0.5$), per e-fold (peak today) and per unit "
        "cosmic time (peak at $z=1.26$), each normalised to its maximum. \\calc", "fig:wz_cpl_clocks"))

# ---------------------------------------------------------------- p2_21
add("p2_21_entanglement_records.tex", "itself: its onset (zero slope for the assumed profile, finite slope for the exponential), and the scaling of $\\tau_{\\rm IAM}$ with temperature and mass.",
    fig("fig_chsh_dephasing", "\\textwidth", "CHSH value of a Bell pair under pointer-basis dephasing",
        "(a) CHSH value against the coherence $c$: $S_{\\max}=2\\sqrt{1+c^2}$ for pointer-basis dephasing (Horodecki criterion~\\cite{Horodecki1995}, "
        "computed from the density matrix) \\derived; $S=\\sqrt2(1+c)$ with the pure-state settings, below 2 for $c<\\sqrt2-1$; $S_{\\max}=2\\sqrt2\\,p$ "
        "for isotropic noise, below 2 for $p<1/\\sqrt2$ \\calc. (b) $S_{\\max}(t)$ for the exponential $c=e^{-t/\\tau_{\\rm IAM}}$ \\calc\\ and for the "
        "assumed profile $c=1-E(t/\\tau_{\\rm IAM})/e$ \\conjecture; points at $t/\\tau_{\\rm IAM}=0.5$, 1, 2, 3. Both fade to 2 and never fall below.",
        "fig:chsh_dephasing"))
rho = 2200.0; EG = lambda m: G * m * m / ((3 * m / (4 * np.pi * rho))**(1 / 3))
tI = lambda m, T: hbar * (C.k * T)**2 * np.log(2) / EG(m)**3; tPD = lambda m: hbar / EG(m)
mx = 10**brentq(lambda l: np.log(tI(10**l, .01) / tPD(10**l)), -20, -5)
t20s = f"{tI(1e-12, .02):,.0f}".replace(",", "{,}")
add("p2_21_entanglement_records.tex", "in microseconds at any temperature; the IAM channel predicts none on that scale, and a time that grows as $T^2$. \\prediction",
    fig("fig_tau_temperature", "\\textwidth", "The IAM time against temperature and mass, beside the Di\\'osi--Penrose time",
        f"Silica spheres, $\\rho=2200$\\,kg\\,m$^{{-3}}$, boundary capacity $\\kB T/E_G$ assumed. (a) $\\tau_{{\\rm IAM}}=\\hbar(\\kB T)^2\\ln2/E_G^3$ against the "
        f"bath temperature for $10^{{-15}}$, $10^{{-13}}$ and $10^{{-12}}$\\,kg (points: {tI(1e-12,.01):.0f}\\,s at 10\\,mK and {t20s}\\,s at 20\\,mK for "
        f"$10^{{-12}}$\\,kg); dotted: the Di\\'osi--Penrose time $\\hbar/E_G$~\\cite{{Diosi1989,Penrose1996}} of each mass, independent of $T$ ({tPD(1e-12)*1e6:.1f}\\,$\\mu$s at "
        f"$10^{{-12}}$\\,kg). (b) $\\tau_{{\\rm IAM}}/\\tau_{{\\rm PD}}$ against mass at 10, 20 and 50\\,mK; at 10\\,mK the two are equal at ${mx/1e-10:.1f}\\times10^{{-10}}$\\,kg. \\calc\\ \\prediction",
        "fig:tau_temperature"))

# ---------------------------------------------------------------- p2_22
add("p2_22_electroweak.tex", "\\calc{} The matter sector exists from the first row. It writes in quantity only from the fifth, once structure forms.",
    fig("fig_ew_timeline", "\\textwidth", "The thermal history and the activation function",
        "(a) The events of the table at their computed times and temperatures: electroweak crossover (159.5\\,GeV~\\cite{DOnofrio2016}, "
        "$9.2\\times10^{-12}$\\,s), QCD confinement (150\\,MeV, $g_*=17.25$), nucleosynthesis (0.1 and 0.07\\,MeV) from $t=0.301\\,g_*^{-1/2}m_P/T^2$ "
        "(dotted: $g_*=106.75$); recombination, $z=30$ and today from the Friedmann integral with radiation, matter and $\\Lambda$ (solid; Planck 2018~\\cite{Planck2018VI}). "
        "(b) $E(a)=\\exp(1-1/a)$ from $z=40$ to today: $9.4\\times10^{-14}$ at $z=30$, $4.5\\times10^{-5}$ at $z=10$, 0.37 at $z=1$; at the electroweak "
        "epoch $\\ln E=-2.0\\times10^{15}$. Equations of \\texttt{docs/verification/scripts/\\allowbreak{}verify\\_entanglement\\_electroweak.py}. \\calc",
        "fig:ew_timeline"))
add("p2_22_electroweak.tex", "hadron writes is not treated. \\openprob",
    fig("fig_cornell", "\\textwidth", "The Cornell potential and its virial weight",
        "The Cornell potential~\\cite{Eichten1978} in units of the radius $r_0=\\sqrt{4\\alpha_s/(3\\sigma)}$ at which its two parts are equal in size, "
        "$V/(\\sigma r_0)=-r_0/r+r/r_0$. (a) The potential and its Coulomb-like ($k=-1$) and linear ($k=+1$) parts. (b) The share of the virial weight "
        "$r\\,dV/dr$ carried by the linear part, $x^2/(1+x^2)$ with $x=r/r_0$: from 0, where $2\\langle K\\rangle=-\\langle V\\rangle$, to 1, where "
        "$2\\langle K\\rangle=+\\langle V\\rangle$. The figure is dimensionless and needs no value of $\\alpha_s$. \\derived", "fig:cornell"))

# ---------------------------------------------------------------- anchor check and write
for e in OUT:
    txt = (S.BOOK / e["file"]).read_text().splitlines()
    n = sum(1 for l in txt if l == e["anchor"])
    assert n == 1, (e["file"], e["anchor"][:60], n)
    e["anchor_line"] = txt.index(e["anchor"]) + 1
    assert e["latex"].count("{") == e["latex"].count("}"), (e["file"], e["latex"][:80])
    for bad in ("the paper", "source paper", "the author", "author", "Mahaffey", "app:errata", "the note", "companion", "separate paper"):
        assert bad not in e["latex"].lower(), (bad, e["file"])
(S.BOOK / "figures_insertions_p2.json").write_text(json.dumps(OUT, indent=1, ensure_ascii=False) + "\n")
print(f"{len(OUT)} insertions written")
LAB = re.compile(r"label\{([^}]+)\}")
for e in OUT:
    kind = "table" if "begin{table" in e["latex"] else "figure"
    print(f"  {e['file']}:{e['anchor_line']}  {kind}  {LAB.search(e['latex']).group(1)}")
