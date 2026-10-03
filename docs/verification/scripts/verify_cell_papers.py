#!/usr/bin/env python3
"""verify_cell_papers.py -- recomputes every number in REVIEW.md and in the DRAFT LaTeX
for the two early cellular papers (cell thermodynamics; vertebrate lifespan).

Section K: numbers proposed for the book (KEEP items).  Section T / V: numbers used in
REVIEW.md to show why an item is out of date (paper value vs recomputed value).
Species data are read from the original figure script (make_figures.py, kept verbatim in
the figscript fig_p4_species_lifespan.py) -- no value is re-typed here except the paper's
printed numbers, which are quoted for comparison.
Run from docs/book:  python ../verification/scripts/verify_cell_papers.py
"""
import math, sys, os, importlib.util
import numpy as np
from scipy import stats

kB, NA, R = 1.380649e-23, 6.02214076e23, 8.314462618
T0 = 310.15
H = lambda x: 0.0 if x <= 0 or x >= 1 else -(x*math.log2(x) + (1-x)*math.log2(1-x))
ok_all = True
def chk(tag, val, ref, tol, note=""):
    global ok_all
    good = abs(val-ref) <= tol
    ok_all &= good
    print(f"{tag:6s} {'PASS' if good else 'FAIL'}  value={val:.6g}  expected={ref:.6g}  {note}")
def show(tag, txt):  # a recomputed number reported against a printed one (no pass/fail)
    print(f"{tag:6s} INFO  {txt}")

# ---------- species data: imported from the figscript (the original lists) ----------
here = os.path.dirname(os.path.abspath(__file__))
fs = os.path.join(here, "..", "..", "book", "figscripts", "fig_p4_species_lifespan.py")
spec = importlib.util.spec_from_file_location("species", fs); sp = importlib.util.module_from_spec(spec)
os.environ["SPECIES_NO_PLOT"] = "1"; spec.loader.exec_module(sp)
MAM, VERT, ORDER_DATA, HMIN_OLD = sp.MAMMALS, sp.VERTEBRATES, sp.ORDER_DATA, sp.H_MIN_OLD

print("=== K: numbers proposed for the book ===")
# K1 floor of the copy error at other temperatures, fixed holding energy (book eq:eps0T)
Eh = 3.41
eps = lambda Tc: 1/(1+math.exp(Eh*T0/(Tc+273.15)))
r = lambda Tc: H(eps(Tc))/H(eps(37.0))
chk("K1a", eps(37.0), 0.0320, 5e-4, "eps0 at 37 C (book p3_08: 0.032)")
chk("K1b", H(eps(37.0)), 0.2043, 5e-4, "H(eps0) at 37 C (book 0.2043)")
chk("K1c", eps(10.0), 0.0233, 5e-4, "eps0 at 10 C (book p4_10: 0.0233)")
chk("K1d", r(10.0), 0.78, 5e-3, "floor ratio at 10 C (book p4_10: 0.78)")
for Tc, ref in ((15, 0.821), (25, 0.902), (32, 0.959), (40, 1.025), (42, 1.041)):
    chk(f"K1{Tc}", r(Tc), ref, 1.5e-3, f"floor ratio at {Tc} C, fixed holding energy")
for Tc, ref in ((15, 0.0248), (25, 0.0280), (32, 0.0303), (40, 0.0330), (42, 0.0337)):
    chk(f"K2{Tc}", eps(Tc), ref, 2e-4, f"eps0 at {Tc} C")
# K3 in vitro DNMT1 preference at a fixed discrimination energy: S(T) = S37 ** (T0/T)
for S37 in (7, 21, 80):
    s15, s42 = S37**(T0/288.15), S37**(T0/315.15)
    show("K3", f"S37={S37}: S(15 C)={s15:.1f}  S(42 C)={s42:.1f}")
chk("K3a", 7**(T0/288.15), 8.1, 0.05, "S(15C) for S37=7")
chk("K3b", 80**(T0/288.15), 111.7, 0.5, "S(15C) for S37=80")
chk("K3c", 80**(T0/315.15), 74.8, 0.3, "S(42C) for S37=80")
# K4 counts in the species figure
chk("K4a", len(MAM), 34, 0, "species in Fig. 1 (paper text says 40 and 43)")
chk("K4b", len(set(m[1] for m in MAM)), 12, 0, "orders in Fig. 1 (paper says 14)")
chk("K4c", len(VERT), 29, 0, "species in Fig. 2 (text says 31)")
hb = [H(m[2]) for m in MAM]
show("K4d", f"H(mean beta) across the 34 mammals: {min(hb):.3f}-{max(hb):.3f} bits")
# K5 literature counts carried (as printed by the sources; checked against abstracts)
show("K5", "Lowe 2018: six mammalian species; Crofts 2024: 42 species; Lu 2023: 11,754 arrays, 59 tissues, 185 species, r>0.96;"
           " Haghani 2023: 15,456 profiles, 348 species; Waterston 2002: ~80% of mouse genes have one human orthologue")
# K6 lifespans in the figure (as tabulated in the script)
d = {m[0]: m for m in MAM}
chk("K6a", d["Bowhead whale"][3], 211, 0, "bowhead lifespan (yr)"); chk("K6b", d["House mouse"][3], 4, 0, "mouse lifespan (yr)")

print("\n=== T: cell thermodynamics paper, printed vs recomputed ===")
ebit = kB*T0*math.log(2)
chk("T1", ebit, 2.968e-21, 1e-24, "kT ln2 at 310.15 K (paper 2.97e-21; book p4_02)")
chk("T2", 19.6e6*ebit, 5.82e-14, 1e-16, "paper floor with N=19.6e6")
chk("T3", 28217448*ebit, 8.37e-14, 1e-16, "book floor with hg19 N=28,217,448")
atp = 54000/NA
chk("T4", atp, 8.97e-20, 1e-22, "ATP per molecule (paper ~9e-20)")
show("T5", f"paper floor in ATP = {19.6e6*ebit/atp:.3g} (paper 'about 1e6'); book {28217448*ebit/atp:.3g}")
chk("T6", 54000/(R*T0), 20.94, 5e-3, "ATP drive / RT (book: M)")
chk("T7", H(0.40)-H(0.60), 0.0, 1e-12, "H symmetric: beta 0.40 and 0.60 carry the same entropy (paper says 0.40 carries more)")
chk("T8", H(0.782), 0.7565, 1e-4, "H(0.782) frontal cortex (paper 0.7565: correct)")
for b, printed in ((0.780, .7951), (0.775, .8058), (0.768, .8175), (0.760, .8325), (0.764, .8215)):
    show("T9", f"H({b}) = {H(b):.4f}  printed {printed}  diff {printed-H(b):+.4f}")
# Supplementary Table S1: every printed H(beta)
S1 = [(0.420,.9789),(0.410,.9741),(0.435,.9834),(0.428,.9811),(0.710,.8680),(0.685,.8936),(0.720,.8565),(0.700,.8816),
      (0.715,.8622),(0.720,.8565),(0.730,.8415),(0.715,.8622),(0.725,.8490),(0.780,.7951),(0.782,.7565),(0.775,.8058),
      (0.768,.8175),(0.760,.8325),(0.730,.8415),(0.725,.8490),(0.720,.8565),(0.728,.8445),(0.695,.8884),(0.730,.8415),
      (0.740,.8265),(0.700,.8816),(0.735,.8340),(0.725,.8490),(0.760,.8325),(0.740,.8265),(0.710,.8680),(0.735,.8340),
      (0.730,.8415),(0.720,.8565),(0.695,.8884),(0.728,.8445),(0.715,.8622)]
bad = [(b, p, round(H(b), 4)) for b, p in S1 if abs(H(b)-p) > 1e-3]
chk("T10", len(S1), 37, 0, "rows in Table S1"); show("T10", f"rows whose printed H(beta) is off by >0.001: {len(bad)} -> {bad}")
show("T11", f"neutrophil beta 0.760 gives H={H(0.760):.4f}, below the printed immune floor 0.8389 (a floor above its own reference cell)")
show("T12", f"DCIS high grade H(0.660)={H(0.660):.4f} (printed 0.929)")
# E(a) = exp(1-1/a): pace dE/da = E/a^2 peaks at a = 1/2
a = np.linspace(0.05, 2, 200001); pace = np.exp(1-1/a)/a**2
chk("T13", a[np.argmax(pace)], 0.5, 1e-4, "peak of dE/da (paper: t_max/2)")

print("\n=== V: vertebrate lifespan paper, printed vs recomputed (from the original script data) ===")
A = lambda b: H(b)/HMIN_OLD
ls = np.array([m[3] for m in MAM]); Av = np.array([A(m[2]) for m in MAM])
rp, pp = stats.pearsonr(np.log(ls), Av); rs, ps = stats.spearmanr(ls, Av)
show("V1", f"Pearson r={rp:.4f} p={pp:.2g} n=34 (text r=-0.9018 p=1.6e-16 n=40; abstract n=43; caption -0.919, 1.6e-14, 34)")
show("V1", f"Spearman rho={rs:.4f} p={ps:.2g} (text -0.9149, 9.5e-18)")
for cut in (20, 35):
    L = Av[ls >= cut]; S = Av[ls < cut]; t, p = stats.ttest_ind(L, S)
    sp_ = math.sqrt(((len(L)-1)*L.var(ddof=1)+(len(S)-1)*S.var(ddof=1))/(len(L)+len(S)-2))
    show("V2", f"split {cut} yr: long n={len(L)} mean {L.mean():.3f}, short n={len(S)} mean {S.mean():.3f}, t={t:.2f}, p={p:.2g}, d={(S.mean()-L.mean())/sp_:.2f};"
              f" long with A>=1.05: {[m[0] for m in MAM if m[3]>=cut and A(m[2])>=1.05]}; short with A<1.05: {[m[0] for m in MAM if m[3]<cut and A(m[2])<1.05]}")
show("V2", "printed: 20-yr split 17 vs 11, t=-21.4, d=1.99 (hard-coded in script); caption 35-yr split 14 vs 20, t=-6.2, d=1.50; figure says 'All 23/23 A<1.05'")
chk("V2c", (1.131-1.006)/math.sqrt((16*0.015**2+10*0.014**2)/26), 8.55, 0.01, "Cohen's d from the paper's own means/SDs, pooled with n=17,11 (paper 1.99; book errata app_B2 8.55)")
order_tab = {"Cetacea": .997, "Proboscidea": .987, "Primates": 1.007, "Artiodactyla": 1.015, "Chiroptera": 1.041, "Carnivora": 1.053,
             "Lagomorpha": 1.114, "Rodentia": 1.125, "Insectivora": 1.157}
for o, v in order_tab.items():
    fromM = np.mean([A(m[2]) for m in MAM if m[1] == o]); nM = sum(m[1] == o for m in MAM)
    fromO = np.mean(ORDER_DATA[o][0]); nO = len(ORDER_DATA[o][0])
    show("V3", f"{o:12s} table {v:.3f} | MAMMALS list n={nM} mean {fromM:.3f} | ORDER_DATA n={nO} mean {fromO:.3f}")
nmr = [m for m in MAM if m[0] == "Naked mole rat"][0]
chk("V4", H(nmr[2])/(HMIN_OLD*((nmr[4]+273.15)/T0)**2), 1.160, 1e-3, "naked mole rat with alpha=2 at 32 C (paper: 'approaches 1.13'; correction RAISES A)")
chk("V5", A(0.601)-A(0.744), 0.1785, 1e-3, "shrew minus bowhead (paper 0.178)")
show("V5", f"lifespan ratio bowhead/shrew = {211/2.5:.1f} with shrew 2.5 yr (Fig.1 list); 105.5 only with shrew 2 yr (Fig.2 list); paper '105-fold'")
Tv = np.array([v[3] for v in VERT]); Ar = np.array([A(v[2]) for v in VERT])
r1, p1 = stats.pearsonr(Tv, Ar); show("V6", f"raw A vs body T: r={r1:.3f} p={p1:.2g} n={len(VERT)} (text +0.80, 7.5e-8, n=31; caption +0.835, 1.8e-8, n=29)")
Ac = lambda al: np.array([H(v[2])/(HMIN_OLD*((v[3]+273.15)/T0)**al) for v in VERT])
red = 1-np.var(Ac(2.0))/np.var(Ar); show("V7", f"variance reduction at alpha=2: {100*red:.1f}% (text 41%, caption 46%)")
als = np.linspace(0, 6, 6001); vv = [np.var(Ac(x)) for x in als]
show("V8", f"alpha minimising cross-class variance on the script's 29 species: {als[int(np.argmin(vv))]:.2f} (paper: 2.0); reduction there {100*(1-min(vv)/np.var(Ar)):.1f}%")
for Ea in (40e3, 60e3, 80e3): show("V9", f"Ea={Ea/1e3:.0f} kJ/mol = {Ea/(R*T0):.1f} kT0 (paper: 4.8/7.2/9.6)")
show("V9", f"paper's alpha formula with 60 kJ/mol: {60e3/(R*T0)*0.25:.1f} (paper 1.8 used 7.2 kT0)")
print("\nALL CHECKS PASS" if ok_all else "\nSOME CHECKS FAIL"); sys.exit(0 if ok_all else 1)
