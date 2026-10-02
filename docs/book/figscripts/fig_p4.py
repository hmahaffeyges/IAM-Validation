"""Part 4 figures and tables added for TODO 9.2 (the cell). Run from any directory:
    python docs/book/figscripts/fig_p4.py            # all figures -> docs/book/figures/part4/fig_p4_*.pdf/.png
    python docs/book/figscripts/fig_p4.py --tables   # prints the computed table rows used in the Part 4 insertions
Every number drawn or printed is read here from the frozen chain v3 files (Biological_Physics/MethylPhys/chain/Runtime Matrices/),
from the record data (doors/data/, doors/PROC_*/, chain_tests/), from the record markdown tables (parsed, not typed), from
book tables (parsed), or computed from CODATA constants (scipy.constants). Nothing is typed by hand except the 2.8e7 CpG
capacity of one haploid genome quoted in Chapter ch:surface and the names of the stages.
No class floors, tiers, provisional breach lines or cohort bands are drawn. Normal (0.95-1.05) is the design tolerance.
"""
import sys, json, re, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, pandas as pd, scipy.constants as C
import _bookstyle as S
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

S.apply()
MP = S.REPO / "Biological_Physics" / "MethylPhys"
RM = MP / "chain" / "Runtime Matrices"
DD = MP / "doors" / "data"
P4 = S.BOOK / "part4"
NORMAL = (0.95, 1.05)
NCOL = S.ALT                     # colour of the Normal band (design tolerance)


def H(x):
    x = np.clip(np.asarray(x, float), 1e-12, 1 - 1e-12)
    return -(x * np.log2(x) + (1 - x) * np.log2(1 - x))


def normal_band(ax, axis="y", label=True):
    f = ax.axhspan if axis == "y" else ax.axvspan
    f(*NORMAL, color=NCOL, alpha=0.12, lw=0, zorder=0)
    (ax.axhline if axis == "y" else ax.axvline)(1.0, color=S.GR, lw=0.6, ls="--", zorder=0)


def canon():
    c = json.load(open(S.REPO / "CANON" / "iam_canon.json"))["constants"]
    return {k: c[k]["value"] for k in ("T_cell", "M_cell", "E_hold_meth", "phi", "eps0_meth", "Met_A_floor_EPIC_neutrophil",
                                       "P_neutrophil_IAM_A", "dG_ATP")}


def floors():
    m = json.load(open(RM / "Met_A_Floors" / "metA_floors_v1_3.json"))["platforms"]["EPIC"]["neutrophils"]
    return m


def nref():
    return json.load(open(RM / "Met_A_Floors" / "neutrophil_reference_v1_1.json"))


def comp():
    return json.load(open(RM / "Met_A_Floors" / "blood_composition_EPIC_v1.json"))


def iama():
    return json.load(open(RM / "IAM_A_Positions" / "iama_positions_v1.json"))


def md_table(path, first_col_startswith=None):
    """Parse the first markdown table in a record (or the one whose header starts with a given word)."""
    rows, take = [], False
    for ln in open(path, encoding="utf-8"):
        if ln.startswith("|"):
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue
            if not rows and first_col_startswith and not cells[0].startswith(first_col_startswith):
                continue
            rows.append(cells); take = True
        elif take:
            break
    return rows


def nums(s):
    return [float(x) for x in re.findall(r"-?\d+\.\d+|-?\d+", s.replace("−", "-").replace("–", " "))]


# ============================== constants ==============================
K = canon()
T0 = K["T_cell"]
LN2 = np.log(2)
EPS0 = 1 / (1 + np.exp(K["E_hold_meth"]))            # 3.41 kT -> 0.032
P_NEU = iama()["cells"]["neutrophils"]["P"]
HREF = floors()["floor"]
CPG_BITS = 2.8e7                                     # Chapter ch:surface: one bit per CpG copy, haploid genome
MSUN = 1.98847e30


def bh(M):
    TH = C.hbar * C.c**3 / (8 * np.pi * C.G * M * C.k)
    S_ = 4 * np.pi * C.G * M**2 * C.k / (C.hbar * C.c)
    return TH, S_


# ============================== p4_03 =================================
def fig_jensen():
    r = nref(); b = np.array(r["profiles_mean_beta"]["neutrophils"]); h = np.array(r["neutrophil_H_mean"])
    meth = b > 0.5
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(S.TEXTW, 2.4), gridspec_kw=dict(width_ratios=[1.35, 1]))
    bins = np.linspace(0, 1, 51)
    a1.hist(h[meth], bins=bins, color=S.IAM, alpha=0.85, label=f"methylated channel ({meth.sum():,} sites)")
    a1.hist(h[~meth], bins=bins, color=S.GOLD, alpha=0.75, label=f"unmethylated channel ({(~meth).sum():,} sites)")
    a1.axvline(h.mean(), color="k", lw=0.9); a1.text(h.mean() + 0.02, a1.get_ylim()[1] * 0.93, f"mean of $H$: {h.mean():.4f} bits", fontsize=7)
    a1.set_xlabel(r"per-site entropy $H(\beta_i)$ (bits; mean over the six reference arrays)"); a1.set_ylabel("identity sites")
    a1.set_xlim(0.15, 0.85); a1.legend(loc="upper right", fontsize=6.5, bbox_to_anchor=(1.0, 0.86)); a1.set_title("Most identity sites carry far less than one bit")
    S.panel_letter(a1, "a")
    labs = ["methylated", "unmethylated", "both"]
    sets = [meth, ~meth, np.ones_like(meth)]
    mh = [h[s].mean() for s in sets]; hm = [float(H(b[s].mean())) for s in sets]
    x = np.arange(3)
    a2.bar(x - 0.18, mh, 0.36, color=S.IAM, label=r"mean of the entropies $\overline{H}$")
    a2.bar(x + 0.18, hm, 0.36, color=S.LIGHT, label=r"entropy of the mean $H(\bar\beta)$")
    for i in range(3):
        a2.text(x[i] - 0.18, mh[i] + 0.02, f"{mh[i]:.3f}", ha="center", va="bottom", fontsize=5.5, rotation=90)
        a2.text(x[i] + 0.18, hm[i] + 0.02, f"{hm[i]:.3f}", ha="center", va="bottom", fontsize=5.5, rotation=90)
    a2.set_xticks(x, labs); a2.set_ylim(0, 1.45); a2.set_ylabel("bits"); a2.legend(loc="upper left", fontsize=6.5)
    a2.set_title("Averaging β first gives about one bit")
    S.panel_letter(a2, "b", dx=-0.16)
    S.save(fig, "part4", "fig_p4_03_jensen")
    return dict(n=[int(s.sum()) for s in sets], mean_beta=[float(b[s].mean()) for s in sets], Hbar=mh, Hofmean=hm)


# ============================== p4_04 =================================
def fig_ledger():
    fig, ax = plt.subplots(figsize=(0.62 * S.TEXTW, 2.6))
    Ms = np.logspace(-1, 10, 50) * MSUN
    TH, S_ = bh(Ms); Nb = S_ / (C.k * LN2); cost = Nb * C.k * TH * LN2
    ax.plot(Nb, cost, color=S.GR, lw=1.0, label=r"black holes: $N k_BT_H\ln2 = T_HS = Mc^2/2$")
    pts = {}
    for M, lab in ((1, "1 M$_\\odot$"), (1e6, "10$^6$ M$_\\odot$")):
        th, s = bh(M * MSUN); n = s / (C.k * LN2); e = n * C.k * th * LN2; pts[lab] = (n, e, M * MSUN * C.c**2 / 2)
        ax.plot(n, e, "o", color=S.GR, ms=4); ax.annotate(lab, (n, e), xytext=(6, -10), textcoords="offset points", fontsize=6.5)
    ecell = CPG_BITS * C.k * T0 * LN2
    ax.plot(CPG_BITS, ecell, "o", color=S.DATA, ms=5)
    ax.annotate(f"cell methylome, 310 K\n{S.sci(CPG_BITS)} bits, {S.sci(ecell)} J", (CPG_BITS, ecell), xytext=(8, 4), textcoords="offset points", fontsize=6.5, color=S.DATA)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1e5, 1e100); ax.set_ylim(1e-16, 1e60)
    ax.set_xlabel("capacity of the surface $N$ (bits)"); ax.set_ylabel(r"cost of holding it, $N k_BT\ln2$ (J)")
    ax.legend(loc="upper left", fontsize=6.5); ax.set_title("The ledger entry of each surface at its own temperature")
    S.save(fig, "part4", "fig_p4_04_ledger")
    th1, s1 = bh(MSUN)
    return dict(ecell=ecell, TS1=th1 * s1, mc2half=MSUN * C.c**2 / 2, smarr_ratio=float(th1 * s1 / (MSUN * C.c**2)),
                K_H=C.physical_constants["Rydberg constant times hc in eV"][0], V_H=-2 * C.physical_constants["Rydberg constant times hc in eV"][0],
                equip=C.k * T0 / 2, pts=pts)


# ============================== p4_05 =================================
def fig_fullsurface():
    r = nref(); h = np.array(r["neutrophil_H_mean"])
    q = np.linspace(0, 1, 101)
    A_meta = ((1 - q) * h.mean() + q * 1.0) / HREF
    Amax = 1 / HREF
    e = np.logspace(np.log10(0.01), np.log10(0.5), 300)
    iam = H(e) / (P_NEU * H(EPS0)); Amax2 = 1 / (P_NEU * H(EPS0))
    eh = float(e[np.argmin(abs(iam - 1))])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(S.TEXTW, 2.4))
    a1.plot(q * 100, A_meta, color=S.IAM); normal_band(a1)
    a1.plot(100, Amax, "o", color=S.IAM, ms=4); a1.annotate(f"surface full: {Amax:.2f}", (100, Amax), xytext=(-62, -4), textcoords="offset points", fontsize=7)
    a1.set_xlabel("share of identity sites moved to a coin flip (%)"); a1.set_ylabel("Met-A"); a1.set_xlim(0, 104); a1.set_ylim(0.8, 3.3)
    a1.set_title("Arrays: Met-A from 1 to 1/0.330263"); S.panel_letter(a1, "a")
    a2.plot(e, iam, color=S.IAM); normal_band(a2)
    a2.plot(EPS0, 1 / P_NEU, "s", color="k", ms=4); a2.annotate(f"floor $H_{{\\min}}$: 1/P = {1/P_NEU:.3f}", (EPS0, 1 / P_NEU), xytext=(8, -12), textcoords="offset points", fontsize=6.5)
    a2.plot(0.5, Amax2, "o", color=S.IAM, ms=4); a2.annotate(f"surface full: {Amax2:.2f}", (0.5, Amax2), xytext=(-64, -2), textcoords="offset points", fontsize=7)
    a2.set_xscale("log"); a2.set_xlim(0.01, 0.6); a2.set_ylim(0.4, 4.9)
    a2.set_xticks([0.01, 0.03, 0.1, 0.3, 0.5], ["0.01", "0.03", "0.1", "0.3", "0.5"]); a2.xaxis.set_minor_formatter(plt.NullFormatter())
    a2.set_xlabel(r"copy error $\varepsilon$ on the methylated channel"); a2.set_ylabel("IAM-A")
    a2.set_title("Molecules: IAM-A from the floor to 4.45"); S.panel_letter(a2, "b")
    S.save(fig, "part4", "fig_p4_05_fullsurface")
    return dict(Amax_meta=Amax, Amax_iama=Amax2, floor=1 / P_NEU, eps_healthy=eh, Hmean=float(h.mean()))


def fig_surfaces():
    th, s = bh(MSUN)
    rows = [("temperature (K)", th, T0), ("capacity (bits)", s / (C.k * LN2), CPG_BITS), ("cost per bit (J)", C.k * th * LN2, C.k * T0 * LN2),
            ("cost of the full surface (J)", th * s, CPG_BITS * C.k * T0 * LN2)]
    fig, ax = plt.subplots(figsize=(0.7 * S.TEXTW, 2.3))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, vh, vc) in zip(y, rows):
        ax.plot([np.log10(vh), np.log10(vc)], [yi, yi], color=S.LIGHT, lw=1.0, zorder=1)
        ax.plot(np.log10(vh), yi, "o", color=S.GR, ms=5, zorder=2); ax.plot(np.log10(vc), yi, "o", color=S.DATA, ms=5, zorder=2)
        ax.annotate(S.sci(vh), (np.log10(vh), yi), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=6, color=S.GR)
        ax.annotate(S.sci(vc), (np.log10(vc), yi), xytext=(0, -10), textcoords="offset points", ha="center", fontsize=6, color=S.DATA)
    ax.set_yticks(y, [r[0] for r in rows]); ax.set_ylim(-0.7, len(rows) + 0.2)
    ax.set_xlabel(r"$\log_{10}$ of the value"); ax.set_xlim(-40, 85)
    ax.plot([], [], "o", color=S.GR, label="horizon of one solar mass"); ax.plot([], [], "o", color=S.DATA, label="cell methylome at 310.15 K")
    ax.legend(loc="upper center", fontsize=6.5, ncol=2); ax.set_title("One equation, parameters 10 to 70 orders apart")
    S.save(fig, "part4", "fig_p4_05_surfaces")
    return {r[0]: (r[1], r[2]) for r in rows}


# ============================== p4_06 / p4_22 ===========================
def imr90():
    return pd.read_csv(MP / "doors" / "PROC_LINES_02_channels" / "imr90_channels.csv")


def fig_imr90_plane():
    d = imr90(); col = {"Proliferating": S.GR, "Senescent": S.IAM, "SV40": S.DATA}
    lab = {"Proliferating": "proliferating (held out)", "Senescent": "replicatively senescent", "SV40": "SV40-immortalised"}
    fig, ax = plt.subplots(figsize=(0.6 * S.TEXTW, 2.8))
    ax.axvspan(*NORMAL, color=NCOL, alpha=0.10, lw=0); ax.axhspan(*NORMAL, color=NCOL, alpha=0.10, lw=0)
    ax.axvline(1, color=S.GR, lw=0.5, ls="--"); ax.axhline(1, color=S.GR, lw=0.5, ls="--")
    for st, g in d.groupby("state", sort=False):
        ax.plot(g.A_meth, g.A_unmeth, "o", color=col[st], ms=4.5, label=lab[st] + f": both channels {g.A_both.min():.3f}–{g.A_both.max():.3f}")
    ax.set_xlabel("methylated channel (reading against proliferating)"); ax.set_ylabel("unmethylated channel")
    ax.set_xlim(0.88, 1.15); ax.set_ylim(0.55, 1.08); ax.legend(loc="center", bbox_to_anchor=(0.5, 0.56), fontsize=6)
    ax.set_title("Two channels, read apart")
    S.save(fig, "part4", "fig_p4_06_imr90_channels")
    return {st: dict(meth=(g.A_meth.min(), g.A_meth.max()), unmeth=(g.A_unmeth.min(), g.A_unmeth.max()), both=(g.A_both.min(), g.A_both.max())) for st, g in d.groupby("state")}


def dnmt():
    return pd.read_csv(DD / "dnmt_arrays_readings.csv")


def fig_dnmt_channels():
    d = dnmt()
    act = d[d.cmpd == "GSK032"].copy(); veh = d[d.cmpd == "DMSO"]
    fig, ax = plt.subplots(figsize=(0.6 * S.TEXTW, 2.7))
    normal_band(ax)
    ax.plot(veh.A_meth, veh.A_unmeth, "o", mfc="none", color=S.GR, ms=4, label="vehicle")
    sc = ax.scatter(act.A_meth, act.A_unmeth, c=np.log10(act.dose_nM.clip(lower=1)), cmap="viridis", s=14, label="active drug (colour: dose)")
    ana = d[~d.cmpd.isin(["DMSO", "GSK032"])]
    names = {"GSK477": ("inactive analogue GSK3510477, 10 µM", S.ALT2), "GSK862": ("second active compound GSK3484862, 1 µM", S.DATA)}
    for cm, g in ana.groupby("cmpd"):
        ax.plot(g.A_meth, g.A_unmeth, "^", color=names[cm][1], ms=4, label=names[cm][0])
    cb = fig.colorbar(sc, ax=ax, pad=0.02); cb.set_label(r"$\log_{10}$ dose (nM)", fontsize=7); cb.ax.tick_params(labelsize=6)
    ax.set_xlabel("methylated channel"); ax.set_ylabel("unmethylated channel"); ax.set_ylim(0.85, 1.35)
    ax.legend(loc="upper right", fontsize=6); ax.set_title("A block of copying moves the methylated channel")
    S.save(fig, "part4", "fig_p4_22_dnmt_channels")
    med_m = float(np.median(d.loc[(d.cmpd == "GSK032") & (d.dose_nM >= 80), "A_meth"] - 1)); med_u = float(np.median(d.loc[(d.cmpd == "GSK032") & (d.dose_nM >= 80), "A_unmeth"] - 1))
    return dict(cmpds=sorted(d.cmpd.unique()), med_m=med_m, med_u=med_u)


# ============================== p4_07 =================================
def fig_heldout():
    loo = pd.read_csv(RM / "Met_A_Floors" / "metA_floors_v1_3_loo.csv")
    acc = pd.read_csv(MP / "chain_tests" / "chain_acceptance.csv").set_index("gsm")
    fig, ax = plt.subplots(figsize=(0.62 * S.TEXTW, 2.4))
    normal_band(ax)
    x = np.arange(len(loo))
    ax.plot(x - 0.15, loo.A_loo, "o", color=S.DATA, ms=5, label=f"sites re-chosen on the other five (SD {loo.A_loo.std(ddof=1):.3f})")
    ax.plot(x + 0.15, loo.A_loo_frozen_sites, "s", color=S.IAM, ms=4, label=f"frozen sites, floor from the other five (SD {loo.A_loo_frozen_sites.std(ddof=1):.3f})")
    ax.plot(x, acc.loc[loo.ref, "A"], "_", color="k", ms=9, mew=1.2, label="in the floor (reads 1 by construction)")
    ax.set_xticks(x, [g.replace("GSM", "GSM\n") for g in loo.ref], fontsize=6); ax.set_ylim(0.93, 1.07); ax.set_ylabel("Met-A")
    ax.legend(loc="upper left", fontsize=6); ax.set_title("Only the held-out readings are evidence")
    S.save(fig, "part4", "fig_p4_07_heldout")
    ni = pd.read_csv(DD / "noise_index.csv")
    fig, ax = plt.subplots(figsize=(0.55 * S.TEXTW, 2.5))
    normal_band(ax)
    ref = ni[ni.set.str.startswith("Salas")]
    ax.axvspan(ref.N.min(), ref.N.max(), color=S.LIGHT, alpha=0.35, lw=0)
    lab = {"Salas floor neutrophils": ("reference arrays (12 deposits, 6 arrays)", S.GR), "GSE247195": ("second laboratory, donor 1", S.IAM),
           "GSE247193": ("second laboratory, donor 2", S.DATA)}
    out = {}
    for k, (l, c) in lab.items():
        g = ni[ni.set == k]; ax.plot(g.N, g.A_own, "o", color=c, ms=3.5, label=l); out[k] = (len(g), g.N.median(), g.A_own.median())
        if k != "Salas floor neutrophils":
            out[k] += (pd.Series(g.N).corr(g.A_own, method="spearman"),)
    ax.set_xlabel("noise index $N$ (bits, 48,528 invariant sites)"); ax.set_ylabel("Met-A, no tare")
    ax.legend(loc="upper left", fontsize=6); ax.set_title("The untared reading follows the array's noise")
    S.save(fig, "part4", "fig_p4_07_noise")
    return dict(loo=loo, acc=acc, ni=ni, noise=out)


# ============================== p4_08 =================================
def fig_iama():
    g = pd.read_csv(MP / "chain_tests" / "iama_floor_granulocytes.csv")
    fig, ax = plt.subplots(figsize=(0.6 * S.TEXTW, 2.4))
    normal_band(ax); ax.axhline(1 / P_NEU, color="k", lw=0.9); ax.text(2.45, 1 / P_NEU - 0.035, f"floor $H_{{\\min}}$ = 1/P = {1/P_NEU:.3f}", fontsize=6.5, ha="right")
    x = np.arange(len(g))
    for dx, col, mk, lab, l2 in ((-0.2, S.GR, "o", "A_phys", "bare floor $\\varepsilon_0$ only"), (0.0, S.IAM, "s", "A_own", "with frozen position $P$ (held out)"),
                                 (0.2, S.DATA, "^", "A_own_damaged", "2 % copy error added")):
        ax.plot(x + dx, g[lab], mk, color=col, ms=5, label=l2)
    ax.vlines(x, g.A_own_odd, g.A_own_even, color=S.IAM, lw=2.5, alpha=0.35)
    ax.set_xticks(x, g.gsm, fontsize=6.5); ax.set_ylabel("IAM-A"); ax.set_ylim(0.85, 1.42); ax.set_xlim(-0.5, 2.5)
    ax.legend(loc="upper left", fontsize=6); ax.set_title("Three granulocyte donors on single molecules")
    S.save(fig, "part4", "fig_p4_08_donors")
    e = np.logspace(np.log10(0.015), np.log10(0.08), 300); iam = H(e) / (P_NEU * H(EPS0))
    fig, ax = plt.subplots(figsize=(0.6 * S.TEXTW, 2.4))
    normal_band(ax); ax.plot(e, iam, color=S.IAM)
    ax.plot(EPS0, 1 / P_NEU, "s", color="k", ms=4); ax.annotate(r"$\varepsilon_0$" + f" = {EPS0:.4f}", (EPS0, 1 / P_NEU), xytext=(-48, -12), textcoords="offset points", fontsize=6.5)
    ax.plot(g.eps, H(g.eps) / (P_NEU * H(EPS0)), "o", color=S.GR, ms=4, label="donor copy error (own reading on this curve)")
    ax.plot(g.eps_damaged, H(g.eps_damaged) / (P_NEU * H(EPS0)), "^", color=S.DATA, ms=4, label="same molecules with 2 % added")
    ax.set_xscale("log"); ax.set_xlabel(r"copy error $\varepsilon$"); ax.set_ylabel(r"$H(\varepsilon)/(P\,H(\varepsilon_0))$")
    ax.legend(loc="upper left", fontsize=6); ax.set_title("IAM-A is steep near the floor")
    S.save(fig, "part4", "fig_p4_08_curve")
    return dict(g=g)


# ============================== p4_09 =================================
def cstat(z, blk=50):
    nb = len(z) // blk; zb = z[:nb * blk].reshape(nb, blk).mean(1)
    return np.var(np.sqrt(blk) * zb) / np.var(z)


def fig_cscore():
    r = nref(); n = len(r["sites_ordered"]); base = r["healthy_clustering_median"]; blk = r["clustering_block"]
    rng = np.random.default_rng(20261002)
    z0 = rng.standard_normal(n)
    za = z0 + 0.5                                        # spread: every site +0.5 SD
    zc = z0.copy(); k = n // 10; st = rng.integers(0, n - k); zc[st:st + k] += 5.0        # same total shift in one tenth of the genome order
    fig, axs = plt.subplots(2, 1, figsize=(S.TEXTW, 2.8), sharex=True)
    out = {}
    for ax, z, lab, col in ((axs[0], za, "departure spread over every site", S.IAM), (axs[1], zc, "same total departure in one region", S.DATA)):
        ax.plot(np.arange(n), z, ",", color=S.LIGHT)
        nb = n // blk; zb = z[:nb * blk].reshape(nb, blk).mean(1)
        ax.step(np.arange(nb) * blk, zb, where="post", color=col, lw=1.0)
        c = cstat(z, blk); out[lab] = (float(z.mean()), c / base)
        ax.set_ylabel("$z_i$"); ax.set_ylim(-4.5, 9.5)
        ax.text(0.01, 0.88, f"{lab}: mean $z$ = {z.mean():.2f}, C = {c/base:.1f}", transform=ax.transAxes, fontsize=7, color=col)
    axs[1].set_xlabel("identity site, genome order (6,000 sites; 50-site block means drawn as steps)")
    axs[0].set_title("Same mean departure, different maps (simulated, fixed seed)")
    S.save(fig, "part4", "fig_p4_09_construction")
    acc = pd.read_csv(MP / "chain_tests" / "chain_acceptance.csv")
    grp = {"healthy purified neutrophil (in floor)": "isolated reference neutrophils", "known mixture, neu>=50%": "DNA mixtures",
           "AML second remission blood (other lab)": "remission bloods, another laboratory"}
    loo = np.array(r["healthy_clustering_LOO"]) / base
    fig, ax = plt.subplots(figsize=(0.62 * S.TEXTW, 2.3))
    ax.axhline(1, color=S.GR, lw=0.6, ls="--")
    sets = [("healthy arrays, each against the other five", loo)] + [(v, acc.loc[acc.group == k, "C"].dropna().values) for k, v in grp.items()]
    for i, (lab, v) in enumerate(sets):
        ax.plot(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, "o", color=S.DATA if i else S.IAM, ms=4)
        out[lab] = (len(v), float(v.min()), float(v.max()))
    short = ["healthy arrays,\nheld out", "isolated\nneutrophils", "DNA\nmixtures", "remission\nbloods"]
    ax.set_xticks(range(len(sets)), short, fontsize=6)
    ax.set_ylabel("Met-A C-score"); ax.set_ylim(0.5, 1.7)
    ax.text(3.45, 1.6, f"far end: {blk}/{base} = {blk/base:.0f}", fontsize=6.5, ha="right")
    ax.set_title("C-score as printed: no band is set")
    S.save(fig, "part4", "fig_p4_09_readings")
    out["far_end"] = blk / base; out["base"] = base; out["blocks"] = n // blk
    return out


# ============================== p4_10 =================================
def fig_fish():
    sal = pd.read_csv(DD / "salmon_readings.csv"); ch = pd.read_csv(DD / "charr_readings.csv"); ri = pd.read_csv(DD / "rimouski_readings.csv")
    ch = ch[ch.qualifying > 1000]                                                   # the failed download (84 molecules) is a failed run
    sets = [("steelhead red cells (RRBS)", sal[sal.tissue == "RBC"].E_kT), ("steelhead sperm (RRBS)", sal[sal.tissue == "Sp"].E_kT),
            ("brook charr sperm (WGBS)", ch.E_kT), ("Atlantic salmon fin, F0 (WGBS)", ri[ri.generation == "F0"].E_kT),
            ("Atlantic salmon fin, F1 (WGBS)", ri[ri.generation == "F1"].E_kT)]
    efix = K["E_hold_meth"] * T0 / 283.15
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(S.TEXTW, 2.6), gridspec_kw=dict(width_ratios=[1.6, 0.8, 0.8], wspace=0.45))
    a1.axvline(K["E_hold_meth"], color=S.GR, lw=0.8, ls="--"); a1.text(K["E_hold_meth"] - 0.02, 4.55, "human cells,\n37 °C: 3.41", fontsize=6, ha="right", color=S.GR)
    a1.axvline(efix, color=S.IAM, lw=0.8, ls=":"); a1.text(efix + 0.02, 4.55, f"fixed energy\nat 10 °C: {efix:.2f}", fontsize=6, color=S.IAM)
    med = {}
    for i, (lab, v) in enumerate(sets):
        y = np.full(len(v), len(sets) - 1 - i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(v))
        a1.plot(v, y, "o", color=S.DATA, ms=2.5, alpha=0.8); m = float(np.median(v)); med[lab] = (len(v), m)
        a1.plot(m, len(sets) - 1 - i, "|", color="k", ms=10, mew=1.2)
    a1.set_yticks(range(len(sets))[::-1], [f"{s[0]}, n={len(s[1])}" for s in sets], fontsize=6)
    a1.set_xlabel(r"$E_{\rm hold}=\ln[(1-\varepsilon)/\varepsilon]$ ($k_BT$)"); a1.set_xlim(3.0, 4.6); a1.set_ylim(-0.6, 5.1)
    a1.set_title("Fish on the human statistic"); S.panel_letter(a1, "a", dx=-0.02)
    rho_c = ch[["dup_frac", "E_kT"]].corr(method="spearman").iloc[0, 1]; rho_r = ri[["conv_fail", "E_kT"]].corr(method="spearman").iloc[0, 1]
    a2.plot(ch.dup_frac * 100, ch.E_kT, "o", color=S.ALT, ms=2.5); a2.set_xlabel("duplicate fraction (%)"); a2.set_ylabel(r"$E_{\rm hold}$ ($k_BT$)")
    a2.set_title(f"charr: ρ = {rho_c:.2f}"); S.panel_letter(a2, "b", dx=-0.3)
    a3.plot(ri.conv_fail * 100, ri.E_kT, "s", color=S.ALT2, ms=2.5); a3.set_xlabel("conversion failure (%)")
    a3.set_title(f"salmon: ρ = {rho_r:.2f}"); S.panel_letter(a3, "c", dx=-0.3)
    S.save(fig, "part4", "fig_p4_10_fish")
    return dict(med=med, efix=efix, rho_c=rho_c, rho_r=rho_r)


# ============================== p4_11 =================================
def map79():
    rows = []
    for ln in open(P4 / "16a_skytools_map79.tex", encoding="utf-8"):
        m = re.match(r"^(\d+) & (.*?) & (.*?) & (.*?) &", ln)
        if m:
            rows.append((int(m.group(1)), m.group(4).strip()))
    return rows


def fig_map79():
    rows = map79(); st = pd.Series([r[1] for r in rows]).value_counts()
    order = ["in use", "built, not in chain", "partial", "calculated", "measured", "reserved", "not built", "open", "analogy", "refused", "out", "does not translate"]
    st = st.reindex([o for o in order if o in st.index])
    col = {"in use": S.IAM, "built, not in chain": S.SKY, "partial": S.SKY, "calculated": S.SKY, "measured": S.SKY, "reserved": S.GOLD, "not built": S.LIGHT,
           "open": S.LIGHT, "analogy": S.ALT2, "refused": S.DATA, "out": S.DATA, "does not translate": S.GR}
    fig, ax = plt.subplots(figsize=(0.62 * S.TEXTW, 2.6))
    y = np.arange(len(st))[::-1]
    ax.barh(y, st.values, color=[col[k] for k in st.index], height=0.7)
    for yi, v in zip(y, st.values):
        ax.text(v + 0.4, yi, str(v), va="center", fontsize=6.5)
    ax.set_yticks(y, st.index, fontsize=6.5); ax.set_xlabel(f"rows of the map (total {len(rows)})"); ax.set_xlim(0, max(st.values) + 4)
    ax.set_title("The 79-row map scored against chain v3")
    S.save(fig, "part4", "fig_p4_11_map79")
    return dict(n=len(rows), counts=st.to_dict())


# ============================== p4_12 =================================
def fig_tare():
    acc = pd.read_csv(MP / "chain_tests" / "chain_acceptance.csv")
    fig, ax = plt.subplots(figsize=(0.55 * S.TEXTW, 2.5))
    normal_band(ax)
    out = {}
    for i, (k, lab, col) in enumerate((("known mixture, neu>=50%", "DNA mixtures,\nreference laboratory", S.IAM),
                                      ("AML second remission blood (other lab)", "remission bloods,\nanother laboratory", S.DATA))):
        g = acc[(acc.group == k) & acc.A_rel_tared.notna()]
        for _, r in g.iterrows():
            ax.plot([2 * i, 2 * i + 1], [r.A, r.A_rel_tared], "-o", color=col, ms=3.5, lw=0.7)
        out[k] = (len(g), g.A.min(), g.A.max(), g.A_rel_tared.min(), g.A_rel_tared.max())
    ax.set_xticks([0, 1, 2, 3], ["untared", "tared", "untared", "tared"]); ax.set_xlim(-0.5, 3.5); ax.set_ylim(0.9, 1.15)
    ax.text(0.5, 1.135, "DNA mixtures,\nreference laboratory", ha="center", fontsize=6.5, color=S.IAM, va="top")
    ax.text(2.5, 1.135, "remission bloods,\nanother laboratory", ha="center", fontsize=6.5, color=S.DATA, va="top")
    ax.set_ylabel("whole-blood Met-A"); ax.set_title("Same-batch tare, acceptance run 3")
    S.save(fig, "part4", "fig_p4_12_tare")
    return out


def covid495():
    d3 = pd.read_csv(DD / "chain_v3_dev3_readings.csv")
    t4 = d3[(d3.test == "T4") & d3.A.notna() & (d3.f_neu >= 0.5)].copy()
    neg = t4[t4.group == "NEGATIVE"]
    X = lambda d: np.c_[np.ones(len(d)), d.f_neu, d.N]
    b = np.linalg.lstsq(X(neg), neg.A, rcond=None)[0]
    pred = X(neg) @ b; r2 = 1 - np.var(neg.A - pred) / np.var(neg.A)
    return t4, neg, b, pred, r2


def fig_noisefit():
    t4, neg, b, pred, r2 = covid495()
    fig, ax = plt.subplots(figsize=(0.5 * S.TEXTW, 2.5))
    lo, hi = 1.05, 1.42
    ax.plot([lo, hi], [lo, hi], color=S.GR, lw=0.6, ls="--")
    ax.plot(pred, neg.A, "o", color=S.DATA, ms=3)
    ax.set_xlabel(f"expected $A$ = {b[0]:.2f} + {b[1]:.3f} $f_{{\\rm neu}}$ + {b[2]:.2f} $N$"); ax.set_ylabel("untared Met-A")
    ax.text(0.04, 0.92, f"{len(neg)} healthy adults, $R^2$ = {r2:.2f}", transform=ax.transAxes, fontsize=6.5)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_title("What the whole-blood reading reads")
    S.save(fig, "part4", "fig_p4_12_noisefit")
    return dict(b=b, r2=r2, n=len(neg))


# ============================== p4_13 =================================
def fig_lowfrac():
    lf = pd.read_csv(DD / "lowfrac_readings.csv"); lf["shift"] = lf.A_dmg - lf.A_raw
    neg = lf[(lf.test == "T4") & (lf.group == "NEGATIVE")].copy()
    X = np.c_[np.ones(len(neg)), neg.f_neu, neg.N]; y = neg.A_raw.values; loo = []
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False; bb = np.linalg.lstsq(X[m], y[m], rcond=None)[0]; loo.append(y[i] / (X[i] @ bb))
    neg["r"] = loo
    bins = [(0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.0)]
    fig, ax = plt.subplots(figsize=(0.62 * S.TEXTW, 2.5))
    ax.plot(lf.f_neu, lf["shift"], "o", color=S.LIGHT, ms=2, label=f"every array ({len(lf)}): shift from a 2 % loss")
    rows = []
    for lo, hi in bins:
        s = neg[(neg.f_neu >= lo) & (neg.f_neu < (hi + 1e-9))]
        sd = float(s.r.std(ddof=1)); sh = float(s["shift"].median()); rows.append((lo, hi, len(s), sd, sh))
        ax.plot([lo, hi], [2 * sd, 2 * sd], color=S.DATA, lw=1.4); ax.plot((lo + hi) / 2, sh, "o", color=S.IAM, ms=4.5)
    ax.plot([], [], color=S.DATA, lw=1.4, label="twice the healthy spread in the bin"); ax.plot([], [], "o", color=S.IAM, label="median shift in the bin")
    ax.axvline(0.20, color="k", lw=0.8, ls=":"); ax.text(0.21, 0.004, "read line 0.20", fontsize=6.5)
    ax.set_xlabel("neutrophil fraction of the specimen"); ax.set_ylabel("rise in Met-A"); ax.set_xlim(0, 1); ax.set_ylim(0, 0.1)
    ax.legend(loc="upper left", fontsize=6); ax.set_title("The signal falls with the fraction; the noise does not")
    S.save(fig, "part4", "fig_p4_13_lowfrac")
    return rows


def fig_markers():
    c = comp(); groups = c["groups"]; mu = np.array([c["mu_markers"][g] for g in groups])   # 8 x 963
    # order markers by the group they separate (largest |mu_g - mean of the others|)
    dev = np.array([np.abs(mu[i] - np.mean(np.delete(mu, i, 0), 0)) for i in range(len(groups))])
    top = np.argmax(dev, 0); order = np.lexsort((-dev.max(0), top))
    fig, ax = plt.subplots(figsize=(S.TEXTW, 1.9))
    im = ax.imshow(mu[:, order], aspect="auto", cmap="cividis", vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(range(len(groups)), groups, fontsize=6.5)
    cnt = pd.Series(top).value_counts().reindex(range(len(groups))).fillna(0).astype(int)
    ax.set_xlabel(f"composition marker ({mu.shape[1]} sites, grouped by the cell each separates)")
    cb = fig.colorbar(im, ax=ax, pad=0.01); cb.set_label(r"$\beta$", fontsize=7); cb.ax.tick_params(labelsize=6)
    ax.set_title("Stage A templates: eight purified blood groups at the 963 markers")
    S.save(fig, "part4", "fig_p4_13_markers")
    return dict(groups=groups, n=mu.shape[1], per_group={groups[i]: int(cnt[i]) for i in range(len(groups))})


# ============================== p4_14 / p4_15 ===========================
def fig_profiles():
    c = comp(); P = {g: np.array(v, float) for g, v in c["profiles_at_neutrophil_sites"].items()}
    neu = P["NEU"]; hn = np.nanmean(H(neu))
    names = {"MONO": "monocytes", "EOS": "eosinophils", "BASO": "basophils", "NK": "NK cells", "B": "B cells", "CD8T": "CD8 T cells", "CD4T": "CD4 T cells"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(S.TEXTW, 2.4), gridspec_kw=dict(wspace=0.55))
    cols = [S.IAM, S.SKY, S.ALT, S.GOLD, S.ALT2, S.DATA, S.GR]
    out = {}
    for (g, nm), col in zip(names.items(), cols):
        d = np.abs(P[g] - neu); d = d[~np.isnan(d)]; xs = np.sort(d); ys = np.arange(1, len(xs) + 1) / len(xs)
        a1.plot(xs, ys, color=col, lw=1.0, label=nm)
        out[nm] = (float(np.median(d)), float((d <= 0.05).mean()), float(np.nanmean(H(P[g])) / hn))
    a1.axvline(0.05, color="k", lw=0.6, ls=":"); a1.set_xscale("log"); a1.set_xlim(1e-4, 0.5)
    a1.set_xlabel(r"$|\Delta\beta|$ from neutrophils at a neutrophil identity site"); a1.set_ylabel("share of the 6,000 sites")
    a1.legend(loc="upper left", fontsize=6); a1.set_title("Close in β ..."); S.panel_letter(a1, "a")
    y = np.arange(len(out))[::-1]
    a2.barh(y, [v[2] for v in out.values()], color=cols, height=0.65); a2.axvline(1, color=S.GR, lw=0.6, ls="--")
    for yi, v in zip(y, out.values()):
        a2.text(v[2] + 0.01, yi, f"{v[2]:.3f}", va="center", fontsize=6)
    a2.set_yticks(y, list(out.keys()), fontsize=6.5); a2.set_xlim(0.9, 1.5)
    a2.set_xlabel("mean $H$ over the neutrophil mean $H$"); a2.set_title("... not in entropy"); S.panel_letter(a2, "b", dx=-0.3)
    S.save(fig, "part4", "fig_p4_14_profiles")
    return out


def fig_window():
    r = nref(); b = np.array(r["profiles_mean_beta"]["neutrophils"]); sd = np.array(r["neutrophil_H_sd_shrunk"]); hm = np.array(r["neutrophil_H_mean"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(S.TEXTW, 2.4), gridspec_kw=dict(wspace=0.65))
    for lo, hi in ((0.05, 0.25), (0.75, 0.95)):
        a1.axvspan(lo, hi, color=NCOL, alpha=0.12, lw=0)
    a1.hist(b, bins=np.linspace(0, 1, 101), color=S.IAM)
    ax2 = a1.twinx(); bb = np.linspace(0.01, 0.99, 300)
    ax2.plot(bb, np.abs(np.log2((1 - bb) / bb)), color=S.DATA, lw=1.0); ax2.set_ylabel(r"$|dH/d\beta|$ (bits per unit $\beta$)", color=S.DATA)
    ax2.spines["right"].set_visible(True); ax2.tick_params(labelsize=6); ax2.set_ylim(0, 7)
    for x in (0.05, 0.25):
        ax2.annotate(f"{abs(np.log2((1-x)/x)):.2f}", (x, abs(np.log2((1 - x) / x))), xytext=(3, 3), textcoords="offset points", fontsize=6, color=S.DATA)
    a1.set_xlabel(r"healthy neutrophil mean $\beta$ at the site"); a1.set_ylabel("identity sites"); a1.set_title("The two windows")
    S.panel_letter(a1, "a")
    a2.plot(b, sd, ",", color=S.IAM, alpha=0.6)
    a2.set_xlabel(r"healthy neutrophil mean $\beta$"); a2.set_ylabel("healthy spread of $H$ (shrunk SD, bits)")
    a2.set_title("Spread of $H$ across the six arrays"); S.panel_letter(a2, "b", dx=-0.22)
    S.save(fig, "part4", "fig_p4_15_window")
    win = ((b >= 0.05) & (b <= 0.25)) | ((b >= 0.75) & (b <= 0.95))
    return dict(n=len(b), in_window=int(win.sum()), meth=int((b > 0.5).sum()), sd_med=float(np.median(sd)), sd_rng=(float(sd.min()), float(sd.max())))


def fig_shared():
    c = comp(); P = {g: np.array(v, float) for g, v in c["profiles_at_neutrophil_sites"].items()}; neu = P["NEU"]
    fig, axs = plt.subplots(1, 2, figsize=(S.TEXTW, 2.5), sharey=True)
    out = {}
    for ax, g, nm, col, l in ((axs[0], "MONO", "monocytes", S.IAM, "a"), (axs[1], "CD4T", "CD4 T cells", S.DATA, "b")):
        x = neu; y = P[g]; ok = ~np.isnan(y)
        ax.plot([0, 1], [0, 1], color=S.GR, lw=0.6, ls="--")
        ax.fill_between([0, 1], [-0.05, 0.95], [0.05, 1.05], color=S.LIGHT, alpha=0.35, lw=0)
        ax.plot(x[ok], y[ok], ",", color=col, alpha=0.7)
        frac = float((np.abs(y[ok] - x[ok]) <= 0.05).mean()); out[nm] = frac
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xlabel(r"healthy neutrophil $\beta$"); ax.set_title(f"{nm}: {100*frac:.1f} % of sites within 0.05")
        S.panel_letter(ax, l, dx=-0.18)
    axs[0].set_ylabel(r"healthy $\beta$ of the other cell")
    S.save(fig, "part4", "fig_p4_15_shared")
    return out


# ============================== p4_16 =================================
def fig_noiseterm():
    r = nref(); sd = np.array(r["neutrophil_H_sd_shrunk"]); hm = np.array(r["neutrophil_H_mean"]); blk = r["clustering_block"]
    n = len(sd); nb = n // blk
    fig, axs = plt.subplots(2, 1, figsize=(S.TEXTW, 2.6), sharex=True)
    for ax, v, lab, col in ((axs[0], hm, "healthy mean $H$ (bits)", S.IAM), (axs[1], sd, "shrunk SD of $H$ (bits)", S.DATA)):
        ax.plot(np.arange(n), v, ",", color=S.LIGHT)
        vb = v[:nb * blk].reshape(nb, blk).mean(1); ax.step(np.arange(nb) * blk, vb, where="post", color=col, lw=1.0)
        ax.set_ylabel(lab, fontsize=7)
    axs[1].set_xlabel("identity site, genome order (6,000 sites; 50-site block means as steps)")
    axs[0].set_title("The sky's reference and noise term come from six purified arrays")
    S.save(fig, "part4", "fig_p4_16_noiseterm")
    return dict(n=n, blk=blk, nb=nb, sd_med=float(np.median(sd)), sd_lo=float(sd.min()), sd_hi=float(sd.max()), hm_med=float(np.median(hm)),
                hm_mean=float(hm.mean()), base=r["healthy_clustering_median"])


# ============================== p4_17 =================================
def fig_repeats():
    t2 = pd.read_csv(DD / "t2_diag.csv")
    t2 = t2[t2.group.isin(["GSE247193", "GSE247195"])].copy()
    t2["zt"] = t2.title.str.extract(r"ZT(\d+)").astype(float)
    fig, ax = plt.subplots(figsize=(0.62 * S.TEXTW, 2.4))
    normal_band(ax)
    lab = {"GSE247195": ("donor 1 (GSE247195)", S.IAM, -0.3), "GSE247193": ("donor 2 (GSE247193)", S.DATA, 0.3)}
    out = {}
    for k, (l, c, dx) in lab.items():
        g = t2[t2.group == k]; ax.plot(g.zt + dx, g.A_own, "o", color=c, ms=3.5, label=f"{l}: SD {g.A_own.std(ddof=1):.3f}, n = {len(g)}")
        out[k] = (len(g), float(g.A_own.mean()), float(g.A_own.std(ddof=1)))
    ax.set_xlabel("time of day (zeitgeber hour)"); ax.set_ylabel("Met-A, no tare"); ax.set_ylim(0.8, 1.35)
    ax.legend(loc="lower right", fontsize=6); ax.set_title("Repeat arrays of two people, a second laboratory")
    S.save(fig, "part4", "fig_p4_17_repeats")
    d3 = pd.read_csv(DD / "chain_v3_dev3_readings.csv"); t3 = d3[(d3.test == "T3")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(S.TEXTW, 2.3), gridspec_kw=dict(wspace=0.35))
    subj = sorted(t3.group.unique())
    for i, sname in enumerate(subj):
        g = t3[t3.group == sname]
        a1.plot(np.full(len(g), i) + np.linspace(-0.15, 0.15, len(g)), g.f_neu, "o", color=S.IAM, ms=3)
        a2.plot(np.full(g.A_rel_tared.notna().sum(), i) + np.linspace(-0.15, 0.15, g.A_rel_tared.notna().sum()), g.A_rel_tared.dropna(), "o", color=S.DATA, ms=3)
        out[sname] = (int(g.A_rel_tared.notna().sum()), float(g.A_rel_tared.mean()), float(g.A_rel_tared.std(ddof=1)), float(g.f_neu.min()), float(g.f_neu.max()))
    a1.axhline(0.5, color=S.GR, lw=0.8, ls="--"); a1.text(3.4, 0.51, "earlier read line 0.50", fontsize=6, ha="right")
    a1.axhline(0.2, color="k", lw=0.8, ls=":"); a1.text(3.4, 0.21, "read line 0.20", fontsize=6, ha="right")
    a1.set_xticks(range(4), [s.replace("subject", "person ") for s in subj], fontsize=6.5); a1.set_ylim(0, 0.7); a1.set_ylabel("neutrophil fraction")
    a1.set_title("Technical replicates (GSE250556)"); S.panel_letter(a1, "a")
    normal_band(a2); a2.set_xticks(range(4), [s.replace("subject", "person ") for s in subj], fontsize=6.5); a2.set_ylim(0.93, 1.07)
    a2.set_ylabel(r"tared $A_{\rm rel}$"); a2.set_title("Read at the 0.20 line, noise-corrected tare"); S.panel_letter(a2, "b", dx=-0.17)
    S.save(fig, "part4", "fig_p4_17_replicates")
    t3r = t3[t3.A_rel_tared.notna()]
    w = t3r.groupby("group").A_rel_tared.transform(lambda s: s - s.mean())
    out["T3_all"] = (len(t3), len(t3r), float(t3r.A_rel_tared.std(ddof=1)), float(np.sqrt((w**2).sum() / (len(t3r) - t3r.group.nunique()))),
                     int(((t3r.A_rel_tared >= 0.95) & (t3r.A_rel_tared <= 1.05)).sum()), float(t3r.A_rel_tared.min()), float(t3r.A_rel_tared.max()),
                     float(t3.f_neu.min()), float(t3.f_neu.max()))
    return out


# ============================== p4_18 =================================
def fig_nulls():
    loo = pd.read_csv(RM / "Met_A_Floors" / "metA_floors_v1_3_loo.csv"); acc = pd.read_csv(MP / "chain_tests" / "chain_acceptance.csv"); d = dnmt()
    rows = [("held-out reference arrays, sites re-chosen", loo.A_loo), ("held-out reference arrays, frozen sites", loo.A_loo_frozen_sites),
            ("DNA mixtures, tared", acc.loc[acc.group.str.startswith("known"), "A_rel_tared"].dropna()),
            ("remission bloods, another laboratory, tared", acc.loc[acc.group.str.startswith("AML"), "A_rel_tared"].dropna()),
            ("vehicle arrays, cell lines", d.loc[d.cmpd == "DMSO", "A"]),
            ("inactive look-alike compound, 10 µM", d.loc[d.cmpd == "GSK477", "A"]),
            ("active drug at 3.2-16 nM", d.loc[(d.cmpd == "GSK032") & (d.dose_nM > 0) & (d.dose_nM <= 16), "A"])]
    fig, ax = plt.subplots(figsize=(0.62 * S.TEXTW, 2.5))
    normal_band(ax, axis="x")
    y = np.arange(len(rows))[::-1]; out = []
    for yi, (lab, v) in zip(y, rows):
        v = np.asarray(v, float); ax.plot([v.min(), v.max()], [yi, yi], color=S.IAM, lw=1.0)
        ax.plot(v, np.full(len(v), yi), "o", color=S.IAM, ms=3); out.append((lab, len(v), float(v.min()), float(v.max())))
        ax.text(1.075, yi, f"n = {len(v)}", va="center", fontsize=6)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=6.5); ax.set_xlim(0.94, 1.09); ax.set_xlabel("Met-A")
    ax.set_title("Null tests: nothing should move, and nothing does")
    S.save(fig, "part4", "fig_p4_18_nulls")
    sc = pd.read_csv(DD / "selfconsist.csv"); sc = sc[sc.f_true >= 0.5]
    fig, ax = plt.subplots(figsize=(0.5 * S.TEXTW, 2.4))
    a = sc.A_nnls_damaged - sc.A_nnls; b = sc.A_fit_damaged - sc.A_fit
    x = np.arange(len(sc))
    ax.bar(x - 0.18, a, 0.36, color=S.IAM, label="fraction from the 963 markers")
    ax.bar(x + 0.18, b, 0.36, color=S.LIGHT, label="fraction re-fitted on the neutrophil sites")
    ax.set_xticks(x, [f"{f:.2f}" for f in sc.f_true], fontsize=6.5); ax.set_xlabel("true neutrophil fraction of the mixture")
    ax.set_ylabel("rise in Met-A from a planted 2 % loss"); ax.legend(loc="upper left", fontsize=6); ax.set_ylim(0, 0.085)
    ax.set_title("Re-fitting the fraction absorbs part of the loss")
    S.save(fig, "part4", "fig_p4_18_planted")
    return dict(nulls=out, planted=(len(sc), float(a.min()), float(a.max()), float(b.min()), float(b.max()), float((sc.f_fit_damaged - sc.f_fit).median())))


# ============================== p4_19 =================================
def fig_detlimit():
    d3 = pd.read_csv(DD / "chain_v3_dev3_readings.csv"); d = d3[d3.det_limit_y.notna()]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(S.TEXTW, 2.3))
    a1.plot(d3.f_neu, d3.shift1, "o", color=S.IAM, ms=2)
    a1.axvline(0.2, color="k", lw=0.8, ls=":"); a1.text(0.21, 0.052, "read line 0.20", fontsize=6.5)
    a1.set_xlabel("neutrophil fraction"); a1.set_ylabel("shift from a 1 % loss of the pattern"); a1.set_xlim(0, 1); a1.set_ylim(0, 0.06)
    a1.set_title("Stage M: what a 1 % loss would do here"); S.panel_letter(a1, "a")
    a2.plot(d.f_neu, d.det_limit_y, "o", color=S.DATA, ms=2)
    a2.set_xlabel("neutrophil fraction"); a2.set_ylabel("detection limit (% loss)"); a2.set_xlim(0, 1); a2.set_ylim(0, 6)
    a2.set_title("Stage T: smallest loss each reading could show"); S.panel_letter(a2, "b", dx=-0.13)
    S.save(fig, "part4", "fig_p4_19_detlimit")
    return dict(n_shift=int(d3.shift1.notna().sum()), n_det=len(d), det_med=float(d.det_limit_y.median()), det_lo=float(d.det_limit_y.min()), det_hi=float(d.det_limit_y.max()),
                sh_lo=float(d3.shift1.min()), sh_hi=float(d3.shift1.max()))


# ============================== p4_20 =================================
def fig_report():
    acc = pd.read_csv(MP / "chain_tests" / "chain_acceptance.csv")
    fig, ax = plt.subplots(figsize=(S.TEXTW, 2.9))
    normal_band(ax, axis="x")
    lab = {"healthy purified neutrophil (in floor)": ("isolated, own floor (untared)", S.GR), "known mixture, neu>=50%": ("DNA mixture (tared)", S.IAM),
           "AML second remission blood (other lab)": ("remission blood (tared)", S.DATA)}
    rows = acc.reset_index(drop=True); y = np.arange(len(rows))[::-1]; out = {"withheld": 0, "Normal": 0}
    for yi, (_, r) in zip(y, rows.iterrows()):
        l, c = lab[r.group]
        v = r.A_rel_tared if pd.notna(r.A_rel_tared) else (r.A if r.group.startswith("healthy") else np.nan)
        if pd.notna(v):
            ax.plot(v, yi, "o", color=c, ms=4); out["Normal"] += int(NORMAL[0] <= v <= NORMAL[1])
        else:
            ax.text(0.952, yi, f"withheld: neutrophil fraction {r.f_neu:.3f} below the run's 0.50 line", fontsize=5.8, va="center", color=S.GR)
            out["withheld"] += 1
        ax.text(1.072, yi, r.gsm, fontsize=5.5, va="center")
    for l, c in lab.values():
        ax.plot([], [], "o", color=c, label=l)
    ax.set_yticks([]); ax.set_xlim(0.95, 1.085); ax.set_ylim(-1, len(rows))
    ax.set_xlabel("reading on the gauge (Normal shaded)"); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, fontsize=6.5)
    ax.set_title("What 22 report pages printed, acceptance run 3")
    S.save(fig, "part4", "fig_p4_20_gauge")
    out["n"] = len(rows)
    # the words the chain printed on the 690 report pages of DEV-CHAIN-V3-RUN3 (690 processed of 692)
    d3 = pd.read_csv(DD / "chain_v3_dev3_readings.csv"); d3 = d3[d3.status == "ok"]
    st = d3.tared_state.fillna("")
    cat = pd.Series("A withheld or untared", index=d3.index)
    cat[st == "Normal"] = "Normal"; cat[st == "above Normal"] = "above Normal"; cat[st == "below Normal"] = "below Normal"
    cat[(st == "") & d3.A.notna()] = "untared: no same-run references"
    cat[(st == "") & d3.A.isna()] = "A withheld (fraction or sites)"
    order = ["Normal", "above Normal", "below Normal", "untared: no same-run references", "A withheld (fraction or sites)"]
    cnt = cat.value_counts().reindex(order).fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(0.55 * S.TEXTW, 2.1))
    yy = np.arange(len(cnt))[::-1]
    ax.barh(yy, cnt.values, color=[NCOL, S.DATA, S.IAM, S.LIGHT, S.LIGHT], height=0.65)
    for yi, v in zip(yy, cnt.values):
        ax.text(v + 6, yi, str(v), va="center", fontsize=6)
    ax.set_yticks(yy, cnt.index, fontsize=6.5); ax.set_xlim(0, cnt.max() * 1.15)
    ax.set_xlabel(f"report pages ({len(d3)} specimens)"); ax.set_title("The only words a page may print")
    S.save(fig, "part4", "fig_p4_20_states")
    out["states"] = cnt.to_dict(); out["n3"] = len(d3)
    return out


# ============================== p4_21 =================================
def fig_dnmt_arrays():
    d = dnmt(); act = d[d.cmpd == "GSK032"]
    lines = sorted(d.line.unique()); cols = dict(zip(lines, [S.IAM, S.DATA, S.ALT]))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(S.TEXTW, 2.5))
    normal_band(a1)
    for ln in lines:
        g = act[act.line == ln].sort_values("dose_nM"); x = g.dose_nM.replace(0, np.nan)
        a1.plot(x, g.A, "o", color=cols[ln], ms=3.5, label=ln); a2.plot(x, g.beta_meth_median, "o", color=cols[ln], ms=3.5, label=ln)
        v = d[(d.line == ln) & (d.cmpd == "DMSO")]
        a1.plot(np.full(len(v), 0.5), v.A, "_", color=cols[ln], ms=7); a2.plot(np.full(len(v), 0.5), v.beta_meth_median, "_", color=cols[ln], ms=7)
    for a in (a1, a2):
        a.set_xscale("log"); a.set_xlim(0.3, 2e4); a.set_xlabel("active drug dose (nM); vehicle at 0.5")
    a1.set_ylabel("Met-A against the line's own vehicle"); a1.legend(loc="upper left", fontsize=6); a1.set_title("A known block of copying reads"); S.panel_letter(a1, "a", dx=-0.16)
    a2.axhline(0.5, color="k", lw=0.8, ls=":"); a2.text(0.35, 0.52, "entropy ceiling, β = 1/2", fontsize=6.5)
    a2.set_ylabel(r"median $\beta$, methylated identity sites"); a2.set_ylim(0.3, 1.0); a2.set_title("Past β = 1/2 the reading turns back"); S.panel_letter(a2, "b")
    S.save(fig, "part4", "fig_p4_21_dnmt_arrays")
    hi = act[act.dose_nM >= 80]; lo = act[(act.dose_nM > 0) & (act.dose_nM <= 16)]
    return dict(n=len(d), veh=(d[d.cmpd == "DMSO"].A.min(), d[d.cmpd == "DMSO"].A.max()), hi=(hi.A.min(), hi.A.max()), lo=(lo.A.min(), lo.A.max()), lines=lines)


def fig_dnmt_molecules():
    rd = pd.read_csv(MP / "doors" / "PROC_DNMT_01_PARTB" / "dnmt_b_readings.csv"); pr = pd.read_csv(MP / "doors" / "PROC_DNMT_01_PARTB" / "dnmt_b_pairs.csv")
    gen = sorted(rd.genotype.unique())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(S.TEXTW, 2.4))
    for i, gname in enumerate(gen):
        for kind, col, dx in (("DMSO", S.GR, -0.12), ("DNMT1i", S.DATA, 0.12)):
            g = rd[(rd.genotype == gname) & (rd.kind == kind)]
            a1.plot(np.full(len(g), i + dx), g.eps_corr, "o", color=col, ms=4, label=("vehicle" if kind == "DMSO" else "DNMT1 inhibitor, 100 nM, 7 days") if i == 0 else None)
    a1.set_xticks(range(len(gen)), gen, fontsize=6, rotation=90); a1.set_ylabel(r"copy error $\varepsilon$ (sequencing error removed)"); a1.set_ylim(0, 0.06)
    a1.legend(loc="lower left", fontsize=6); a1.set_title("EM-seq, one leukaemia line"); S.panel_letter(a1, "a")
    normal_band(a2)
    a2.plot(np.arange(len(pr)), pr.A, "o", color=S.DATA, ms=4.5)
    a2.set_xticks(np.arange(len(pr)), [f"{g} {r}" for g, r in zip(pr.genotype, pr.rep)], fontsize=5.5, rotation=90)
    a2.set_ylabel("IAM-A against own vehicle"); a2.set_ylim(0.9, 2.1); a2.set_title("8 of 8 above Normal"); S.panel_letter(a2, "b", dx=-0.16)
    S.save(fig, "part4", "fig_p4_21_dnmt_molecules")
    return dict(veh=(rd[rd.kind == "DMSO"].eps_corr.min(), rd[rd.kind == "DMSO"].eps_corr.max()), trt=(rd[rd.kind == "DNMT1i"].eps_corr.min(), rd[rd.kind == "DNMT1i"].eps_corr.max()),
                A=(pr.A.min(), pr.A.max()), conv=float(pr.conv_diff.max()))


# ============================== p4_22 =================================
def fig_fraction_free():
    t4, neg, b, pred, r2 = covid495()
    t4 = t4.copy(); X = np.c_[np.ones(len(t4)), t4.f_neu, t4.N]; t4["corr"] = t4.A / (X @ b)
    Xn = np.c_[np.ones(len(neg)), neg.f_neu, neg.N]; y = neg.A.values; loo = []
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False; bb = np.linalg.lstsq(Xn[m], y[m], rcond=None)[0]; loo.append(y[i] / (Xn[i] @ bb))
    t4.loc[neg.index, "corr"] = loo
    rho_raw = t4[["f_neu", "A"]].corr(method="spearman").iloc[0, 1]; rho = t4[["f_neu", "corr"]].corr(method="spearman").iloc[0, 1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(S.TEXTW, 2.3), sharex=True)
    a1.plot(t4.f_neu, t4.A, "o", color=S.LIGHT, ms=2.2); a1.set_ylabel("untared Met-A"); a1.set_title(f"Untared (ρ with fraction {rho_raw:.2f})")
    normal_band(a2); a2.plot(t4.f_neu, t4["corr"], "o", color=S.IAM, ms=2.2); a2.set_ylabel("divided by its own expectation")
    a2.set_title(f"Over its own expectation (ρ {rho:.2f})"); a2.set_ylim(0.88, 1.15)
    for a, l in ((a1, "a"), (a2, "b")):
        a.set_xlabel("neutrophil fraction"); S.panel_letter(a, l, dx=-0.16)
    S.save(fig, "part4", "fig_p4_22_fraction")
    return dict(n=len(t4), rho_raw=rho_raw, rho=rho)


# ============================== p4_23 =================================
def fig_tumour():
    tp = pd.read_csv(DD / "tumour_pairs.csv"); tp = tp[tp.eps_corr_tumour.notna()]
    fig, ax = plt.subplots(figsize=(0.55 * S.TEXTW, 2.7))
    lim = (0.03, 0.05); ax.plot(lim, lim, color=S.GR, lw=0.6, ls="--")
    sty = {("EOCRC", "WGBS"): ("o", S.DATA, "colorectal, WGBS"), ("OSCC", "WGBS"): ("s", S.IAM, "oral, WGBS"),
           ("OSCC", "oxWGBS"): ("D", S.SKY, "oral, 5hmC removed")}
    for (st, asy), g in tp.groupby(["study", "assay"]):
        mk, c, l = sty[(st, asy)]; ax.plot(g.eps_corr_normal, g.eps_corr_tumour, mk, color=c, ms=4.5, label=l)
    lim5 = tp[~tp.instrument_ok]
    ax.plot(lim5.eps_corr_normal, lim5.eps_corr_tumour, "o", mfc="none", mec="k", ms=7, label="conversion diff. > 0.005")
    ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_aspect("equal")
    ax.set_xlabel(r"normal tissue, copy error $\varepsilon$"); ax.set_ylabel(r"tumour, same person, $\varepsilon$")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.33), fontsize=5.8); ax.set_title("Above the diagonal: more copy error")
    S.save(fig, "part4", "fig_p4_23_tumour")
    return tp


def fig_plasma():
    txt = open(MP / "doors" / "PROC_MOLECULE_01_OUTCOME.md", encoding="utf-8").read()
    m = re.search(r"detected in (\d+)/20 at 10 %, (\d+)[–-](\d+)/20 at 3 %,\s*(\d+)[–-](\d+)/20 at 1 %, (\d+)/20 at 0\.1 %", txt)
    hits = [(10, int(m.group(1)), int(m.group(1))), (3, int(m.group(2)), int(m.group(3))), (1, int(m.group(4)), int(m.group(5))), (0.1, int(m.group(6)), int(m.group(6)))]
    m2 = re.search(r"10 %: K562 (\d+)/20, GM12878 (\d+)/20, HepG2 (\d+)/20", txt)
    thr10 = [int(x) for x in m2.groups()]
    fig, ax = plt.subplots(figsize=(0.55 * S.TEXTW, 2.4))
    for f, lo, hi in hits:
        ax.plot([f, f], [lo, hi], color=S.IAM, lw=2.2); ax.plot([f], [(lo + hi) / 2], "o", color=S.IAM, ms=4)
    ax.plot([], [], "-o", color=S.IAM, label="total isolated-error count (range over 3 cancer lines)")
    ax.plot([10] * 3, thr10, "x", color=S.DATA, ms=5, label="per-molecule threshold, 10 %")
    ax.plot([1], [0], "x", color=S.DATA, ms=5); ax.annotate("threshold at 1 %: 0/20", (1, 0), xytext=(6, 4), textcoords="offset points", fontsize=6, color=S.DATA)
    ax.set_xscale("log"); ax.set_xlim(0.07, 15); ax.set_ylim(-1, 21.5); ax.set_xlabel("cancer DNA in the constructed mixture (%)")
    ax.set_ylabel("mixtures detected (of 20)"); ax.legend(loc="center left", fontsize=6); ax.set_title("Constructed mixtures, 0.5-1.5 M molecules")
    S.save(fig, "part4", "fig_p4_23_plasma")
    return dict(hits=hits, thr10=thr10)


# ============================== p4_24 =================================
def status_counts():
    txt = open(P4 / "p4_24_status.tex", encoding="utf-8").read()
    body = txt.split(r"\endhead", 1)[1]
    rows = [r for r in body.split(r"\\") if "&" in r and "multicolumn" not in r]
    cnt = {}
    for r in rows:
        st = r.split("&")[1]
        for macro, name in ((r"\calc", "calculated"), (r"\derived", "derived"), (r"\conjecture", "conjecture"), (r"\analogy", "analogy"),
                            (r"\calibrated", "calibrated"), (r"\measured", "measured"), (r"\openprob", "open problem"), (r"\prediction", "prediction"),
                            (r"\fitted", "fitted")):
            if macro in st:
                cnt[name] = cnt.get(name, 0) + 1
        plain = re.sub(r"\\[a-z]+(\{[^}]*\})?", "", st).replace("~\\%", " %").strip(" ,")
        if plain and not any(m in st for m in (r"\calc", r"\derived", r"\conjecture", r"\analogy", r"\calibrated", r"\measured", r"\openprob", r"\prediction", r"\fitted")):
            cnt["other: " + plain] = cnt.get("other: " + plain, 0) + 1
    return len(rows), cnt


def fig_status():
    n, cnt = status_counts()
    lab = [k for k in cnt if not k.startswith("other")]; oth = sorted([k for k in cnt if k.startswith("other")])
    keys = sorted(lab, key=lambda k: -cnt[k]) + oth
    fig, ax = plt.subplots(figsize=(0.62 * S.TEXTW, 2.6))
    y = np.arange(len(keys))[::-1]
    ax.barh(y, [cnt[k] for k in keys], color=[S.LIGHT if k.startswith("other") else S.IAM for k in keys], height=0.7)
    for yi, k in zip(y, keys):
        ax.text(cnt[k] + 0.15, yi, str(cnt[k]), va="center", fontsize=6)
    ax.set_yticks(y, [k.replace("other: ", "") for k in keys], fontsize=6.2); ax.set_xlabel(f"labels in Table tab:status ({n} rows)")
    ax.set_xlim(0, max(cnt.values()) + 2); ax.set_title("What carries which label")
    S.save(fig, "part4", "fig_p4_24_labels")
    return n, cnt


def fig_summary():
    loo = pd.read_csv(RM / "Met_A_Floors" / "metA_floors_v1_3_loo.csv"); acc = pd.read_csv(MP / "chain_tests" / "chain_acceptance.csv"); d = dnmt()
    pr = pd.read_csv(MP / "doors" / "PROC_DNMT_01_PARTB" / "dnmt_b_pairs.csv"); tp = pd.read_csv(DD / "tumour_pairs.csv").dropna(subset=["ratio"]); im = imr90()
    g = pd.read_csv(MP / "chain_tests" / "iama_floor_granulocytes.csv")
    rows = [("held-out reference neutrophils (Met-A)", loo.A_loo, S.IAM), ("healthy granulocytes, held out (IAM-A)", g.A_own, S.IAM),
            ("tared DNA mixtures and remission bloods (Met-A)", acc.A_rel_tared.dropna(), S.IAM),
            ("senescent IMR90, unmethylated channel", im.loc[im.state == "Senescent", "A_unmeth"], S.GOLD),
            ("SV40 IMR90, methylated channel", im.loc[im.state == "SV40", "A_meth"], S.DATA),
            ("DNMT1 inhibitor ≥ 80 nM (Met-A)", d.loc[(d.cmpd == "GSK032") & (d.dose_nM >= 80), "A"], S.DATA),
            ("DNMT1 inhibitor 100 nM (IAM-A)", pr.A, S.DATA)]
    fig, ax = plt.subplots(figsize=(S.TEXTW, 2.4))
    normal_band(ax, axis="x")
    y = np.arange(len(rows))[::-1]; out = []
    for yi, (lab, v, c) in zip(y, rows):
        v = np.asarray(v, float); ax.plot([v.min(), v.max()], [yi, yi], color=c, lw=1.0); ax.plot(v, np.full(len(v), yi), "o", color=c, ms=3)
        out.append((lab, len(v), float(v.min()), float(v.max())))
    ax.set_xscale("log"); ax.set_xlim(0.6, 2.2); ax.set_xticks([0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 2.0], ["0.6", "0.7", "0.8", "0.9", "1", "1.25", "1.5", "2"])
    ax.minorticks_off(); ax.set_yticks(y, [r[0] for r in rows], fontsize=6.3); ax.set_xlabel("reading on the one gauge (log scale; healthy = 1)")
    ax.set_title("Measured readings of Part 4 on one gauge (each against its own healthy reference)")
    S.save(fig, "part4", "fig_p4_24_summary")
    return out


ALL = [fig_jensen, fig_ledger, fig_fullsurface, fig_surfaces, fig_imr90_plane, fig_heldout, fig_iama, fig_cscore, fig_fish, fig_map79, fig_tare, fig_noisefit,
       fig_lowfrac, fig_markers, fig_profiles, fig_window, fig_shared, fig_noiseterm, fig_repeats, fig_nulls, fig_detlimit, fig_report, fig_dnmt_arrays,
       fig_dnmt_molecules, fig_dnmt_channels, fig_fraction_free, fig_tumour, fig_plasma, fig_status, fig_summary]

def tex_sci(x, d=2, dollars=True):
    e = int(np.floor(np.log10(abs(x)))); m = x / 10**e
    s = f"{m:.{d-1}f}\\times10^{{{e}}}"
    return f"${s}$" if dollars else s


def tables():
    """Return the LaTeX bodies of the Part 4 tables added for TODO 9.2; every number is computed here."""
    T = {}
    # p4_01: surfaces
    rows = []
    for lab, Tk, nb in (("horizon, one solar mass", bh(MSUN)[0], bh(MSUN)[1] / (C.k * LN2)),
                        ("horizon, $10^6$ solar masses", bh(1e6 * MSUN)[0], bh(1e6 * MSUN)[1] / (C.k * LN2)),
                        ("superconducting qubit, 20 mK", 0.020, None), ("cell nucleus, 310.15 K", T0, CPG_BITS)):
        e = C.k * Tk * LN2
        rows.append(f"{lab} & {tex_sci(Tk)} & {tex_sci(e)} & " + (f"{tex_sci(nb)} & {tex_sci(nb*e)}" if nb else "one per qubit & ---") + r" \\")
    T["p4_01_surfaces"] = "\n".join(rows)
    # p4_03: Jensen on the identity sites
    j = fig_jensen(); names = ["methylated channel", "unmethylated channel", "both channels"]
    T["p4_03_jensen"] = "\n".join(f"{names[i]} & {j['n'][i]:,} & {j['mean_beta'][i]:.3f} & {j['Hofmean'][i]:.3f} & {j['Hbar'][i]:.3f} \\\\" for i in range(3))
    T["_jensen"] = j
    # p4_04: the halves
    l = fig_ledger()
    T["p4_04_halves"] = "\n".join([
        f"hydrogen, ground state & $2\\langle K\\rangle=-\\langle V\\rangle$ & $\\langle K\\rangle={l['K_H']:.2f}$ eV, $\\langle V\\rangle={l['V_H']:.2f}$ eV & \\derived \\\\",
        "slow contraction of a self-gravitating body & $E=U/2$ & half of $|\\Delta U|$ heats, half is radiated & \\derived \\\\",
        "charging a capacitance from a fixed supply & drawn $CV^2$ & $\\tfrac12CV^2$ stored, $\\tfrac12CV^2$ dissipated & \\derived \\\\",
        f"Schwarzschild horizon, one solar mass & $Mc^2=2T_HS$ & $T_HS={tex_sci(l['TS1'],3,False)}$ J; $Mc^2/2={tex_sci(l['mc2half'],3,False)}$ J & \\calc \\\\",
        f"equipartition, 310.15 K & $\\tfrac12\\kB T$ per quadratic term & {tex_sci(l['equip'],3)} J (a different half) & \\derived \\\\"])
    T["_ledger"] = l
    # p4_07: the six reference arrays
    h = fig_heldout(); loo, acc, ni = h["loo"], h["acc"], h["ni"].set_index("gsm")
    fl = floors(); sent = {r.split("_", 1)[0]: r.split("_", 1)[1] for r in fl["refs"]}
    rr = []
    for _, r in loo.iterrows():
        g = r.ref; n_ = ni.loc[g, "N"] if g in ni.index else np.nan
        rr.append(f"{g} & {sent[g].replace('_', chr(92)+'_')} & {acc.loc[g,'A']:.4f} & {r.A_loo:.3f} & {r.A_loo_frozen_sites:.3f} & {n_:.4f} & {acc.loc[g,'C']:.2f} \\\\")
    T["p4_07_refarrays"] = "\n".join(rr)
    # p4_09: C-score
    c = fig_cscore()
    T["p4_09_cscore"] = "\n".join([
        f"healthy reference arrays, each against the other five & {c['healthy arrays, each against the other five'][0]} & {c['healthy arrays, each against the other five'][1]:.2f}--{c['healthy arrays, each against the other five'][2]:.2f} & \\calibrated \\\\",
        f"isolated reference neutrophils, acceptance run 3 & {c['isolated reference neutrophils'][0]} & {c['isolated reference neutrophils'][1]:.2f}--{c['isolated reference neutrophils'][2]:.2f} & \\measured \\\\",
        f"DNA mixtures, acceptance run 3 & {c['DNA mixtures'][0]} & {c['DNA mixtures'][1]:.2f}--{c['DNA mixtures'][2]:.2f} & \\measured \\\\",
        f"remission bloods, another laboratory & {c['remission bloods, another laboratory'][0]} & {c['remission bloods, another laboratory'][1]:.2f}--{c['remission bloods, another laboratory'][2]:.2f} & \\measured \\\\",
        f"far end: every 50-site block moving as one & --- & {c['far_end']:.0f} & \\calc \\\\"])
    T["_cscore"] = c
    # p4_16: inputs of the sky
    k = fig_noiseterm()
    T["p4_16_skyinputs"] = "\n".join([
        f"identity sites, genome order & {k['n']:,} \\\\", f"block size; blocks per map & {k['blk']}; {k['nb']} \\\\",
        f"healthy mean $H$ per site: median (mean) & {k['hm_med']:.4f} ({k['hm_mean']:.4f}) bits \\\\",
        f"shrunk SD $s_i$: median (range) & {k['sd_med']:.4f} ({k['sd_lo']:.4f}--{k['sd_hi']:.4f}) bits \\\\",
        f"healthy clustering baseline $c_{{\\rm healthy}}$ & {k['base']} \\\\"])
    T["_sky"] = k
    # p4_17: the change floor so far
    r = fig_repeats(); a = r["T3_all"]
    T["p4_17_changefloor"] = "\n".join([
        f"isolated neutrophils, donor 1, another laboratory, no tare & {r['GSE247195'][0]} & SD {r['GSE247195'][2]:.3f} & \\measured \\\\",
        f"isolated neutrophils, donor 2, same laboratory, no tare & {r['GSE247193'][0]} & SD {r['GSE247193'][2]:.3f} & \\measured \\\\",
        f"whole-blood technical replicates (GSE250556), 0.20 line, noise-corrected tare & {a[1]} of {a[0]} & within-person SD {a[3]:.3f}; all {a[2]:.3f}; {a[4]} of {a[1]} in Normal & \\measured \\\\",
        "two remission draws of one person (PROC-AML-SERIAL-01, bar S5) & 10 pairs & 10 of 10 within 0.05 & \\measured \\\\"])
    T["_repeats"] = r
    # p4_19: gates
    it = json.load(open(RM / "Intake" / "intake_thresholds_v1.json")); ip = iama()["cells"]["neutrophils"]; nr_ = nref()
    T["p4_19_gates"] = "\n".join([
        f"Stage 0 & call rate: proceed $\\ge{it['call_rate']['proceed_at_or_above']}$; flagged {it['call_rate']['quarantine_below']}--{it['call_rate']['proceed_at_or_above']}; quarantine $<{it['call_rate']['quarantine_below']}$ & \\texttt{{intake\\_thresholds\\_v1.json}} \\\\",
        f"Stage 0 & bisulfite conversion $\\ge{it['bisulfite_conversion']['min']}$: printed, not gated & \\texttt{{intake\\_thresholds\\_v1.json}} \\\\",
        "platform & EPIC v1 only; more than 700,000 probes & \\texttt{conductor\\_v3.py} \\\\",
        f"Stage A & at least 867 of the {len(comp()['markers'])} markers measured & \\texttt{{blood\\_composition\\_EPIC\\_v1.json}} \\\\",
        f"Stage M & neutrophils $\\ge20$~\\%; at least 90~\\% of the {floors()['n_sites']:,} identity sites & \\texttt{{conductor\\_v3.py}}, \\texttt{{metA\\_floors\\_v1\\_3.json}} \\\\",
        f"Stage MC & blocks of {nr_['clustering_block']} sites; at least ten blocks; baseline {nr_['healthy_clustering_median']} & \\texttt{{neutrophil\\_reference\\_v1\\_1.json}} \\\\",
        "Stage T & median tare with 3--19 references; noise-corrected tare with $\\ge20$ & \\texttt{conductor\\_v3.py} \\\\",
        f"Stage Q & pipeline \\texttt{{{ip['pipeline'].replace('_', chr(92)+'_')}}}; $P={ip['P']}$; at least 100,000 opportunities & \\texttt{{iama\\_positions\\_v1.json}} \\\\"])
    # p4_23: tumour pairs
    tp = pd.read_csv(DD / "tumour_pairs.csv"); tp = tp[tp.eps_corr_tumour.notna()]
    T["p4_23_tumour"] = "\n".join(f"{x.patient} & {x.assay} & {x.eps_corr_normal:.5f} & {x.eps_corr_tumour:.5f} & {x.ratio:.3f} & {x.conv_diff:.4f}{'' if x.instrument_ok else ' (over 0.005)'} \\\\" for x in tp.itertuples())
    return T


if __name__ == "__main__":
    if "--tables" in sys.argv:
        for k, v in tables().items():
            if not k.startswith("_"):
                print(f"% ---- {k}\n{v}\n")
    else:
        for f in ALL:
            f()
