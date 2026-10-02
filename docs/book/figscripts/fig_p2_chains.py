"""Part 2, Chapters 'The matter ledger of the horizon' (p2_04_dualsector_chains.tex) and 'Why matter and light are in different sectors'
(p2_05_dual_sector_note.tex).
fig_two_hubble: H0 for the photon sector (Level 2 Run A posterior) and the matter sector (x sqrt(1 + beta_m)), against Planck 2018 and SH0ES.
fig_sigma8_shift: sigma8 and H0 per data combination, LambdaCDM against the fixed coupling (mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv).
fig_mu0_constraints: the growth amplitude mu0: published constraints as quoted in p2_07, the book's Level 1 free-mu0 posteriors
(median and 90 % interval, chains in mgcamb_validation/chains; the prior ends at +0.2), the prediction and the Euclid full-survey forecast width."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, csv
import _bookstyle as S, _chains as CH
import matplotlib.pyplot as plt

bm = 0.15765
rows = {r["chain"]: r for r in csv.DictReader(open(S.REPO / "mgcamb_validation" / "CHAIN_EXTRACTION_FINAL.csv"))}
H0g, H0g_sd = float(rows["iam_level2_runA"]["H0"]), float(rows["iam_level2_runA"]["H0_sd"])
H0m, H0m_sd = H0g * np.sqrt(1 + bm), H0g_sd * np.sqrt(1 + bm)
planck, shoes = (67.36, 0.54), (73.04, 1.04)
print(f"photon {H0g:.2f} +/- {H0g_sd:.2f}; matter {H0m:.2f} +/- {H0m_sd:.2f}; vs Planck {(H0g-planck[0])/planck[1]:+.2f} sigma; vs SH0ES {(H0m-shoes[0])/shoes[1]:+.2f} sigma")
S.apply()
fig, ax = plt.subplots(figsize=(0.75 * S.TEXTW, 1.9))
ent = [("SH0ES (Cepheid ladder)", *shoes, S.DATA, "s"), ("matter sector, $67.16\\sqrt{1+\\beta_m}$", H0m, H0m_sd, S.IAM, "o"),
       ("Planck 2018 (CMB)", *planck, S.DATA, "s"), ("photon sector, Level 2 chain", H0g, H0g_sd, S.IAM, "o")]
for i, (lab, v, e, col, mk) in enumerate(ent):
    ax.errorbar(v, i, xerr=e, fmt=mk, color=col, ms=5, capsize=2, lw=1.0)
    ax.annotate(f"{v:.2f} ± {e:.2f}", (v, i), xytext=(0, 7), textcoords="offset points", fontsize=7, ha="center")
ax.set_yticks(range(4)); ax.set_yticklabels([e[0] for e in ent], fontsize=7)
ax.set_xlim(65.5, 75.5); ax.set_ylim(-0.6, 3.7)
ax.set_xlabel("$H_0$ (km s$^{-1}$ Mpc$^{-1}$)")
ax.set_title("Two rates, each consistent with its own probe ($-0.37\\sigma$, $-0.75\\sigma$)")
S.save(fig, "part2", "fig_two_hubble")

pairs = [("Planck", "lcdm_baseline", "iam_fixed_mu0 (r2 final)"), ("Planck + RSD", "planck_rsd_lcdm_baseline", "planck_rsd_iam_fixed"),
         ("Planck + BAO", "planck_bao_lcdm_baseline", "planck_bao_iam_fixed"), ("Planck + Pantheon+", "planck_pantheon_lcdm_baseline", "planck_pantheon_iam_fixed"),
         ("Level 2, Planck", "iam_level2_runC_lcdm", "iam_level2_runA")]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(S.TEXTW, 2.3), sharey=True, gridspec_kw=dict(wspace=0.08))
for i, (lab, l, m) in enumerate(pairs[::-1]):
    for ax, key in ((a1, "sigma8"), (a2, "H0")):
        vl, el = float(rows[l][key]), float(rows[l][key + "_sd"]); vm, em = float(rows[m][key]), float(rows[m][key + "_sd"])
        ax.plot([vl, vm], [i, i], color=S.LIGHT, lw=1.0, zorder=1)
        ax.errorbar(vl, i + 0.12, xerr=el, fmt="s", color=S.GR, ms=4, capsize=1.5, lw=0.8)
        ax.errorbar(vm, i - 0.12, xerr=em, fmt="o", color=S.IAM, ms=4, capsize=1.5, lw=0.8)
        if key == "sigma8":
            print(f"{lab:18s} sigma8 {vl:.4f} -> {vm:.4f} ({100*(vm/vl-1):+.2f} %)  H0 {float(rows[l]['H0']):.2f} -> {float(rows[m]['H0']):.2f}")
a1.set_yticks(range(5)); a1.set_yticklabels([p[0] for p in pairs[::-1]], fontsize=7)
a1.set_xlabel("$\\sigma_8$"); a2.set_xlabel("$H_0$ (km s$^{-1}$ Mpc$^{-1}$)")
a1.set_xlim(0.788, 0.83); a2.set_xlim(66.2, 68.3); a1.set_xticks([0.79, 0.80, 0.81, 0.82]); a2.set_xticks([66.5, 67.0, 67.5, 68.0])
a1.set_title("$\\sigma_8$ falls in every combination"); a2.set_title("$H_0$ does not move")
a2.errorbar([], [], xerr=[], fmt="s", color=S.GR, ms=4, label="$\\Lambda$CDM"); a2.errorbar([], [], xerr=[], fmt="o", color=S.IAM, ms=4, label="$\\beta_m=\\Omega_m/2$ fixed")
a2.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7)
a1.set_ylim(-0.5, 4.9)
S.panel_letter(a1, "a", dx=-0.45); S.panel_letter(a2, "b", dx=-0.04)
S.save(fig, "part2", "fig_sigma8_shift")

# ---------------- mu0 constraints -----------------
free = {}
for lab, (l, fx, fr) in CH.L1.items():
    X = CH.load(*fr); w = X.weight.values; v = X.mu0.values
    free[lab] = (CH.wquant(v, w, 0.5), CH.wquant(v, w, 0.05), float(np.sum(w[v < -0.135]) / w.sum()), CH.wquant(v, w, 0.95))
    print(f"{lab:18s} mu0 median {free[lab][0]:+.3f}  5-95 % {free[lab][1]:+.3f} to {free[lab][3]:+.3f}  P(mu0<-0.135) {free[lab][2]:.2f}")
pub = [("DES Y3 + external", 0.08, 0.19, 0.21), ("DESI 2024 full shape + BAO + BBN", 0.11, 0.54, 0.45),
       ("DESI full shape + CMB + DES Y3", 0.04, 0.22, 0.22), ("ACT + WMAP + SDSS + SN", 0.02, 0.19, 0.19)]
fig, ax = plt.subplots(figsize=(S.TEXTW, 2.9))
ax.axvspan(-0.136 - 0.04, -0.136 + 0.04, color=S.SKY, alpha=0.3, lw=0)
ax.axvline(-0.136, color=S.IAM, lw=1.0); ax.axvline(0, color=S.GR, lw=0.8, ls="--")
labels = []
y = 0
for lab, (med, lo, p, hi) in list(free.items())[::-1]:
    ax.plot([lo, hi], [y, y], color=S.ALT, lw=1.4); ax.plot(med, y, "o", color=S.ALT, ms=4)
    ax.plot(0.2, y, marker="|", color=S.ALT, ms=7)
    labels.append(f"this book, Level 1, {lab}"); y += 1
for lab, v, em, ep in pub[::-1]:
    ax.errorbar(v, y, xerr=[[em], [ep]], fmt="s", color=S.DATA, ms=4, capsize=2, lw=1.0)
    labels.append(lab); y += 1
ax.set_yticks(range(y)); ax.set_yticklabels(labels, fontsize=7)
ax.text(-0.185, y - 0.35, "prediction $-0.136$\n(band: Euclid $\\sigma\\approx0.04$)", color=S.IAM, fontsize=7, ha="right", va="bottom")
ax.text(0.01, y - 0.35, "general relativity", color=S.GR, fontsize=7, ha="left", va="bottom")
ax.set_ylim(-0.6, y + 0.9); ax.set_xlim(-0.8, 0.65)
ax.set_xlabel("$\\mu_0=\\mu(z{=}0)-1$")
ax.set_title("Neither the prediction nor general relativity is excluded yet")
for lab, v, em, ep in pub: print(f"{lab}: prediction at {(v+0.136)/em:.2f} sigma (lower error), GR at {v/em:.2f} sigma")
S.save(fig, "part2", "fig_mu0_constraints")
