"""DEV-HAMMER-01 step 1 (2026-10-10, before any Hammer-seq or HeLa file is read): can the copy error a cell holds be predicted from
measured maintenance kinetics, as Hammer-seq (GSE131098) records them?
Model (book D1; E_hold = k_BT ln(g/f)): on a held molecule, per copy, a site opposite a methylated parent site fails with probability f;
a site opposite an unmethylated parent site is gained with probability g. Steady state: eps = f/(f+g).
Measurement, as the paper does it: parent and daughter are joined by a hairpin; the strand with MORE methylated CpGs is called the parent,
ties at random; per site the event is MM, MU, UU or UM (parent first). Coverage filter MM+MU >= 3 is per site and does not change the
pooled ratios used here, so it is not modelled. Stage Q's eps on the same molecules: isolated errors on molecules >= 6 calls, >= 80 % methylated.
Molecules: 8 CpGs each (hairpin reads of ~150 bp in methylated territory). Maintenance in time: m(t) = (1-f)(1 - 0.5 e^{-t/0.05} - 0.5 e^{-t/2}),
gain g(t) = g(1 - e^{-t/2}) (hours; >50 % at 4 min, >80 % by 30 min, done by 10 h, as reported).
Estimators of f and g from the recorded events, then eps_pred = f/(f+g), compared with the true eps and with Stage Q's eps.
Usage: python3 hammer_01.py"""
import numpy as np
rg = np.random.default_rng(20261010); L = 8; N = 400_000
def stageq(P):
    called = P.shape[1]; held = P.mean(1) >= 0.8; err = opp = 0
    Q = P[held]
    for k in range(1, called - 1):
        opp += len(Q); err += int(((Q[:, k] == 0) & (Q[:, k - 1] == 1) & (Q[:, k + 1] == 1)).sum())
    return err / opp
def run(f, g, t):
    eps = f / (f + g)
    par = (rg.random((N, L)) >= eps).astype(np.int8)                                   # 1 = methylated, stationary parent
    m = (1 - f) * (1 - 0.5 * np.exp(-t / 0.05) - 0.5 * np.exp(-t / 2)); gt = g * (1 - np.exp(-t / 2))
    u = rg.random((N, L)); dau = np.where(par == 1, u < m, u < gt).astype(np.int8)
    swap = (dau.sum(1) > par.sum(1)) | ((dau.sum(1) == par.sum(1)) & (rg.random(N) < 0.5))
    P = np.where(swap[:, None], dau, par); D = np.where(swap[:, None], par, dau)
    MM = int(((P == 1) & (D == 1)).sum()); MU = int(((P == 1) & (D == 0)).sum()); UU = int(((P == 0) & (D == 0)).sum()); UM = int(((P == 0) & (D == 1)).sum())
    fh = MU / (MM + MU); gh = UM / (UU + UM)
    return dict(f=f, g=g, t_h=t, eps_true=round(eps, 4), stageQ_eps=round(stageq(par), 4), f_hat=round(fh, 4), g_hat=round(gh, 4),
                eps_pred=round(fh / (fh + gh), 4) if fh + gh > 0 else None, parent_U_called=round(1 - P.mean(), 4))
print("case                 t     eps_true StageQ  f_hat   g_hat   eps_pred  called-parent U")
for f, g in ((0.010, 0.30), (0.020, 0.60), (0.005, 0.15), (0.030, 0.30)):
    for t in (0.07, 0.5, 2, 10, 24):
        r = run(f, g, t)
        print(f"f {f:.3f} g {g:.2f}   {t:5.2f}h  {r['eps_true']:.4f}  {r['stageQ_eps']:.4f}  {r['f_hat']:.4f}  {r['g_hat']:.4f}  {r['eps_pred']}   {r['parent_U_called']}", flush=True)
