#!/usr/bin/env python3
"""The Locator's healthy reference and the measured null for the 'where' stage, per Law Sphere unit.
Loss: held errors at the unit (err over opp, IAM-A's rule); gain: marks at unmarked sites (gerr over gopp). Each healthy atlas library is
self-referenced: its own overall rate s (sum err / sum opp) divides out, so a unit's healthy RELATIVE rate per type is
  r[u, c] = sum_l err[u, l] / sum_l (opp[u, l] * s_l)   over the type's atlas libraries.
A sample with Deconvolver shares w expects at unit u:  E[u] = opp[u] * s_sample * sum_c w_c r[u, c] / sum_c w_c   (types with no count at u drop out).
Statistic z[u] = (obs - E) / sqrt(E). The null is MEASURED: every atlas library read against the reference built without it (leave one out);
z is pooled across libraries by bins of E, and each bin keeps its variance inflation (phi) and its upper tail quantiles, so a p-value is read
from the healthy libraries themselves, not assumed. Places: Benjamini-Hochberg at 5 % per sample and side. Usage:
  build_where_null.py build ATLAS_CSV(gsm,cell) COUNTS_DIR OUT_DIR      -> where_rates_v1.npz, where_null_v1.json
  build_where_null.py read OUT_DIR COUNTS.npz [shares.json]           -> places (json lines)"""
import sys, os, json, glob, numpy as np, pandas as pd
EB = np.array([0, 0.5, 1, 2, 4, 8, 16, 32, 64, 1e12]); Q = [0.9, 0.99, 0.999, 0.9999]
def load(f):
    Z = np.load(f, allow_pickle=False); m = json.loads(str(Z["meta"])) if "meta" in Z.files else {}
    return Z["unit"], {k: Z[k].astype(np.float64) for k in ("err", "opp", "gerr", "gopp")}, m.get("shares", {})
def build(atlas_csv, cdir, out):
    A = pd.read_csv(atlas_csv); L = []
    for g, c in zip(A.gsm, A.cell):
        f = glob.glob(os.path.join(cdir, f"*{g}_counts.npz"))
        if f: u, C, _ = load(f[0]); L.append((g, c, u, C))
    U = np.unique(np.concatenate([x[2] for x in L])); ix = {v: i for i, v in enumerate(U)}; n = len(U)
    types = sorted({x[1] for x in L}); ti = {c: i for i, c in enumerate(types)}
    NE = np.zeros((2, len(types), n)); DE = np.zeros((2, len(types), n)); per = []
    for g, c, u, C in L:
        j = np.array([ix[v] for v in u]); t = ti[c]; row = {}
        for side, (e, o) in enumerate((("err", "opp"), ("gerr", "gopp"))):
            s = C[e].sum() / max(C[o].sum(), 1); NE[side, t, j] += C[e]; DE[side, t, j] += C[o] * s; row[side] = (j, C[e], C[o] * s)
        per.append((g, t, row))
    # leave-one-out null
    Z = {0: [], 1: []}; Ev = {0: [], 1: []}
    for g, t, row in per:
        for side in (0, 1):
            j, e, os_ = row[side]; ne = NE[side, t, j] - e; de = DE[side, t, j] - os_
            ok = de > 0; r = np.where(ok, ne / np.maximum(de, 1e-300), np.nan); E = os_ * r
            m = ok & (E > 0); Z[side].append((e[m] - E[m]) / np.sqrt(E[m])); Ev[side].append(E[m])
    null = {}
    for side, nm in ((0, "loss"), (1, "gain")):
        z = np.concatenate(Z[side]); E = np.concatenate(Ev[side]); b = np.digitize(E, EB) - 1; bins = []
        for k in range(len(EB) - 1):
            zz = z[b == k]
            bins.append(dict(lo=float(EB[k]), hi=float(EB[k + 1]), n=int(len(zz)), phi=float(np.var(zz)) if len(zz) > 50 else None,
                             tail={str(q): float(np.quantile(zz, q)) for q in Q} if len(zz) > 1000 else None, z_sorted_sample=np.sort(zz)[::max(1, len(zz) // 2000)].round(4).tolist() if len(zz) > 50 else []))
        null[nm] = bins
    R = np.where(DE > 0, NE / np.maximum(DE, 1e-300), np.nan).astype(np.float32)
    os.makedirs(out, exist_ok=True); np.savez_compressed(os.path.join(out, "where_rates_v1.npz"), unit=U, loss=R[0], gain=R[1], types=np.array(types))
    json.dump(dict(kind="IAM 'where' healthy reference and measured null v1", libraries=len(L), types=len(types), units=int(n), bins=null,
                   rule="z = (obs - E)/sqrt(E); p from the healthy leave-one-out z of the same expected-count bin; Benjamini-Hochberg 5 % per sample and side"),
              open(os.path.join(out, "where_null_v1.json"), "w"))
    return len(L), n
def pval(obs, E, bins):
    """Upper-tail p of an observed count: negative binomial with mean E and variance phi x E, phi measured on the healthy leave-one-out
    null of the same expected-count bin (phi <= 1: Poisson). P(X >= obs)."""
    from scipy.stats import nbinom, poisson
    p = np.ones(len(obs)); b = np.digitize(E, EB) - 1
    for k, B in enumerate(bins):
        m = b == k
        if not m.any(): continue
        phi = B.get("phi") or 1.0
        if phi <= 1.0001: p[m] = poisson.sf(obs[m] - 1, E[m])
        else: r = E[m] / (phi - 1); p[m] = nbinom.sf(obs[m] - 1, r, r / (r + E[m]))
    return np.clip(p, 1e-300, 1)
def bh(p, q=0.05):
    n = len(p); o = np.argsort(p); k = np.nonzero(p[o] <= q * np.arange(1, n + 1) / n)[0]
    keep = np.zeros(n, bool)
    if len(k): keep[o[:k[-1] + 1]] = True
    qv = np.empty(n); qv[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]; return keep, np.minimum(qv, 1)
def read(refdir, counts, shares=None):
    Rz = np.load(os.path.join(refdir, "where_rates_v1.npz"), allow_pickle=False); N = json.load(open(os.path.join(refdir, "where_null_v1.json")))
    u, C, sh = load(counts); sh = shares or sh; types = list(Rz["types"]); w = np.array([sh.get(t, 0.0) for t in types]); w = w / w.sum() if w.sum() > 0 else np.ones(len(types)) / len(types)
    ix = {v: i for i, v in enumerate(Rz["unit"])}; j = np.array([ix.get(v, -1) for v in u]); ok = j >= 0; out = {}
    for side, (e, o, nm) in enumerate((("err", "opp", "loss"), ("gerr", "gopp", "gain"))):
        R = Rz[nm][:, np.where(ok, j, 0)].astype(np.float64); W = np.where(np.isfinite(R), w[:, None], 0); r = np.nansum(R * W, 0) / np.maximum(W.sum(0), 1e-12)
        s = C[e].sum() / max(C[o].sum(), 1); E = C[o] * s * r; m = ok & (W.sum(0) > 0) & (E > 0)
        z = np.full(len(u), np.nan); z[m] = (C[e][m] - E[m]) / np.sqrt(E[m]); p = np.ones(len(u)); p[m] = pval(C[e][m], E[m], N["bins"][nm])
        keep, qv = bh(p[m]); P = np.where(m)[0][keep]
        out[nm] = dict(tested=int(m.sum()), flagged=int(len(P)), places=[dict(unit=int(u[i]), obs=float(C[e][i]), expected=round(float(E[i]), 3), z=round(float(z[i]), 2), q=float(qv[np.searchsorted(np.where(m)[0], i)])) for i in P[np.argsort(-z[P])]])
    return out
if __name__ == "__main__":
    if sys.argv[1] == "build": print(build(*sys.argv[2:5]))
    else: print(json.dumps(read(sys.argv[2], sys.argv[3], json.load(open(sys.argv[4])) if len(sys.argv) > 4 else None))[:4000])


def split(rates_npz, out_dir, per_part=19):
    """Split where_rates_v1.npz by groups of cell types (each part holds every unit) so each file stays under the repository's upload limit."""
    Z = np.load(rates_npz); T = Z["types"]; parts = [list(range(i, min(i + per_part, len(T)))) for i in range(0, len(T), per_part)]
    for k, g in enumerate(parts, 1): np.savez_compressed(os.path.join(out_dir, f"where_rates_v1_part{k}.npz"), unit=Z["unit"], types=T[g], loss=Z["loss"][g], gain=Z["gain"][g])
    return len(parts)
