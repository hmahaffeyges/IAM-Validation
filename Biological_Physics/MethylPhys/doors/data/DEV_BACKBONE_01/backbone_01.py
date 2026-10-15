"""DEV-BACKBONE-01 scorer (rule: doors/DEV_BACKBONE_01.md, written 2026-10-15 before any restricted reading).
Usage: python backbone_01.py synthetic        (planted pairs, no data)
       python backbone_01.py real <dir>       (the six EOCRC WGBS pairs; RUN ONLY with the author's go: steps 1-3 first)
Constants (written to backbone_01_constants.json): the healthy firm-territory range 0.0136-0.0226 is PROC-CHANNEL-01 section 7
(56 healthy cell types on the 26,800 sites every cell keeps fully methylated); the loose cut 0.70 on 100 kb bins is DEV-ARCH-01's.
Positions in the tables are chromosome << 32 | coordinate."""
import os, sys, json, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__))
K = dict(healthy_lo=0.0136, healthy_hi=0.0226, loose_cut=0.70, bin_bp=100_000, min_pooled_t=50, min_opp=5000, se_allow=2.0,
         source="PROC_CHANNEL_01_OUTCOME.md section 7 (range); DEV_ARCH_01 arch_box.py A1 (cut)")
json.dump(K, open(os.path.join(H, "backbone_01_constants.json"), "w"), indent=1)
PAIRS = ["CRC1", "CRC2", "CRC3", "CRC4", "CRC5", "CRC6"]

def bins(T): return (T.pos.values >> 32) * 100_000 + (T.pos.values & 0xFFFFFFFF) // K["bin_bp"]

def score(tabs, sub_err):
    """tabs: {(patient, tissue): per-site table}; sub_err: {(patient, tissue): sequencing-error rate}. Returns rows and verdict."""
    N = pd.concat([tabs[(p, "normal")].assign(b=bins(tabs[(p, "normal")])) for p in PAIRS])
    g = N.groupby("b")[["m", "t"]].sum(); g = g[g.t >= K["min_pooled_t"]]; firm = set(g.index[g.m / g.t >= K["loose_cut"]])
    rows = []
    for p in PAIRS:
        for tis in ("normal", "tumour"):
            T = tabs[(p, tis)]; b = bins(T); f = np.isin(b, list(firm)); kn = np.isin(b, list(g.index))
            o = float(T.opp_A[f].sum() + T.opp_B[f].sum()); e = float(T.err_A[f].sum() + T.err_B[f].sum())
            lo = kn & ~f; ol = float(T.opp_A[lo].sum() + T.opp_B[lo].sum()); el = float(T.err_A[lo].sum() + T.err_B[lo].sum())
            eps = e / max(o, 1) - sub_err[(p, tis)]
            rows.append(dict(patient=p, tissue=tis, firm_opp=o, firm_eps=eps, firm_se=np.sqrt(max(e, 1)) / max(o, 1),
                             loose_eps=(el / ol - sub_err[(p, tis)]) if ol else np.nan, loose_opp=ol))
    R = pd.DataFrame(rows); n = R[R.tissue == "normal"].set_index("patient"); t = R[R.tissue == "tumour"].set_index("patient")
    Nok = ((n.firm_eps + K["se_allow"] * n.firm_se >= K["healthy_lo"]) & (n.firm_eps - K["se_allow"] * n.firm_se <= K["healthy_hi"])).sum()
    T1 = int((t.firm_eps > n.firm_eps).sum()); T2 = int((t.firm_eps > K["healthy_hi"]).sum()); low = int((R.firm_opp < K["min_opp"]).sum())
    if low > 2 or Nok < 5: v = "INCONCLUSIVE"
    elif T1 < 5: v = "NOT MET"
    else: v = "MET" if T2 >= 4 else "NOT MET"
    return R, dict(N=int(Nok), T1=T1, T2=T2, low_opp_samples=low, verdict=v)

def planted(tumour_eps, seed):
    rng = np.random.default_rng(seed); tabs = {}; sub = {}
    for i, p in enumerate(PAIRS):
        for tis in ("normal", "tumour"):
            n = 60_000; chrom = rng.integers(1, 23, n); coord = rng.integers(0, 50_000_000, n); pos = (chrom.astype(np.int64) << 32) | coord
            b = coord // K["bin_bp"]; loose = (b % 3 == 0)                      # a third of bins loose
            m = rng.binomial(4, np.where(loose, 0.45, 0.92)); t = np.full(n, 4); opp = rng.poisson(np.where(loose, 0.3, 1.2))
            e_true = 0.016 if tis == "normal" else tumour_eps[i]
            err = rng.binomial(opp, np.where(loose, 0.05, e_true)); half = err // 2
            tabs[(p, tis)] = pd.DataFrame(dict(pos=pos, m=m, t=t, opp_A=opp - opp // 2, err_A=err - half, opp_B=opp // 2, err_B=half))
            sub[(p, tis)] = 0.0
    return tabs, sub

if __name__ == "__main__":
    if sys.argv[1] == "synthetic":
        L = ["DEV-BACKBONE-01 synthetic test (planted pairs; no data read)"]
        cases = [("tumors worse than nameplate (5 of 6 at 0.030, one at 0.016)", [0.030] * 5 + [0.016], "MET"),
                 ("no tumor excess on firm territory (all at 0.016)", [0.016] * 6, "NOT MET")]
        ok = True
        for name, te, want in cases:
            R, V = score(*planted(te, 7)); ok &= V["verdict"] == want
            L.append(f"{name}: N {V['N']}/6, T1 {V['T1']}/6, T2 {V['T2']}/6, firm opportunities per sample min {int(R.firm_opp.min())} -> {V['verdict']} (expected {want})")
        L.append("SYNTHETIC: " + ("MET" if ok else "NOT MET"))
        open(os.path.join(H, "backbone_01_synthetic.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
    elif sys.argv[1] == "real":
        raise SystemExit("REFUSED: run only with the author's go (steps 1-3 on paper first); remove this line in that commit")
