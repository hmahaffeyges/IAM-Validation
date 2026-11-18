import subprocess as _sp, sys as _sy
_sp.run([_sy.executable, '-m', 'pip', 'install', '-q', 'numpy'], check=True)
#!/usr/bin/env python3
"""KR1 (register KR1, exploration; the author: a Koide-style record 'cannot be fully reconstructed ... perhaps time cannot move backwards because the record literally
cannot be fully reconstructed'). Runs ON A GITHUB MACHINE (numpy only, minutes). Two simulated objects whose laws run both ways:
  (1) classical: a 3-vector v (like (sqrt m_e, sqrt m_mu, sqrt m_tau)) carried by a rigid rotation about a fixed axis that is not (1,1,1);
  (2) quantum: a qutrit under a fixed random unitary; v = sqrt of its populations, so Q = exp(-H_1/2(p)) exactly.
Three records written each tick on a tile of B bits per number: KOIDE = (|v|, Q); KOIDE+PHASE = (|v|, Q, delta); FULL = v itself.
Measures, per record: (a) bits lost per write: the entropy of the true state's delta bin given the record (Koide: about B; with the phase: 0);
(b) rewind: from the record at t alone, pick every state it allows, run each back one tick with the known law, and compare with the true state at t-1
(spread of the rewound Q); (c) time symmetry of the record itself: is the record sequence of the time-reversed run the reversed sequence? (d) finite tile:
one record kept and overwritten each tick; the information lost per overwrite H(R_t-1 | R_t), priced kT ln2 per bit.
Prediction (written in the register before the run): (a) Koide loses about B bits per write, the phase record 0, FULL 0; (b) rewinding from Koide spreads
over the allowed circle, from Koide+phase is exact (up to the 6 orderings/signs it cannot tell apart); (c) the record sequence alone is time-symmetric for
both objects (compression alone gives no arrow); (d) the arrow appears with the overwrite: H(R_t-1|R_t) > 0 per tick, paid in heat."""
import numpy as np, json, os
rng = np.random.default_rng(11); B = 8; NT = 400; NS = 64          # bits per number, ticks, states allowed per record sampled for the rewind
u = np.ones(3) / np.sqrt(3)
def basis():
    e1 = np.array([1, -1, 0]) / np.sqrt(2); e2 = np.cross(u, e1); return e1, e2
E1, E2 = basis()
def koide(v):
    r = np.linalg.norm(v); c = v @ u / r; Q = 1 / (3 * c * c)
    w = v - (v @ u) * u; d = np.arctan2(w @ E2, w @ E1); return r, Q, d
def from_koide(r, Q, d):
    c = 1 / np.sqrt(3 * Q); s = np.sqrt(max(0.0, 1 - c * c)); return r * (c * u + s * (np.cos(d) * E1 + np.sin(d) * E2))
def q(x, lo, hi): return int(np.clip((x - lo) / (hi - lo) * 2 ** B, 0, 2 ** B - 1))
def rot(axis, ang):
    a = axis / np.linalg.norm(axis); K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]]); return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K
def run(step, back, v0):
    V = [v0]
    for _ in range(NT): V.append(step(V[-1]))
    return np.array(V)
out = {}
objs = {}
R = rot(np.array([1.0, 0.3, -0.5]), 0.071); objs["classical rotation"] = (lambda v: R @ v, lambda v: R.T @ v, np.array([0.3, 1.1, 2.4]))
H = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)); H = (H + H.conj().T) / 2; w_, P_ = np.linalg.eigh(H)
U = P_ @ np.diag(np.exp(-1j * 0.05 * w_)) @ P_.conj().T
psi0 = rng.normal(size=3) + 1j * rng.normal(size=3); psi0 /= np.linalg.norm(psi0)
objs["qutrit"] = (lambda s: U @ s, lambda s: U.conj().T @ s, psi0)
for name, (fwd, bwd, x0) in objs.items():
    vec = (lambda s: np.sqrt(np.abs(s) ** 2)) if name == "qutrit" else (lambda s: s)
    X = run(fwd, bwd, x0)
    V = np.array([vec(x) for x in X]); K = np.array([koide(v) for v in V])
    rlo, rhi = K[:, 0].min() * 0.987, K[:, 0].max() * 1.019   # off-centre: a conserved scale must not sit on a bin edge (2026-11-18 fix)
    recK = [(q(r, rlo, rhi), q(Qv, 1 / 3, 1)) for r, Qv, _ in K]
    recP = [(a, b, q(d, -np.pi, np.pi)) for (a, b), (_, _, d) in zip(recK, K)]
    # (a) bits lost per write, MEASURED (2026-11-18 fix; was set by construction): H(delta bin | Koide record) over 200,000 random states, 4 bits per number
    from collections import Counter as _C, defaultdict as _D
    _g = _D(_C); _B4 = 4; _q4 = lambda x, lo, hi: int(np.clip((x - lo) / (hi - lo) * 2 ** _B4, 0, 2 ** _B4 - 1))
    for _ in range(200000):
        if name == 'qutrit': _s = rng.normal(size=3) + 1j * rng.normal(size=3); _s /= np.linalg.norm(_s)
        else: _s = np.abs(rng.normal(size=3)); _s /= np.linalg.norm(_s); _s *= rng.uniform(0.5, 2.0)
        _r, _Q, _d = koide(vec(_s)); _g[(_q4(_r, 0.0, 2.1), _q4(_Q, 1 / 3, 1))][_q4(_d, -np.pi, np.pi)] += 1
    _n = sum(sum(c.values()) for c in _g.values())
    lostK = float(-sum(v / _n * np.log2(v / sum(c.values())) for c in _g.values() for v in c.values())); lostP = 0.0
    # (b) rewind from the record at t: sample the states the record allows, run each back one tick
    spreadK, spreadP = [], []
    for t in range(1, NT):
        r, Qv, d = K[t]; Qtrue_prev = K[t - 1][1]
        if name == "qutrit":
            # the record allows any phase pattern and any delta: sample both, rebuild psi, run back
            errs = []
            for _ in range(NS):
                vv = from_koide(r, Qv, rng.uniform(-np.pi, np.pi)); vv = np.abs(vv); vv /= np.linalg.norm(vv)
                s = vv * np.exp(1j * rng.uniform(0, 2 * np.pi, 3)); errs.append(koide(np.abs(bwd(s)))[1] - Qtrue_prev)
            spreadK.append(np.std(errs))
            errs = []
            for _ in range(NS):
                vv = from_koide(r, Qv, d); s = vv * np.exp(1j * rng.uniform(0, 2 * np.pi, 3)); errs.append(koide(np.abs(bwd(s)))[1] - Qtrue_prev)
            spreadP.append(np.std(errs))
        else:
            errs = [koide(bwd(from_koide(r, Qv, rng.uniform(-np.pi, np.pi))))[1] - Qtrue_prev for _ in range(NS)]; spreadK.append(np.std(errs))
            spreadP.append(abs(koide(bwd(from_koide(r, Qv, d)))[1] - Qtrue_prev))
    # (c) time symmetry of the record: run backward from the end; compare with the reversed forward record
    Xr = [X[-1]]
    for _ in range(NT): Xr.append(bwd(Xr[-1]))
    Kr = [koide(vec(x)) for x in Xr]; recKr = [(q(r, rlo, rhi), q(Qv, 1 / 3, 1)) for r, Qv, _ in Kr]
    sym = float(np.mean([a == b for a, b in zip(recKr, recK[::-1])]))
    sym_state = float(np.mean([np.allclose(x, y, atol=1e-9) for x, y in zip(Xr, X[::-1])]))   # the rewind itself, at full precision
    # (d) finite tile: one record kept, overwritten each tick; H(R_t-1 | R_t) over an ensemble of 20,000 random starts, records at 4 bits per number
    from collections import Counter, defaultdict
    B4 = 4; qq = lambda x, lo, hi: int(np.clip((x - lo) / (hi - lo) * 2 ** B4, 0, 2 ** B4 - 1)); pairs = defaultdict(Counter)
    for _ in range(20000):
        if name == "qutrit": s0 = rng.normal(size=3) + 1j * rng.normal(size=3); s0 /= np.linalg.norm(s0)
        else: s0 = np.abs(rng.normal(size=3)); s0 /= np.linalg.norm(s0)
        k0, k1 = koide(vec(s0)), koide(vec(fwd(s0)))
        pairs[qq(k1[1], 1 / 3, 1)][qq(k0[1], 1 / 3, 1)] += 1
    n = sum(sum(c.values()) for c in pairs.values())
    Hc = -sum(cnt / n * np.log2(cnt / sum(c.values())) for c in pairs.values() for cnt in c.values())
    out[name] = dict(bits_lost_per_write_koide=lostK, bits_lost_per_write_with_phase=lostP,
                     rewind_spread_Q_koide=float(np.median(spreadK)), rewind_spread_Q_with_phase=float(np.median(spreadP)),
                     record_time_symmetric_share=sym, rewind_state_match_share=sym_state, bits_lost_resolution='4 bits per number (of 4 possible for delta)', overwrite_bits_lost_per_tick=float(Hc), distinct_records=len(set(recK)), Q_range=[float(K[:, 1].min()), float(K[:, 1].max())])
    print("::notice::KR1 " + name + ": " + json.dumps(out[name]))
os.makedirs("xout", exist_ok=True); json.dump(out, open("xout/kr1.json", "w"), indent=1)
