"""IAM's Law Sphere (sequencing; development, DEV-SPHERE-01). The information sphere of the IAM Sequencing Atlas, read against IAM's Law.

The sphere's points are the atlas's own units: each UCSC hg19 CpG island is one unit, and outside islands each block of 50 consecutive CpGs is
one; a unit has a place when its levels across the 58 healthy types vary beyond its own person-to-person and counting noise. Its job
(territory) is the family of the type where it departs most from its own level. The runtime reference (Runtime Matrices/Seq_Atlas/law_sphere)
carries, per unit and type, the pooled level and the share of the unit's CpGs that the type holds.

A sample is read from its molecules by IAM-A's own rule (stage_q2_globe.count_pat_regions; stage_q2_selfref quality floor: a molecule of at
least 6 calls is held when at least 80 % of them are methylated, unmarked when at most 20 %, otherwise loose; a held error is an interior T
between two C's on a held molecule, a gain an interior C between two T's on an unmarked molecule; even lines give the levels, odd lines the
molecule classes, so no molecule is used twice). The law's expected point at each unit is built in the sample's own cell shares (the
Deconvolver): held CpGs at 1 - p, where p is IAM-A's own predicted held error (the self-reference line from the sample's loose share), unmarked
CpGs at the law's stray level 1/(1 + D) less the eraser's clean-up, plus the meter. Every layer is per unit and per job, each judged against its
own counting floor (stage_info_sphere.counting_floor, edit 198)."""
import os, json, numpy as np, pandas as pd
import stage_q2_selfref as Q2

HERE = os.path.dirname(os.path.abspath(__file__))
REF_DIR = os.path.join(HERE, "Runtime Matrices", "Seq_Atlas", "law_sphere")
EPS_LAW = 0.04454        # A4: the held error from M and the two-coin push (driven passage, x = 2M)
M_COIN = 20.94           # kicks per ATP coin (dG_ATP 54 kJ/mol at 310.15 K)
PUSH_LAW = 2 * M_COIN    # two coins per restore
D_SELECT = 60            # the writer's selection (A15): stray rate 1/(1 + D)
ERASER_CLEAN = 0.18      # share of strays the eraser cleans (TET2 + TET3)
CONV_DEFAULT = 0.0044    # conversion failure when the sample's own is not given (A17)
MIN_OPP = 200            # a unit's hold layers are drawn at >= 200 held-error opportunities (as stage_q2_globe)
MIN_CALLS = 20           # a unit's level layers are drawn at >= 20 calls
LAYERS = [
    ("angle_law", "Angle to the law", "rad", "the angle between the sample's level and the law's expected level, in the sample's own cells"),
    ("alarm_law", "Angle to the law over its floor", "x floor", "the angle to the law divided by the angle counting alone would give a healthy sample read this deeply"),
    ("fidelity_law", "Fidelity to the law", "0 to 1", "cos^2 of the angle to the law: 1 is exactly the law's state"),
    ("power_factor", "Power-factor reading", "cos", "cos of the angle to the law (fidelity is its square); 1 is exactly in phase with the law (an analogy to the grid's power factor)"),
    ("angle_law_atlas", "Angle between the law and the atlas", "rad", "the angle between the law's expected point and the healthy atlas's, in the sample's cells: how far healthy sits from the law"),
    ("angle_atlas", "Angle to the healthy atlas", "rad", "the angle between the sample's level and the healthy atlas level of the same cells (drift toward another job)"),
    ("hold_vs_self", "Hold against IAM-A's own line", "x predicted", "the unit's held error, from molecules, over the held error the sample's own loose share predicts"),
    ("hold_vs_law", "Hold against the law", "x law", "the unit's held error over the law's 0.0445 (A4)"),
    ("kicks_held", "Energy held per mark", "kT", "ln((1 - h)/h): what each held mark carries, in thermal kicks (law: ln 21.45 = 3.07)"),
    ("phi", "Share of each coin kept as hold", "phi", "kicks held over the coin's 20.94 (law 0.1464, P4)"),
    ("coins_per_restore", "Implied coins per restore", "coins", "the push the slow-write rule needs to give the measured hold, in coins (law 2)"),
    ("koide_q", "Koide balance", "Q", "(1 + lost/held)/3 with lost/held = h/(1 - h): 1/3 fully gentle, 2/3 Koide's sudden write (cell's law 0.349; identification conjectured)"),
    ("write_vs_law", "Stray marks against the law", "x law", "the unit's gain error on unmarked molecules over the law's stray level, meter included"),
    ("loose_share", "Loose share", "share", "molecules neither held nor unmarked: marks still being settled"),
    ("halves", "Halves agreement over its floor", "x floor", "the angle between the levels of the two halves of the file over their counting floor: the instrument's own check"),
    ("depth", "Calls per unit", "calls", "how deeply the unit was read: sets every floor")]

def unit_map(isl_first, isl_last, block, n):
    """CpG index (1-based) -> unit: islands first, then blocks of `block` consecutive CpGs outside islands."""
    u = np.full(n + 2, -1, np.int64)
    for j, (a, b) in enumerate(zip(isl_first, isl_last)):
        if a > 0 and b >= a: u[a:b + 1] = j
    nI = len(isl_first); idx = np.arange(1, n + 1); free = u[1:n + 1] < 0
    u[1:n + 1] = np.where(free, nI + (idx - 1) // block, u[1:n + 1]); return u

def load_ref(ref_dir=None):
    """The runtime reference: units.npz (island bounds, block size) and the per-unit table (seq_sphere_ref_part*.parquet)."""
    d = ref_dir or REF_DIR; Z = np.load(os.path.join(d, "units.npz"))
    parts = sorted(f for f in os.listdir(d) if f.startswith("seq_sphere_ref_part") and f.endswith(".parquet"))
    T = pd.concat([pd.read_parquet(os.path.join(d, f)) for f in parts]).sort_values("unit").reset_index(drop=True)
    return dict(isl_first=Z["isl_first"], isl_last=Z["isl_last"], block=int(Z["block"][0]), n=int(Z["n_cpg"][0]), table=T,
                cells=sorted({c.split("|")[0] for c in T.columns if c.endswith("|level")}))

def count_pat_units(lines, umap):
    """IAM-A's molecule rule per unit (see the module text). Returns per-unit arrays."""
    nU = int(umap.max()) + 1; n = len(umap) - 2
    K = ("err", "opp", "gerr", "gopp", "meth_a", "calls_a", "meth_b", "calls_b", "held", "unm", "loose")
    A = {k: np.zeros(nU) for k in K}; buf = {k: [] for k in K}
    def flush():
        for k, b in buf.items():
            if b:
                x = np.array(b, dtype=np.int64).reshape(-1, 2); x = x[(x[:, 0] >= 1) & (x[:, 0] <= n)]
                A[k] += np.bincount(umap[x[:, 0]], x[:, 1], nU); b.clear()
    for k, line in enumerate(lines):
        p = line.rstrip("\n").split("\t")
        if len(p) < 4: continue
        i, pat, w = int(p[1]), p[2], int(p[3]); pos = [i + q for q, x in enumerate(pat) if x != "."]; s = [x for x in pat if x != "."]
        if not s: continue
        side = "a" if k % 2 == 0 else "b"
        for q, x in zip(pos, s):
            buf["calls_" + side].extend((q, w))
            if x == "C": buf["meth_" + side].extend((q, w))
        if side == "b" and len(s) >= Q2.QUAL_MIN_CALLS:
            nc = s.count("C")
            if nc >= Q2.QUAL_MIN_METH * len(s):
                buf["held"].extend((pos[0], w))
                for q in range(1, len(s) - 1):
                    buf["opp"].extend((pos[q], w))
                    if s[q] == "T" and s[q - 1] == "C" and s[q + 1] == "C": buf["err"].extend((pos[q], w))
            elif nc <= (1 - Q2.QUAL_MIN_METH) * len(s):
                buf["unm"].extend((pos[0], w))
                for q in range(1, len(s) - 1):
                    buf["gopp"].extend((pos[q], w))
                    if s[q] == "C" and s[q - 1] == "T" and s[q + 1] == "T": buf["gerr"].extend((pos[q], w))
            else: buf["loose"].extend((pos[0], w))
        if len(buf["calls_a"]) > 4_000_000 or len(buf["calls_b"]) > 4_000_000: flush()
    flush(); return A

def _theta(b, e):
    return np.arccos(np.clip(np.sqrt(np.clip(b, 0, 1) * np.clip(e, 0, 1)) + np.sqrt(np.clip(1 - b, 0, 1) * np.clip(1 - e, 0, 1)), -1, 1))

_XG = np.logspace(-3, 5, 4000)
_OG = (_XG / 2) / (1 - (1 - np.exp(-_XG)) / _XG)
def push_for_odds(odds):
    """Invert the slow-write rule held/lost = (x/2)/(1 - (1 - e^-x)/x) (identity with A4's driven passage) for the push x, in kicks."""
    o = np.asarray(odds, float); return np.where(np.isfinite(o) & (o > _OG[0]), np.interp(np.log(np.clip(o, _OG[0], _OG[-1])), np.log(_OG), _XG), np.nan)

def reading(C, ref, shares, loose_share, conv=None):
    """One sample on IAM's Law Sphere. C: count_pat_units output; ref: load_ref(); shares: {type: share} from the Deconvolver (types outside
    the reference are left out and the rest renormalised); loose_share: IAM-A's. Returns (record, per-unit layers)."""
    import stage_info_sphere as IS
    T = ref["table"]; cells = ref["cells"]; u = T["unit"].to_numpy()
    s = {c: float(v) for c, v in (shares or {}).items() if c in cells and v and v > 0}; tot = sum(s.values())
    if not s: return {"status": "NOT_RUN", "reason": "no cell share in the reference"}, None
    s = {c: v / tot for c, v in s.items()}
    p = float(Q2.predict(loose_share)); cv = CONV_DEFAULT if conv is None else float(conv)
    stray = (1 / (1 + D_SELECT)) * (1 - ERASER_CLEAN); u_law = stray + cv * (1 - stray)
    e = np.zeros(len(T)); a = np.zeros(len(T))
    for c, w in s.items():
        hf = T[f"{c}|held"].to_numpy(np.float64); hf = np.where(np.isfinite(hf), hf, (T[f"{c}|level"].to_numpy(np.float64) >= 0.5))
        e += w * (hf * (1 - p) + (1 - hf) * u_law); a += w * T[f"{c}|level"].to_numpy(np.float64)
    ma, ca, mb, cb = (C[k][u] for k in ("meth_a", "calls_a", "meth_b", "calls_b")); calls = ca + cb
    b = np.where(calls > 0, (ma + mb) / np.maximum(calls, 1), np.nan); okL = calls >= MIN_CALLS
    th = np.where(okL, _theta(b, e), np.nan); fl = np.sqrt(np.where(okL, IS.counting_floor(calls, e), np.nan))
    th_la = _theta(e, a)   # edit 206: the angle between the law's and the atlas's expected points (no sample in it)
    err, opp = C["err"][u], C["opp"][u]; okH = opp >= MIN_OPP; h = np.where(okH, err / np.maximum(opp, 1), np.nan)
    odds = np.where(okH & (h > 0), (1 - h) / np.maximum(h, 1e-12), np.nan)
    gerr, gopp = C["gerr"][u], C["gopp"][u]; okG = gopp >= MIN_OPP
    nm = C["held"][u] + C["unm"][u] + C["loose"][u]
    ba = np.where(ca > 0, ma / np.maximum(ca, 1), np.nan); bb = np.where(cb > 0, mb / np.maximum(cb, 1), np.nan)
    okHv = (ca >= MIN_CALLS) & (cb >= MIN_CALLS); hfl = np.sqrt(np.where(okHv, IS.counting_floor(ca, e) + IS.counting_floor(cb, e), np.nan))
    kicks = np.log(odds)
    Lay = dict(angle_law=th, alarm_law=th / fl, fidelity_law=np.cos(th) ** 2, power_factor=np.cos(th), angle_law_atlas=th_la, angle_atlas=np.where(okL, _theta(b, a), np.nan),
               hold_vs_self=h / p, hold_vs_law=h / EPS_LAW, kicks_held=kicks, phi=kicks / M_COIN, coins_per_restore=push_for_odds(odds) / M_COIN,
               koide_q=(1 + h / np.maximum(1 - h, 1e-12)) / 3, write_vs_law=np.where(okG, (gerr / np.maximum(gopp, 1)) / u_law, np.nan),
               loose_share=np.where(nm >= 20, C["loose"][u] / np.maximum(nm, 1), np.nan),
               halves=np.where(okHv, _theta(ba, bb) / hfl, np.nan), depth=calls)
    def summ(m):
        r = dict(units=int(m.sum()))
        if not m.any(): return r
        t2, f2 = th[m & okL], fl[m & okL]
        if len(t2): r.update(angle_law_rms=round(float(np.sqrt(np.nanmean(t2 ** 2))), 5), floor_rms=round(float(np.sqrt(np.nanmean(f2 ** 2))), 5),
                             alarm_law=round(float(np.sqrt(np.nanmean(t2 ** 2)) / np.sqrt(np.nanmean(f2 ** 2))), 3),
                             fidelity_law=round(float(np.cos(np.sqrt(np.nanmean(t2 ** 2))) ** 2), 5),
                             angle_atlas_rms=round(float(np.sqrt(np.nanmean(Lay["angle_atlas"][m & okL] ** 2))), 5))
        r.update(angle_law_atlas_rms=round(float(np.sqrt(np.nanmean(th_la[m] ** 2))), 5))
        if "angle_law_rms" in r: r["power_factor"] = round(float(np.cos(r["angle_law_rms"])), 5)
        mh = m & okH
        if mh.any():
            hp = float(err[mh].sum() / max(opp[mh].sum(), 1)); od = (1 - hp) / max(hp, 1e-12)
            r.update(held_err=round(hp, 5), hold_vs_self=round(hp / p, 4), hold_vs_law=round(hp / EPS_LAW, 4), kicks_held=round(float(np.log(od)), 4),
                     phi=round(float(np.log(od)) / M_COIN, 5), coins_per_restore=round(float(push_for_odds(od)) / M_COIN, 4), koide_q=round((1 + hp / (1 - hp)) / 3, 5),
                     held_opp=int(opp[mh].sum()))
        mg = m & okG
        if mg.any(): r.update(write_vs_law=round(float(gerr[mg].sum() / max(gopp[mg].sum(), 1)) / u_law, 4))
        if (m & (nm > 0)).any(): r.update(loose_share=round(float(C["loose"][u][m].sum() / max(nm[m].sum(), 1)), 4))
        return r
    terr = T["territory"].to_numpy(); jobs = {t: summ(terr == t) for t in sorted(set(terr))}
    rec = dict(stage="IAM's Law Sphere (sequencing; development, DEV-SPHERE-01)", status="OK", units_in_reference=int(len(T)),
               predicted_held_err=round(p, 5), law_held_err=EPS_LAW, law_stray_as_read=round(u_law, 5), conversion=cv, shares_used={c: round(v, 4) for c, v in s.items()},
               whole=summ(np.ones(len(T), bool)), jobs=jobs, layers=[dict(key=k, name=nm_, unit=un, about=ab) for k, nm_, un, ab in LAYERS],
               law=dict(eps=EPS_LAW, kicks_held=round(float(np.log((1 - EPS_LAW) / EPS_LAW)), 4), phi=0.1464, coins_per_restore=2.0, koide_q=0.349))
    return rec, Lay
