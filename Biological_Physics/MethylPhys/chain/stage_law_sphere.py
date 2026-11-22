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
import os, json, math, numpy as np, pandas as pd
import stage_q2_selfref as Q2

HERE = os.path.dirname(os.path.abspath(__file__))
REF_DIR = os.path.join(HERE, "Runtime Matrices", "Seq_Atlas", "law_sphere")
EPS_LAW = 0.04454        # A4: the held error from M and the two-coin push (driven passage, x = 2M)
M_COIN = 20.94           # kicks per ATP coin (dG_ATP 54 kJ/mol at 310.15 K)
PUSH_LAW = 2 * M_COIN    # two coins per restore
D_SELECT = 60            # the writer's selection (A15): stray rate 1/(1 + D)
ERASER_CLEAN = 0.18      # share of strays the eraser cleans (TET2 + TET3)
CONV_DEFAULT = 0.0044    # conversion failure when the sample's own is not given (A17)
R_GAS = 8.314462618     # J/(mol K)
CARD = dict(   # edit 207: every law input, defaults from the cell simulator's law card (minicell.law_card); --law-set KEY=VALUE or --law-card FILE
    T_K=310.15,          # temperature (K); with dG_ATP gives M = dG_ATP / (R T)
    dG_ATP=54000.0,      # the coin's size, J/mol (measured; minicell dG_ATP)
    coins=2.0,           # ATP coins per restore (derived, A2; minicell SAM_cost)
    push=None,           # suddenness x in kicks; when set it replaces coins x M
    held="iama",         # 'iama': held error from IAM-A's own line for this sample; 'law': from the push (driven passage / slow-write rule)
    line_firm=None, line_slope=None,   # IAM-A's line (held error = firm + slope x loose share); None = the chain's committed line
    D=60.0,              # writer's selection: stray rate 1/(1 + D) (measured; minicell D_select)
    eraser_share=0.0,    # share of held losses made by the eraser (minicell eraser_share): held error = restore error / (1 - share)
    eraser_clean=0.18,   # share of strays the eraser cleans (measured; minicell eraser_stray_clean)
    conv=None,           # conversion failure; None = 0.0044 (measured; minicell conv_fail)
    by_type={},          # per cell type, any of: coins, push, eraser_share, held_mult (A8 written as a multiplier on held error)
    by_job={})           # per job (territory), held_mult: a job-specific multiplier
def card_of(over=None):
    """The default card with overrides (dict, or 'KEY=VALUE' strings); unknown keys are refused."""
    c = json.loads(json.dumps(CARD))
    if isinstance(over, (list, tuple)):
        o = {}
        for kv in over:
            k, v = kv.split("=", 1); k = k.strip()
            try: v = json.loads(v)
            except Exception: pass
            o[k] = v
        over = o
    for k, v in (over or {}).items():
        if k not in c: raise KeyError(f"unknown law-card input {k!r}; known: {sorted(c)}")
        c[k] = v
    return c
def odds_of_push(x):
    """Held over lost for a write of x kicks: (x/2)/(1 - (1 - e^-x)/x), the slow-write rule (identical to A4's driven passage)."""
    x = float(x); return (x / 2) / (1 - (1 - np.exp(-x)) / x)
def loss_of_eps(eps):
    """The per-call loss q whose isolated losses q(1 - q)^2 equal the counter's held error eps (IAM-A's counter and the book's eps count a
    loss only with both neighbours marked). Assumes losses are independent along the molecule; clustering (coupling) would change it."""
    e = np.asarray(eps, float); q = e.copy()
    for _ in range(30): f = q * (1 - q) ** 2 - e; d = (1 - q) * (1 - 3 * q); q = np.clip(q - f / d, 0, 1 / 3)
    return q
def eps_of(card, cell=None, loose_share=None):
    """The held error the card predicts for one cell type."""
    t = (card.get("by_type") or {}).get(cell, {}) if cell else {}
    if card["held"] == "iama" and "coins" not in t and "push" not in t:
        if card.get("line_firm") is not None: e = float(card["line_firm"]) + float(card.get("line_slope") or 0) * float(loose_share or 0)
        else: e = float(Q2.predict(loose_share))
    else:
        M = card["dG_ATP"] / (R_GAS * card["T_K"]); x = t.get("push", card["push"] if card["push"] is not None else t.get("coins", card["coins"]) * M)
        e = 1.0 / (1.0 + odds_of_push(x))
    sh = float(t.get("eraser_share", card["eraser_share"])); e = e / max(1e-9, 1 - sh)
    return min(0.5, e * float(t.get("held_mult", 1.0)))
MIN_OPP = 200            # a unit's hold layers are drawn at >= 200 held-error opportunities (as stage_q2_globe)
PILEUP_X = 10            # edit 218: a unit read deeper than 10 x the sample's median calls per unit is a depth pile-up (repeats), left out of pooled hold
def pileup_mask(calls):
    """Edit 218. True where a unit's calls exceed PILEUP_X times the median calls over the units read (a depth pile-up). A chain rule, not physics."""
    c = np.asarray(calls, float); r = c[c > 0]
    return c > PILEUP_X * float(np.median(r)) if len(r) else np.zeros(len(c), bool)
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

def reading(C, ref, shares, loose_share, conv=None, card=None):
    """One sample on IAM's Law Sphere. C: count_pat_units output; ref: load_ref(); shares: {type: share} from the Deconvolver (types outside
    the reference are left out and the rest renormalised); loose_share: IAM-A's. Returns (record, per-unit layers)."""
    import stage_info_sphere as IS
    T = ref["table"]; cells = ref["cells"]; u = T["unit"].to_numpy()
    s = {c: float(v) for c, v in (shares or {}).items() if c in cells and v and v > 0}; tot = sum(s.values())
    if not s: return {"status": "NOT_RUN", "reason": "no cell share in the reference"}, None
    s = {c: v / tot for c, v in s.items()}
    card = card_of(card) if not isinstance(card, dict) or set(card) != set(CARD) else card
    ls_src = "IAM-A"
    if loose_share is None:   # edit 210: the sphere's own loose share when IAM-A gives none
        _nm = C["held"][u] + C["unm"][u] + C["loose"][u]; loose_share = float(C["loose"][u].sum() / max(_nm.sum(), 1)); ls_src = "the sphere's own units (IAM-A gave none)"
    M_c = float(card["dG_ATP"]) / (R_GAS * float(card["T_K"]))   # edit 237: the law's constants from the card, not fixed
    x_c = float(card["push"]) if card["push"] is not None else float(card["coins"]) * M_c
    eps_c = 1.0 / (1.0 + odds_of_push(x_c)); floor_c = 1.0 / (1.0 + math.exp(min(M_c, 700.0)))
    p = float(Q2.predict(loose_share)); cv = float(card["conv"]) if card["conv"] is not None else (CONV_DEFAULT if conv is None else float(conv))
    stray = (1 / (1 + float(card["D"]))) * (1 - float(card["eraser_clean"])); u_law = stray + cv * (1 - stray)
    pt = {c: eps_of(card, c, loose_share) for c in s}; terr0 = T["territory"].to_numpy()
    jm = np.ones(len(T))
    for j, v in (card.get("by_job") or {}).items(): jm[terr0 == j] = float((v or {}).get("held_mult", 1.0))
    e = np.zeros(len(T)); a = np.zeros(len(T))
    for c, w in s.items():
        hf = T[f"{c}|held"].to_numpy(np.float64); hf = np.where(np.isfinite(hf), hf, (T[f"{c}|level"].to_numpy(np.float64) >= 0.5))
        e += w * (hf * (1 - loss_of_eps(np.minimum(0.148, pt[c] * jm))) + (1 - hf) * u_law); a += w * T[f"{c}|level"].to_numpy(np.float64)
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
               hold_vs_self=h / p, hold_vs_law=h / eps_c, kicks_held=kicks, phi=kicks / M_c, coins_per_restore=push_for_odds(odds) / M_c,
               koide_q=(1 + h / np.maximum(1 - h, 1e-12)) / 3, write_vs_law=np.where(okG, (gerr / np.maximum(gopp, 1)) / u_law, np.nan),
               loose_share=np.where(nm >= 20, C["loose"][u] / np.maximum(nm, 1), np.nan),
               halves=np.where(okHv, _theta(ba, bb) / hfl, np.nan), depth=calls)
    pile = pileup_mask(calls)
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
        mh = m & ~pile & (opp > 0)   # edit 218: all units of the set, pile-ups left out (was m & okH)
        if mh.any():
            hp = float(err[mh].sum() / max(opp[mh].sum(), 1)); od = (1 - hp) / max(hp, 1e-12)
            r.update(held_err=round(hp, 5), hold_vs_self=round(hp / p, 4), hold_vs_law=round(hp / eps_c, 4), kicks_held=round(float(np.log(od)), 4),
                     phi=round(float(np.log(od)) / M_c, 5), coins_per_restore=round(float(push_for_odds(od)) / M_c, 4), koide_q=round((1 + hp / (1 - hp)) / 3, 5),
                     held_opp=int(opp[mh].sum()))
        mv = m & okHv & ~pile   # edit 218: halves, the strand-to-strand angle over its counting floor
        if mv.any():
            _t = _theta(ba[mv], bb[mv]); r.update(halves=round(float(np.sqrt(np.nanmean(_t ** 2)) / np.sqrt(np.nanmean(hfl[mv] ** 2))), 3), halves_units=int(mv.sum()))
        r.update(pileup_units=int((m & pile).sum()))
        mg = m & okG & ~pile
        if mg.any(): r.update(write_vs_law=round(float(gerr[mg].sum() / max(gopp[mg].sum(), 1)) / u_law, 4))
        if (m & (nm > 0)).any(): r.update(loose_share=round(float(C["loose"][u][m].sum() / max(nm[m].sum(), 1)), 4))
        return r
    terr = T["territory"].to_numpy(); jobs = {t: summ(terr == t) for t in sorted(set(terr))}
    rec = dict(stage="IAM's Law Sphere (sequencing; development, DEV-SPHERE-01)", status="OK", units_in_reference=int(len(T)), card=card, loose_share=round(float(loose_share), 4), loose_share_source=ls_src,
               predicted_held_err_by_type={c: round(v, 5) for c, v in pt.items()},
               predicted_held_err=round(p, 5), law_held_err=round(eps_c, 6), law_stray_as_read=round(u_law, 5), conversion=cv, shares_used={c: round(v, 4) for c, v in s.items()},
               whole=summ(np.ones(len(T), bool)), jobs=jobs, layers=[dict(key=k, name=nm_, unit=un, about=ab) for k, nm_, un, ab in LAYERS],
               law=dict(eps=round(eps_c, 6), M=round(M_c, 5), push=round(x_c, 4), floor=floor_c, kicks_held=round(float(np.log((1 - eps_c) / eps_c)), 4),
                        phi=round(float(np.log((1 - eps_c) / eps_c)) / M_c, 5), coins_per_restore=round(x_c / M_c, 4), koide_q=round((1 + eps_c / (1 - eps_c)) / 3, 5),
                        held_to_lost=round((1 - eps_c) / eps_c, 4), lost_over_held=round(eps_c / (1 - eps_c), 6)))   # edit 241
    rec["_units"] = u.tolist()   # edit 208: unit order of the layers (popped before the bundle is written)
    return rec, Lay


COUNT_KEYS = ("err", "opp", "gerr", "gopp", "meth_a", "calls_a", "meth_b", "calls_b", "held", "unm", "loose")
def save_counts(path, C, ref, shares, loose_share, conv=None, sample=""):
    """Edit 207: the sample's per-unit molecule counts on the sphere's units, with its shares and loose share, so any card can be read again
    in seconds without the molecule file (count once, evaluate many)."""
    u = ref["table"]["unit"].to_numpy()
    np.savez_compressed(path, unit=u, **{k: C[k][u].astype(np.float32) for k in COUNT_KEYS},
                        meta=np.array(json.dumps(dict(shares=shares, loose_share=loose_share, conv=conv, sample=sample))))
def load_counts(path, ref):
    """The saved counts, laid back onto unit indexes as count_pat_units returns them; and the saved shares, loose share, conversion."""
    Z = np.load(path, allow_pickle=False); u = Z["unit"]; n = int(max(u.max(), ref["table"]["unit"].max())) + 1
    C = {}
    for k in COUNT_KEYS: a = np.zeros(n); a[u] = Z[k]; C[k] = a
    return C, json.loads(str(Z["meta"]))
def sweep(C, ref, shares, loose_share, key, values, base=None, conv=None, cell=None):
    """One input over a range (in by_type[cell] when cell is given); every whole-sample and per-job summary at each value."""
    out = []
    for v in values:
        c = card_of(base)
        if cell: c["by_type"] = dict(c.get("by_type") or {}); c["by_type"][cell] = dict(c["by_type"].get(cell, {}), **{key: v})
        else: c[key] = v
        r, _ = reading(C, ref, shares, loose_share, conv, c); out.append(dict(value=v, whole=r["whole"], jobs=r["jobs"]))
    return out
def fit(C, ref, shares, loose_share, key, lo, hi, n=41, base=None, conv=None, cell=None, target="alarm_law"):
    """The value of one input that brings the sample closest to the law: the alarm (angle over its counting floor) at its minimum, over a
    grid of n values from lo to hi, for the whole sample and for each job. Returns the curve and the best value for each."""
    vals = list(np.linspace(float(lo), float(hi), int(n)))
    S = sweep(C, ref, shares, loose_share, key, vals, base, conv, cell)
    def best(get):
        y = np.array([get(x) if get(x) is not None else np.nan for x in S], float)
        if not np.isfinite(y).any(): return None
        i = int(np.nanargmin(y)); return dict(value=round(vals[i], 6), **{target: round(float(y[i]), 5)}, at_edge=i in (0, len(vals) - 1))
    jobs = sorted(S[0]["jobs"]) if S else []
    return dict(key=key, cell=cell, target=target, values=[round(v, 6) for v in vals],
                whole=best(lambda x: x["whole"].get(target)), jobs={j: best(lambda x, j=j: (x["jobs"].get(j) or {}).get(target)) for j in jobs},
                curve=[dict(value=round(x["value"], 6), whole=x["whole"].get(target)) for x in S])

# ---- edit 208: the Law Sphere as a page (the website's globe viewer, one layer per output, each a coded difference from its healthy value)
TEX_H, TEX_W = 128, 256
GROUPS = [("Against the law", ["alarm_law", "angle_law", "fidelity_law", "power_factor"]),
          ("Hold (the record)", ["hold_vs_law", "hold_vs_self", "kicks_held", "phi", "coins_per_restore", "koide_q"]),
          ("Write and settling", ["write_vs_law", "loose_share"]),
          ("Healthy and the law", ["angle_law_atlas", "angle_atlas"]),
          ("Instrument", ["halves", "depth"])]
LAW_VALUE = dict(alarm_law=1.0, hold_vs_law=1.0, hold_vs_self=1.0, write_vs_law=1.0, halves=1.0, kicks_held=3.0657, phi=0.1464, coins_per_restore=2.0,
                 koide_q=0.3489, angle_law_atlas=0.0)
def departure(key, Lay):
    """Each unit's departure from its healthy value: the law's value where there is one; for the angles, from the angle counting alone gives
    (angle / alarm); for loose share and depth, from the sample's own median (no law value; said in the layer's note)."""
    v = np.asarray(Lay[key], float)
    if key in LAW_VALUE: return v - LAW_VALUE[key], "the law's value " + f"{LAW_VALUE[key]:g}"
    fl = np.asarray(Lay["angle_law"], float) / np.asarray(Lay["alarm_law"], float)
    if key in ("angle_law", "angle_atlas"): return v - fl, "the angle counting alone gives"
    if key == "fidelity_law": return v - np.cos(fl) ** 2, "the fidelity counting alone allows"
    if key == "power_factor": return v - np.cos(fl), "the power-factor reading counting alone allows"
    if key == "depth": lv = np.log10(np.where(v > 0, v, np.nan)); return lv - np.nanmedian(lv), "the sample's own median depth (log10 calls)"
    return v - np.nanmedian(v), "the sample's own median"
def texture(X, v, H=TEX_H, W=TEX_W):
    """The mean of each unit's value in every cell of an equirectangular grid; direction (cos lat cos lon, sin lat, cos lat sin lon), as
    info_sphere_viewer.territories. NaN where no unit with a value falls."""
    X = np.asarray(X, float); ok = np.isfinite(v)
    lat = np.arcsin(np.clip(X[:, 1], -1, 1)); lon = np.arctan2(X[:, 2], X[:, 0])
    r = np.clip(((0.5 - lat / np.pi) * H).astype(int), 0, H - 1); c = np.clip(((lon + np.pi) / (2 * np.pi) * W).astype(int), 0, W - 1); k = r * W + c
    s = np.bincount(k[ok], v[ok], H * W); n = np.bincount(k[ok], minlength=H * W)
    return np.where(n > 0, s / np.maximum(n, 1), np.nan).reshape(H, W)
def spec(rec, Lay, positions, sample=""):
    """A globe_viewer spec: one group per family of outputs, one 'diff' layer per output (0 no unit; 1..255 = -0.5..+0.5 x scale)."""
    import io, base64
    from PIL import Image
    pu, X = positions["unit"], positions["X"]; order = {int(u): i for i, u in enumerate(rec.get("_units", []))}
    names = {k: (n, un, ab) for k, n, un, ab in LAYERS}; groups = []
    for gname, keys in GROUPS:
        layers = []
        for key in keys:
            if key not in Lay: continue
            d, against = departure(key, Lay); vv = np.full(len(pu), np.nan)
            idx = np.array([order.get(int(u), -1) for u in pu]); m = idx >= 0; vv[m] = d[idx[m]]
            T = texture(X, vv); fin = np.abs(T[np.isfinite(T)])
            if not len(fin): continue
            sc = float(max(2 * np.quantile(fin, 0.95), 1e-6)); code = np.where(np.isfinite(T), np.clip(np.round(128 + 254 * T / sc), 1, 255), 0).astype(np.uint8)
            b = io.BytesIO(); Image.fromarray(code, "L").save(b, "PNG", optimize=True)
            above = [round(float((fin > t * sc / 100).mean()), 4) for t in range(51)]
            n_, un, ab = names[key]
            layers.append(dict(kind="diff", name=n_, src="data:image/png;base64," + base64.b64encode(b.getvalue()).decode(), scale=round(sc, 6), above=above, start=0,
                               note=f"{ab}. Colour: the departure from {against} ({un}); the slider hides departures smaller than the chosen size.",
                               share_text="of the sphere's measured cells depart by more than this"))
        if layers: groups.append(dict(name=gname, layers=layers))
    return dict(id="law_sphere", name=f"{sample}: IAM's Law Sphere".strip(": "), groups=groups,
                about=("IAM's Law Sphere (development). Each point of the sphere is a unit of the IAM Sequencing Atlas (a CpG island, or 50 CpGs "
                       "outside islands), placed by how it behaves across the 58 healthy cell types. Each layer shows, for this sample, how far each "
                       "unit departs from its healthy value: the law's value where the law gives one. Departures are read from the sample's own molecules."))

# ---- edit 212: the hold alarm (register F245-F248)
def type_positions():
    """Each healthy type's measured position on IAM-A's derived line (type_positions_v1.json; measured, not derived)."""
    return json.load(open(os.path.join(REF_DIR, "type_positions_v1.json"), encoding="utf-8"))
def hold_alarm(C, ref, cell, predicted_derived, blocks=100):
    """The sample's held error (all units, IAM-A's molecule rule) against the law, IAM-A's own derived line, and the line x the type's measured
    position. z = log(held / reference) / sqrt(se^2 + u^2): se from a block jackknife over the units in genome order (the measured spread between
    units and molecules), u the type's band uncertainty (only for the type reference). One cell type only: the line is defined per type."""
    if predicted_derived is None or not cell:
        return dict(status="NOT_RUN", reason="needs IAM-A's derived prediction for one cell type (mixtures are not read by the hold alarm)")
    u = ref["table"]["unit"].to_numpy(); err = np.asarray(C["err"], float)[u]; opp = np.asarray(C["opp"], float)[u]
    _pl = pileup_mask(np.asarray(C["calls_a"], float)[u] + np.asarray(C["calls_b"], float)[u])   # edit 218: pile-ups out of the alarm
    err = np.where(_pl, 0.0, err); opp = np.where(_pl, 0.0, opp)
    E, O = float(err.sum()), float(opp.sum())
    if O < 1e4: return dict(status="NOT_RUN", reason=f"too few held opportunities ({int(O)})")
    p = E / O; idx = np.array_split(np.arange(len(err)), blocks)
    jk = np.array([np.log((E - err[i].sum()) / max(O - opp[i].sum(), 1)) for i in idx]); se = float(np.sqrt((blocks - 1) / blocks * ((jk - jk.mean()) ** 2).sum()))
    P = type_positions()["types"].get(cell); out = dict(status="OK", held_err=round(p, 5), held_opp=int(O), se_jackknife=round(se, 4), cell=cell, refs={})
    for k, e, ub in (("law", EPS_LAW, 0.0), ("own_line", float(predicted_derived), 0.0), ("type_position", float(predicted_derived) * P["centre"] if P else None, P["u_rel_k1"] if P else None)):
        if e is None: out["refs"][k] = dict(status="NOT_RUN", reason="no measured position for this type"); continue
        z = float(np.log(p / e) / np.sqrt(se ** 2 + ub ** 2)); out["refs"][k] = dict(expected=round(e, 5), ratio=round(p / e, 4), z=round(z, 2), u_band=ub,
            state="within" if abs(z) <= 2 else ("above" if z > 0 else "below"))
    if P: out["type_position"] = dict(centre=P["centre"], u_rel_k1=P["u_rel_k1"], n=P["n"], basis="measured (A8-POSITION, F132; F248 out of sample)")
    return out

# ---- edit 215: the islands no healthy cell type writes
def unused_islands(C, shares, conv=None, q_ref=None):
    """Stray writing at the islands written in no healthy cell type (unused_islands_v1.json, measured from the IAM Sequencing Atlas). Every
    methylated call there is a stray. Returns the sample's pooled rate, the law's stray level as read (writer 1/(1 + D) after the eraser's
    clean-up, plus conversion failure) and the atlas's healthy level in the sample's own cell shares, each as a ratio with z from counting."""
    J = json.load(open(os.path.join(REF_DIR, "unused_islands_v1.json"), encoding="utf-8"))
    out = dict(status="OK", label="DEVELOPMENT - not commissioned", basis=J.get("basis"), islands=len(J["units"]), off_array=len(J["off_array"]))
    cv = CONV_DEFAULT if conv is None else float(conv); u_law = (1 / (1 + D_SELECT)) * (1 - ERASER_CLEAN); u_law = u_law + cv * (1 - u_law)
    S = {c: float(w) for c, w in (shares or {}).items() if w and c in J["atlas_stray_by_type"]}; tw = sum(S.values())
    e_atlas = sum(w * J["atlas_stray_by_type"][c] for c, w in S.items()) / tw if tw > 0.5 else None
    for name, ids in (("all", J["units"]), ("off_array", J["off_array"])):
        ids = np.asarray(ids, int); m = float((np.asarray(C["meth_a"], float)[ids] + np.asarray(C["meth_b"], float)[ids]).sum())
        n = float((np.asarray(C["calls_a"], float)[ids] + np.asarray(C["calls_b"], float)[ids]).sum())
        if n < 1000: out[name] = dict(status="NOT_RUN", reason=f"too few calls ({int(n)})"); continue
        r = m / n; r_at = r if (conv is None or q_ref is None) else max(0.0, q_ref + (r - float(conv)) * (1 - q_ref) / (1 - float(conv)))   # edit 246: on the atlas's conversion scale
        rec = dict(calls=int(n), stray=round(r, 6), stray_on_atlas_scale=round(r_at, 6))
        for k, e in (("law", u_law), ("atlas", e_atlas)):
            if e is None: rec[k] = dict(status="NOT_RUN", reason="cell shares cover less than half the sample"); continue
            rr = r if k == "law" else r_at; z = (rr - e) / np.sqrt(e * (1 - e) / n); rec[k] = dict(expected=round(e, 6), ratio=round(rr / e, 4), z_counting=round(float(z), 2))
        out[name] = rec
    return out
