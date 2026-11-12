"""The mini cell, version 2 (DEV-MINICELL-01; development; a simulation of the model, not data).

WHERE IAM'S LAW IS (physics first; everything else in this file serves it):
  the coin       M = dG_ATP / (R T_cell): the drive in kicks at the encoding surface (chromatin) at the cell's own temperature   [IAM coin_M]
  the delivery   E_hold = phi * M * kT: the share of each coin delivered as holding                                            [IAM delivery_phi]
  the hold       eps = 1 / (1 + exp(E_hold/kT)): the held error set by what maintenance costs, not by the environment            [IAM hold_energy]
  the floor      kT ln2 per bit at the encoding surface                                                                         [established, Landauer; IAM places it]
  So with IAM on, the held level at held-marked sites, and how it moves with temperature and with the drive, come from the law; the
  maintenance kinetics below (pair states, copying, writing, erasing, lag) and the meter are established kinetics and our own method,
  and only deliver that level. With IAM off, the level is the measured 37 C value and does not move with temperature or the drive
  (the kinetic rates are held fixed): the difference between the two runs is exactly the IAM claim.
  Not claimed: the 37 C level itself (E_hold = 3.084 kT is MEASURED); only its dependence on T and dG_ATP is the law's.

Built like the methylation field's maintenance models (each CpG pair, a dyad, is unmethylated U, hemimethylated H or fully methylated M;
per cell cycle: copying, maintenance of hemimethylated pairs, de novo writing, erasing/loss), plus a bisulfite read simulator that writes
molecules in the .pat form the chain reads, with each library's molecule-end fault by distance from the molecule's end.

Site classes (each with its own rates): held_marked (the cell holds the mark), held_unmarked (the cell holds the site empty), loose.
Levers (simcore.LEVERS): restore = maintenance m (per cycle, of a hemimethylated pair), copy = one replication per cycle,
writer = de novo d (per unmethylated strand per cycle), eraser/loss = e (per methylated strand per cycle), lag = how late in the cycle
maintenance completes (copier lag), meter = the library's errors.
"""
import math, gzip, random
import numpy as np
from simcore import Param, ModelCard, op, sweep, contingency, sensitivity

CLASSES = ("held_marked", "held_unmarked", "loose")


def default_card(name="mini cell v1 (healthy, assumed rates)"):
    A, Me = "assumed", "measured"
    P = {}
    # rates per cell cycle by class; assumed (literature ranges for maintenance and de novo in somatic cells), to be calibrated on spent data
    for c, m, d, e in (("held_marked", 0.97, 0.05, 0.01), ("held_unmarked", 0.97, 0.004, 0.10), ("loose", 0.90, 0.10, 0.10)):
        P[f"{c}.m"] = Param(m, "per cycle", A, "maintenance of a hemimethylated pair; literature range 0.9-0.99", "restore")
        P[f"{c}.d"] = Param(d, "per strand per cycle", A, "de novo writing on an unmethylated strand", "writer")
        P[f"{c}.e"] = Param(e, "per strand per cycle", A, "erasing or loss of a methylated strand", "eraser")
    P["share.held_marked"] = Param(0.55, "fraction of identity sites", A, "to be taken from the atlas column")
    P["share.held_unmarked"] = Param(0.35, "fraction of identity sites", A, "to be taken from the atlas column")
    P["lag"] = Param(0.10, "cell cycles (mean)", A, "copier lag: time after copying for maintenance to complete", "restore")
    # the meter (one library); end excess per call from the nearer end, marked side and unmarked side (B-cell library GSE3963003, spent;
    # fixed-length molecules, DEV_ATLAS_V21_01.md); interior errors from the hairpin no-methyltransferase floor (GSE116482)
    P["meter.false_mark"] = Param(0.004, "per call", Me, "unconverted unmethylated C read as marked; GSE116482 cells without methyltransferases", "meter")
    P["meter.false_unmark"] = Param(0.002, "per call", A, "methylated C read as unmarked inside the molecule", "meter")
    P["meter.end_m"] = Param((0.198,), "excess per call 1..k", Me, "B-cell library, marked side, call 1 above the plateau", "meter")
    P["meter.end_u"] = Param((0.034, 0.009, 0.004), "excess per call 1..k", Me, "B-cell library, unmarked side, calls 1-3 above the plateau", "meter")
    P["meter.calls"] = Param(12, "CpG calls per molecule", A, "fixed length for v1", "meter")
    P["meter.depth"] = Param(30, "molecules per region", A, "read depth", "meter")
    return ModelCard(name, "cell", P)


def _cycle_matrix(m, d, e, lag, t_obs=None, K=100):
    """Dyad states [U, H, M] (H = one strand marked). One cell cycle: copying at t=0, then K steps of writing/erasing on each strand and
    maintenance of H with a hazard set so that the cycle completes a fraction m, front-loaded with mean delay 'lag'.
    Returns the full-cycle matrix and, if t_obs is given, the matrix up to t_obs (for a snapshot partway through the cycle)."""
    C = np.array([[1, 0, 0], [0.5, 0.5, 0], [0, 1, 0]], float)          # copying: U->U; H->U or H (each half); M->H (new strand empty)
    dt = 1.0 / K
    # maintenance hazard h(t) with completion by end of cycle = m and mean delay ~ lag: h(t) = lam * exp(-t/lag) style front loading
    tau = max(lag, 1e-3); tot = -math.log(max(1 - m, 1e-12)); norm = tau * (1 - math.exp(-1 / tau))
    def step(t):
        hm = tot * math.exp(-t / tau) / norm * dt; pm = 1 - math.exp(-hm)
        pd, pe = 1 - (1 - d) ** dt, 1 - (1 - e) ** dt
        S = np.zeros((3, 3))
        # U: each of two strands written with pd
        S[0, 0] = (1 - pd) ** 2; S[0, 1] = 2 * pd * (1 - pd); S[0, 2] = pd ** 2
        # H: maintenance completes (->M), else the empty strand written (pd) or the marked strand erased (pe)
        a = pm + (1 - pm) * pd; b = pe
        S[1, 2] = a * (1 - b); S[1, 0] = (1 - a) * b; S[1, 1] = 1 - S[1, 2] - S[1, 0]
        # M: each strand erased with pe
        S[2, 2] = (1 - pe) ** 2; S[2, 1] = 2 * pe * (1 - pe); S[2, 0] = pe ** 2
        return S
    Tfull = C.copy(); Tobs = None
    for i in range(K):
        if t_obs is not None and Tobs is None and i * dt >= t_obs: Tobs = Tfull.copy()
        Tfull = Tfull @ step(i * dt)
    return Tfull, (Tobs if Tobs is not None else Tfull)


def class_state(m, d, e, lag, K=100, phases=20):
    """Steady state of one class: dyad shares [U, H, M], averaged over the cell-cycle phase at sampling (asynchronous population)."""
    T, _ = _cycle_matrix(m, d, e, lag, K=K)
    w, v = np.linalg.eig(T.T); p0 = np.real(v[:, np.argmin(abs(w - 1))]); p0 = p0 / p0.sum()     # state at the start of a cycle
    snaps = [p0 @ _cycle_matrix(m, d, e, lag, t_obs=(j + 0.5) / phases, K=K)[1] for j in range(phases)]
    return np.mean(snaps, axis=0)


def model(card):
    """Operating point: per class, the dyad shares, the strand level and the held error; the hairpin metric (fully held pairs per pair with
    one strand off, at held_marked sites); the whole-region level. Meter readings (bulk, no molecule simulation) by side."""
    out = {}
    for c in CLASSES:
        U, Hh, M = class_state(card.v(f"{c}.m"), card.v(f"{c}.d"), card.v(f"{c}.e"), card.v("lag"))
        lvl = M + Hh / 2; out[f"{c}.U"], out[f"{c}.H"], out[f"{c}.M"], out[f"{c}.level"] = U, Hh, M, lvl
    out["held_error_marked"] = 1 - out["held_marked.level"]          # marks missing at held-marked sites
    out["held_error_unmarked"] = out["held_unmarked.level"]          # marks present at held-empty sites
    out["restores_per_loss"] = out["held_marked.M"] / max(out["held_marked.H"], 1e-12)   # hairpin metric MM / (MU + UM)
    s1, s2 = card.v("share.held_marked"), card.v("share.held_unmarked"); s3 = max(0.0, 1 - s1 - s2)
    out["loose_share"] = s3
    fm, fu = card.v("meter.false_mark"), card.v("meter.false_unmark")
    for side, lv in (("marked", out["held_marked.level"]), ("unmarked", out["held_unmarked.level"])):
        out[f"read_interior_{side}"] = lv * (1 - fu) + (1 - lv) * fm
    out["E_hold_kT"] = math.log(out["held_marked.level"] / max(1 - out["held_marked.level"], 1e-12))   # kT ln(on/off) at held-marked sites
    return out


def end_profile(card, L=None):
    """The meter's expected error per call by distance from the nearer end (1..L/2), marked and unmarked side, as the fault-location
    profile measures it (the share of calls read against the side's held state)."""
    o = model(card); L = L or card.v("meter.calls"); fm, fu = card.v("meter.false_mark"), card.v("meter.false_unmark")
    em, eu = card.v("meter.end_m"), card.v("meter.end_u"); prof = {"marked": [], "unmarked": []}
    for dist in range(1, L // 2 + 1):
        xm = em[dist - 1] if dist <= len(em) else 0.0; xu = eu[dist - 1] if dist <= len(eu) else 0.0
        lm = o["held_marked.level"]; lu = o["held_unmarked.level"]
        pu_m = fu + xm; pm_u = fm + xu                                   # flip probabilities at this distance
        prof["marked"].append(round((1 - lm) * (1 - pm_u) + lm * pu_m, 5))
        prof["unmarked"].append(round(lu * (1 - pu_m) + (1 - lu) * pm_u, 5))
    return prof


def simulate_pat(card, n_regions=200, path=None, seed=20261111, start_index=1):
    """Molecules as .pat lines (chrom, first CpG index, calls C/T/., count): each region is a run of meter.calls consecutive CpGs of one
    class (by the card's shares); each molecule draws each pair's state from the class's steady state, reads one strand, and passes each call
    through the meter (interior flips plus the end excess by distance from the nearer end). Development: synthetic CpG indices."""
    g = random.Random(seed); o = model(card); L = card.v("meter.calls"); depth = card.v("meter.depth")
    sh = [card.v("share.held_marked"), card.v("share.held_unmarked")]; sh.append(max(0.0, 1 - sum(sh)))
    fm, fu = card.v("meter.false_mark"), card.v("meter.false_unmark"); em, eu = card.v("meter.end_m"), card.v("meter.end_u")
    lines = []; idx = start_index
    for r in range(n_regions):
        c = g.choices(CLASSES, weights=sh)[0]; pU, pH, pM = o[f"{c}.U"], o[f"{c}.H"], o[f"{c}.M"]
        for _ in range(depth):
            s = []
            for j in range(L):
                u = g.random(); st = 0 if u < pU else (1 if u < pU + pH else 2)
                strand = 1 if st == 2 else (0 if st == 0 else int(g.random() < 0.5))
                dist = min(j, L - 1 - j) + 1
                xm = em[dist - 1] if dist <= len(em) else 0.0; xu = eu[dist - 1] if dist <= len(eu) else 0.0
                if strand: read = 0 if g.random() < fu + xm else 1
                else: read = 1 if g.random() < fm + xu else 0
                s.append("C" if read else "T")
            lines.append((idx, "".join(s)))
        idx += L + 5
    if path:
        from collections import Counter
        cnt = Counter(lines)
        with gzip.open(path, "wt") as f:
            for (i, s), n in sorted(cnt.items()): f.write(f"chr1\t{i}\t{s}\t{n}\n")
    return lines


def calibrate(card, restores_per_loss, held_level, cls="held_marked", lo=(0.5, 1e-5), hi=(0.99999, 0.9)):
    """State estimation from measured numbers (never from a target the model is then said to predict): solve the class's maintenance m
    so the hairpin metric (fully held pairs per pair with one strand off) matches restores_per_loss, and its writer d so the strand level
    matches held_level, with its eraser e and the lag held at their card values. Alternates the two one-dimensional solves to convergence."""
    from scipy.optimize import brentq
    m, d = card.v(f"{cls}.m"), card.v(f"{cls}.d"); e, lag = card.v(f"{cls}.e"), card.v("lag")
    def rpl(mm, dd): U, Hh, M = class_state(mm, dd, e, lag); return M / Hh
    def lvl(mm, dd): U, Hh, M = class_state(mm, dd, e, lag); return M + Hh / 2
    for _ in range(12):
        m = brentq(lambda x: rpl(x, d) - restores_per_loss, lo[0], hi[0], xtol=1e-7)
        d = brentq(lambda x: lvl(m, x) - held_level, lo[1], hi[1], xtol=1e-8)
    c = card.set(**{f"{cls}.m": m, f"{cls}.d": d})
    c[f"{cls}.m"].label = "derived"; c[f"{cls}.m"].source = f"calibrated: hairpin restores per loss {restores_per_loss} (lag, e assumed)"
    c[f"{cls}.d"].label = "derived"; c[f"{cls}.d"].source = f"calibrated: held level {held_level} (lag, e assumed)"
    return c


# ---------------------------------------------------------------------------------------------------------------------------------------
# version 2: the law's ledger first
import simcore as _sc
R_GAS = 8.314462618   # J/(mol K), exact


def law_card(name="mini cell v2 (healthy, law first)"):
    """The v1 card plus the ledger's inputs, every one with its origin. The maintenance m is calibrated on the hairpin wild type with the
    lag and eraser assumed (calibrate()); the writer d is then solved so the kinetics deliver the held level (the law's, or the measured)."""
    c = default_card(name); P = dict(c)
    P["T_cell"] = Param(310.15, "K", "measured", "37 C, the cell's own temperature (canon T_cell)", "kick", origin="measured")
    P["dG_ATP"] = Param(54000.0, "J/mol", "measured", "cytosolic free energy of ATP hydrolysis (canon dG_ATP)", "coin", origin="established")
    P["phi"] = Param(0.1473, "fraction of one coin", "measured", "phi = E_hold/M, canon (56 healthy cell types)", "restore", origin="measured")
    P["iam"] = Param(True, "switch", "definition", "IAM on: the held level from the law; off: the measured 37 C level, fixed", origin="ours")
    P["eps_measured_37C"] = Param(0.04376, "held error", "measured", "canon eps0_meth at 37 C (used only with IAM off)", origin="measured")
    P["baseline_candidate"] = Param(False, "switch", "definition", "on (with IAM on): the held level from the driven passage (PHY-083, candidate) instead of measured phi", origin="ours")
    P["SAM_cost"] = Param(2.0, "ATP coins per SAM", "derived", "canon SAM_cost_ATP: exactly 2 where adenylate kinase is at equilibrium (minicell/sam_cost_ak.py)", "coin", origin="established")
    P["held_marked.m"].value = 0.9661; P["held_marked.m"].label = "derived"; P["held_marked.m"].origin = "measured"
    P["held_marked.m"].source = ("calibrated on maintenance alone: hairpin genotype without writers or erasers (Tet1-3 + Dnmt3a/b/c knockout), restores per "
                                 "loss 16.38 (GSE116482, mouse stem cells), lag assumed 0.1 cycle")
    P["held_marked.e"].value = 0.10; P["held_marked.e"].label = "derived"; P["held_marked.e"].origin = "measured"
    P["held_marked.e"].source = ("first calibration: wild type 12.11 needs about 10 % of held marks erased and rewritten per cycle (reaches 12.4 at the human "
                                 "level 0.956; the stem cells' own level at those sites not yet used)")
    return ModelCard(name, "cell", P)


def ledger(card):
    """The law's ledger for this card (IAM readings name the relation they used)."""
    Sc = _sc; T = card.v("T_cell"); kT = Sc.K_B * T
    M = Sc.Reading(card.v("dG_ATP") / (R_GAS * T), "IAM", ["coin_M", "encoding_surface"], "the drive in kicks at the encoding surface (chromatin) at the cell's own temperature")
    if "baseline_candidate" in card and card.v("baseline_candidate"):
        w = card.v("SAM_cost") * float(M); odds = (w / 2) / (1 - (1 - math.exp(-w)) / w)
        E = Sc.Reading(math.log(odds), "IAM", ["coin_M", "driven_passage"], "holding energy in kT from the driven passage (candidate)")
    else:
        E = Sc.Reading(card.v("phi") * M, "IAM", ["coin_M", "delivery_phi"], "holding energy in kT")
    eps = Sc.Reading(1 / (1 + math.exp(E)), "IAM", list(E.relations) + ["hold_energy"], "held error the law sets")
    return dict(M=M, M_landauer=Sc.Reading(M / Sc.LN2, "IAM", ["coin_M", "encoding_surface"], "the coin in Landauer units"), E_hold_kT=E, eps_hold=eps,
                landauer_floor_J=Sc.Reading(kT * Sc.LN2, "established", note="Landauer, kT ln2 at the cell's temperature"),
                eps_ceiling=Sc.Reading(0.5, "established", note="the coin flip"))


def model2(card):
    """Operating point, law first: the ledger, then the kinetics solved to deliver the held level, then the readings, each with its origin."""
    from scipy.optimize import brentq
    Sc = _sc; L = ledger(card); iam = bool(card.v("iam"))
    eps = float(L["eps_hold"]) if iam else card.v("eps_measured_37C")
    m, e, lag = card.v("held_marked.m"), card.v("held_marked.e"), card.v("lag")
    d = brentq(lambda x: (lambda u: u[2] + u[1] / 2)(class_state(m, x, e, lag)) - (1 - eps), 1e-6, 0.999, xtol=1e-9)
    U, Hh, Mm = class_state(m, d, e, lag)
    lab = ("IAM", ["coin_M", "delivery_phi", "hold_energy"]) if iam else ("measured", [])
    out = dict(L) if iam else {k: v for k, v in L.items() if v.origin != "IAM"}
    out["held_error_marked"] = Sc.Reading(1 - (Mm + Hh / 2), lab[0], lab[1], "delivered by the kinetics at the level the law sets" if iam else "the measured 37 C level, fixed")
    out["writer_d_required"] = Sc.Reading(d, lab[0], lab[1], "the writing the kinetics need to deliver that level (per strand per cycle)")
    kin = (lab[0], lab[1]) if iam else ("established", [])
    out["hemi_share"] = Sc.Reading(Hh, kin[0], kin[1], "shape from established maintenance kinetics; with IAM on it moves only through the level the law sets")
    out["restores_per_loss"] = Sc.Reading(Mm / Hh, kin[0], kin[1], "hairpin metric (maintenance calibrated on 12.11 at 37 C); with IAM on it moves only through the law's level")
    return out
