#!/usr/bin/env python3
"""Cellular Performance Gauge — conductor v3. DEVELOPMENT - not commissioned. Scope: neutrophils, EPIC v1 arrays.

Runs after Stage 0 (intake) and Stage 1 (IDAT calibration), both driven by MethylPhys_Interface/run_sample.py:
  platform  EPIC v1 only: refused when Stage 0 reports another array type, when probe names carry the EPIC v2 design suffix,
            or when the beta vector holds 700,000 probes or fewer (450K; floor pending)
  Stage A   composition (whole blood): NNLS on the 963 markers of blood_composition_EPIC_v1.json (8 groups from purified EPIC
            cells; no marker is a neutrophil identity site), sum 1; >= 90 % of the markers must be measured
  Stage M   Met-A: isolated / sorted neutrophils -> healthy reference (stage_m_met_a.read); whole blood -> composition-matched healthy
            expectation, read when the neutrophil fraction is >= MIN_READ_FRACTION and >= 90 % of the 6000 sites are measured
  Stage MC  Met-A C-score: clustering of the neutrophil residual map over the healthy baseline (development: band not set)
  Stage T   step 1, self-tare II (adopted by the author 2026-10-04, DEV-SELFTARE-02; wired here 2026-10-04): each probe design mapped
            onto the reference arrays' scale by this array's own low and high fixed-site anchors (dev_stages.selftare_map), before
            Stages A, M and MC; no references; nothing is fitted. Step 2, the median tare against >= 3 healthy references run the same
            way (same slide, else same batch), whole blood and isolated alike: A_rel = A / median(reference A); nothing is fitted.
            Without references the reading is reported as untared (step 2 not run).
  Noise     Stage M records the array's noise index N (noise_sites_EPIC_v1.json). If N > N_max (noise_gate_EPIC_v1.json, the top of
            the reference arrays' range) the gauge state is withheld unless the reading is tared (DEV-NOISE-01 rule). If fewer than 90 % of
            the noise sites are measured, N cannot be formed and the gauge state is withheld with the reason (author decision A, 2026-10-04).
  Specimen  whole blood and isolated / sorted / purified neutrophils only (stage_0_intake.specimen_refusal, author decision L, 2026-10-04).
Frozen inputs (Runtime Matrices/Met_A_Floors): metA_floors_v1_3.json, metA_floors_v1_3_loo.csv, neutrophil_reference_v1_1.json,
blood_composition_EPIC_v1.json, noise_sites_EPIC_v1.json, noise_gate_EPIC_v1.json. neutrophil_reference_v1_1.json keys profiles_mean_beta and profile_map are record only (not read).
IAM-A (sequencing) is Stage Q, stage_q_iam_a.py, called by run_sample.py with --pat or --site-table.
Formulas (canon: Met-A, C-score):
  H(b) = -b log2 b - (1-b) log2(1-b)
  isolated:    Met-A = mean_i H(beta_i) / floor
  whole blood: Met-A = mean_i H(beta_i) / mean_i H(e_i),  e_i = sum_g f_g mu_g,i  (f: this specimen's fractions; mu: purified healthy profiles)
  residual z_i = (H(beta_i) - H(ref_i)) / s_i  (ref_i = neutrophil mean H, or H(e_i) in blood; s_i = shrunk healthy SD of H among neutrophils)
  C = var(block means of clustering_block consecutive sites x sqrt(block)) / var(z)  /  healthy median clustering
"""
import json, os
import numpy as np, pandas as pd
import stage_m_met_a as SM
HERE = os.path.dirname(os.path.abspath(__file__)); RM = os.path.join(HERE, "Runtime Matrices", "Met_A_Floors")
BUILD = "DEVELOPMENT - not commissioned (chain v3, neutrophils only)"
MIN_READ_FRACTION = 0.20      # below this the 1 % shift is under 0.01 and too few sites carry the cell (DEV-LOWFRAC-01: 5 healthy arrays under 0.40)
MIN_MARKER_FRACTION = 0.9     # Stage A needs >= 90 % of blood_composition_EPIC_v1.json "markers" measured (867 of 963)
MIN_REFS = 3                  # Stage T needs >= 3 same-run reference arrays (median tare)
MIN_NOISE_FRACTION = 0.9      # noise index N needs >= 90 % of the 48,528 noise sites measured
ACCEPTED_ARRAY_TYPES = ("EPIC_v1",)
_REF = None
def ref():
    global _REF
    if _REF is None: _REF = json.load(open(os.path.join(RM, "neutrophil_reference_v1_1.json")))
    return _REF
H = SM._H
ISOLATED = ("isolated neutrophils", "sorted neutrophils", "purified neutrophils", "neutrophils")

_NS = None
def noise_sites():
    """Runtime Matrices/Met_A_Floors/noise_sites_EPIC_v1.json: 48,528 EPIC sites every purified blood group holds fixed
    (group mean <= 0.03 or >= 0.97, group SD <= 0.02; no neutrophil identity site). Their entropy is the array's own noise."""
    global _NS
    if _NS is None: _NS = json.load(open(os.path.join(RM, "noise_sites_EPIC_v1.json")))
    return _NS

_NG = None
def noise_gate():
    """Runtime Matrices/Met_A_Floors/noise_gate_EPIC_v1.json: N_max = top of the reference arrays' noise range (DEV-NOISE-01)."""
    global _NG
    if _NG is None: _NG = json.load(open(os.path.join(RM, "noise_gate_EPIC_v1.json")))
    return _NG

def noise_index(beta):
    """N = mean H(beta) over the noise sites measured on this array; None when fewer than MIN_NOISE_FRACTION are measured."""
    S = noise_sites()["sites"]; x = beta.reindex(S); ok = x.notna(); n = int(ok.sum())
    rec = {"noise_sites_measured": n, "noise_sites_total": len(S)}
    rec["noise_index"] = round(float(H(x[ok]).mean()), 5) if n >= MIN_NOISE_FRACTION * len(S) else None
    return rec

_BC = None
def _bc():
    global _BC
    if _BC is None: _BC = json.load(open(os.path.join(RM, "blood_composition_EPIC_v1.json")))
    return _BC

def min_markers():
    return int(np.ceil(MIN_MARKER_FRACTION * len(_bc()["markers"])))

def stage_a_composition(beta):
    """EPIC blood composition (blood_composition_EPIC_v1): 8 groups from Salas purified EPIC cells; markers exclude the neutrophil sites;
    NNLS, sum 1. Same platform and same reference as the expectation profiles. Refused (fractions None) when fewer than
    MIN_MARKER_FRACTION of the markers are measured."""
    from scipy.optimize import nnls
    B = _bc(); y = beta.reindex(B["markers"]); M = pd.DataFrame(B["mu_markers"], index=B["markers"]); ok = y.notna()
    rec = {"stage": "A", "method": "EPIC blood NNLS (blood_composition_EPIC_v1)", "n_markers_used": int(ok.sum()),
           "n_markers_required": min_markers(), "n_markers_total": len(B["markers"])}
    if ok.sum() < min_markers():
        rec.update(fractions=None, reason=f"only {int(ok.sum())} of {len(B['markers'])} composition markers measured "
                                          f"(>= {min_markers()} required): composition not solved"); return rec
    f, res = nnls(M[ok].values, y[ok].values); f = f / f.sum() if f.sum() > 0 else f
    rec.update(fractions=dict(zip(B["groups"], [float(v) for v in f])),
               residual_mae=float(np.abs(y[ok].values - M[ok].values @ f).mean()))
    return rec

def _clustering(z, w=None):
    w = int(w or ref()["clustering_block"])
    o = z.dropna().values; nb = len(o) // w
    if nb < 10: return None
    b = o[:nb * w].reshape(nb, w).mean(1) * np.sqrt(w); return float(np.var(b) / np.var(o))

def _state(A):
    return "Normal" if SM.NORMAL[0] <= A <= SM.NORMAL[1] else ("above Normal" if A > SM.NORMAL[1] else "below Normal")

def stage_m_blood(beta, comp):
    """Whole blood: Met-A = mean H(beta) / mean H(e) at the neutrophil sites, e = sum_g f_g mu_g (EPIC purified group profiles).
    The untared value carries a composition and laboratory offset; the gauge state is printed only after the tare against
    healthy whole bloods run the same way (Stage T)."""
    B = _bc(); S = pd.Index(B["neutrophil_sites"]); P = {g: pd.Series(v, index=S, dtype="float64") for g, v in B["profiles_at_neutrophil_sites"].items()}
    rec = {"stage": "M", "reading": "Met-A", "cell": "neutrophils", "specimen": "whole blood", "fraction": None, "build": BUILD,
           "band": "Normal 0.95-1.05 (after tare)", "A": None}
    fractions = comp.get("fractions")
    if fractions is None:
        rec["reason"] = f"{comp.get('reason')}: A withheld"; return rec, None
    fn = fractions.get("NEU", 0.0); rec["fraction"] = round(fn, 4)
    if fn < MIN_READ_FRACTION:
        rec["reason"] = f"neutrophil fraction {fn:.3f} < {MIN_READ_FRACTION}: fraction reported, A withheld"; return rec, None
    x = beta.reindex(S); e = sum(v * P[g] for g, v in fractions.items() if g in P); ok = x.notna() & e.notna()
    need = int(np.ceil(SM.SITE_COVERAGE_MIN * len(S)))
    if ok.sum() < need:
        rec["reason"] = f"only {int(ok.sum())} of {len(S)} neutrophil sites measured (>= {need} required): A withheld"; return rec, None
    A = float(H(x[ok]).mean() / H(e[ok]).mean())
    mu = P["NEU"]; xd = x + fn * 0.01 * (0.5 - mu)            # a known 1 % loss of the neutrophils' pattern, at this specimen's own fraction
    rec["shift_per_1pct_loss"] = round(float(H(xd[ok].clip(1e-6, 1 - 1e-6)).mean() / H(e[ok]).mean()) - A, 5)
    rec.update(_ceiling(x[ok], mu[ok]))
    rec.update(A=round(A, 4), n_sites=int(ok.sum()), expectation="composition-matched healthy (EPIC purified group profiles x this specimen's fractions)",
               state="untared: read A_rel (Stage T)")
    R = ref(); Sr = pd.Index(R["sites_ordered"])
    z = (H(beta.reindex(Sr)) - H(e.reindex(Sr))) / pd.Series(R["neutrophil_H_sd_shrunk"], index=Sr)
    return rec, z

def _ceiling(x, mu):
    """Entropy ceiling: per-site H peaks at beta = 0.5. Met-A is monotone in pattern loss only while the cell's methylated sites stay above 0.5
    (DNMT-01: A_meth saturates at 1/H(floor) once the methylated sites reach beta ~0.5)."""
    hi = mu > 0.5
    if not hi.any(): return {}
    m = float(x[hi].mean())
    return {"methylated_sites_mean_beta": round(m, 4), "past_entropy_ceiling": bool(m < 0.5)}

def stage_m_isolated(beta):
    """Isolated / sorted neutrophils against their healthy reference. The healthy-reference state is kept as state_own_floor; the reading's state is
    'untared' until Stage T reads it against >= 3 same-run references."""
    rec = SM.read(beta, "neutrophils", specimen="isolated neutrophils")
    R = ref(); S = pd.Index(R["sites_ordered"])
    B = _bc(); Sb = pd.Index(B["neutrophil_sites"]); mu = pd.Series(B["profiles_at_neutrophil_sites"]["NEU"], index=Sb, dtype="float64")
    x = beta.reindex(Sb); ok = x.notna() & mu.notna(); rec.update(_ceiling(x[ok], mu[ok]))
    if rec.get("A") is None: return rec, None
    xd = (x + 0.01 * (0.5 - mu))[ok].clip(1e-6, 1 - 1e-6); rec["shift_per_1pct_loss"] = round(float(H(xd).mean() / H(x[ok].clip(1e-6, 1 - 1e-6)).mean() * rec["A"]) - rec["A"], 5)
    rec["state_own_floor"] = rec.pop("state"); rec["state"] = f"untared (healthy-reference state: {rec['state_own_floor']}): read A_rel (Stage T)"
    z = (H(beta.reindex(S)) - pd.Series(R["neutrophil_H_mean"], index=S)) / pd.Series(R["neutrophil_H_sd_shrunk"], index=S)
    return rec, z

def stage_mc_cscore(z):
    """Met-A C-score (stage 6): clustering of the residual z map in genomic order - variance of the means of blocks of clustering_block
    consecutive sites (x sqrt(block)) over the site variance, divided by the healthy median clustering of neutrophil_reference_v1_1.json.
    Healthy = 1; development (band not set). No residual map -> C None with the reason."""
    R = ref(); c = _clustering(z) if z is not None else None
    if c is None: return {"stage": "MC", "reading": "Met-A C-score", "C": None, "reason": "no residual map"}
    return {"stage": "MC", "reading": "Met-A C-score", "C": round(c / R["healthy_clustering_median"], 4), "clustering": round(c, 4),
            "healthy_baseline": R["healthy_clustering_median"], "n_healthy_baseline": len(R["healthy_clustering_LOO"]),
            "block_sites": int(R["clustering_block"]),
            "healthy_range": [min(R["healthy_clustering_LOO"]) / R["healthy_clustering_median"], max(R["healthy_clustering_LOO"]) / R["healthy_clustering_median"]],
            "status": "development: healthy band not yet set", "frac_abs_z_gt3": round(float((z.abs() > 3).mean()), 4)}

def _ref_records(ref_A, sample_id=None):
    """References as plain A values or as records {A, f_neu, N[, id]}. Returns (records, n_self_excluded)."""
    recs, n_self = [], 0
    for r in (ref_A or []):
        if isinstance(r, dict):
            if sample_id is not None and str(r.get("id", r.get("gsm", ""))) == str(sample_id): n_self += 1; continue
            a = r.get("A")
            if a is None or (isinstance(a, float) and np.isnan(a)): continue
            g = lambda k: (None if r.get(k) is None or (isinstance(r.get(k), float) and np.isnan(r.get(k))) else float(r[k]))
            recs.append({"A": float(a), "f_neu": g("f_neu"), "N": g("N")})
        elif r is not None and not (isinstance(r, float) and np.isnan(r)):
            recs.append({"A": float(r), "f_neu": None, "N": None})
    return recs, n_self

def stage_t_tare(A, ref_A, shift_1pct=None, sample_id=None):
    """Same-run tare against >= MIN_REFS healthy references of the same specimen type run the same way (same slide, else same batch).
    references: untared A values, or records {A[, id]}. A_rel = A / median(reference A); spread = SD of reference A / median.
    Nothing is fitted. Detection limit = 2 x reference spread / shift per 1 % loss."""
    recs, n_self = _ref_records(ref_A, sample_id)
    if A is None: return {"stage": "T", "A_rel": None, "reason": "no A"}
    if len(recs) < MIN_REFS:
        return {"stage": "T", "A_rel": None, "n_self_excluded": n_self, "reason": f"untared: {len(recs)} same-run reference arrays (>= {MIN_REFS} required)"}
    a_ = np.array([r["A"] for r in recs]); m = float(np.median(a_)); Ar = A / m; sd = float(np.std(a_ / m, ddof=1))
    dl = (2 * sd / shift_1pct) if shift_1pct and shift_1pct > 0 else None
    return {"stage": "T", "n_refs": len(recs), "n_self_excluded": n_self, "method": "median tare (same-run healthy references)",
            "reference_median": round(m, 4), "A_rel": round(Ar, 4), "reference_spread_sd": round(sd, 4),
            "detection_limit_pct_loss": (round(dl, 2) if dl is not None else None),
            "detection_note": "smallest loss of the cell's pattern (percent) this specimen could show: 2 x reference spread / shift per 1 % loss",
            "state": _state(Ar)}

def stage_t_selftare_ii(beta):
    """Stage T step 1, self-tare II (adopted 2026-10-04, DEV-SELFTARE-02): per probe design, beta' = Lr + (beta - L)(Ur - Lr)/(U - L) from this
    array's own fixed-site anchors L, U and the reference arrays' Lr, Ur (dev_stages.selftare_map; a design with anchors missing or U - L <= 0.1
    is left unmapped). Returns (beta', record); when the runtime file is missing, (beta, record with status NOT_RUN)."""
    import dev_stages as DV
    b2, info = DV.selftare_map(beta)
    if b2 is None: return beta, {"step": "1 self-tare II", "status": "NOT_RUN", **info}
    return b2, {"step": "1 self-tare II", "status": "OK", **info,
                "note": "Stages A, M and MC read this array's betas mapped onto the reference arrays' scale by its own fixed-site anchors; the noise index reads the unmapped betas"}

def platform_refusal(beta, array_type=None):
    """None when the specimen is EPIC v1; otherwise the refusal text."""
    if array_type is not None and array_type not in ACCEPTED_ARRAY_TYPES:
        return f"array type {array_type}: chain v3 reads EPIC v1 arrays only (no frozen neutrophil floor for this platform)"
    p = SM.platform_of(beta)
    if p == "EPIC_v2":
        return "EPIC v2 probe names (design suffix, e.g. cg..._TC21): chain v3 reads EPIC v1 arrays only (no frozen neutrophil floor for this platform)"
    if p != "EPIC":
        return f"{len(beta):,} probes (450K or incomplete vector): chain v3 reads EPIC v1 arrays only (450K neutrophil floor pending)"
    return None

def run_neutrophil(beta, specimen="whole blood", ref_A=None, array_type=None, sample_id=None):
    """beta: pd.Series from Stage 1 (EPIC v1). ref_A: same-run healthy references - their Met-A before the median tare (met_a.A, which
    carries Stage T step 1, self-tare II, since 2026-10-04), or records {A, f_neu, N[, id]}
    (Stage T; plain A values or records with A). array_type: Stage 0's array type (header, else declared), or None for a beta table. sample_id: excluded from the
    references if a record carries it. Returns the v3 bundle."""
    beta = beta.copy(); beta.index = beta.index.astype(str)
    out = {"build": BUILD, "specimen": specimen, "platform": SM.platform_of(beta), "array_type": array_type, "scope": "neutrophils only",
           "floors_version": SM._floors()["version"], "reference_version": ref()["version"]}
    import stage_0_intake as S0
    r = S0.specimen_refusal(specimen)
    if r: out["refusal"] = r; out["refusal_code"] = "SPECIMEN_REFUSED"; return out
    r = platform_refusal(beta, array_type)
    if r: out["refusal"] = r; out["refusal_code"] = "PLATFORM_REFUSED"; return out
    if S0.normalise_specimen(specimen) == "constructed dna mixture": out["note"] = "constructed DNA mixture of blood cells (test material): read as whole blood"
    b_st, st1 = stage_t_selftare_ii(beta)   # Stage T step 1, self-tare II (adopted 2026-10-04): A, M and MC read the mapped betas
    if S0.ACCEPTED_SPECIMENS.get(S0.normalise_specimen(specimen)) == "isolated neutrophils":
        m, z = stage_m_isolated(b_st); out["composition"] = {"stage": "A", "note": "isolated neutrophils: composition not solved"}
    else:
        a = stage_a_composition(b_st); out["composition"] = a
        m, z = stage_m_blood(b_st, a)
    m.update(noise_index(beta))   # the noise index reads the betas before self-tare II (step 1): N and the noise gate are unchanged by Stage T
    t = stage_t_tare(m.get("A"), ref_A, m.get("shift_per_1pct_loss"), sample_id=sample_id)   # Stage T step 2, the median tare
    t["selftare_ii"] = st1
    if t.get("A_rel") is not None: m["state"] = "tared: read A_rel (Stage T)"
    g = noise_gate(); N = m.get("noise_index"); m["noise_gate_N_max"] = g["N_max"]
    m["noise_gate"] = ("not measured" if N is None else ("pass" if N <= g["N_max"] else "above the reference arrays' range"))
    if N is None and m.get("A") is not None:   # author decision A (2026-10-04): below 90 % noise-site coverage the noise is unknown -> no state
        need = int(np.ceil(MIN_NOISE_FRACTION * m["noise_sites_total"]))
        m["noise_gate"] = "not measured: noise-site coverage below 90 %"
        m["state"] = (f"withheld: only {m['noise_sites_measured']:,} of the {m['noise_sites_total']:,} noise sites were measured on this array "
                      f"(at least {need:,}, 90 %, are needed). The noise index N is this array's own noise, read at sites every blood cell holds fixed; "
                      f"without it the gauge cannot tell a change in the cell from noise on the array, so no gauge state is shown, tared or not. "
                      f"A is printed as a number only. Probes are lost when Stage 1 finds them at background: re-hybridise the specimen or check the "
                      f"array's signal (Stage 1 detection line).")
    elif N is not None and N > g["N_max"] and t.get("A_rel") is None and m.get("A") is not None:   # no A -> its own reason stands (2026-10-03)
        m["state"] = f"withheld: noise index {N} > {g['N_max']} and no same-run tare; A printed as a number only"
    out["met_a"] = m; out["met_a_cscore"] = stage_mc_cscore(z); out["tare"] = t
    out["withheld"] = ["tier lines beyond Normal (not yet measured on this scale)", "other cell types (outside commissioning scope)"]
    return out
