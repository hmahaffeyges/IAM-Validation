"""The mini cosmos, version 0 (DEV-MINICOSMOS-01; development; a simulation of the model, not data). Design and skeleton, 2026-11-12.

WHERE IAM'S LAW IS (physics first). IAM enters only through the entropy functional; the Friedmann equation is never modified.
  the surface    the cosmic apparent horizon r_A = c/H: the nearest encoding surface for distributed structure       [IAM encoding_surface]
  the kick       the Gibbons-Hawking temperature T_H = hbar H / (2 pi k_B), the horizon's own temperature              [established]
  the bit price  dE_rec(a) = k_B T_H(a) ln2 per record written on the horizon (epoch-dependent; conjecture)         [IAM encoding_surface]
  the record     E(a) = exp(1 - 1/a), the cumulative record, kept on the horizon ledger (conjecture; needs rate exponent n = 7/2) [IAM activation]
  the partition  beta_m = Omega_m / 2 (the virial half at the cosmic horizon)                                       [IAM virial_half]
  the ratio      R(a) = Omega_m a^-3 / (beta_m E(a)): the potential half against the record (kinetic) half          [IAM virial_half]
Everything else (H(a) of the background with IAM off, Planck parameters, unit constants) is established or measured, labeled as such.

TIME AXIS. The clock is the scale factor a (a = 1 today); time is read along a from the start of accumulated decoherence (electroweak
symmetry breaking) into the far future, where E(a) -> e. Every reading is a function of a, so the simulator draws each one over cosmic time.

THE SWITCH. iam=False: the background and its readings from established physics alone (flat LambdaCDM, Planck 2018 inputs), which must
reproduce what the field computes. iam=True: adds the ledger (bit price, record, partition, ratio); the difference between the two is
exactly the IAM content. The background expansion itself is the established one in v0; the book's two rulers (H0 by the photon sector
67.16, by the matter sector 72.26; canon locked results) are read, not refitted.

SELF-TESTS (minicosmos_selftest() below): the numbers the canon's four-domains table prints for the cosmos row, recomputed here.
ADDED 2026-11-18: the ledger's density identity and each sector's deceleration and acceleration onset (dark sector).
ADDED 2026-11-15 (v0.1): matter's coupling as its own black-hole half, Lambda's own-horizon temperature, held/paid = 1/2 for settled surfaces,
and a black-hole module (Hawking temperature, kappa R/c^2 = 1/2, the CMB balance mass): one simulator for the universe and its black holes,
which the same nearest-surface reading treats as local surfaces inside the cosmic ledger.
NEXT (v1): the record rate along a (I_dot ~ rho_m D^n f H, n = 7/2) integrated with the growth factor D(a) of the established background,
to show E(a) arising rather than being put in; then the two-ruler reading of H0 as a time series."""
import math, os, json, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simcore as Sc
M_P = 1.67262192e-27; HBAR = 1.054571817e-34; KB = 1.380649e-23; C = 299792458.0; G = 6.67430e-11; MPC = 3.0856775814913673e22; LN2 = math.log(2)
_CANON = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "CANON", "iam_canon.json")))["constants"]
BETA_M = _CANON["beta_m"]["value"]; OMEGA_M = 2 * BETA_M          # canon: beta_m = Omega_m / 2, Omega_m = 0.3153 (Planck 2018)

def H_of(a, H0_kms, om=OMEGA_M):
    """Established flat LambdaCDM background (radiation neglected in v0), in 1/s."""
    H0 = H0_kms * 1e3 / MPC; return H0 * math.sqrt(om * a ** -3 + (1 - om))

def E_act(a): return math.exp(1 - 1 / a) if a > 0 else 0.0

def ledger(a=1.0, H0_kms=67.16, iam=True):
    H = H_of(a, H0_kms); out = {}
    out["H"] = Sc.Reading(H, "established", note="flat LambdaCDM background, 1/s")
    out["T_H"] = Sc.Reading(HBAR * H / (2 * math.pi * KB), "established", note="Gibbons-Hawking temperature, K")
    rA = C / H; out["horizon_bits"] = Sc.Reading(math.pi * rA ** 2 / (G * HBAR / C ** 3) / LN2, "established", note="area/(4 l_P^2 ln2)")
    if iam:
        out["bit_price_J"] = Sc.Reading(KB * float(out["T_H"]) * LN2, "IAM", ["encoding_surface"], "k_B T_H ln2 per record, kept on the horizon ledger (writing is local)")
        out["E_record"] = Sc.Reading(E_act(a), "IAM", ["encoding_surface"], "cumulative record E(a) = exp(1 - 1/a) (conjecture)")
        out["R_ratio"] = Sc.Reading(OMEGA_M * a ** -3 / (BETA_M * E_act(a)), "IAM", ["virial_half"], "potential half over the record half")
        H2 = OMEGA_M * a ** -3 + (1 - OMEGA_M)                       # (H/H0)^2 of the established background
        out["mu_matter"] = Sc.Reading(H2 / (H2 + BETA_M * E_act(a)), "IAM", ["virial_half"], "matter's coupling mu = H^2/(H^2 + beta_m E(a) H0^2); "
                                       "1 when the record is empty, falling as it accumulates (book ch:threeway)")
        out["R_cluster"] = Sc.Reading(1 + BETA_M * E_act(a) / H2, "IAM", ["virial_half"], "lensing-to-hydrostatic mass ratio R = 1/mu: light "
                                       "(Sigma = 1) against matter (mu); the deciding test is the sign of dR/dz (IAM negative)")
        out["Sigma_light"] = Sc.Reading(1.0, "IAM", ["encoding_surface"], "light keeps Sigma = 1: the photon writes no record (unbroken U(1)); a free "
                                       "wave holds equal electric and magnetic energy at 376.73 ohm, permanently at natural load")
        # 2026-11-15 (register X6, R13, X1; leads, checked in sympy 2026-11-14 night): the nearest-surface reading
        om_a = OMEGA_M * a ** -3 / (OMEGA_M * a ** -3 + (1 - OMEGA_M))
        M_H = C ** 3 / (2 * G * H)                                   # the mass of the Hubble sphere at the critical density, kg
        out["beta_m_own_hole"] = Sc.Reading(om_a / 2, "IAM", ["settled_half"], "matter's mass in the Hubble sphere priced as its own black hole at "
                                            "Hawking's temperature holds T S = M c^2 / 2 (Smarr), so matter's share is Omega_m(a)/2; equals beta_m today (lead)")
        out["lambda_T_ratio"] = Sc.Reading(math.sqrt(1 - om_a), "IAM", ["encoding_surface"], "Lambda's record on its own de Sitter horizon: T_dS/T_H = "
                                            "sqrt(Omega_L(a)), the book's factor in the Lambda chapter, fixed by the nearest-surface rule (lead)")
        out["held_per_paid"] = Sc.Reading(0.5, "IAM", ["settled_half"], "a settled area-law surface holds 1/2 of what it paid (T S against the integral of T dS)")
        out["held_per_inside"] = Sc.Reading(1.0, "IAM", ["settled_half"], "the cosmic horizon's T S over the energy inside (M_H c^2): 1, the canon's ratio; "
                                            f"inside energy {M_H * C ** 2:.3e} J is half of what was paid")
        # 2026-11-18 (register R13; dark-sector spider DS1, GitHub run 44323118220): the ledger's density identity and the matter sector's own deceleration
        V_H = 4 * math.pi / 3 * rA ** 3; S_geo = KB * math.pi * rA ** 2 / (G * HBAR / C ** 3)
        out["ledger_density_ratio"] = Sc.Reading(float(out["T_H"]) * S_geo / V_H / (3 * H ** 2 / (8 * math.pi * G) * C ** 2), "established",
                                                  note="T_H S_geo / V_H over rho_crit c^2 = 1 exactly: anything booked on the horizon ledger lands at the critical density")
        out["q_matter"] = Sc.Reading(q_sector(a, matter=True), "IAM", ["virial_half"], "matter sector's own deceleration from H_m^2 = H_LCDM^2 + beta_m E(a) H0^2 "
                                     "(the book's eq:Hm); light keeps LCDM's q (Sigma = 1)")
        out["q_light"] = Sc.Reading(q_sector(a, matter=False), "established", note="flat LambdaCDM deceleration (the photon ruler)")
        out["write_rate_over_bekenstein"] = Sc.Reading(E_act(a) * a ** 2, "IAM", ["encoding_surface"], "the record rate the ledger needs per unit of matter's "
                                                       "rest energy, pi E(a) a^2 m c^2/hbar, over Bekenstein's maximum pi m c^2/hbar (1981): exactly 1 today; below it in the past, above it after today unless the history saturates (finding TIME AND RATE)")
        out["proton_write_nats_s"] = Sc.Reading(math.pi * M_P * C ** 2 * E_act(a) * a ** 2 / HBAR, "IAM", ["encoding_surface"], "nats per second each proton must write for the ledger (4.478e24 today)")
        out["matter_clock_rate"] = Sc.Reading(1 / math.sqrt(1 + BETA_M * E_act(a) / (OMEGA_M * a ** -3 + 1 - OMEGA_M)), "IAM", ["virial_half"],
                                              "matter's clock against light's, H/H_m (the book's dual sector): 0.929 today, 0.991 at z = 1; matter's clock runs slower")
    return out

def _growth_table(om, orad=9.1e-5, ai=1e-6, a_end=3.0, steps=6000):
    """Linear growth D(a) and f = dlnD/dlna of flat LambdaCDM with radiation (the book's fig_p1_law_derivations.py setup: Meszaros mode deep in the
    radiation era), RK4 in ln a. Returns lists (lna, D, f)."""
    ol = 1 - om - orad
    def rhs(lna, y):
        a = math.exp(lna); h2 = om / a ** 3 + orad / a ** 4 + ol; dlnh = -(1.5 * om / a ** 3 + 2 * orad / a ** 4) / h2
        return (y[1], -(2 + dlnh) * y[1] + 1.5 * om / a ** 3 / h2 * y[0])
    x0, x1 = math.log(ai), math.log(a_end); h = (x1 - x0) / steps; y0 = ai * om / orad; y = (1 + 1.5 * y0, 1.5 * y0); x = x0
    X, D, F = [x], [y[0]], [y[1] / y[0]]
    for _ in range(steps):
        k1 = rhs(x, y); k2 = rhs(x + h / 2, (y[0] + h / 2 * k1[0], y[1] + h / 2 * k1[1])); k3 = rhs(x + h / 2, (y[0] + h / 2 * k2[0], y[1] + h / 2 * k2[1]))
        k4 = rhs(x + h, (y[0] + h * k3[0], y[1] + h * k3[1])); y = (y[0] + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]), y[1] + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])); x += h
        X.append(x); D.append(y[0]); F.append(y[1] / y[0])
    return X, D, F

def rate_law_power(n, a_lo, a_hi, om=None, orad=9.1e-5):
    """The book's sec:law_exponent calculation (fig_p1_law_derivations.py): dS_info/dln a ~ rho_m D^n f / (T_H A_H) ~ a^-3 D^n f H, fitted as a power of a
    over [a_lo, a_hi] on 400 log-spaced points from a = 10^-2.5 to 2. E(a) = exp(1 - 1/a) needs the power -1."""
    om = OMEGA_M if om is None else om; X, D, F = _growth_table(om, orad); ol = 1 - om - orad
    def at(a):
        x = math.log(a); i = min(int((x - X[0]) / (X[1] - X[0])), len(X) - 2); t = (x - X[i]) / (X[i + 1] - X[i])
        return D[i] + t * (D[i + 1] - D[i]), F[i] + t * (F[i + 1] - F[i])
    L = [-2.5 + k * (math.log10(2) + 2.5) / 399 for k in range(400)]; pts = []
    for l in L:
        a = 10 ** l
        if a_lo <= a <= a_hi:
            d, f = at(a); pts.append((math.log(a), math.log(a ** -3 * d ** n * f * math.sqrt(om / a ** 3 + orad / a ** 4 + ol))))
    mx = sum(p[0] for p in pts) / len(pts); my = sum(p[1] for p in pts) / len(pts)
    return sum((p[0] - mx) * (p[1] - my) for p in pts) / sum((p[0] - mx) ** 2 for p in pts)

def q_sector(a, matter=True):
    """Deceleration q = -1 - d ln H / d ln a of one sector: light sees flat LambdaCDM; matter adds beta_m E(a) H0^2 (E(a) = exp(1 - 1/a), d ln E/d ln a = 1/a)."""
    H2 = OMEGA_M * a ** -3 + (1 - OMEGA_M); dH2 = -3 * OMEGA_M * a ** -3
    if matter: H2 += BETA_M * E_act(a); dH2 += BETA_M * E_act(a) / a
    return -1 - 0.5 * dH2 / H2

def onset_z(matter=True):
    """Redshift where the sector's q crosses zero (bisection in a)."""
    lo, hi = 0.2, 1.0
    for _ in range(80):
        m = (lo + hi) / 2
        if q_sector(m, matter) > 0: lo = m
        else: hi = m
    return 1 / ((lo + hi) / 2) - 1

def black_hole(M_kg):
    """2026-11-15 (register X1, X23): a Schwarzschild black hole priced at its own (Hawking) temperature. Established physics plus the settled_half
    reading. T_CMB today 2.7255 K: a hole colder than the CMB gains more than it radiates."""
    Rs = 2 * G * M_kg / C ** 2; T = HBAR * C ** 3 / (8 * math.pi * G * M_kg * KB); S = math.pi * Rs ** 2 / (G * HBAR / C ** 3)   # S in units of k_B
    kappa = C ** 4 / (4 * G * M_kg)
    return dict(R_s=Sc.Reading(Rs, "established", note="m"), T_Hawking=Sc.Reading(T, "established", note="K"),
                held_per_inside=Sc.Reading(KB * T * S / (M_kg * C ** 2), "IAM", ["settled_half"], "T S / M c^2 = kappa R/c^2 = 1/2 (Smarr)"),
                kappa_R_c2=Sc.Reading(kappa * Rs / C ** 2, "established", note="surface gravity times radius over c^2"),
                membrane_ohm=Sc.Reading(4e-7 * math.pi * C, "established", note="membrane paradigm (Damour; Znajek): the horizon's surface "
                                        "resistivity equals the impedance of free space, 376.73 ohm, matched to infinity (register E38)"),
                held_per_paid=Sc.Reading(0.5, "IAM", ["settled_half"], "1/2, as every settled surface"),
                colder_than_CMB=Sc.Reading(T < 2.7255, "established", note="True: the hole is a net absorber today"))

def collapse_fill(M_kg, R_m):
    """Black-hole formation (collapse spider, 2026-11-18): the fill f = r_s/R of a body of mass M and radius R. f reaches 1 exactly as the first trapped
    surface forms (Oppenheimer-Snyder dust). A static body cannot exceed the causal cap 0.708 (Lattimer 2012) nor Buchdahl's 8/9 (1959)."""
    f = 2 * G * M_kg / (C ** 2 * R_m)
    return dict(fill=Sc.Reading(f, "established", note="r_s/R"), static_possible=Sc.Reading(f < 8 / 9, "established", note="below Buchdahl's 8/9"),
                trapped=Sc.Reading(f >= 1, "established", note="first trapped surface: the new writing surface"))

CAUSAL_CAP = 0.708; BUCHDAHL_CAP = 8 / 9

I_F = math.pi ** 4 / 8

def suddenness(duration_s, own_time_s):
    """How sudden a write is: x = duration / the system's own clock. The classical split it implies for a field switched on smoothly is
    radiated/held = 2 I_f x^-3 (I_f = pi^4/8), capped at 1: the exact half (spent = held) needs x <= (2 I_f)^(1/3) = 2.8985; slower writes keep the
    record and spend almost nothing (surfaces_checks.py)."""
    x = duration_s / own_time_s
    return dict(x=Sc.Reading(x, "established", note="duration over own clock"),
                split=Sc.Reading(min(1.0, 2 * I_F / x ** 3), "established", note="radiated/held, 1 = the exact half"))

def ew_crossover_suddenness(Tc_GeV=159.5, width_GeV=5.0, gstar=106.75, mW_GeV=80.377):
    """The electroweak crossover (D'Onofrio and Rummukainen 2016): duration 2 (dT/T_c) t_c in the radiation era, own clock hbar/(m_W c^2)."""
    hbar_GeV_s = 6.582119569e-25; MPl = 1.220890e19
    t_c = 0.3012 * MPl / (math.sqrt(gstar) * Tc_GeV ** 2) * hbar_GeV_s
    return suddenness(2 * width_GeV / Tc_GeV * t_c, hbar_GeV_s / mW_GeV)

def collapse_suddenness(R0_over_rs):
    """Oppenheimer-Snyder dust collapse from rest at R0: proper time (pi/2)(R0/r_s)^(3/2) in units of the light-crossing time r_s/c."""
    return suddenness(math.pi / 2 * R0_over_rs ** 1.5, 1.0)

M_CMB_BALANCE = HBAR * C ** 3 / (8 * math.pi * G * KB * 2.7255)   # kg: the mass whose Hawking temperature equals today's CMB (about 4.5e22 kg)

def history(a_grid, H0_kms=67.16, iam=True):
    """Every reading along the clock a."""
    return [dict(a=a, z=1 / a - 1, **{k: float(v) for k, v in ledger(a, H0_kms, iam).items()}) for a in a_grid]

def minicosmos_selftest():
    ok = True
    def chk(name, cond): nonlocal ok; ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name)
    L = ledger(1.0, 67.16)
    chk("T_GH = 2.65e-30 K at H0 67.16 (four domains, temperature)", abs(float(L["T_H"]) - 2.65e-30) < 0.005e-30)
    chk("bit price 2.53e-53 J at H0 67.16 (four domains, bit_price)", abs(float(L["bit_price_J"]) - 2.53e-53) < 0.005e-53)
    chk("horizon bits about 3.3e122 at 67.16 (four domains, bits)", abs(float(L["horizon_bits"]) / 1e122 - 3.3) < 0.05)
    chk("horizon bits about 2.8e122 at 72.26", abs(float(ledger(1.0, 72.26)["horizon_bits"]) / 1e122 - 2.8) < 0.05)
    for z, R in ((0.0, 2.0), (0.3, 5.9), (0.7, 19.8), (2.0, 399)):
        r = float(ledger(1 / (1 + z))["R_ratio"]); chk(f"R(z={z}) = {R} (four domains, ratio_R)", abs(r - R) / R < 0.01)
    chk("E(1) = 1, E(1/2) inflection value e^-1", abs(E_act(1.0) - 1) < 1e-12 and abs(E_act(0.5) - math.exp(-1)) < 1e-12)
    chk("mu(1) = 1/(1 + beta_m) = 0.864 (four domains, reference)", abs(1 / (1 + BETA_M) - 0.864) < 0.0005)
    chk("IAM off: no IAM readings", all(v.origin != "IAM" for v in ledger(1.0, iam=False).values()))
    for z, R in ((0.0, 1.1577), (0.3, 1.0848), (2.0, 1.0023)):
        chk(f"R_cluster(z={z}) = {R} (three-way clusters)", abs(float(ledger(1 / (1 + z))["R_cluster"]) - R) < 0.0005)
    _h = 1e-5; _d = (float(ledger(1 / 1.30001)["R_cluster"]) - float(ledger(1 / 1.29999)["R_cluster"])) / 2e-5
    chk(f"dR/dz at z = 0.3 is negative, -0.183 (three-way clusters slope test): {_d:.4f}", abs(_d + 0.1832) < 0.002)
    chk("mu(1) reading equals 1/(1 + beta_m)", abs(float(L["mu_matter"]) - 1 / (1 + BETA_M)) < 1e-12)
    chk("beta_m from matter's own black hole equals the canon beta_m today (X6 lead)", abs(float(L["beta_m_own_hole"]) - BETA_M) < 1e-9)
    chk("Lambda's own-horizon temperature ratio sqrt(Omega_L) = 0.8274 (Lambda chapter)", abs(float(L["lambda_T_ratio"]) - 0.8274) < 0.0005)
    B = black_hole(10 * 1.989e30)
    chk("black hole: T S / M c^2 = 1/2 and kappa R / c^2 = 1/2 (Smarr; X1)", abs(float(B["held_per_inside"]) - 0.5) < 1e-9 and abs(float(B["kappa_R_c2"]) - 0.5) < 1e-12)
    chk("horizon membrane resistivity 376.73 ohm = free-space impedance (E38)", abs(float(B["membrane_ohm"]) - 376.73) < 0.01)
    chk("a 10 solar-mass hole is colder than the CMB", bool(B["colder_than_CMB"].value if hasattr(B["colder_than_CMB"], "value") else float(B["colder_than_CMB"])))
    chk("CMB balance mass about 4.5e22 kg (saturation chapter)", abs(M_CMB_BALANCE / 4.5e22 - 1) < 0.02)
    chk("ledger density identity T_H S_geo / V_H = rho_crit c^2 (dark sector DS1)", abs(float(L["ledger_density_ratio"]) - 1) < 1e-9)
    chk("matter-sector q today -0.6595 (dark sector; light -0.527)", abs(float(L["q_matter"]) + 0.6595) < 0.0005 and abs(float(L["q_light"]) + 0.5271) < 0.0005)
    chk(f"acceleration onset: matter z {onset_z(True):.3f}, light z {onset_z(False):.3f} (dark sector 0.736, 0.631)", abs(onset_z(True) - 0.736) < 0.002 and abs(onset_z(False) - 0.631) < 0.002)
    for a_, v in ((1.0, 1.0), (0.5, 0.0920), (1.25, 1.908), (2.0, 6.594)):
        chk(f"record rate over Bekenstein's maximum at a = {a_}: {v} (time and rate)", abs(float(ledger(a_)["write_rate_over_bekenstein"]) - v) < 0.001 * max(v, 1))
    chk("per-proton write rate today 4.478e24 nats/s", abs(float(L["proton_write_nats_s"]) / 4.478e24 - 1) < 0.001)
    chk("matter clock rate 0.929 today, 0.991 at z = 1 (two clocks)", abs(float(L["matter_clock_rate"]) - 0.9294) < 0.0005 and abs(float(ledger(0.5)["matter_clock_rate"]) - 0.9911) < 0.0005)
    _cf = collapse_fill(1.4 * 1.989e30, 1.2e4)
    chk("neutron star 1.4 Msun, R 12 km: fill 0.34, static allowed", abs(float(_cf["fill"]) - 0.3446) < 0.002 and bool(_cf["static_possible"].value if hasattr(_cf["static_possible"], "value") else float(_cf["static_possible"])))
    chk("caps: causal 0.708 below Buchdahl 8/9 below trapping 1", CAUSAL_CAP < BUCHDAHL_CAP < 1)
    for n_, lo_, hi_, v_ in ((2.5, 0.01, 0.1, -2.02), (3.0, 0.01, 0.1, -1.52), (3.5, 0.01, 0.1, -1.02), (4.0, 0.01, 0.1, -0.53), (3.5, 0.25, 1.0, -1.57)):
        chk(f"rate law n = {n_}: power of dS_info/dln a over a {lo_}-{hi_} = {v_} (book p1_02 sec:law_exponent)", abs(rate_law_power(n_, lo_, hi_) - v_) < 0.006)
    chk("required writing per e-fold today beta_m S_geo = 5.2e121 bits at H0 67.16 (eq:law_Sneed)", abs(BETA_M * float(L["horizon_bits"]) / 5.2e121 - 1) < 0.01)
    _ew = ew_crossover_suddenness()
    chk("suddenness: electroweak crossover x = 7.05e13 write times, split ~7e-41 (adiabatic)", abs(float(_ew["x"]) / 7.05e13 - 1) < 0.01 and float(_ew["split"]) < 1e-39)
    chk("suddenness: the exact half at x = 2.8985", abs(float(suddenness(2.8985, 1.0)["split"]) - 1) < 0.001)
    _c2, _c10 = collapse_suddenness(2.0), collapse_suddenness(10.0)
    chk("suddenness: collapse from 2 r_s x = 4.443, split 0.278; from 10 r_s x = 49.67, split 1.99e-4", abs(float(_c2["x"]) - 4.443) < 0.002 and abs(float(_c2["split"]) - 0.2777) < 0.002
        and abs(float(_c10["x"]) - 49.67) < 0.02 and abs(float(_c10["split"]) / 1.987e-4 - 1) < 0.01)
    print("ALL PASS" if ok else "SOME FAILED"); return ok

if __name__ == "__main__":
    sys.exit(0 if minicosmos_selftest() else 1)
