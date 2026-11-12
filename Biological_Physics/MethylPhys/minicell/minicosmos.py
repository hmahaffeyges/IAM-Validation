"""The mini cosmos, version 0 (DEV-MINICOSMOS-01; development; a simulation of the model, not data). Design and skeleton, 2026-11-12.

WHERE IAM'S LAW IS (physics first). IAM enters only through the entropy functional; the Friedmann equation is never modified.
  the surface    the cosmic apparent horizon r_A = c/H: the nearest encoding surface for distributed structure       [IAM encoding_surface]
  the kick       the Gibbons-Hawking temperature T_H = hbar H / (2 pi k_B), the horizon's own temperature              [established]
  the bit price  dE_rec(a) = k_B T_H(a) ln2 per record written on the horizon (epoch-dependent; conjecture)         [IAM encoding_surface]
  the record     E(a) = exp(1 - 1/a), the cumulative record on the horizon (conjecture; needs rate exponent n = 7/2) [IAM activation]
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
NEXT (v1): the record rate along a (I_dot ~ rho_m D^n f H, n = 7/2) integrated with the growth factor D(a) of the established background,
to show E(a) arising rather than being put in; then the two-ruler reading of H0 as a time series."""
import math, os, json, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simcore as Sc
HBAR = 1.054571817e-34; KB = 1.380649e-23; C = 299792458.0; G = 6.67430e-11; MPC = 3.0856775814913673e22; LN2 = math.log(2)
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
        out["bit_price_J"] = Sc.Reading(KB * float(out["T_H"]) * LN2, "IAM", ["encoding_surface"], "k_B T_H ln2 per record on the horizon")
        out["E_record"] = Sc.Reading(E_act(a), "IAM", ["encoding_surface"], "cumulative record E(a) = exp(1 - 1/a) (conjecture)")
        out["R_ratio"] = Sc.Reading(OMEGA_M * a ** -3 / (BETA_M * E_act(a)), "IAM", ["virial_half"], "potential half over the record half")
    return out

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
    print("ALL PASS" if ok else "SOME FAILED"); return ok

if __name__ == "__main__":
    sys.exit(0 if minicosmos_selftest() else 1)
