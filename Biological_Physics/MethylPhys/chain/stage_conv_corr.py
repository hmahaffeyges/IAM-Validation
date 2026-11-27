"""Laboratory correction for bisulfite non-conversion, from the sample's own reads (the November 7 rule; finding F275; design PRIVATE-CONVCORR).

An unmethylated CpG that escapes conversion reads as methylated with probability q; a methylated CpG always reads methylated. A site's observed
level is therefore o = m + (1 - m) q. The sample's own q (q_s) is its islands meter: methylated calls at the CpG islands no healthy cell type
methylates, counted on its own molecules (stage_q2_selfref.count_pat_sites, register T1). The IAM Sequencing Atlas libraries carry their own
rate; q_ref is the median of their meters. A reading is put on the atlas's own scale by removing q_s and applying q_ref:
    m = (o - q_s) / (1 - q_s);   o_ref = m + (1 - m) q_ref = q_ref + (o - q_s)(1 - q_ref) / (1 - q_s)
At q_s = q_ref nothing changes. The same transform applies to a molecule-counted gain rate (marks at unmarked places).
Readings taken from whole molecules against themselves (IAM-A, held error, line locality, copy timing, coupling, runs) do not need it: F275
measured them within 3 % under 1 % planted non-conversion.

Reference file: Runtime Matrices/Lab_Correction/conversion_ref_v1.json (q_ref, its spread, the libraries it came from)."""
import os, json, math

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "Runtime Matrices", "Lab_Correction", "conversion_ref_v1.json")

def load_ref(path=REF):
    return json.load(open(path))

def to_ref(o, q_s, q_ref):
    """One level or rate on the atlas's conversion scale. Values are clipped to [0, 1]."""
    if o is None or q_s is None or q_ref is None: return o
    v = q_ref + (o - q_s) * (1.0 - q_ref) / (1.0 - q_s)
    return min(1.0, max(0.0, v))

def meter(site_counts):
    """q_s and its binomial standard error from the islands meter's counts (gain_err = marks over opportunities, gain_opp)."""
    isl = (site_counts or {}).get("islands") or {}
    q, n = isl.get("gain_err"), isl.get("gain_opp")
    if q is None or not n: return None, None, 0
    return float(q), math.sqrt(max(q * (1 - q), 1e-12) / n), int(n)

def record(q_s, se, n, ref, applied, what):
    return dict(stage="laboratory correction (bisulfite non-conversion, from the sample's own reads)", applied=applied, q_sample=q_s, q_sample_se=se, q_sample_opportunities=n,
                q_atlas=ref.get("q_ref"), q_atlas_libraries=ref.get("libraries"), shift=None if q_s is None else round(q_s - ref["q_ref"], 6), corrected=what,
                formula="o_ref = q_atlas + (o - q_sample)(1 - q_atlas)/(1 - q_sample)", reference_file="Runtime Matrices/Lab_Correction/conversion_ref_v1.json")

def apply_levels(beta, site_counts, ref, clip=True):   # edit 276: clip=False returns the corrected levels unclipped (F302)
    """Correct the sample's site levels (dict or Series of CpG -> level) once, before the Deconvolver; every level reading after it inherits."""
    q_s, se, n = meter(site_counts)
    if q_s is None: return beta, record(None, None, 0, ref, False, "not applied: the islands meter was not counted")
    q_r = ref["q_ref"]; f = (1.0 - q_r) / (1.0 - q_s)
    if hasattr(beta, "items") and not hasattr(beta, "clip"):
        out = {k: (None if v is None or (isinstance(v, float) and math.isnan(v)) else ((min(1.0, max(0.0, q_r + (v - q_s) * f))) if clip else (q_r + (v - q_s) * f))) for k, v in beta.items()}
    else:
        out = (q_r + (beta - q_s) * f); out = out.clip(0, 1) if clip else out
    return out, record(q_s, se, n, ref, True, "every site level, before the Deconvolver (Met-A, Met-A C-Score, territories, Deconvolver and mixture check, distance to maximum disorder)")

def apply_rate(rate, site_counts, ref):
    """Correct one molecule-counted gain rate (clock gain, Law Sphere writes, unused-island strays)."""
    q_s, _, _ = meter(site_counts)
    return to_ref(rate, q_s, ref["q_ref"]) if q_s is not None else rate

if __name__ == "__main__":   # closed-form check: known m, planted q, recovered on the reference scale
    import random
    ref = dict(q_ref=0.0043, libraries=1); random.seed(1); worst = 0.0
    for _ in range(2000):
        m = random.random(); q_true = random.uniform(0, 0.03)
        o = m + (1 - m) * q_true
        want = m + (1 - m) * ref["q_ref"]
        worst = max(worst, abs(to_ref(o, q_true, ref["q_ref"]) - want))
    print("closed form: worst error", worst)
    print("never-methylated island at the planted 1 %:", to_ref(0.01405, 0.01405, 0.0043), "(expected 0.0043)")
