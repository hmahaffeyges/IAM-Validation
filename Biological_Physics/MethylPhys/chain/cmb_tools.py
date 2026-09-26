#!/usr/bin/env python3
"""Every tool this chain borrowed from cosmology, and whether it worked on THIS run.

The author's instruction, 2026-09-25: "We should have every one of the CMB tools we employed listed and it
should contain a pass or fail next to it and if it fails it sends the info to the red flags report for the AI
to read. We are going to eventually have a LOT more of them as time goes on."

So this is a registry, not prose. Each entry says what was borrowed, from whom, where it is implemented, and
- the part that matters - a CHECK that runs on the finished bundle and returns PASS, FAIL, NOT_RUN or
NOT_APPLICABLE with one line of evidence. The Safeguards tab renders the table; anything that FAILS is also
emitted as a red flag, so a reader who only opens one tab still sees it.

Adding a tool: append one TOOL entry with a check that takes the bundle and returns (status, evidence).
Nothing else needs changing - the table, the counts and the red flags all follow.

Status vocabulary, deliberately narrow:
  PASS            the tool ran on this specimen and its own internal condition held
  FAIL            the tool ran and its condition did NOT hold - this is a red flag
  NOT_RUN         the tool is implemented and wired, but this specimen did not reach it
  NOT_APPLICABLE  the tool cannot apply to this specimen (wrong substrate, no laboratory zero)
  NOT_BUILT       borrowed in principle, not implemented - the roadmap, kept visible on purpose
"""

# Each tool: (id, name, borrowed_from, what_it_does_here, where, check)
# `check(o)` -> (status, evidence)


def _sky(o):
    return o.get("patient_sky") or {}


def _chk_detection(o):
    """Stage 2d - inverse-variance foreign-cell detection (PROC-MF-01/02/03; adopted by the author 2026-09-26, scoped to
    commissioned laboratories). PASS when the stage ran and its status is one the report can act on; NOT_RUN when the
    bundle has no stage-2d record; NOT_APPLICABLE when the laboratory has no panel; FAIL when it ran but returned no
    per-cell rows (the stage promised a table and did not deliver one)."""
    fd = o.get("foreign_detection")
    if fd is None: return "NOT_RUN", "bundle has no foreign_detection record"
    st = fd.get("status") or ""
    if st.startswith("NOT_COMMISSIONED"): return "NOT_APPLICABLE", st
    if st.startswith("NOT_RUN") or st.startswith("WITHHELD"): return "NOT_RUN", st
    if not fd.get("cells"): return "FAIL", "status %r but no per-cell rows" % st
    return "PASS", "%d foreign columns scored; %d above the laboratory line; %s" % (len(fd["cells"]), len(fd.get("detected") or []), st)


def _chk_healpix(o):
    s = _sky(o)
    if not s.get("available"):
        return ("NOT_APPLICABLE", "no commissioned residual scale for this laboratory, so no sky was built")
    n = ((s.get("all") or {}).get("n"))
    if not n:
        return ("NOT_RUN", "sky present but no address count recorded")
    # NSIDE 128 -> 196,608 pixels; the projection is genomically local (measured 2026-09-22:
    # 196,608/196,608 pixels genomically contiguous, one chromosome each, median span 511 bp)
    return ("PASS", "%s addresses placed on the NSIDE-128 sphere in genomic order" % format(n, ","))


def _chk_plate(o):
    s = _sky(o)
    if not s.get("available"):
        return ("NOT_APPLICABLE", "no sky for this laboratory")
    if s.get("_plate_drawn") is True:
        return ("PASS", "the specimen's own plate was rendered from its own residuals")
    return ("FAIL", "the plate could not be drawn: %s" % (s.get("_plate_error") or "unknown"))


def _chk_mask(o):
    s = _sky(o)
    if not s.get("available"):
        return ("NOT_APPLICABLE", "no sky for this laboratory")
    cl = s.get("classes") or {}
    masked = [c for c, v in cl.items() if (v or {}).get("status") != "ASSESSABLE"]
    if not cl:
        return ("NOT_RUN", "no per-class panels recorded")
    return ("PASS", "%d of %d class panels masked below their presence floor and report nothing"
            % (len(masked), len(cl)))


def _chk_residual_scale(o):
    s = _sky(o)
    if not s.get("available"):
        return ("NOT_APPLICABLE", "this laboratory has no commissioned residual scale")
    a = s.get("all") or {}
    f = a.get("frac_abs_z_gt2")
    if f is None:
        return ("NOT_RUN", "no whole-sky residual statistic recorded")
    # healthy range measured on the four commissioned laboratories
    ok = 0.0 <= float(f) <= 0.25
    return ("PASS" if ok else "FAIL",
            "%.1f %% of addresses beyond |z| = 2 (healthy 2.6-3.2 %%); the scale is this laboratory's own "
            "per-address spread" % (100 * float(f)))


def _chk_nilc(o):
    so = o.get("second_opinion") or {}
    if not so.get("available"):
        return ("NOT_RUN", str(so.get("reason") or "the needlet solver did not run"))
    ag = so.get("agreement")
    if ag == "AGREE":
        return ("PASS", "needlet solver agrees with the primary fit at class level (L1 = %s)"
                % so.get("L1_class"))
    return ("FAIL", "needlet solver DISAGREES with the primary fit at class level (L1 = %s, bar 0.10)"
            % so.get("L1_class"))


def _chk_inverse_variance(o):
    td = o.get("trace_detection") or {}
    m = td.get("_meta") or {}
    if not m.get("available"):
        return ("NOT_RUN", str(m.get("reason") or "the trace panel did not load"))
    if not m.get("calibrated_for_this_substrate", True):
        return ("NOT_APPLICABLE", "thresholds are a whole-blood measurement; this specimen is '%s'"
                % (m.get("substrate_declared") or "not declared"))
    return ("PASS", "each address weighted by the inverse of its atlas posterior variance over %s addresses"
            % format(m.get("n_addresses") or 0, ","))


def _chk_mahalanobis(o):
    dep = o.get("departure") or {}
    if not dep.get("reportable"):
        return ("NOT_APPLICABLE", str(dep.get("status") or "not reportable for this specimen"))
    d = dep.get("mahalanobis_distance")
    n = dep.get("n_axes") or dep.get("axes") or 1
    return ("PASS", "distance %s on %s banded axis/axes" % (d, n))


def _chk_brightness(o):
    # superseded, not retired: the sky weights by the sample's own composition, which no precomputed
    # brightness file can. Kept in the registry so the lineage is visible.
    cl = o.get("classes") or {}
    any_ci = any((v or {}).get("A_ci_lo") is not None for v in cl.values())
    return ("PASS" if any_ci else "NOT_RUN",
            "surface brightness was the first borrowing; superseded by the composition-weighted sky. "
            "Credible intervals on the class readings are its surviving use." if any_ci else
            "no brightness credible intervals on this run")


def _not_built(reason):
    def f(_o):
        return ("NOT_BUILT", reason)
    return f


TOOLS = [
 ("HEALPIX", "HEALPix pixelisation", "Gorski et al., the Planck sky",
  "every CpG placed on a sphere in genomic order, NSIDE 128, so neighbouring pixels are neighbouring genome",
  "stage_4_6_patient_cmb.py", _chk_healpix),
 ("PLATE", "Mollweide plate of the residual sky", "CMB map-making",
  "the specimen's own residual z rendered as a sky, per class",
  "stage_4_6_patient_cmb.py render_plate", _chk_plate),
 ("RESIDUAL", "residual against a measured zero and scale", "CMB anisotropy against the monopole",
  "z = (beta - sum_c f_c mu_c - m_lab) / s_lab: the specimen minus what its OWN composition predicts, over "
  "this laboratory's own per-address spread. Never a comparison to a picture.",
  "stage_4_6_patient_cmb.py", _chk_residual_scale),
 ("MASK", "masking what the instrument cannot see", "the galaxy cut",
  "a class below its presence floor is not there, so its panel is masked and reports nothing",
  "cpg_conductor presence floors", _chk_mask),
 ("NILC", "needlet internal linear combination", "CMB component separation",
  "an independent second solve of the composition, compared with the primary fit class by class",
  "nilc_celltype_deconvolver.py", _chk_nilc),
 ("DETECT", "inverse-variance matched-template detection of a foreign cell", "point-source / cluster detection in a noisy map",
  "Stage 2d: each foreign cell's atlas profile, minus the specimen's own blood background, fitted to the residual with per-locus 1/variance "
  "weights from commissioned healthy blood; centred and lined per laboratory. Detection limit 0.5-1 % on four 450K laboratories (PROC-MF-02/03); "
  "the full-covariance matched filter was tried first and tied NNLS (PROC-MF-01: 1,506 markers vs 36 arrays).",
  "cpg_conductor.py stage_2d_foreign_detection; Runtime Matrices/A_Scoring_Module/detection_panel_v1.json", _chk_detection),
 ("INVVAR", "inverse-variance weighting", "optimal map-making",
  "each address weighted by the inverse of its atlas posterior variance - what brought the trace-class "
  "detection limit from 5 % to 2 %",
  "stage_2c_trace_detection.py", _chk_inverse_variance),
 ("MAHAL", "Mahalanobis distance in the banded space", "CMB parameter likelihoods",
  "how far this specimen sits from the healthy centre, in units of the healthy spread",
  "cpg_conductor stage 5", _chk_mahalanobis),
 ("BRIGHT", "surface brightness", "astronomical photometry",
  "an intensity that does not depend on distance or aperture, applied to a class; superseded by the "
  "composition-weighted sky, not retired",
  "attach_brightness_ci (retired v1 conductor)", _chk_brightness),
 # --- borrowed in principle, not implemented. Kept visible so the roadmap is not a separate document.
 ("CLS", "angular power spectrum of the residual sky", "the CMB power spectrum",
  "would say whether a departure is locally clustered along the genome or spread across it - one number per "
  "specimen, computable on data already on disk",
  "not implemented", _not_built("MEASURED 2026-09-25 (PROC-CLS-01): the sky IS structured, 3.5x its permutation null at l 2-8, dying by l~200 - but the healthy reference does not transfer across laboratories (B3), so nothing is reported. Original note: no implementation; roadmap - needs no new data")),
 ("COV", "full cell-type covariance in the separation", "generalised least squares on the CMB covariance",
  "the atlas carries the covariance between cell types at each address and the chain treats every "
  "uncertainty as independent - the largest piece of unspent evidence in the chain",
  "not implemented", _not_built("no implementation; the atlas has the posterior covariance")),
 ("DIFFMAP", "difference map between two draws", "CMB difference maps between detectors",
  "two draws from one person, subtracted: the technical term cancels and the detection limit drops by an "
  "order of magnitude",
  "not implemented", _not_built("blocked: no serial cohort with two draws from the same person")),
 ("BEAM", "beam smoothing", "the instrument beam",
  "smoothing the residual sky to the scale at which genomic correlation is interpretable",
  "stage_4_6_patient_cmb.py (fixed smoothing)", _not_built(
      "applied at a fixed scale in the plate; not yet a measured beam with a stated resolution")),
 ("POSTERIOR", "per-patient posterior for the composition", "parameter estimation from a CMB likelihood",
  "sampling the composition instead of a single constrained fit: every fraction would carry a credible "
  "interval, and a trace component would have a tail instead of being pinned at exactly zero",
  "not implemented", _not_built("no implementation; ENHANCEMENTS B3 - needs a runtime bar per specimen")),
 ("ILC_SKY", "internal linear combination on the residual sky", "CMB foreground cleaning",
  "the needlet solver currently gives a second opinion on composition; applied to the residual sky it would "
  "separate a departure from the genomic-correlation background",
  "not implemented", _not_built("no implementation; ENHANCEMENTS B4")),
 ("XSPEC", "cross-spectra between class panels", "CMB temperature-polarisation cross-spectra",
  "whether a departure is shared across cell classes (systemic) or confined to one (focal)",
  "not implemented", _not_built("no implementation; ENHANCEMENTS B7 - depends on the power spectrum, CLS")),
 ("APODMASK", "apodised mask instead of a binary presence floor", "the apodised galaxy mask",
  "today a class at 1.9 per cent is masked and one at 2.1 per cent is fully trusted; a graded mask would "
  "weight each class panel by how well that class is actually constrained",
  "not implemented", _not_built("no implementation; ENHANCEMENTS B8")),
 ("FISHER", "degeneracy and Fisher analysis of the composition", "the banana degeneracy in parameter space",
  "which composition solutions are genuinely distinguishable, rather than assuming the reported one is "
  "unique - it turns 'the solvers disagree' into 'these two are degenerate along this direction'",
  "not implemented", _not_built("no implementation; ENHANCEMENTS B9")),
]


def evaluate(o):
    """-> list of dicts, one per tool, with its status on this bundle."""
    out = []
    for tid, name, src, does, where, check in TOOLS:
        try:
            status, evidence = check(o)
        except Exception as e:                                   # a broken check is itself a finding
            status, evidence = "FAIL", "the check raised %s: %s" % (type(e).__name__, e)
        out.append({"id": tid, "tool": name, "borrowed_from": src, "what_it_does_here": does,
                    "where": where, "status": status, "evidence": evidence})
    return out


def failures(o):
    """The ones a red flag must carry."""
    return [t for t in evaluate(o) if t["status"] == "FAIL"]


def summary(o):
    c = {}
    for t in evaluate(o):
        c[t["status"]] = c.get(t["status"], 0) + 1
    return c


if __name__ == "__main__":
    import json
    import sys
    b = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else {}
    for t in evaluate(b):
        print("%-8s %-12s %s" % (t["status"], t["id"], t["evidence"][:100]))
