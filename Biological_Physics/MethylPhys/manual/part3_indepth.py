#!/usr/bin/env python3
"""Part III of Issue 003 - the commissioned chain, the atlas and the cosmology toolkit, in depth.

Written 2026-09-22 on the author's instruction: *"everything in the interface we just added should be in this with
the full chain and and stage detailed in depth. This is the place to do it. The interface is the shorter condensed
version. We should include the in depth exploration of the CMB tools we applied and so forth too."*

Two of the four chapters are GENERATED rather than written, so they cannot drift from the instrument:

  * the stage-by-stage chapter reads `cpg_conductor.py` itself - the stage list, the call order in run_full() and
    each stage's own docstring - so a stage added, removed or re-documented in the code appears here on the next
    build, and a stage that is not in the conductor cannot appear at all;
  * the atlas chapter reads IAMAtlasREBUILD_celltype_to_class.json and percell_reference_v0_3.json, so the cell
    counts, the per-entry marker counts, exclusivities and healthy ranges are the ones the chain will actually use.

The toolkit chapter is authored, with every number in it traceable to a sealed procedure or a dated working note.
"""
import os, re, json, ast, collections

HERE = os.path.dirname(os.path.abspath(__file__))
CHAIN = os.path.normpath(os.path.join(HERE, "..", "chain"))
ATLAS = os.path.normpath(os.path.join(HERE, "..", "atlas"))


# ── generated inputs ────────────────────────────────────────────────────────────────────────────────────────────
def conductor_stages():
    """(name, docstring, call_order) for every stage function in the conductor, in run_full() order."""
    src = open(os.path.join(CHAIN, "cpg_conductor.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    docs = {n.name: (ast.get_docstring(n) or "").strip()
            for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("stage")}
    run = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_full"), None)
    called, seen = [], set()
    if run is not None:
        for node in ast.walk(run):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in docs:
                if node.func.id not in seen:
                    seen.add(node.func.id); called.append(node.func.id)
    for name in docs:                      # stages defined but not called by run_full are named as such
        if name not in seen: called.append(name)
    return [(n, docs[n], (i + 1) if n in seen else None) for i, n in enumerate(called)]


def atlas_cells():
    """{class: [cell, ...]} and the per-entry reference, both as the chain will read them."""
    c2c = json.load(open(os.path.join(ATLAS, "IAMAtlasREBUILD_celltype_to_class.json"), encoding="utf-8"))
    by = collections.defaultdict(list)
    for cell, cl in c2c.items(): by[cl].append(cell)
    for cl in by: by[cl].sort()
    ref = {}
    p = os.path.join(CHAIN, "Runtime Matrices", "Percell_Reference", "percell_reference_v0_3.json")
    if os.path.exists(p):
        ref = json.load(open(p, encoding="utf-8")).get("entries", {})
    return dict(by), ref


def cell_roster_rows(cls_key):
    """Rows for one class's cell table: cell, markers, resolvable, exclusivity, healthy A range across labs."""
    by, ref = atlas_cells()
    rows = []
    for cell in by.get(cls_key, []):
        e = ref.get(cell)
        if not e:
            rows.append((cell, "-", "no per-entry reference yet", "-", "-")); continue
        labs = e.get("labs", {})
        p10 = [v["A_p10"] for v in labs.values() if "A_p10" in v]
        p90 = [v["A_p90"] for v in labs.values() if "A_p90" in v]
        mk = [v["n_markers_found"] for v in labs.values() if "n_markers_found" in v]
        rng = f"{min(p10):.3f} - {max(p90):.3f}" if p10 and p90 else "-"
        rows.append((cell,
                     f"{min(mk)}-{max(mk)}" if mk else str(e.get("n_markers_panel", "-")),
                     "yes" if e.get("individually_resolvable") else "NO - in a collinearity group",
                     f"{e.get('exclusivity', 0):.3f}" if e.get("exclusivity") is not None else "-",
                     rng))
    return rows


# ── the CMB toolkit, authored: tool, what it does in cosmology, what it does here, status ────────────────────────
TOOLKIT = [
 ("HEALPix spherical pixelisation",
  "Planck's equal-area tessellation of the sky. Every pixel covers the same solid angle, so a statistic computed "
  "pixel by pixel is not biased by where on the sphere it sits.",
  "The 450K/EPIC CpG set is projected onto an N_side = 128 sphere, 196,608 pixels, by genomic coordinate from the "
  "Illumina manifest. Each pixel holds the residual of a contiguous run of CpGs on one chromosome.",
  "MEASURED 2026-09-22: 196,608 of 196,608 pixels hold genomically contiguous CpGs, all on one chromosome; median "
  "span within a pixel 511 bp (90th percentile 20,481 bp). Neighbouring pixels are a median 1,256 CpGs apart in "
  "genomic order, 25 per cent within 10 CpGs, under 1 per cent beyond 10,000 - against about 161,000 for a random "
  "assignment. The projection is a genuine locality-preserving reindexing, which is what licenses reading "
  "structure in the map as genomic rather than as an artefact of the layout."),
 ("Internal linear combination (ILC)",
  "Planck's component separation: form a weighted combination of frequency maps that minimises variance subject to "
  "unit response to the component wanted. It needs no model of the foregrounds, only their statistics.",
  "The composition step solves for cell-type fractions as the combination of atlas reference profiles that "
  "minimises residual variance under a non-negativity and sum constraint - Walther's constrained NNLS - with the "
  "needlet ILC variant run beside it as a second opinion.",
  "COMMISSIONED. The constrained solver is the composition the report stands on; the ILC variant is compared with "
  "it at class level with an agreement bar of L1 <= 0.10 and reported as a flag (PROC-NILC-01). Cell-level "
  "disagreement inside one lineage is expected and is not scored, because the atlas cannot separate the members of "
  "a collinearity group."),
 ("Needlets (wavelets on the sphere)",
  "Localised both in position and in angular scale, so a foreground that is strong at one scale in one region can "
  "be down-weighted there without discarding that scale everywhere.",
  "The second solver weights each atlas entry by the inverse of its own posterior standard deviation, which is the "
  "same idea one level down: an address the atlas pinned down precisely carries more weight than one it did not.",
  "LIVE inside the second opinion. The full needlet decomposition of the methylome map - scale-by-scale "
  "separation rather than a single inverse-variance weight - is not built."),
 ("Masking and the confidence mask",
  "Planck does not report the sky behind the galaxy. The galactic plane is masked, and every statistic is computed "
  "on the unmasked fraction with the mask stated.",
  "A class whose estimated presence falls below its detection floor is masked: the chain reports nothing for it "
  "rather than a number computed from noise. Loci absent on the sample's array type are masked the same way.",
  "COMMISSIONED, and the disanalogy is stated where it appears: the galaxy hides a sky that is really there, "
  "whereas a class below its floor is not present in the specimen at all. Both refuse to report where the "
  "instrument cannot see; only one has something behind the mask."),
 ("Beam and transfer functions",
  "Every instrument filters the true sky. Planck's likelihood carries the beam window function explicitly, because "
  "a measured power spectrum is the true one multiplied by what the instrument did to it.",
  "The deconvolver is the chain's beam: it explains away signal that looks like a change in cell composition. This "
  "was found empirically before it had a name - depleted chromosomes in an early validation run were CpGs the "
  "solver had absorbed into composition.",
  "NAMED and recorded as a lesson; not yet a forward-modelled transfer function. Deconvolving the instrument out "
  "of the reading, the way a beam is deconvolved from a spectrum, is on the roadmap."),
 ("Per-pixel noise covariance",
  "Planck carries a noise covariance per pixel, not one number for the map, because sensitivity varies across the "
  "sky with the scan pattern.",
  "Each atlas entry carries its own posterior standard deviation from the MCMC build, and each laboratory its own "
  "measured spread; the gauge is read against the laboratory's own zero rather than a global one.",
  "COMMISSIONED per entry and per laboratory. What is NOT used is the covariance BETWEEN cell types at the same "
  "address - the MCMC produced it, and the chain currently treats every entry's uncertainty as independent. That "
  "is the conservative choice and the wasteful one: a full covariance is what lets a separation say these two are "
  "individually uncertain but their sum is well determined, which is exactly the blood situation. It is the "
  "largest piece of unspent evidence in the chain."),
 ("Null tests and shuffled skies",
  "Before a cosmological detection is believed, the same pipeline is run on simulations with no signal, on split "
  "halves, and on rotated or shuffled maps, and must return nothing.",
  "The null runner shuffles the sky, permutes labels, splits panels and runs constructed specimens through the "
  "whole chain; a reading that survives a shuffle is a reading of the pipeline, not of the patient.",
  "COMMISSIONED - nulls N1 to N8 live, and every family-A validation carries its null result."),
 ("Surface brightness (the first borrowing)",
  "Astronomy's way of stating an intensity that does not depend on distance or aperture, so two sources can be "
  "compared without knowing how far away they are.",
  "The brightness layer applied that to an architecture class: an intensity per class that does not depend on how "
  "much of the class is in the specimen.",
  "SUPERSEDED, NOT RETIRED. The sky now builds its expectation from the sample's own composition, which no "
  "precomputed per-class file can do - but the first import is still load-bearing one layer down, and its lineage "
  "is stated on the interface's sky page."),
 ("Difference maps",
  "Two observations of the same sky differenced leave instrument and systematics behind and show only what "
  "changed. It is the strongest design in the toolkit.",
  "Two draws from the same person, processed the same way, differenced address by address: everything constant "
  "about that person and that laboratory cancels.",
  "NOT YET POSSIBLE - no cohort held here has a repeat draw. The technical floor is therefore an upper bound "
  "inferred from cross-sectional spread: between-person spread per address has a median of 0.0289 and a 5th "
  "percentile of 0.0089, which bounds the technical term. On that bound a paired two-sigma difference resolves "
  "0.025 at one address, 0.011 across five, 0.0056 across a CpG island of twenty, and 0.0008 across a thousand - "
  "an order of magnitude below published island-scale effects. Finding a serial cohort is the first item on the "
  "roadmap."),
 ("Angular power spectrum",
  "The two-point statistic of the sky: how much structure there is at each angular scale, and the object every "
  "cosmological parameter is fitted to.",
  "The same statistic on the methylome sphere would say at what genomic scale a specimen's departure lives - "
  "single addresses, islands, domains, or whole chromosomes - in one curve rather than a list of regions.",
  "NOT BUILT. It is the natural next statistic now that the projection is measured to be genomically local, and "
  "it is on the roadmap."),
]

REFUSALS = [
 ("A class below its detection floor", "reports nothing for that class", "Planck's confidence mask"),
 ("A laboratory with no measured zero", "refuses the absolute reading; relative only", "an uncalibrated detector"),
 ("An array type whose loci are absent", "masks the missing loci and states the count", "incomplete sky coverage"),
 ("A per-entry band wider than the gauge's normal band", "prints the number, withholds the tier word", "an error bar wider than the effect"),
 ("A cell in a collinearity group", "reports the group, never the member", "degenerate parameters in a fit"),
 ("A single array asked for a cellular age in years", "refuses - the trajectory is a population measurement", "a one-pixel cosmology"),
]


# ── renderers ───────────────────────────────────────────────────────────────────────────────────────────────────
def render(story, L, tbl, SP, PageBreak, Paragraph):
    """Part III: the chain stage by stage, the atlas and its cells, the cosmology toolkit, and the refusals."""
    story.append(PageBreak())
    story.append(Paragraph("PART III - THE COMMISSIONED CHAIN, IN DEPTH", L.sSect))
    story.append(Paragraph("Every stage as the code documents itself, the atlas cell by cell, and the cosmology "
                           "toolkit tool by tool. The researcher interface is the condensed form of this part; "
                           "this is the long form.", L.sSub))
    story.append(Paragraph("Two of the four chapters below are generated from the instrument rather than written: "
        "the stage chapter reads the conductor's own stage list, call order and docstrings, and the atlas chapter "
        "reads the cell-to-class map and the per-entry reference. A stage that is not in the conductor cannot "
        "appear here, and a cell count printed here is the one the chain will use.", L.sMut))

    # chapter 1 - the stages
    story.append(SP(0.10))
    story.append(Paragraph("III.1 &nbsp; The chain, stage by stage", L.sSect2))
    stages = conductor_stages()
    called = [s for s in stages if s[2]]
    not_called = [s for s in stages if not s[2]]
    story.append(Paragraph(f"run_full() calls {len(called)} stages in the order below. {len(not_called)} further "
        f"stage function{'s' if len(not_called) != 1 else ''} {'are' if len(not_called) != 1 else 'is'} defined in "
        f"the conductor but deliberately not called: "
        + ", ".join(f"<b>{n}</b>" for n, _, _ in not_called) + ".", L.sBody))
    for name, doc, order in stages:
        head = f"{order}. {name}" if order else f"{name} - defined, NOT called by run_full"
        story.append(SP(0.045))
        story.append(Paragraph(f"<b>{head}</b>", L.sSect2))
        body = re.sub(r"\s+", " ", doc).strip() or "(no docstring)"
        story.append(Paragraph(body.replace("<", "&lt;").replace(">", "&gt;"), L.sBodySm))

    # chapter 2 - the atlas and its cells
    by, ref = atlas_cells()
    story.append(PageBreak())
    story.append(Paragraph("III.2 &nbsp; The atlas, and the cells it can speak about", L.sSect2))
    total = sum(len(v) for v in by.values())
    story.append(Paragraph(f"IAMAtlasREBUILD holds <b>{total} cell types</b> mapped to the eight architecture "
        f"classes, and <b>{len(ref)}</b> of them now carry a per-entry healthy reference measured on 40-array "
        f"panels per laboratory. The {total - len(ref)} without one are reported at class level only. Every "
        f"per-cell number in this document and in the interface comes from these two files.", L.sBody))
    story.append(SP(0.05))
    story.append(tbl([["class", "cells", "with per-entry reference", "class floor H_min"]] +
        [[cl, str(len(by[cl])), str(sum(1 for c in by[cl] if c in ref)), "see the class card"]
         for cl in sorted(by, key=lambda c: -len(by[c]))], [0.34, 0.16, 0.30, 0.20], fs=7.5))
    story.append(SP(0.05))
    story.append(Paragraph("<b>Two limits on a per-cell claim, and they are not the same limit.</b> The first is "
        "separability: an entry in a collinearity group cannot be told from its neighbours by this atlas at any "
        "precision, so the chain reports the group. The second is the width of the entry's own healthy band: where "
        "that band is wider than the gauge's normal band, the number is printed and the tier word withheld. The "
        "per-class rosters on the cards state both, entry by entry.", L.sBodySm))
    story.append(Paragraph("<b>The duplicate-label problem, stated plainly.</b> The same lineage appears under "
        "several atlas entries whose per-cell readings differ by more than anything biological could explain, "
        "purely because different reference panels defined their markers. Until those entries are merged to one "
        "lineage each, no reading should be quoted as 'which cell moved'. It is the blocker on per-cell reporting "
        "and the second item on the roadmap.", L.sBodySm))

    # chapter 3 - the toolkit
    story.append(PageBreak())
    story.append(Paragraph("III.3 &nbsp; The cosmology toolkit, tool by tool", L.sSect2))
    story.append(Paragraph("None of these tools was invented here. Each was built to read a faint signal out of a "
        "noisy sky by people who had one sky, no repeat observations and no control group - which is the position a "
        "clinician is in with one specimen. For each: what it does in cosmology, what it does here, and where it "
        "stands.", L.sBody))
    for name, cosmo, here, status in TOOLKIT:
        story.append(SP(0.05))
        story.append(Paragraph(f"<b>{name}</b>", L.sSect2))
        story.append(tbl([["in cosmology", "in this chain"], [cosmo, here]], [0.5, 0.5], fs=7.2))
        story.append(Paragraph(status, L.sBodySm))

    # chapter 4 - refusals
    story.append(PageBreak())
    story.append(Paragraph("III.4 &nbsp; What the chain refuses, and the cosmology twin of each refusal", L.sSect2))
    story.append(Paragraph("A measurement instrument is defined as much by what it declines to report as by what "
        "it reports. Each refusal below is enforced in code, not by convention, and each has a twin in the "
        "practice the toolkit came from.", L.sBody))
    story.append(tbl([["the situation", "what the chain does", "the cosmology twin"]] +
                     [list(r) for r in REFUSALS], [0.34, 0.38, 0.28], fs=7.5))
    story.append(SP(0.05))
    story.append(Paragraph("NOT REPORTABLE is a result. It is printed with the reason and the number that "
        "triggered it, so a reader can see what would have to change for the reading to become available.", L.sMut))


# ── III.5  the engine, exactly: every formula the chain computes, read from the runtime ──────────────────────────
def runtime(name):
    import glob
    g = glob.glob(os.path.join(CHAIN, "**", name), recursive=True)
    return json.load(open(g[0], encoding="utf-8")) if g else {}


def render_engine_spec(story, L, tbl, SP, PageBreak, Paragraph):
    """The formulas as the code computes them, with the four-laboratory constants they are read against.

    Every number here is loaded from the runtime matrices at build time, so the page is a specification of the
    engine rather than a description of it: if a constant changes, this page changes with it.
    """
    band = runtime("identity_band_v3.json"); meta = band.get("_meta", {})
    pooled = band.get("pooled", {}); age01 = runtime("age01_results.json")
    curve = runtime("reference_age_curve_v1.json"); cmeta = curve.get("_meta", {})
    coh = meta.get("cohorts", {})
    if isinstance(coh, str):
        import ast as _a; coh = _a.literal_eval(coh)
    sigma = (pooled.get("p90", 0) - pooled.get("p10", 0)) / (2 * 1.2816) if pooled else None

    story.append(PageBreak())
    story.append(Paragraph("III.5 &nbsp; The engine, exactly - every formula the chain computes", L.sSect))
    story.append(Paragraph("Loaded from the runtime matrices at build time, so this page is the specification and "
                           "not a description of one: if a constant changes, this page changes with it.", L.sSub))

    story.append(Paragraph("<b>1. The entropy of a methylation fraction.</b> For a mean beta value on a locus set, "
        "the chain computes the binary Shannon entropy in bits, with beta clamped to (1e-12, 1 - 1e-12):", L.sBody))
    story.append(Paragraph("H(&#946;) = -&#946; log<sub>2</sub>&#946; - (1-&#946;) log<sub>2</sub>(1-&#946;)", L.sSect2))

    story.append(Paragraph("<b>2. The reported gauge</b> (<font name='Courier'>stage_b_identity</font>, "
        "PROC-SWITCH-01; SOP s41/s106). Beta is averaged over the class's IDENTITY loci - not a marker union - and "
        "divided by that class's MCMC floor. The haematopoietic-progenitor joint component uses the progenitor "
        "floor (PREREG s3):", L.sBody))
    story.append(Paragraph("A_mapped = H(&#946;&#772;) / H_min(class),&nbsp;&nbsp; &#946;&#772; = mean &#946; over "
                           "the class identity loci", L.sSect2))
    story.append(Paragraph("<b>3. The two corrections that make it absolute.</b> The donor's decade and the "
        "laboratory's own zero are subtracted, which is what turns a relative reading into one that can be compared "
        "against a fixed line:", L.sBody))
    story.append(Paragraph("A_abs = A_mapped - c(decade) - z_lab", L.sSect2))
    story.append(Paragraph(f"c(decade) comes from <font name='Courier'>reference_age_curve_v1.json</font> "
        f"(PROC-PANEL-03: per-decade median of mapped immune identity-loci A about the grand median, decades with "
        f"n &gt;= 30 only, n = {cmeta.get('n', '1,379')}). z_lab is the laboratory's own median on its 40-array "
        f"healthy panel (<font name='Courier'>lab_zero.py</font>). <b>With no laboratory zero the chain refuses:</b> "
        f"A_mapped is still returned, A_abs and the placement are None, and the reading is marked not reportable.",
        L.sBodySm))

    story.append(SP(0.05))
    story.append(Paragraph("<b>4. The healthy line and the band it is read against</b> "
        f"(<font name='Courier'>identity_band_v3.json</font>, four zeroed healthy cohorts, n = {meta.get('n','')}):",
        L.sBody))
    story.append(tbl([["quantity", "value", "what it is"],
        ["p10", f"{pooled.get('p10','')}", "lower edge of the healthy band"],
        ["p50", f"{pooled.get('p50','')}", "the healthy line - A_abs = 1.000 by construction"],
        ["p90", f"{pooled.get('p90','')}", "upper edge"],
        ["width", f"{pooled.get('width','')}", "p90 - p10, one band width"],
        ["sigma", f"{sigma:.5f}" if sigma else "-", "(p90 - p10) / (2 x 1.2816) - the band read as a normal spread"]],
        [0.16, 0.16, 0.68], fs=7.5))
    story.append(Paragraph("Placement is BELOW_BAND below p10, ABOVE_BAND above p90, IN_BAND between; the tier word "
        "comes from <font name='Courier'>cpg_tiers.tier_of(A_abs, H_min)</font> (Stage 7, PROC-TIER-01).", L.sBodySm))

    story.append(SP(0.05))
    story.append(Paragraph("<b>5. The four laboratories the line is built from</b> - each contributes its own zero, "
        "and its own measured false-alarm rate is what a departure is judged against:", L.sBody))
    rows = [["laboratory", "n", "z_lab (full cohort)", "false alarm at p95", "at p99", "chip median SD"]]
    for k in sorted(coh):
        v = coh[k]
        rows.append([k.replace("_", " "), str(v.get("n", "")), f"{v.get('z_lab_full_cohort', '')}",
                     f"{v.get('tail_p95', '')}", f"{v.get('tail_p99', '')}", f"{v.get('chip_median_sd', '')}"])
    story.append(tbl(rows, [0.28, 0.10, 0.18, 0.16, 0.12, 0.16], fs=7.2))
    tr = meta.get("four_lab_tail_range_p95", [])
    if tr:
        story.append(Paragraph(f"<b>The false-alarm rate is a measured property of the laboratory, not a constant.</b> "
            f"Across the four it ranges from {tr[0]} to {tr[1]} at p95 - so a departure just past the threshold means "
            f"something different in Karolinska ({coh.get('GSE42861_Karolinska',{}).get('tail_p95','')}) than in UCLA "
            f"({coh.get('GSE111629_UCLA',{}).get('tail_p95','')}), and the chain prints the sample's own laboratory "
            f"rate beside the distance rather than a single number. Centring on the chip recovers a good part of it "
            f"(the <font name='Courier'>tail_p95_if_chip_centred</font> column of the same file), which is why chip "
            f"position is on the roadmap as a covariate.", L.sBodySm))

    story.append(SP(0.05))
    story.append(Paragraph("<b>6. The departure</b> (<font name='Courier'>stage_5_mahalanobis</font>, PROC-MAHA-01; "
        "SOP s47-51). Each assessable component is standardised against the healthy line, and the components are "
        "combined as a distance:", L.sBody))
    story.append(Paragraph("z = (A_abs - 1.000) / &#963;,&nbsp;&nbsp; D = &#8730;(&#931; z&#178;) over the n "
                           "assessable components", L.sSect2))
    story.append(Paragraph("Thresholds are the chi-square quantiles for n degrees of freedom - D<sub>95</sub> = "
        "&#8730;&#967;&#178;(0.95, n), D<sub>99</sub> = &#8730;&#967;&#178;(0.99, n) - so the bar rises with the "
        "number of axes and a distance cannot be inflated by adding components. A component with no commissioned "
        "band is not assessable: the haematopoietic-progenitor joint and every non-blood class today. <b>On whole "
        "blood n = 1</b>, so the distance is |z| on the immune axis and the report says so in those words. The "
        "chain also prints <font name='Courier'>band_widths_from_line</font> = (A_abs - 1.000) / (p90 - p10), "
        "because a reader wants the departure in units of the healthy spread as well as in sigmas.", L.sBodySm))

    story.append(SP(0.05))
    story.append(Paragraph("<b>7. What the same four laboratories forbid: cellular age in years</b> "
        f"(<font name='Courier'>stage_6_cellular_age</font>, PROC-AGE-01). The healthy immune gauge rises "
        f"{age01.get('slope_per_yr', 0)*1000:.2f} mA per year against a within-laboratory spread of "
        f"{age01.get('sd_A', 0):.4f}, so inverting the curve resolves age to about "
        f"{round(age01.get('resolution_yr', 0))} years for one array: {age01.get('A1_within10', 0)*100:.1f} per cent "
        f"of healthy donors land within ten years of their own age (Spearman rho = {age01.get('A3_rho', 0):.3f} on "
        f"n = {meta.get('n','')}). The trajectory is real and reproduced - it IS the reference curve above - but it "
        f"is a population measurement, and the stage reports the resolution rather than an age.", L.sBodySm))
    story.append(Paragraph("<b>8. The second opinion</b> (<font name='Courier'>stage_2b_second_opinion</font>): the "
        "needlet ILC solver runs beside the constrained NNLS, both folded to classes, and the chain reports "
        "agreement when the class-level L1 difference is at or below 0.10. Cell-level disagreement inside one "
        "lineage is expected and is not scored.", L.sBodySm))


# ── the table of contents, generated against the rendered document ───────────────────────────────────────────────
TOC_PAGES_FILE = os.path.join(HERE, "toc_pages.json")


def toc_entries(L, D):
    """(group, exact heading as it renders, probe text) - the probe is what is searched for in the built PDF."""
    e = [("Front matter", "What's new in this issue", "WHAT'S NEW"),
         ("Front matter", "What this paper is not", "WHAT THIS PAPER IS NOT"),
         ("Front matter", "What this document claims, and what it does not", "WHAT THIS DOCUMENT CLAIMS, AND WHAT IT DOES NOT"),
         ("Front matter", "Prior art - the door into the conversation", "PRIOR ART"),
         ("Part I - the instrument", "s1 Reconciliation: Issue 002 to repository HEAD", "RECONCILIATION"),
         ("Part I - the instrument", "s2 The IAM Atlas: 115 cell types, eight classes", "THE IAM ATLAS"),
         ("Part I - the instrument", "s3 Two instruments and the presence rule", "TWO INSTRUMENTS"),
         ("Part I - the instrument", "s4 Framework from Issue 002: ranking, substrates, saturation", "FIVE-SUBSTRATE FRAMEWORK"),
         ("Part I - the instrument", "s5 The physics: Landauer, the Mahaffey number, the reference, the gauge", "SECTION 5 THE PHYSICS"),
         ("Part I - the instrument", "s5A Where the tools come from", "SECTION 5A WHERE THE TOOLS COME FROM"),
         ]
    for c in L.CARDS:
        e.append(("The eight architecture-class cards",
                  f"#{c['pos']} {c['name']}", f"#{c['pos']} · {c['name'].upper()}"))
    e += [("Part III - the chain in depth", "III.1 The chain, stage by stage", "The chain, stage by stage"),
          ("Part III - the chain in depth", "III.2 The atlas, and the cells it can speak about", "The atlas, and the cells it can speak about"),
          ("Part III - the chain in depth", "III.3 The cosmology toolkit, tool by tool", "The cosmology toolkit, tool by tool"),
          ("Part III - the chain in depth", "III.4 What the chain refuses, and its cosmology twin", "What the chain refuses, and the cosmology twin"),
          ("Part III - the chain in depth", "III.5 The engine, exactly - every formula the chain computes", "The engine, exactly - every formula the chain computes"),
          ("Record and appendices", "s7 Substrate characterisation: the substrate x class grid", "SUBSTRATE CHARACTERIZATION"),
          ("Record and appendices", "s8 Procedures", "SECTION 8 PROCEDURES"),
          ("Record and appendices", "s9 Operating rules", "OPERATING RULES"),
          ("Record and appendices", "s10 Falsification record", "FALSIFICATION RECORD"),
          ("Record and appendices", "s11 Engine map: every stage, and whether this issue covers it", "ENGINE MAP"),
          ("Record and appendices", "s12 For the clinician: the instrument in plain terms", "FOR THE CLINICIAN"),
          ("Record and appendices", "Appendix V - the validation index", "APPENDIX V"),
          ("Record and appendices", "Appendix VI - the CMB to methylome translation map", "APPENDIX VI"),
          ("Record and appendices", "Appendix VII - the completion sprint, scored", "APPENDIX VII"),
          ("Record and appendices", "Future goals - what is worth the effort, in order", "FUTURE GOALS — WHAT IS WORTH THE EFFORT"),
          ("Record and appendices", "Glossary", "GLOSSARY"),
          ]
    return e


def render_toc(story, L, D, tbl, SP, PageBreak, Paragraph):
    pages = {}
    if os.path.exists(TOC_PAGES_FILE):
        pages = json.load(open(TOC_PAGES_FILE, encoding="utf-8"))
    story.append(Paragraph("CONTENTS", L.sSect))
    story.append(Paragraph("Every chapter by its exact name, with the page it starts on. Generated against the "
                           "rendered document, so a page number here is the page the section is on.", L.sSub))
    last = None
    for group, label, probe in toc_entries(L, D):
        if group != last:
            story.append(SP(0.045)); story.append(Paragraph(group.upper(), L.sSect2)); last = group
        pg = pages.get(probe)
        dots = "&nbsp;.&nbsp;" * 2
        story.append(Paragraph(
            f"{label}{dots}<b>{pg}</b>" if pg else f"{label}{dots}<font color='#666'>-</font>", L.sBodySm))
    if not pages:
        story.append(Paragraph("(page numbers are filled in by the second pass: build once, then again)", L.sMut))


def collect_toc_pages(pdf_path, L, D):
    """Search the built PDF for each probe and write toc_pages.json for the next build."""
    import pypdfium2 as pdfium
    d = pdfium.PdfDocument(pdf_path)
    texts = [re.sub(r"\s+", " ", d[i].get_textpage().get_text_range()) for i in range(len(d))]
    # the contents page lists every chapter name, so it matches every probe: exclude it by its own subtitle
    texts = ["" if "Generated against the rendered document" in t else t for t in texts]
    out, missing = {}, []
    for _g, _label, probe in toc_entries(L, D):
        hit = next((i + 1 for i, t in enumerate(texts) if probe in t), None)
        if hit: out[probe] = hit
        else: missing.append(probe)
    json.dump(out, open(TOC_PAGES_FILE, "w", encoding="utf-8"), indent=1)
    return out, missing
