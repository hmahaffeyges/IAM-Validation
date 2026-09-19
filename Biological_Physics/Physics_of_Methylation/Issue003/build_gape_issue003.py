#!/usr/bin/env python3
"""build_gape_issue003.py — GAPE Issue 003 (September 2026).

Composed from: gape002_lib.py (every Issue 002 rendering primitive, card, and section, extracted
verbatim) + data003.py (every constant loaded from the runtime files at repo HEAD, plus the dated
runs of 2026-09-19) + the new sections written here.

Run:  CPG_TRIAL=<path to CPG_TRIAL_CODE> python build_gape_issue003.py [out.pdf]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gape002_lib as L
import data003 as D
from gape002_lib import (Paragraph, Table, Spacer, PageBreak, HRFlowable, KeepTogether,
                         S, P, PH, Pb, Ps, SP, HR, tbl_style, PW, inch, colors, letter,
                         SimpleDocTemplate, sTitle, sSub, sSect, sSect2, sLabel, sBody, sBodySm,
                         sMut, sDisc, sCode, LAV, LAV_M, LAV_D, MUTED, MUTED2, TEXT, WHITE, TEAL,
                         GREEN, AMBER, RED_C, ORANGE, SURF, SURF2, BORDER, BG, W, H, CLS_COLS, FillRect)

ISSUE = "Issue 003"; DATE = "September 2026"

def make_canvas(canvas, doc):
    canvas.saveState(); canvas.setFillColor(BG); canvas.rect(0, 0, W, H, fill=1, stroke=0)
    canvas.setStrokeColor(LAV_D); canvas.setLineWidth(0.5)
    canvas.line(0.5*inch, 0.45*inch, W - 0.5*inch, 0.45*inch)
    canvas.setFillColor(MUTED2); canvas.setFont('Helvetica', 7)
    canvas.drawString(0.5*inch, 0.30*inch, f'IAMPerformance  ·  GAPE {ISSUE}  ·  DRAFT')
    canvas.drawCentredString(W/2, 0.30*inch, 'Patents pending 64/012,720 and 64/014,568')
    canvas.drawRightString(W - 0.5*inch, 0.30*inch, f'Page {canvas.getPageNumber()}')
    canvas.restoreState()

def opener(story, num, title, blurb):
    story.append(PageBreak())
    story.append(FillRect(PW, 0.80*inch, colors.HexColor('#0a081a'), r=5)); story.append(Spacer(1, -0.80*inch)); story.append(Spacer(1, 14))
    story.append(Paragraph(num, S('opL', fontName='Helvetica-Bold', fontSize=9, textColor=LAV, leading=12)))
    story.append(Paragraph(title, S('opT', fontName='Helvetica-Bold', fontSize=22, textColor=WHITE, leading=26)))
    story.append(Spacer(1, 10)); story.append(Paragraph(blurb, sBody)); story.append(SP(0.08))

def tbl(rows, widths, fs=7.5, head=True):
    data = [[PH(c) for c in rows[0]]] + [[P(c) if not isinstance(c, Paragraph) else c for c in r] for r in rows[1:]] if head else [[P(c) for c in r] for r in rows]
    t = Table(data, colWidths=[PW*w for w in widths], repeatRows=1 if head else 0); t.setStyle(tbl_style(fs)); return t

def status_p(s):
    col = {"CONF": GREEN, "MEAS": TEAL, "COMP": AMBER, "ABS": MUTED2, "SAT": MUTED2,
           "STRUCT": ORANGE, "DATA": LAV_M, "NONE": RED_C, "OPEN": ORANGE}
    c = next((v for k, v in col.items() if s.startswith(k)), TEXT)
    return Paragraph(f'<font color="{c.hexval()}"><b>{s}</b></font>', L._sTD)

# ═══════════════════════════════════════════════════════════════════════════════
def cover(story):
    story.append(Paragraph('IAMPerformance', sTitle))
    story.append(Paragraph('Physics-Derived Cellular Fidelity Intelligence', sSub)); story.append(SP(0.06))
    story.append(HRFlowable(width='100%', thickness=1, color=LAV, spaceAfter=5))
    story.append(Paragraph(f'<b>{ISSUE}  ·  {DATE}</b>  ·  The Healthy Range of the Cellular Write Process — '
        'Eight Architecture Classes, Forty H_min Values, One Atlas of 115 Cell Types, and the First Written '
        'Specification of the Running Engine', S('cv', fontSize=10.5, textColor=TEXT, leading=15)))
    story.append(SP(0.08))
    story.append(Paragraph('Heath W. Mahaffey  ·  IAMPerformance  ·  Entiat, Washington', sMut))
    story.append(Paragraph('Repository: github.com/hmahaffeyges/IAM-Validation  ·  engine at commit 66f37fe (2026-07-03)', sMut))
    story.append(Paragraph('Prepared with Claude Science, 2026-09-19. <b>DRAFT.</b> Every number is loaded from the runtime '
        'files or from a dated run named in the text. Rows marked OPEN are unresolved and are printed as such.', sMut))
    story.append(SP(0.12))
    story.append(Paragraph("WHAT'S NEW IN ISSUE 003", sSect))
    news = [
     ("The claim, restated.", "This is not a detection tool and not a treatment. Cells compute and write to a two-dimensional "
      "surface as a semiconductor or a quantum processor does; the physics of that write process has a healthy operating range "
      "per architecture class, calculated once and frozen (2026-04-06); departures from that range correlate with the hypo- and "
      "hyper-methylation patterns seen in disease <i>and</i> with normal variation in healthy individuals. Disease is corroboration, not product."),
     ("The IAM Atlas.", f"IAMAtlasREBUILD: {D.N_CPGS:,} CpGs, {D.N_CELLTYPES} cell types mapped to the eight classes, built {D.ATLAS_BUILD}. "
      "Every class card now lists its cell types. The atlas did not exist when Issue 002 was written."),
     ("Two instruments, stated.", "The GAUGE (entropy of one mean β over per-class identity loci) and the SEPARATION statistic "
      "(mean per-CpG entropy over per-cell discriminative markers) are different instruments answering different questions. Issue 002 had one A-score."),
     ("Healthy is a band, not a line.", "A = 1.0 is the architectural commitment line. Healthy reads on the age-matched p10–p90 band "
      "of age_reference_matrix.json. The per-class input offset explored in July is retired."),
     ("Presence before score.", "A class's A-score is a reading only if the deconvolver finds the class in the sample "
      "(DETECT_FLOOR). Scoring absent classes produced spurious BREACH calls in 4/4 healthy plasma samples on 2026-09-19; the rule is now printed."),
     ("Where the tools come from (§5A).", "The Mahaffey number (20.94 = dG_ATP/RT, the cell's ATP budget over k_B T; the name n_bio retired); the forty MCMC-confirmed floors and their forty ceilings; the atlas as a posterior; the four Planck tools borrowed for the methylome (brightness sky, masking, matched filter, cross-method deconvolution — the last one cut in July and recorded as such); and why this is not a cohort method."),
     ("Reconciliation table (§1).", "Every constant and rule that changed between April and repo HEAD, with the commit that changed it. Nothing silently updated."),
     ("Sections 7–8 replaced.", "The detection-trajectory and deployment-readiness sections are replaced by Substrate Characterization "
      "(§7) and Procedures (§8): what each substrate carries, what is verified, what is open, and the exact steps that reproduce each verified number."),
     ("Stage 1 executed from raw IDATs - the chain is closed.", "All eleven test IDAT pairs through the repo\'s own calibrator reproduce the cached betas bit-for-bit (11/11, max diff 0.000000, both array types). Raw IDAT -> beta -> class fractions -> sealed anchor: every link now reproduced from the repository on a stranger\'s machine (PROC-CAL-01)."),
     ("The end-to-end simulation caught the production gauge reading the wrong CpG set.", "A synthetic healthy patient - a pure mixture of healthy Atlas posteriors - reads BREACH through the conductor, because Stage B computes H(beta_mean) over the bimodal class marker union, not the identity loci its docstring and the SOP name; the age band was compiled the same way, so real blood never showed it. The identity-loci gauge reads the synthetic at 0.99 and real adenoma at 1.10 but has no band yet. Phase 1 becomes a prerequisite (PROC-N7-01, RECON A4)."),
     ("The sealed breast anchor reproduced from raw GEO.", "GSE51032, 460 samples, 115 cell types: mean-of-H over the v0_2 discriminative markers returns the sealed CSV at r = 1.00000, max diff 0.00004, on a machine that had never seen the project. H(beta_mean) on the same markers does not (r = 0.45). The seal rests on the repo-HEAD markers, not the chrX-removed copy (PROC-ANCHOR-01)."),
     ("The aggregation conflict between the current SOP and the running code, measured.", "SOP v1.4.0 says never H(β_mean); the chain at HEAD computes H(β_mean); the age band is compiled as H(β_mean). On whole blood with the class present the two aggregations are rank-identical with a +0.029 offset; on mixed or absent-class panels H(β_mean) inflates as §105 says. Ruled in §1.5: one aggregation per surface, fixed by the reference each was built with (RECON A3) - and the analyst's own first reading of the tissue values is retracted in §10."),
     ("First real-data ground truth for the deconvolver.", "Nine known genomic-DNA mixes (Moss 2018): the terminal class recovers a neuronal spike (r = 0.945); secretory and cycling do not recover hepatocyte or colon spikes. Recorded as PASS/FAIL in §8 and §10, not smoothed over."),
     ("Operating rules and falsification record (§9–10).", "The lessons that govern how the instrument may be read, dated and sourced; and every outcome that went against the framework, kept on the record."),
    ]
    for h, b in news:
        story.append(Paragraph(f'<b>{h}</b> {b}', sBodySm))
    story.append(SP(0.1)); story.append(HR())
    story.append(Paragraph('WHAT THIS ISSUE DOES NOT YET COVER', sLabel))
    story.append(Paragraph('Of the twelve chain stages at HEAD (0, 1, 2, 3, 4, 4.5, 4.6, 5, 6, 7, 8, 9) this issue documents three — Stages 2, 4 and 7 — and the atlas. The other nine (0, 1, 3 [not built], 4.5, 4.6, 5, 6, 8, 9) are named in §11 with their files and left for the next issue. '
        'The disease-signature matrix, the crown-jewel wall, the second chain, the directional AD detector, cellular age and the report builders are therefore not described here.', sBodySm))
    story.append(SP(0.06))
    story.append(Paragraph('WHAT THIS PAPER IS NOT', sLabel))
    story.append(Paragraph('Not clinical validation. Not a diagnostic. Nothing here should inform patient care. The instrument is at a research stage; '
        'any clinical use requires prospective validation, regulatory review and qualified clinical oversight. Public retrospective cohorts only; '
        'sample sizes in the new runs are small and are printed beside every number.', sBodySm))

def toc(story):
    story.append(PageBreak()); story.append(Paragraph('TABLE OF CONTENTS', sSect))
    rows = [("Front", "Cover · What's new · What this paper is not · Contents"),
            ("§1", "Reconciliation — Issue 002 → repo HEAD, every changed constant and rule"),
            ("§2", "The IAM Atlas — 115 cell types, eight classes, provenance, identity loci, markers"),
            ("§3", "Two Instruments and the Presence Rule"),
            ("§4", "Framework (from Issue 002) — global ranking, five substrates, MCMC↔bootstrap, saturation"),
            ("Cards", "Eight architecture-class cards (from Issue 002) — each with its atlas addendum: cell types, identity loci, age band"),
            ("§1.7", "NEW — The reporting rule (extends the §3.2 presence gate): a class gauge is put on the report only where the class is determined and present; whole blood reports immune plus one haematopoietic-progenitor component. stem_adult has carried no finding in nine VALs"),
            ("§1.6", "NEW — What the cosmology tools found that cohorts could not: the standing evidence ledger (nine rows to date, N7 first) that pre-empts the circularity objection"),
            ("App. VI–IX", "NEW — the CMB→methylome translation map (79 rows, scored: what got built, what was cut, what was refused); the completion sprint scored, with the lesson that the bones must be trusted first; Future Goals — the CMB items worth the effort, in gated order; the Part II outline"),
            ("Glossary", "NEW — CMB and Chain Terms (Cosmic Methylome Background, brilliance, HEALPix, component separation, matched filter, Mahalanobis Option A, the eight nulls, synthetic patients, PREREG/seal, Jensen gap, flatness ...) and Chain Links: one line per runtime file, tagged FLOOR / RULER / BAND / CODE / DATA"),
            ("App. V", "NEW — Validation index: all 103 VAL identifiers in the repository with title, date, cohort, stated decision, record completeness and path"),
            ("§5A", "NEW — Where the tools come from: Mahaffey number 20.94, forty MCMC floors, the atlas posterior, the CMB toolkit, not a cohort method"),
            ("§5", "Physics & Methodology (Issue 002 Section 2) — H_min derivation, substrates, saturation, inversions, C1/C2/C3"),
            ("§6", "Evidence, Baselines, Scenarios, Predictions (Issue 002 Sections 3–6)"),
            ("§7", "NEW — Substrate Characterization: the substrate × class grid"),
            ("§8", "NEW — Procedures: PROC entries, one per verified cell of the grid"),
            ("§9", "NEW — Operating Rules: the lessons as rules, dated and sourced"),
            ("§10", "NEW — Falsification Record"),
            ("§11", "NEW — Engine Map: every stage of the running chain and whether this issue covers it"),
            ("§12", "NEW — For the clinician: the instrument in plain terms"),
            ("Back", "Master predictions · Data sources · Glossary · A final note")]
    story.append(tbl([("", "")] + rows, [0.10, 0.90], head=False))

# ═══════════════════════════════════════════════════════════════════════════════
def sec1_recon(story):
    opener(story, 'SECTION 1', 'RECONCILIATION — ISSUE 002 → REPO HEAD',
        'Issue 002 was written in April 2026, before the atlas and before the engine was rebuilt (commits 2026-06-25 to 2026-07-03). '
        'This table states, for every constant and rule that differs, what 002 said, what the running engine says, why the engine is '
        'current, and where to check. Rows marked <b>OPEN</b> are real inconsistencies at HEAD that this document does not resolve. '
        'The chain-of-custody SOP v1.3.x predates the rebuild; commit d7b0e1f explicitly supersedes its §41. Until this section, the '
        'commits were the only specification of the running engine.')
    rows = [("#", "Item", "Issue 002", "Repo HEAD (66f37fe)", "Why HEAD is current", "Evidence")]
    for r in D.RECON:
        rows.append(tuple(Ps(x) if i in (4, 5) else P(x) for i, x in enumerate(r)))
    story.append(tbl(rows, [0.05, 0.14, 0.17, 0.26, 0.20, 0.18], fs=7))
    story.append(SP(0.08))
    story.append(Paragraph('Reading the table', sLabel))
    story.append(Paragraph('The forty H_min values (row H1) are unchanged: the floors were frozen on 2026-04-06 and the engine at HEAD carries them '
        'byte-for-byte. What changed is everything <i>around</i> the floors — where healthy sits (A1), what a score means when the class is not '
        'in the sample (D1), which floor a substrate is read against (S1), and the existence of a second instrument (I1). Five rows are OPEN: two tier '
        'vocabularies (T3), two presence floors (D2), an atlas count that disagrees with the SOP (K2), two files with one name (M1), and the absence of '
        'any post-atlas SOP (SOP). Each is a question a reviewer will ask; each is answered here by being stated rather than hidden.', sBodySm))

# ═══════════════════════════════════════════════════════════════════════════════
def sec1b_rulings(story):
    story.append(Paragraph('1.5  Rulings on the two open decisions', sSect2))
    story.append(Paragraph('The author delegated the two decisions the reconciliation left open. Both are recorded here with their reasoning so they can be overruled with the same specificity.', sBodySm))
    for R in (D.RULING_A3, D.RULING_M1B):
        story.append(Paragraph(R["title"], sLabel))
        story.append(Paragraph(f'<b>Rule.</b> {R["rule"]}', sBodySm))
        for k in ("gauge","separation","guards","sop","why","reseal"):
            if k in R: story.append(Paragraph(f'<b>{k.capitalize()}.</b> {R[k]}', sBodySm))
        story.append(Paragraph(R["status"], sMut)); story.append(SP(0.08))

def sec2_atlas(story):
    opener(story, 'SECTION 2', 'THE IAM ATLAS',
        f'IAMAtlasREBUILD — {D.N_CPGS:,} CpGs × {D.N_CELLTYPES} cell types, each mapped to one of the eight architecture classes. '
        f'Built {D.ATLAS_BUILD} from per-class MCMC (batch 5000, immune 1500), duplicate reconciliation, and merge; the predecessor '
        'IAMAtlas.csv.xz was retired for a collapse/flatness bug. It ships compressed in the repository (98 MB .xz → 577 MB .csv) and is the '
        'deconvolver reference, the source of the gauge identity loci, and the source of the separation markers. H_min values in the atlas are the frozen 2026-04-06 set.')
    story.append(Paragraph('2.1  Cell types by class', sSect2))
    rows = [("Class", "n cell types", "Identity loci (gauge)", "H_min methyl", "Ceiling 1/H_min", "Cell types (atlas names)")]
    for c in D.CLASS_ORDER:
        cells = D.CELLS_BY_CLASS[c]; idl = D.IDENTITY[c]
        rows.append((Pb(c), P(str(len(cells))), P(f"{idl['n_loci']:,}"), P(f"{idl['H_min']}"), P(f"{D.ceiling(c):.4f}"), Ps(", ".join(cells))))
    story.append(tbl(rows, [0.10, 0.07, 0.10, 0.08, 0.08, 0.57], fs=6.8))
    story.append(SP(0.06))
    story.append(Paragraph('Two structural facts a reader should take from this table. <b>Stromal</b> has 2,294 identity loci against 29,000–57,000 for every other '
        'class and only five cell types; on colorectal tissue it returned exactly 0.0000 in 4/4 samples with 1,441 of 7,090 class markers matched — a coverage '
        'gap, not a biological absence (§7). <b>Immune</b> holds 51 of the 115 cell types; a pooled immune gauge therefore requires a single cell type to '
        'contribute roughly eight times an equal share before its own β shift moves the class A by one tier (§3.3).', sBodySm))
    story.append(Paragraph('2.2  Identity loci (the gauge panel)', sSect2))
    story.append(Paragraph(f'Derived from the frozen atlas by the criterion <font face="Courier">{D.IDENTITY_PROV.get("criterion","|class_mean − H_min_beta| ≤ 0.05")}</font>: '
        'CpGs where a healthy cell of the class sits at its characteristic β. Substrate: methyl. The gauge takes ONE mean β over these loci and returns '
        'H(β<sub>mean</sub>)/H_min(class, substrate). These are not discriminative markers and must never be swapped for them (§3).', sBodySm))
    story.append(Paragraph('2.3  Discriminative markers (the separation panel)', sSect2))
    sx = D.MARKERS_META.get("_sex_marker_removal", {})
    story.append(Paragraph(f'iamatlas_celltype_markers_v0_2.json — one-vs-rest markers per cell type, 100 per cell in the SOP specification. '
        f'{sx.get("removed_chrX","131")} chrX markers were removed on {sx.get("date","2026-06-11")}: <i>"{sx.get("reason","")}"</i>. '
        'Two files with this name exist at different sizes (RECON M1); the one in the trial bundle is 200,311 B, the one at repo HEAD 237,545 B. Which is current must be stated before either is cited.', sBodySm))
    story.append(Paragraph('2.4  The deconvolver', sSect2))
    story.append(Paragraph('walther_iam_deconvolver.py: non-negative least squares over the atlas, one streaming pass selecting class markers (7,114 on the 2026-09-19 runs) '
        'and cell-type markers, returning class fractions, cell-type fractions and a residual MAE. NILC, the second deconvolver of SOP §9, was cut on 2026-07-02 (commit c1be0c3). '
        'Validation: N7 on synthetic Dirichlet mixtures, MAE 0.0076–0.0093 across eight classes; conformance against the project answer key on three real EPIC tissue samples, '
        'MAE 0.0004/0.0002/0.0002 (PROC-DECON-01, §8). It has not been validated against real tissue of known composition. LESSON-DECONV-01 (§9) governs how sparse cell-level output is read.', sBodySm))

# ═══════════════════════════════════════════════════════════════════════════════
def sec3_instruments(story):
    opener(story, 'SECTION 3', 'TWO INSTRUMENTS AND THE PRESENCE RULE',
        'Issue 002 had one A-score. The running engine has two statistics that share the template A = H/H_min and nothing else - and, as §3.3 shows, they measure different things. '
        'Confusing them produced the all-BREACH bug of 2026-06-11. Reading either without the presence rule produced the spurious plasma breaches of 2026-09-19.')
    rows = [("", "GAUGE (the call)", "SEPARATION (disease matching)"),
            ("CpG set", "per-class IDENTITY loci", "per-cell one-vs-rest DISCRIMINATIVE markers"),
            ("Statistic", "A = H(β<sub>mean</sub>)/H_min — entropy of ONE mean β over the panel", "mean_i H(β_i)/H_min — mean of per-CpG entropies"),
            ("Resolution", "class-level", "cell-level"),
            ("Tier of trust", "RELIABLE", "INDICATIVE (v0.1 atlas; LESSON-DECONV-01)"),
            ("Reads against", "age-matched p10–p90 band (placement) + severity ladder (tier)", "disease signature matrix v1.13 (direction only)"),
            ("Answers", "where does this class sit on the ruler", "how separable is this patient from healthy, and in which direction"),
            ("Must never receive", "discriminative markers (bimodal → β<sub>mean</sub> → 0.5 → false ceiling)", "identity loci")]
    story.append(tbl(rows, [0.16, 0.42, 0.42]))
    story.append(SP(0.08)); story.append(Paragraph('3.1  Placement and severity are two axes', sSect2))
    story.append(Paragraph('PLACEMENT: A below p10 of the age-matched band → BELOW BAND; within → IN BAND; above p90 → ABOVE BAND. '
        'SEVERITY (gauge ladder, unchanged from Issue 002): NORMAL &lt; 1.01 ≤ MARGINAL &lt; 1.05 ≤ DETECTABLE &lt; 1.07 ≤ URGENT &lt; 1.10 ≤ FLOOR BREACH, ceiling 1/H_min, '
        'saturation flagged within 0.005 of the ceiling. A far below the age cohort is INVERSION — a finding (seminoma, senescence, aged HSC), not an error. '
        'The Stage 7 customer vocabulary (tier_breakpoints.json v1.3) uses SUPPRESSED/NORMAL/ELEVATED/SIGNIFICANTLY ELEVATED/BREACH on the same breakpoints minus the 1.05 split; RECON T3.', sBodySm))
    story.append(Paragraph('3.2  The presence rule', sSect2))
    story.append(Paragraph('<font face="Courier">cpg_conductor.py</font> (2026-07): <i>"per-cell A always paired with its deconvolved fraction so presence and score are read together. '
        'DETECT_FLOOR = 0.01 — a cell below this is treated as absent (fraction sets presence)."</i> An A-score computed over the identity loci of a class that is not in the sample is '
        'entropy of noise, not a reading. On 2026-09-19, before this rule was applied, stromal read BREACH (A 1.103–1.119) in 4/4 healthy plasma samples at fraction 0.0000; after it, '
        'those calls vanished and two of four healthy samples were clean on every present class. The README for the Mahalanobis adjudicator uses 3%; the two floors give different '
        'answers for terminal in healthy plasma (present at 1.5–2.7%). One floor must be chosen (RECON D2).', sBodySm))
    story.append(Paragraph('3.3  The aggregation question, measured (PROC-FORMULA-01)', sSect2))
    story.append(Paragraph('The current SOP (v1.4.0, 2026-06-30, §105 LESSON-ASCORE-02) states the A-score is the mean of per-CpG entropies and is <i>never</i> the entropy of the mean β. '
        'The chain wired at HEAD one day later (d7b0e1f) computes exactly H(β<sub>mean</sub>) over identity loci, and SOP v1.3.3 - same date as v1.4.0 - says so too. The most recent SOP and the running code disagree. '
        'Rather than pick a docstring, all four aggregation × loci combinations were computed on the same eleven Stage-1-calibrated samples:', sBodySm))
    rows=[("sample","", "identity: H(β_mean)","identity: mean H","discrim.: H(β_mean)","discrim.: mean H")]
    for gsm,lab in D.FORMULA_LABELS.items():
        r=D.FORMULA_2X2.get(f"immune|{gsm}")
        if r: rows.append((gsm,lab,f"{r['id_Hm']:.4f}",f"{r['id_mH']:.4f}",f"{r['disc_Hm']:.4f}",f"{r['disc_mH']:.4f}"))
    story.append(Paragraph('Immune class, H_min 0.838889, ceiling 1.1921. Identity loci n≈32,000 (450K) / 20,000–30,000 (EPIC tissue); discriminative class-union n≈2,400 / 1,800–2,700.', sMut))
    story.append(tbl(rows,[0.14,0.20,0.165,0.165,0.165,0.165],fs=7))
    for k,v in D.FORMULA_FINDINGS: story.append(Paragraph(f'<b>{k}.</b> {v}', sBodySm))
    story.append(Paragraph(f'<b>Verdict.</b> {D.FORMULA_VERDICT}', sBodySm))
    story.append(Paragraph('3.4  What a pooled class gauge can and cannot resolve', sSect2))
    story.append(Paragraph('The secretory class pools 18 cell types (Breast, Prostate, Hepatocytes, Pancreatic β, Thyroid …) into one panel and one mean. It cannot distinguish organs by construction. '
        'Arithmetic on the frozen H_min: for a single cell type shifting its own β by −0.20 to move the pooled secretory A by one tier width (+0.05), that cell type must contribute '
        '16.1% of the DNA in the sample — 2.9× an equal share. For immune (51 cell types) the figure is 15.7%, 8× an equal share. Organ resolution is the separation instrument\'s job, '
        'and on 2026-09-19 that instrument routed all shed epithelium to gastric references (§7.3). <b>No organ-level claim is supported today by either channel.</b> '
        'The gauge answers "did the compartment depart"; that is the claim this document makes.', sBodySm))

# ═══════════════════════════════════════════════════════════════════════════════
def card_addendum(story, key):
    cells = D.CELLS_BY_CLASS[key]; idl = D.IDENTITY[key]; hm = D.H_MIN_TABLE[key]
    story.append(KeepTogether([
        Paragraph(f'ISSUE 003 ADDENDUM — {key.upper()} IN THE ATLAS', sLabel),
        Paragraph(f'<b>{len(cells)} cell types:</b> {", ".join(cells)}', sBodySm),
        Paragraph(f'<b>Identity loci:</b> {idl["n_loci"]:,} (methyl)  ·  <b>H_min by substrate:</b> methyl {hm[0]:.6f} · nucl {hm[1]:.6f} · fuzz {hm[2]:.6f} · wps {hm[3]:.6f} · frag {hm[4]:.6f}  ·  '
                  f'<b>methyl ceiling</b> {D.ceiling(key):.4f}{"  — cannot reach BREACH on methylation alone" if D.ceiling(key) < 1.10 else ""}', sBodySm)]))
    rows = [("Age (midpoint)", "n", "A mean", "p10", "p90", "β mean", "Source")]
    for e in D.AGE_REF[key]:
        rows.append((str(e["age_midpoint"]), str(e["n_samples"]), f'{e["A_mean"]:.4f}', f'{e["A_p10"]:.4f}', f'{e["A_p90"]:.4f}', f'{e["beta_mean"]:.3f}', e["source_citation"]))
    story.append(Paragraph('Healthy age-matched band (age_reference_matrix.json). A = 1.0 is the commitment line; healthy reads on this band.', sMut))
    story.append(tbl(rows, [0.14, 0.07, 0.12, 0.12, 0.12, 0.12, 0.31], fs=7))

# ═══════════════════════════════════════════════════════════════════════════════
def sec7_substrates(story):
    opener(story, 'SECTION 7', 'SUBSTRATE CHARACTERIZATION',
        'Replaces Issue 002 Sections 7–8. Every (substrate, class) pair has its own floor and its own healthy range. This section is the grid: what each substrate '
        'carries, which cells of the grid have been measured with the canonical files, which are absent by biology, which are structurally unreadable, and which are open. '
        'Each open cell names the experiment and the public data that would fill it. This is the hand-off to a researcher.')
    story.append(Paragraph('7.1  What each substrate carries', sSect2))
    story.append(tbl([("Substrate", "Reads", "Floor", "Status 2026-09-19"),
        ("whole blood / buffy coat", "immune ARCHITECTURE of the leukocytes present; epithelial composition absent by biology", "methyl", "n=7 both channels; 3/3 healthy BELOW age band — OPEN"),
        ("plasma cfDNA", "SHED tissue composition", "frag / wps / nucl — never methyl", "composition measured n=33; gauge withdrawn (wrong floor)"),
        ("tissue", "positive control; heterogeneity marker (glioma-LL-002)", "methyl", "conformant, MAE 0.0004"),
        ("urine sediment", "shed urothelial / prostate", "to be established", "GSE119260 in hand, not run"),
        ("CSF", "shed CNS (terminal class)", "to be established", "GSE292312, GSE269403 located; supplementary files"),
        ("stool", "shed colonic epithelium", "to be established", "no public array data exists")], [0.18, 0.36, 0.18, 0.28]))
    story.append(SP(0.08)); story.append(Paragraph('7.2  The grid', sSect2))
    rows = [("", *D.CLASS_ORDER)]
    for sub in D.SUBSTRATES_GRID:
        r = [Pb(sub)]
        for c in D.CLASS_ORDER:
            st = D.GRID_STATUS.get((sub, c)) or D.GRID_STATUS.get((sub, "*")) or ("NOT ESTABLISHED", "")
            tok={"CONFORMANT":"CONF","MEASURED":"MEAS","COMPOSITION":"COMP","ABSENT":"ABS","STRUCTURAL":"STRUCT","SATURATED":"SAT","DATA":"DATA","NO":"NONE"}
            k=st[0].split(" ")[0]; r.append(status_p(tok.get(k,"—") if st[0]!="NOT ESTABLISHED" else "—"))
        rows.append(tuple(r))
    story.append(tbl(rows, [0.16] + [0.105]*8, fs=6.5))
    story.append(Paragraph('CONF = reproduces the project answer key · MEAS = run on canonical files, result recorded · COMP = deconvolver composition only (gauge not applicable on the methyl floor) · '
        'ABS = fraction ≈ 0 by biology · STRUCT = zero by marker coverage · SAT = methyl ceiling below BREACH · DATA = public cohort located, not run · NONE = no public data, primary collection required · — = not established', sDisc))
    story.append(SP(0.06)); story.append(Paragraph('7.3  What was measured', sSect2))
    story.append(Paragraph('<b>Whole blood, n=7</b> (test-data IDATs, Stage-1 calibrated, both channels). Epithelial-class fraction 0.0000–0.0111 in all seven; immune 0.803–0.966; residual MAE 0.043–0.060. '
        'Two rheumatoid-arthritis cases sit inside the non-RA range on both channels (gauge 0.8743/0.8789 vs 0.8066–0.8924; immune fraction 0.8395/0.8686 vs 0.8027–0.9663). '
        '<b>The three age-labelled healthy donors read BELOW the age-matched band:</b> 43M A=0.8739 (band 0.9021–0.9934, z −2.07); 58M 0.8066 (0.9083–1.0029, z −4.03); 67F 0.8326 (0.9166–1.0138, z −3.49). '
        'Either the Stage-1 path and the literature-compiled reference matrix are on different scales, or these donors are below band. n=3 cannot separate the two. This is PROC-WB-IMMUNE-01\'s open verdict.', sBodySm))
    story.append(tbl([("GSM", "label", "epithelial", "immune", "progenitor", "resid MAE", "immune gauge A", "age band")] +
        [(g, lab, f'{e:.4f}', f'{im:.4f}', f'{pr:.4f}', f'{rm:.4f}', f'{A:.4f}',
          (lambda b: f'{b["A_p10"]:.3f}–{b["A_p90"]:.3f} → {"BELOW" if A < b["A_p10"] else "IN" if A <= b["A_p90"] else "ABOVE"}')(D.age_band("immune", age)) if age else "age not in GEO")
         for g, lab, age, e, im, pr, rm, A in D.WHOLE_BLOOD], [0.13, 0.15, 0.10, 0.09, 0.10, 0.10, 0.13, 0.20], fs=6.8))
    story.append(SP(0.06))
    story.append(Paragraph('<b>Plasma cfDNA, GSE122126 (Moss 2018), n=33 plus 22 sepsis</b> — composition only. ' + D.PLASMA_NOTE, sBodySm))
    pc = D.PLASMA_COMPOSITION
    story.append(tbl([("condition", "n", "epithelial fraction (cycling+secretory+terminal+stromal)", "residual MAE")] +
        [(k, str(v["n"]), f'{v["epi_mean"]:.3f} (mean)' if "epi_mean" in v else f'median {v["epi_median"]:.3f}, p75 {v["epi_p75"]:.3f}, max {v["epi_max"]:.3f}', f'{v["resid"]:.3f}' if v.get("resid") else "—")
         for k, v in pc.items()], [0.20, 0.08, 0.50, 0.22]))
    story.append(Paragraph('Lung cancer reads epithelial-null in the correct substrate: nothing shed, nothing to read, instrument says so. Sepsis sheds epithelium in a minority of samples '
        '(multi-organ injury, established biology). Residual MAE tracks epithelial fraction (r = 0.98 across all samples) — a second readout the chain computes and discards; not yet checked against the corpus. '
        '<b>Cell-level routing:</b> ' + D.CELL_ROUTING, sBodySm))
    story.append(SP(0.04))
    story.append(Paragraph('<b>Tissue</b> — conformance, PROC-DECON-01 (§8). Whole-blood versus tissue integrity gate (CHK-3.1) on Stage-1-calibrated data:', sBodySm))
    story.append(tbl([("GSM", "substrate", "% β &lt;0.05 or &gt;0.95", "% β in 0.40–0.60")] + [(g, s, f'{e:.1f}', f'{m:.1f}') for g, s, e, m in D.CHK31], [0.15, 0.35, 0.25, 0.25]))
    story.append(Paragraph('The &gt;30% extreme-β criterion written for whole blood fails every properly calibrated tissue sample. Integrity thresholds must be stated per substrate (§9, CCL-032 amended).', sDisc))
    story.append(SP(0.06)); story.append(Paragraph('7.4  The researcher hand-off — one experiment per open row', sSect2))
    story.append(tbl([("substrate", "prediction to test", "data", "who can run it"),
        ("whole blood × immune", "healthy donors read IN the age band once Stage-1 and the reference are on one scale; if not, the reference is re-anchored to the atlas scale (open VAL-053)", "test-data IDATs + any public healthy blood 450K/EPIC", "any methylation lab; no clinical access needed"),
        ("plasma cfDNA", "epithelial classes appear in cancer plasma on frag/wps/nucl floors and stay absent in healthy; in-vitro mixes recover at MAE ≤ 0.01", "GSE122126 (has 14 known mixes); GSE311578 (70 AD plasma)", "cfDNA group with WGS for the non-methyl substrates"),
        ("urine", "prostate signal appears in the secretory class of urine sediment in the same men whose tissue shows it; plasma of the same men reads lower", "GSE119260: 4 pts × tissue/plasma/urine", "urology or GU-oncology group"),
        ("CSF", "terminal-class signal present in CSF and absent from the same patient's blood", "GSE292312 (24 CSF + 157 tumour); GSE269403 (39 CSF + 17 paired blood)", "neuro-oncology group"),
        ("stool", "colonic cycling-class signal in stool DNA", "none public — primary collection", "GI group with a stool-DNA protocol")], [0.16, 0.40, 0.24, 0.20]))

# ═══════════════════════════════════════════════════════════════════════════════
def proc(story, pid, title, fields):
    story.append(KeepTogether([Paragraph(f'<b>{pid}</b> — {title}', sSect2),
        tbl([("field", "value")] + fields, [0.16, 0.84], fs=7.2)]))
    story.append(SP(0.08))

def sec8_procedures(story):
    opener(story, 'SECTION 8', 'PROCEDURES',
        'One entry per verified cell of the grid, in the form a stranger can run: input, operation, expected, tolerance, observed, verdict. Nothing else. '
        'Every entry names the exact files at repo commit 66f37fe. An entry whose verdict is OPEN is printed as OPEN. Entries not yet written are listed at the end with what they need. '
        'The template is PROC-DECON-01, the first procedure in this project to pass on a machine that had never seen the author\'s setup.')
    proc(story, 'PROC-DECON-01', 'Deconvolver conformance — tissue, EPIC', [
        ("input", "betas_cache.pkl (Stage-1 noob-calibrated β, TEST_DATA); IAMAtlasREBUILD.csv decompressed from repo IAM_Atlas/IAMAtlasREBUILD.csv.xz; IAMAtlasREBUILD_celltype_to_class.json"),
        ("operation", "WaltherIAMDeconvolver(atlas, celltype_class_map=map).deconvolve(beta_dict) for GSM8772491, GSM5065990, GSM5065985"),
        ("expected", "class fractions in TEST_DATA_MANIFEST.md: GSM8772491 cycling .354 immune .253 stem_pluri .161 secretory .122 terminal .112"),
        ("tolerance", "class-fraction MAE ≤ 0.001 over documented classes"),
        ("observed 2026-09-19", "GSM8772491: cycling .3535 immune .2530 stem_pluri .1605 secretory .1215 terminal .1115 — MAE 0.0004, max |err| 0.0005. GSM5065990 MAE 0.0002. GSM5065985 MAE 0.0002. Status OK all three; 7,114 class markers, 4,000 cell markers."),
        ("verdict", "PASS. Deltas are the manifest's 3-decimal rounding."),
        ("scope", "verifies deconvolver + atlas + map. Does NOT verify Stage 1 (cached β used, raw IDATs not decoded). See PROC-CAL-01.")])
    proc(story, 'PROC-CAL-01', 'Stage 1 - raw IDAT to calibrated beta, executed', [
        ("input", D.CAL01["input"]), ("operation", D.CAL01["operation"]), ("expected", D.CAL01["expected"])]
        + [(r["gsm"], f"{r['array']} {r['n_cpgs']:,} CpGs, {r['secs']} s; vs cache r = {r['r']:.6f}, max |diff| {r['maxdiff']:.6f}, CpGs differing >1e-4: {r['n_gt_1e4']}") for r in D.STAGE1]
        + [("verdict", D.CAL01["verdict"]), ("consequence", D.CAL01["consequence"])])
    proc(story, 'PROC-N7-01', 'End-to-end synthetic simulation - the gauge as wired reads the wrong CpG set', [
        ("input", D.N7_01["input"]), ("R1 composition", D.N7_01["R1 composition"]), ("R2 gauge", D.N7_01["R2 gauge"]),
        ("root cause", D.N7_01["root cause"])]
        + [(s_, f"{a} | {m}") for s_, a, m in D.N7_01["same samples, both statistics"]]
        + [(f"consequence {k+1}", c) for k, c in enumerate(D.N7_01["consequences"])]
        + [("verdict", D.N7_01["verdict"])])
    proc(story, 'PROC-NILC-01', 'The retired second deconvolver, rerun as designed', [(k, D.NILC_01[k]) for k in ("input","conditioning","result","reading","recommendation")])
    proc(story, 'PROC-SEP-01', 'Is the HSC / progenitor information in the Atlas?', [(k, D.SEP_01[k]) for k in ("question","measurement","diagnosis")] + [(a, c) for a, c in D.SEP_01["remedy"]] + [("framing", D.SEP_01["framing"])])
    proc(story, 'PROC-WB-IMMUNE-01', 'Whole blood × immune — healthy reads in band', [
        ("input", "betas_cache.pkl for GSM2333901 (58M), GSM2333905 (67F), GSM2333950 (43M), GSM1051533, GSM1051534 (RA-study controls), GSM1051525, GSM1051526 (RA); iamatlas_gauge_identity_loci_v1_0.json; age_reference_matrix.json; IAMAtlasREBUILD.csv + map"),
        ("operation", "(1) deconvolve → require immune fraction ≥ DETECT_FLOOR (it is: 0.80–0.97); (2) β<sub>mean</sub> over the 42,134 immune identity loci (36,290 present on 450K); (3) A = H(β<sub>mean</sub>)/0.838889; (4) place against the immune p10–p90 band at the donor's age; (5) NO input offset (retired)"),
        ("expected", "healthy donors IN BAND; RA cases not required to differ (chronic autoimmune; no prior)"),
        ("tolerance", "IN BAND = p10 ≤ A ≤ p90 of age_reference_matrix immune at nearest decade midpoint"),
        ("observed 2026-09-19", "43M A=0.8739 vs [0.9021, 0.9934] z −2.07 BELOW · 58M 0.8066 vs [0.9083, 1.0029] z −4.03 BELOW · 67F 0.8326 vs [0.9166, 1.0138] z −3.49 BELOW · RA 0.8789/0.8743 and controls 0.8924/0.8673 (ages not in GEO). Composition: epithelial 0.0000–0.0111 all seven."),
        ("verdict", "OPEN. 3/3 healthy BELOW BAND. Two hypotheses, not separable at n=3: (a) age_reference_matrix (compiled from Hannum/Horvath/Alisch literature β) and the Stage-1 noob path are on different scales — the ~0.1 gap is in the direction the retired 0.055 offset addressed; (b) the donors are below band. Next: run ≥30 public healthy whole-blood 450K/EPIC through the same path; if the median sits ~0.1 below p50 at every decade, (a) is confirmed and the reference is re-anchored to the atlas scale (VAL-053)."),
        ("do not", "compute a group mean and call it the result; the reading is per donor against the band")])
    proc(story, 'PROC-WB-COMP-01', 'Whole blood — composition is immune, epithelium absent', [
        ("input", "as PROC-WB-IMMUNE-01, deconvolver only"),
        ("expected", "immune + progenitor + stem_adult ≥ 0.95; epithelial (cycling+secretory+terminal+stromal) ≤ 0.02; residual MAE below tissue (0.108–0.153)"),
        ("observed 2026-09-19", "immune 0.8027–0.9663, progenitor 0–0.174, epithelial 0.0000–0.0111, residual 0.0434–0.0596 — all seven"),
        ("verdict", "PASS on n=7 including two RA cases. This is the substrate specification (README 'Key validated facts'), not a limitation. It says nothing about the immune-ARCHITECTURE channel, which is PROC-WB-IMMUNE-01.")])
    proc(story, 'PROC-PLASMA-COMP-01', 'Plasma cfDNA — shed epithelium by condition', [
        ("input", "GSE122126-GPL21145 series matrix (Moss 2018, EPIC): healthy 4, colon 4, breast 3, lung 4, CUP 4, sepsis 22, in-vitro mixes 14; author-processed β (no Stage 1)"),
        ("operation", "deconvolve each sample; epithelial = cycling+secretory+terminal+stromal"),
        ("expected", "healthy ≤ 0.05; solid-tumour plasma &gt; healthy; in-vitro mixes recover declared proportions"),
        ("observed 2026-09-19", "healthy 0.030 · colon 0.463 · breast 0.466 · lung 0.010 · CUP 0.293 · sepsis median 0.040 (p75 0.233, max 0.520). Residual MAE tracks epithelial fraction r=0.98. Mixes NOT yet scored."),
        ("verdict", "PASS on direction for colon/breast; lung null recorded; sepsis sheds in a minority. NOT a gauge result — any A on this cohort must use frag/wps/nucl floors (RECON S1), which array data cannot supply. Cell-level tissue-of-origin FAILS: all epithelium → gastric references."),
        ("caveat", "author-processed GEO β without Stage 1; n per cancer 3–4")])
    proc(story, 'PROC-FORMULA-01', 'A-score aggregation - SOP v1.4.0 (mean of per-CpG H) vs code at HEAD (H of mean beta)', [
        ("input", "betas_cache.pkl (7 Stage-1 whole blood incl. 2 RA; 4 EPIC colorectal tissue); iamatlas_gauge_identity_loci_v1_0.json; iamatlas_celltype_markers_v0_2.json aggregated to class unions; age_reference_matrix.json"),
        ("operation", "per class {immune, cycling, secretory}, per sample: (a) H(beta_mean)/H_min over identity loci [code at HEAD]; (b) mean_i H(beta_i)/H_min over identity loci [SOP v1.4.0 formula on the HEAD loci]; (c),(d) the same two over the discriminative class union. Then Spearman(a,b) and the offset a-b, separately for whole blood and tissue. Then check which aggregation age_reference_matrix.json was compiled with."),
        ("expected", "if the two aggregations are equivalent on the declared substrate, rank agreement ~1 and a constant offset; if s105's mechanism is real, the offset should grow on mixed or absent-class panels; the band must match one of them exactly"),
        ("observed 2026-09-19", "WHOLE BLOOD immune: Spearman +1.000, offset +0.029 +/- 0.002. WHOLE BLOOD cycling/secretory (absent classes): offset +0.16. TISSUE immune: offset +0.24 +/- 0.07. age_reference_matrix A_mean = H(beta_mean)/H_min to 5 decimals in 80/80 cells. Discriminative mean-of-H tracks presence (cycling WB 0.35-0.41 vs tissue 0.57-0.62)."),
        ("verdict", "OPEN - author decision required. On the declared substrate with the class present, formula = constant offset; both valid IF band and runtime agree, and the band is H(beta_mean). On mixed/absent panels H(beta_mean) inflates as s105 says - never read it as architecture (presence gate; glioma-LL-002). The current SOP and the running code disagree and one must be amended: keep H(beta_mean) and scope s105's NEVER, or adopt mean-of-H and recompile the band."),
        ("retraction", "the first draft of this PROC (same day) read the tissue H(beta_mean) values as a disease ordering; that was composition inflation and is withdrawn (s10)."),
        ("caveat", "n = 11, two substrates; RA-study samples read +0.04 on discriminative H(beta_mean) for cases and controls alike - a GEO cohort offset (CCL-004), not disease")])
    proc(story, 'PROC-PLASMA-MIX-01', 'Deconvolver against real known mixtures (Moss 2018 Table 6 / Table 8)', [
        ("input", "GSE122126 GPL21145: in_vitro_mix_9..17 (9 genomic-DNA mixes: leukocytes 85-96% + hepatocytes/lung/neurons/colon 3.5-10%, Supp Data 1 Table 6) and cfDNA_mix_1,3,5,6,7 (colon-cancer cfDNA CC2 at 0-10% into healthy cfDNA, Table 8). Author-processed β."),
        ("operation", "deconvolve; compare declared spike per tissue to the class the atlas assigns that tissue (Hepatocytes→secretory, Cortical_neurons→terminal, Colon_epithelial_cells→cycling). GEO title→Mix mapping by order, corroborated by max-terminal landing on Mix2 (10% neurons) and max-cycling on Mix4 (10% colon)."),
        ("expected", "declared non-leukocyte fraction recovered (slope ~1, bias ~0); each spiked tissue rises in its own class"),
        ("tolerance", "proposed: per-class MAE ≤ 0.02 against declared; r ≥ 0.9 per spiked tissue"),
        ("observed 2026-09-19", f"TOTAL non-leukocyte: r = {D.MIX_STATS['total_r']:.3f}, MAE {D.MIX_STATS['total_mae']:.4f}, bias {D.MIX_STATS['total_bias']:+.4f} (over-reads by ~7 points in every mix). "
         f"NEURONS→terminal r = {D.MIX_STATS['neur_terminal_r']:+.3f} (10% → 6.5%, under-read, ordering correct). HEPATOCYTES→secretory r = {D.MIX_STATS['hep_secretory_r']:+.3f}, p = {D.MIX_STATS['hep_secretory_p']:.2f} (10% → 1.7%). "
         f"COLON→cycling r = {D.MIX_STATS['colon_cycling_r']:+.3f}, p = {D.MIX_STATS['colon_cycling_p']:.2f} (cycling reads 5–10% whether or not colon is present). Cancer-cfDNA spike (n=5): epithelial r = {D.MIX_T8_R[0]:+.3f}, p = {D.MIX_T8_R[1]:.2f} — monotone but not interpretable as recovery (CC2's own load unknown)."),
        ("verdict", "SPLIT. Terminal class recovers a real neuronal spike (PASS on direction, under-reads magnitude). Secretory does NOT register hepatocyte DNA and cycling does NOT register colon DNA — both FAIL on the first real ground truth the deconvolver has faced. Consistent with §7.3 cell routing (all epithelium → gastric) and LESSON-DECONV-01 (reference mismatch): the atlas profiles for liver and colon do not match real liver and colon DNA closely enough to earn weight. This is atlas work, not solver work."),
        ("consequence", "the plasma colon/breast epithelial fractions of PROC-PLASMA-COMP-01 (0.46) are real departures from healthy (0.03) but their class assignment is not trustworthy; the tissue-of-origin claim is withdrawn until the atlas recovers Table 6.")])
    proc(story, 'PROC-ANCHOR-01', 'Sealed GSE51032 per-cell anchor, reproduced from raw GEO betas', [
        ("input", D.ANCHOR["input"]), ("operation", D.ANCHOR["operation"]),
        ("expected", "r = 1.0000, maxdiff = 0.0000 against the sealed CSV, as SOP v1.4.0 s105 reports for the vault module")]
        + [(f"observed {r[0]}", f"{r[1]}; {r[2]}; {r[3]} - {r[4]}") for r in D.ANCHOR["rows"]]
        + [("also observed", D.ANCHOR["extras"]), ("verdict", D.ANCHOR["verdict"])])
    proc(story, 'PROC-QC-01', 'Stage 0-1 intake and calibration gates (SOP v1.4.0 s11-25) - specified; PROC-CAL-01 executes it', [(k,v) for k,v in D.STAGE01_QC]+[
        ("verdict", "SPECIFIED, NOT RUN - every gate above is a pass/fail with a threshold; PROC-CAL-01 will record each on the seven test-data IDATs")])
    story.append(Paragraph('Not yet written — and what each needs', sSect2))
    story.append(tbl([("PROC", "needs"),
        ("PROC-WB-PROGENITOR-01 / STEM_ADULT-01", "the CHANGELOG's suspected breach artifact in peripheral blood; requires the presence-gated class ranking fix (design decision on threshold pending)"),
        ("PROC-TISSUE-N-COMP-01", "any tissue claim must report immune-class A beside the target class (glioma-LL-002); write once CPG-NEW-001 is re-run presence-gated"),
                ("PROC-URINE-01, PROC-CSF-01", "GSE119260 in hand; CSF cohorts need supplementary-file fetch; floors for these substrates not yet established"),
        ("PROC-N5-01", "plate/sentrix position null from raw IDAT filenames for GSE51032/GSE51057 — first thing a methylation referee asks for on the breast pre-dx result")], [0.30, 0.70]))

# ═══════════════════════════════════════════════════════════════════════════════
def sec9_rules(story):
    opener(story, 'SECTION 9', 'OPERATING RULES',
        'The lessons that govern how the instrument may be read, promoted from appendix to rules. Each carries its date, source, and supersession status. A rule that is broken produces a wrong number, not a style complaint; several of these were broken in the runs of 2026-09-19 and the resulting wrong numbers are in §10.')
    rows = [("ID", "Rule", "Content", "Source", "Status")]
    for r in D.RULES: rows.append((Pb(r[0]), Pb(r[1]), Ps(r[2]), Ps(r[3]), P(r[4])))
    story.append(tbl(rows, [0.10, 0.20, 0.44, 0.16, 0.10], fs=7))

def sec10_falsification(story):
    opener(story, 'SECTION 10', 'FALSIFICATION RECORD',
        'Every outcome that did not go the framework\'s way, kept on the record. A framework that cannot show its failures cannot show that its passes mean anything. '
        'Historical VALs are quoted from their sealed OUTCOME files; the 2026-09-19 rows are from this session and include the author\'s own failed predictions.')
    story.append(tbl([("Test", "What happened", "Outcome")] + [(Pb(a), P(b), Pb(c)) for a, b, c in D.FALSIFICATION], [0.20, 0.60, 0.20]))
    story.append(SP(0.08))
    story.append(Paragraph('Also on the record from this session, as method failures rather than results: the same analyst computed group effect sizes where the framework requires per-sample absolute readings (three times); '
        'scored classes the deconvolver had not found in the sample; scored plasma on the methyl floor; and read the pre-atlas SOP as the specification of the post-atlas engine. '
        'Each is now a rule in §9 or a row in §1. That is what the rules are for.', sBodySm))


def sec11_engine_map(story):
    opener(story, 'SECTION 11', 'ENGINE MAP — WHAT THE RUNNING CHAIN CONTAINS, AND WHAT THIS ISSUE DOES NOT YET COVER',
        'The flowchart at HEAD (flowchart_vKISS.html) lists the twelve stages below (0 through 9, with 4.5 and 4.6 as separate wired stages), plus orchestration and test data. Issue 002 had no chain at all — it was class cards and physics — so every stage is new territory for 003, '
        'and this issue documents only the stages that were executed with the canonical files during its preparation. The right-hand column says so, stage by stage. '
        'A reader who needs Stage 4.5, 5, 6, 8 or 9 will not find them here; they are the next issue\'s work, and the files that implement them are named so they can be read now.')
    rows=[("Stage","Files at HEAD","Status","What it does (from source)","In Issue 003?")]
    for st,f,status,what,cov in D.ENGINE_MAP:
        rows.append((Pb(st),Ps(f),P(status),Ps(what),Pb(cov) if cov.startswith("YES") else P(cov)))
    story.append(tbl(rows,[0.14,0.22,0.12,0.34,0.18],fs=6.8))
    story.append(SP(0.08)); story.append(Paragraph('Findings from the 2026-09 review that are recorded here only as reconciliation rows, not yet as sections', sSect2))
    story.append(Paragraph('The origin-gate fail-open in walther_clinical.py (F3); two stale strings describing the retired gauge, one on the patient report (F4); the gauge\'s missing sign (F5); '
        'two inequivalent A-score definitions (F6); the immune H_min revision history (F7); the seminoma value carried as both 0.67 and 0.755 in different files; the breast pre-diagnostic result '
        '(n = 47 cases / 601 controls across GSE51032 + GSE51057; matched-filter ρ = +0.058, CI [+0.001, +0.114]) and its unrun plate-position null; the colorectal cohort test CPG-NEW-001 in full '
        '(its P1/P4 failures are in §10, its method and N-random/N-comp results are not here); the two-observables test on GSE48684/GSE139404; and the disease-signature matrix v1.13 with its 81 rows. '
        'Each is in the project record (CPG_first_read.md, CPG_LEDGER.md, CPG_breast_CRC_learned.md, OUTCOME_CPG-NEW-001.md) and none has been re-verified for this issue.', sBodySm))

# ═══════════════════════════════════════════════════════════════════════════════
def sec5a_tools(story):
    opener(story, 'SECTION 5A', 'WHERE THE TOOLS COME FROM',
        'Issue 002 stated the physics of the floor. Since April the engine acquired an atlas, a deconvolver, a sky projection, a matched filter and a second '
        'statistic, and every one of them was borrowed from somewhere specific: from the semiconductor and quantum-computing applications of the same law, and from the '
        'Planck CMB pipeline. A reader who is going to spend time on this work will want to know where each idea came from and what it is doing here. This section says so, '
        'in the order a physicist would ask. Nothing here is a new claim; the new claims are in §5 (reproduced from 002) and the measurements are in §7–§8.')
    # 5A.1 Mahaffey number (author's definition, 2026-09-19)
    story.append(Paragraph('5A.1  The Mahaffey number', sSect2))
    story.append(Paragraph(f'<b>M = {D.MAHAFFEY["value"]}.</b> {D.MAHAFFEY["definition"]}', sBodySm))
    story.append(Paragraph(D.MAHAFFEY["not_hmin"], sBodySm))
    story.append(Paragraph(D.MAHAFFEY["n_bio"], sBodySm))
    story.append(Paragraph('The same construction - a measured drive energy over the local thermal quantum - is what SCAPE computes for a transistor and QAPE for a qubit; those figures are quoted in Section 5A.8 exactly as their own reports state them, with their own denominators, and are not re-derived here.', sBodySm))
    story.append(Paragraph(f'<b>Open.</b> {D.MAHAFFEY["open"]}', sDisc))
    story.append(Paragraph(D.MAHAFFEY["ln2_note"], sDisc))
    story.append(SP(0.10))
    # 5A.2 40 floors
    story.append(Paragraph('5A.2  Forty floors — eight classes × five substrates, each confirmed by MCMC', sSect2))
    story.append(Paragraph('The cellular H_min is not one number. Each of the eight architecture classes has its own floor on each of five physically distinct lab measurements — methylation β, '
        'nucleosome occupancy, nucleosome fuzziness, window protection score, and cfDNA fragment size — giving the forty-cell table below. The methylation column was fixed by the G-002 chain '
        '(17 chains, R-hat &lt; 1.001) and the other four by G-003b (5 × 32-walker ensembles); all forty were frozen 2026-04-06 and the engine at HEAD carries them byte-for-byte (RECON H1). '
        'Because every floor has its own ceiling 1/H_min, every class has <b>five saturation limits</b>, and the substrate a sample arrives on selects which five of the forty apply to it. '
        'The runtime saturation rule flags any A within 0.005 of its ceiling; a class whose methyl ceiling is below 1.10 (stem_pluri, 1.018) cannot register BREACH on methylation and must be read on another substrate.', sBodySm))
    SUBS=["methyl","nucl","fuzz","wps","frag"]
    SH={"methyl":"meth","nucl":"nucl","fuzz":"fuzz","wps":"wps","frag":"frag"}
    rows=[("class",)+tuple(f"H_min<br/>{SH[s]}" for s in SUBS)+tuple(f"ceil.<br/>{SH[s]}" for s in SUBS)]
    for c in D.CLASS_ORDER:
        hm=D.H_MIN_TABLE[c]; rows.append((Pb(c),)+tuple(f"{h:.6f}" for h in hm)+tuple(f"{1/h:.3f}" for h in hm))
    story.append(tbl(rows,[0.12]+[0.088]*10,fs=6.4))
    story.append(Paragraph('Posterior widths matter for tier calls: the methylation σ(H_min) from G-002 is ±0.0012 (Issue 002 cards), so a tier boundary 0.01 wide is ~8σ of floor uncertainty — the floor is not the limiting error; Stage-1 calibration and the age band are.', sDisc))
    story.append(Paragraph('The sampler, so a reader can re-run it', sLabel))
    story.append(tbl([("setting","value")]+[(Pb(k),v) for k,v in D.MCMC_HMIN],[0.24,0.76],fs=6.8))
    # 5A.3 Atlas as posterior
    story.append(Paragraph('5A.3  The IAM Atlas is a posterior, not a lookup table', sSect2))
    story.append(Paragraph(f'IAMAtlasREBUILD ({D.N_CPGS:,} CpGs × {D.N_CELLTYPES} cell types) was built by per-class MCMC over the public reference atlases — immune alone from 63 cell types across 25 source atlases, '
        '4,987,426 observations, batch 1500, ~146,000 s; stromal, the smallest, from 5 cell types and 17 atlases in ~2,550 s. For every CpG and every class the atlas stores the posterior mean, SD and credible '
        'interval (8 "brightness files", 246 posterior columns), plus per-cell-type means. Three consumers read it: the deconvolver takes the means as its reference matrix; the gauge identity loci are the CpGs '
        'where the class mean sits within 0.05 of the class H_min_β; and the Stage 4.6 sky (5A.4) uses the per-CpG SD as its noise model. The predecessor IAMAtlas.csv.xz was retired for a collapse/flatness bug — '
        'which is what the first two lines of the CHANGELOG are about, and why the provenance file names the build.', sBodySm))
    story.append(Paragraph('How the atlas was built', sLabel))
    story.append(tbl([("item","value")]+[(Pb(k),v) for k,v in D.ATLAS_BUILD],[0.22,0.78],fs=6.8))
    # 5A.4 CMB toolkit
    story.append(Paragraph('5A.4  The CMB toolkit, applied to the methylome', sSect2))
    story.append(Paragraph('The methylome and the microwave sky pose the same problem: one noisy map, a physical reference for what it should look like, several foregrounds, and the need to say '
        'whether a departure at a given position is real. The engine borrows four Planck tools, each named with its origin and its methylome analogue.', sBodySm))
    story.append(tbl([("Planck tool","what it does there","methylome analogue in the engine","status at HEAD"),
        ("Brightness / temperature map with per-pixel noise","T(pixel) vs the ΛCDM expectation, standardised by the pixel noise","Stage 4.6 <b>Personal Cosmic Methylome</b> (cpg_patient_cmb.py): z(CpG) = (β_patient − μ_class)/max(σ_class, floor), σ from the atlas posterior, so z is departure relative to healthy biological variation at that locus; projected to a HEALPix sky by atlas row order, one Mollweide per class","BUILT 2026-06-29"),
        ("Assessability / masking","galactic mask: don\'t read where the foreground dominates","a class is ASSESSABLE from blood only if its DNA is present; when it is not, the whole panel reads as a uniform offset and is drawn reference-only. Self-determined: median|z| &lt; ASSESS_MAX. The presence rule (§3.2) is the same idea one stage earlier","BUILT"),
        ("Matched filter","correlate the map against a template to detect a known signal below the noise","Stage 5 <b>residual-map matched filter</b>: Pearson r between the patient\'s per-CpG departure from the derived atlas baseline and a disease\'s signed residual map, with Fisher CI; fires when the CI clears zero in either direction. Breast: ρ = +0.058, CI [+0.001, +0.114] on the pre-dx cohort (a detection whose lower bound is 0.001 above zero — marginal, and printed as such)","BUILT; AD residual map removed 2026-07 (was acting as AD\'s detector)"),
        ("Component separation cross-check (Commander / NILC / SMICA / SEVEM must agree)","four independent algorithms had to agree before any CMB result shipped","<b>NILC methylome deconvolver</b>: weights derived once from the data\'s own covariance, independent of Walther\'s NNLS; the two had to agree on class fractions before L5+ was trusted (Walther vs NILC ρ = +0.74 immune, +0.82 progenitor on GSE51032)","<b>CUT 2026-07-02</b> (commit c1be0c3, Walther alone). The cross-method discipline is no longer enforced in the chain; the code remains in NILC Deconvolver/")],
        [0.18,0.22,0.42,0.18],fs=6.8))
    story.append(Paragraph('The honest note on the last row: the Planck discipline was the strongest methodological borrow in the engine, and it was removed to simplify. Whether it returns is a decision this document records as OPEN.', sDisc))
    from reportlab.platypus import Image as RLImage
    _fp=os.path.join(os.path.dirname(os.path.abspath(__file__)),"fig_four_skies.png")
    if os.path.exists(_fp):
        story.append(RLImage(_fp, width=PW, height=PW*0.66)); story.append(Paragraph(D.FOUR_SKIES_CAP, sDisc)); story.append(SP(0.10))
    # 5A.5 not cohort people
    story.append(Paragraph('5A.5  Why this is not a cohort method', sSect2))
    story.append(Paragraph('Standard biomarker work standardises a patient against a population — a z-score against controls, a classifier trained on cases. The IAM instrument does not. '
        'The reference is <b>physical</b> (forty frozen floors) and <b>derived</b> (the atlas posterior and the age band compiled from healthy references only, before any disease data were seen). '
        'A patient is read against that reference alone; cohorts enter only to establish the <i>direction</i> a disease moves a class (L-5, §9) and never as a baseline. '
        'What this buys: no population standardisation, no training set, one frozen reference that a stranger can download and check, and a per-patient absolute reading rather than a percentile. '
        'What it costs, stated here because the measurements of §7 found it: the healthy range is a function of class <i>and substrate</i>, and the same immune panel with the same floor reads ~0.85 in whole blood and ~1.03 in plasma. '
        'A physical reference has to be anchored per substrate before a departure means anything — which is PROC-WB-IMMUNE-01\'s open verdict. The analyst preparing this issue built a cohort comparison three times '
        'before internalising this rule; it is now a rule with a mechanism (presence-gated, per-sample scoring functions) rather than a reminder.', sBodySm))
    # 5A.6 two statistics
    story.append(Paragraph('5A.6  Two surfaces, two aggregations - ruled in s1.5', sSect2))
    story.append(Paragraph('§3.3 shows all four aggregation × loci combinations on the same eleven samples. Identity loci carry the class gauge; discriminative markers carry presence and the sealed anchors. RULING A3 (s1.5) fixes one aggregation per surface: the gauge on identity loci is H(β<sub>mean</sub>)/H<sub>min</sub>, because H<sub>min</sub> and the age band are both defined that way; the separation statistic on discriminative markers is the mean of per-CpG H, because those panels are bimodal and the sealed anchors are that statistic. On the declared substrate the two differ by a constant +0.03; on mixed or absent panels they diverge, and the gauge now refuses those panels by a measured Jensen-gap test rather than by rule.', sBodySm))

    # 5A.7 hull
    story.append(Paragraph('5A.7  The Mahalanobis hull - Stage 5, specified, not yet run here', sSect2))
    story.append(tbl([("item","value")]+[(Pb(k),v) for k,v in D.HULL],[0.22,0.78],fs=6.8))
    # 5A.8 nulls
    story.append(Paragraph('5A.8  The null suite - what a sealed result has beaten', sSect2))
    story.append(tbl([("null","criterion")]+[(Pb(k),v) for k,v in D.NULLS],[0.26,0.74],fs=6.8))
    # 5A.9 downstream chain
    story.append(Paragraph('5A.9  The downstream chain as specified (SOP v1.4.0), for the stages this issue does not execute', sSect2))
    story.append(tbl([("stage / rule","specification")]+[(Pb(k),v) for k,v in D.CHAIN_RULES],[0.22,0.78],fs=6.6))
    story.append(Paragraph(f'<b>Pipeline relativity of absolute A.</b> {D.XU538}', sBodySm))
    story.append(Paragraph(f'<b>A cross-species prediction.</b> {D.CANINE}', sBodySm))
    # 5A.10 catalogue
    story.append(Paragraph('5A.10  The validation catalogue as recorded - NOT re-verified in this issue', sSect2))
    story.append(Paragraph('These are the effect sizes the corpus records for the sealed VALs. This issue re-ran none of them; they are listed so the reader knows what exists and can ask for the sealed record. Where two instruments give different signs for one result, both are shown.', sBodySm))
    story.append(tbl([("result","VAL","recorded effect","note")]+[(k,v,e,n) for k,v,e,n in D.CATALOGUE],[0.26,0.12,0.40,0.22],fs=6.4))
    # 5A.11 siblings
    story.append(Paragraph('5A.11  QAPE and SCAPE - the sibling applications, structure and published results only', sSect2))
    story.append(Paragraph('The same statistic on two other substrates. Their calibration inputs are not part of this document.', sBodySm))
    story.append(tbl([("engine","A","anchors","what sets the floor","published prediction record")]+[tuple(r) for r in D.QAPE_SCAPE],[0.08,0.24,0.18,0.30,0.20],fs=6.4))

# ═══════════════════════════════════════════════════════════════════════════════
def sec12_clinician(story):
    opener(story, 'SECTION 12', 'FOR THE CLINICIAN - THE INSTRUMENT IN PLAIN TERMS',
        'Two of the source documents (The Cellular Margin; What Is Astro-Genetics) were written for readers who are not physicists. This section carries their content, corrected where the errata in s10 apply, so a physician reading only this section understands what a departure is and is not.')
    for k in ["margin","floor","reading","substrate","life","errata"]:
        story.append(Paragraph(k.upper(), sLabel)); story.append(Paragraph(D.CLINICIAN[k], sBodySm)); story.append(SP(0.05))
    story.append(Paragraph('The stellar analogy the author uses: white dwarfs and neutron stars sit below a mass limit (Chandrasekhar 1.4 M_sun, TOV ~2.3 M_sun) set by physics, not by a survey of stars; the Sun\'s eventual core at 0.54 M_sun reads 0.38 of the limit, Procyon B 0.42, PSR J0740+6620 0.90. The cellular floor is the same kind of number: a limit from the physics, against which each object is read individually.', sBodySm))

def secVI_translation_map(story):
    story.append(PageBreak())
    story.append(Paragraph('APPENDIX VI — THE CMB → METHYLOME TRANSLATION MAP, SCORED', sSect))
    story.append(Paragraph('Written by the author before the chain was built: the 32-section CMB-analysis curriculum walked module by module, each assigned a methylome analog and a status '
        '(HAVE · ROADMAP · ADD = should add · REINTERP = translates with reinterpretation · NO = does not translate). The last column was added on 2026-09-19 and records what the chain actually did. '
        'Two rows went <i>against</i> the map and are the most instructive: a second deconvolver (row 20, built then cut) and de-aging (row 47, built then refused). '
        'Row 1 is the frame for the whole chain: the 115 cell types are the harmonic basis, and the deconvolver is the projection onto it. '
        'This appendix exists so that a geneticist can see where each tool came from, and a cosmologist can see where the methylome stops behaving like the sky.', sBodySm))
    for sec_ in D.TRANSLATION_MAP:
        story.append(Paragraph(sec_["title"], sLabel))
        rows=[("#","CMB module","CPG analog","status","author's note","2026-09-19 status")]+[tuple(r) for r in sec_["rows"]]
        story.append(tbl(rows,[0.03,0.15,0.23,0.10,0.20,0.29], fs=5.6)); story.append(SP(0.08))

def secVII_sprint(story):
    story.append(PageBreak())
    story.append(Paragraph('APPENDIX VII — THE COMPLETION SPRINT, SCORED', sSect))
    story.append(Paragraph('The spring-2026 plan to take the chain "from C- to A" in 12–15 sessions: Phase A (L9 nulls) → B (L4 foregrounds) → C (L5 correlation structure) → D (L6 covariance) → E (L7–L8 likelihood and inference) → F (audit). '
        f'Scored against the repository on 2026-09-19. The author\'s verdict: <i>"{D.SPRINT_VERDICT}"</i>', sBodySm))
    rows=[("phase","deliverable","what happened")]+[tuple(r) for r in D.SPRINT_SCORED]
    story.append(tbl(rows,[0.08,0.30,0.62], fs=6.0))
    story.append(SP(0.1)); story.append(Paragraph('The lesson', sLabel))
    for para in D.SPRINT_LESSON.split("\n\n"):
        story.append(Paragraph(para.replace("**","").replace("`",""), sBodySm)); story.append(SP(0.05))

def secVIII_part2(story):
    story.append(PageBreak())
    story.append(Paragraph('PART II — THE CHAIN, STAGE BY STAGE (FORTHCOMING)', sSect))
    story.append(Paragraph('Issue 003 documents the engine and its evidence. Part II, written once the chain carries its seal, will teach it: one chapter per stage, each giving the purpose, the cosmology it borrows, '
        'what was tried first and why it failed, the runtime files it reads, the procedure that confirms it, and what a researcher trained on bootstrapping needs to know before touching MCMC output or a Mahalanobis distance. '
        'The outline as it stands:', sBodySm))
    rows=[("chapter","content")]+[tuple(r) for r in D.PART_II_OUTLINE]
    story.append(tbl(rows,[0.30,0.70], fs=6.4))

def secIX_future(story):
    story.append(PageBreak())
    story.append(Paragraph('FUTURE GOALS — WHAT IS WORTH THE EFFORT, IN ORDER', sSect))
    story.append(Paragraph('Drawn from the scored translation map (Appendix VI) and the scored sprint (Appendix VII): only items not yet built, kept only where the data and tools in hand can support them, '
        'each with the gate it waits on. The order is the lesson of Appendix VII applied — nothing above the bands is scheduled until the bands are rebuilt from one cohort through one pipeline. '
        '"Not now" is a list, not a refusal; "does not translate" is the author\'s own ruling.', sBodySm))
    rows=[("gate","goal","from","why it is realistic","needs")]+[tuple(r) for r in D.FUTURE_GOALS]
    story.append(tbl(rows,[0.10,0.20,0.11,0.43,0.16], fs=5.8))

def sec_cosmo_evidence(story):
    story.append(PageBreak())
    story.append(Paragraph('WHAT THE COSMOLOGY TOOLS FOUND THAT COHORTS COULD NOT', sSect))
    story.append(Paragraph('The objection this section pre-empts is the one every reviewer will make: <i>"your healthy reference was calibrated on cohorts and validated on cohorts - that is circular."</i> '
        'The answer is not a denial. It is correct, and a cohort-only pipeline cannot detect the circularity - which is why this project does not use cohorts to validate the instrument. '
        'It validates the way a CMB experiment does: synthetic data with known truth, injection-recovery, split-half cross-checks, convergence and distinctness tests, look-elsewhere correction, sealed pre-registration. '
        'Below is what those methods found that no cohort could have. It is a ledger, kept current by rule (Reproduction Kit RUNBOOK §10), and it includes the reversals.', sBodySm))
    story.append(Paragraph(D.COSMO_EVIDENCE_RULE, sMut)); story.append(SP(0.08))
    rows=[("date","CMB method","why a cohort is blind to it","what was found","record")]+[tuple(r) for r in D.COSMO_EVIDENCE]
    story.append(tbl(rows,[0.10,0.18,0.19,0.38,0.15], fs=5.8))

def sec_presence(story):
    story.append(PageBreak())
    story.append(Paragraph('THE REPORTING RULE - WHICH CLASSES A SUBSTRATE MAY REPORT', sSect))
    story.append(Paragraph('Asked how often stem_adult had been the deciding class, the record was checked: every OUTCOME file, the VAL index, every disease card.', sBodySm))
    story.append(tbl([("where","stem_adult result","what carried the finding")]+[tuple(r) for r in D.STEM_ADULT_RECORD],[0.28,0.36,0.36], fs=6.0))
    story.append(SP(0.1))
    for k in ("statement","scope","whole blood","myeloid check","rationale","path back","keep"):
        story.append(Paragraph(f'<b>{k}.</b> {D.PRESENCE_RULE[k]}', sBodySm))

def sec0_scope(story):
    story.append(Paragraph('WHAT THIS DOCUMENT CLAIMS, AND WHAT IT DOES NOT', sSect))
    story.append(Paragraph('<b>Not claimed.</b>', sBodySm))
    for x in D.SCOPE["not_claimed"]: story.append(Paragraph(f'&bull; {x}', sBodySm))
    story.append(SP(0.06)); story.append(Paragraph('<b>Claimed.</b> ' + D.SCOPE["claimed"], sBodySm))
    story.append(SP(0.06)); story.append(Paragraph('<b>Invitation.</b> ' + D.SCOPE["invitation"], sBodySm))
    story.append(SP(0.1))

def sec_chain_terms(story):
    story.append(PageBreak())
    story.append(Paragraph('GLOSSARY — CMB AND CHAIN TERMS', sSect))
    story.append(Paragraph('For a reader who knows bootstrapping but not MCMC, and has never met a Mahalanobis hull or a HEALPix sky. Each definition is taken from the source file named in brackets.', sBodySm))
    for term, defn in D.CHAIN_TERMS:
        story.append(Paragraph(f'<b>{term}</b> — {defn}', sBodySm)); story.append(SP(0.04))

def sec_chain_links(story):
    story.append(PageBreak())
    story.append(Paragraph('GLOSSARY — CHAIN LINKS', sSect))
    story.append(Paragraph('One line per runtime file the conductor touches, tagged by what kind of thing it holds. The tags matter more than the names: a '
        '<b>FLOOR</b> is physics and is one number; a <b>RULER</b> says where the gauge reads and is derived from the Atlas; a <b>BAND</b> is a cohort statistic and is only '
        'as good as its n and pipeline; <b>CODE</b> runs; <b>DATA</b> is read. Every "healthy reads wrong" case found in September 2026 traced to a BAND, never to a FLOOR (RECON B1).', sBodySm))
    rows=[("file","kind","what it is")]+[(f,k,w) for f,k,w in D.CHAIN_LINKS]
    story.append(tbl(rows,[0.27,0.07,0.66], fs=6.4))

def secV_val_index(story):
    story.append(PageBreak())
    story.append(Paragraph('APPENDIX V - VALIDATION INDEX: EVERY VAL IN THE REPOSITORY, WITH ITS PATH', sSect))
    story.append(Paragraph(D.VAL_INDEX_NOTE, sBodySm)); story.append(SP(0.08))
    rows=[("VAL","title","date","cohorts","decision","record","path (repo, 66f37fe)")]
    for o in D.VAL_INDEX:
        t=o["title"].replace("CPG-VAL-","").replace(o["val"]+" ","")[:64]
        rows.append((o["val"], t, o["date"][:10], o["cohorts"], o["decision"][:34], o["record"]+(" · RETIRED" if o["retired"] else ""), o["path"].replace("Biological_Physics/","BP/")))
    story.append(Paragraph("Path prefix <font face=\"Courier\">BP/</font> = <font face=\"Courier\">Biological_Physics/</font> (tree reorganized 2026-09-19: Testing_and_Code/VAL_PreAtlas, VAL_PostAtlas, RETIRED). Paths are directories; the prereg, outcome and results files sit inside them.", sMut))
    story.append(tbl(rows,[0.082,0.225,0.078,0.105,0.105,0.085,0.32], fs=5.4))

def build(out_path):
    doc = SimpleDocTemplate(out_path, pagesize=letter, leftMargin=0.5*inch, rightMargin=0.5*inch, topMargin=0.45*inch, bottomMargin=0.55*inch)
    story = []
    cover(story); toc(story)
    story.append(PageBreak()); sec0_scope(story)
    sec1_recon(story); sec1b_rulings(story); sec_cosmo_evidence(story); sec_presence(story); sec2_atlas(story); sec3_instruments(story)
    # §4 framework from Issue 002
    L.blk_ranking(story); L.blk_framework(story); L.blk_mcmc(story); L.blk_bodytemp_saturation(story)
    # cards
    L.render_cascade_section(story)
    for card in L.CARDS:
        L.render_card(story, card); card_addendum(story, card['key'])
    sec5a_tools(story)
    # §5 physics (002 §2) with a one-paragraph preface
    story.append(PageBreak())
    story.append(Paragraph('PREFACE TO SECTION 5 (ISSUE 002 SECTION 2), ADDED IN ISSUE 003', sLabel))
    story.append(Paragraph('The derivation below is reproduced from Issue 002 as written. It was written before the G-002 and G-003b chains were run; the chains (17 methylation chains R-hat &lt; 1.001; '
        '5 × 32-walker chains for the four other substrates) then returned the forty H_min values that the engine at HEAD carries byte-for-byte. The derived-vs-calibrated ledger of the companion '
        'physics reference (IAM_for_physicists, §4) is available to any reader who wants each quantity\'s provenance stated separately; it is not reproduced here.', sBodySm))
    L.render_section_2_physics(story)
    L.render_section_3_evidence(story); L.render_section_4_baselines(story); L.render_section_5_scenarios(story); L.render_section_6_predictions(story)
    # new sections
    sec7_substrates(story); sec8_procedures(story); sec9_rules(story); sec10_falsification(story); sec11_engine_map(story); sec12_clinician(story)
    # back matter from 002
    secV_val_index(story); secVI_translation_map(story); secVII_sprint(story); secIX_future(story); secVIII_part2(story)
    L.blk_master_predictions(story); L.blk_data_sources(story); L.blk_glossary(story); sec_chain_terms(story); sec_chain_links(story)
    story.append(Paragraph(D.GLOSSARY_NOTE_MAHAFFEY, sDisc))
    L.blk_final_note(story)
    doc.build(story, onFirstPage=make_canvas, onLaterPages=make_canvas)
    return out_path

if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else 'IAMPerformance_GAPEIssue003_DRAFT.pdf'
    print(build(out))
