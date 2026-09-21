"""data003.py — every number Issue 003 prints, loaded from the runtime files or from
a dated, sourced run. Nothing here is typed from memory. Each block names its source.

Runtime files: CPG_TRIAL_CODE.zip (user-supplied 2026-09-19) == repo HEAD 66f37fe for every
file except iamatlas_celltype_markers_v0_2.json (see RECON row M1).
"""
import json, re, ast, os, collections
T = os.environ.get("CPG_TRIAL", "trial/CPG_TRIAL_CODE")

def _j(name): return json.load(open(os.path.join(T, name)))

# ── engine constants (cpg_gauge_engine.py @ HEAD) ─────────────────────────────
_eng = open(os.path.join(T, "cpg_gauge_engine.py"), encoding="utf-8").read()
def _lit(name):
    m = re.search(rf"^{name}\s*=\s*(\{{[^\n]*\}})\s*(#.*)?$", _eng, re.M)          # one-line dict
    if not m: m = re.search(rf"^{name}\s*=\s*(\{{.*?^\}})", _eng, re.M | re.S)   # multi-line dict
    return ast.literal_eval(m.group(1))
H_MIN_TABLE      = _lit("H_MIN_TABLE")
HEALTHY_BASELINE = _lit("HEALTHY_BASELINE")
AUC_W            = _lit("AUC_W")
HEALTHY_SD       = _lit("HEALTHY_SD")
SUB_ORDER        = ['methyl', 'nucl', 'fuzz', 'wps', 'frag']
BASELINE_CLASSES = ['cycling','secretory','immune','terminal','stromal','stem_adult','progenitor','stem_pluri']
CLASS_ORDER      = ['terminal','secretory','immune','progenitor','cycling','stromal','stem_adult','stem_pluri']  # Issue 002 card order
BREACH = 1.10; SATURATION_MARGIN = 0.005
GAUGE_TIERS = [("NORMAL", None, 1.01), ("MARGINAL", 1.01, 1.05), ("DETECTABLE", 1.05, 1.07),
               ("URGENT", 1.07, 1.10), ("FLOOR BREACH", 1.10, None)]

# ── atlas ─────────────────────────────────────────────────────────────────────
PROV = _j("IAMAtlasREBUILD_provenance.json")
C2C  = _j("IAMAtlasREBUILD_celltype_to_class.json")
CELLS_BY_CLASS = collections.defaultdict(list)
for ct, c in C2C.items(): CELLS_BY_CLASS[c].append(ct)
for c in CELLS_BY_CLASS: CELLS_BY_CLASS[c].sort()
N_CELLTYPES = len(C2C); N_CPGS = PROV["n_cpgs"]; ATLAS_BUILD = PROV["build_date"][:10]

# ── age reference, tiers ──────────────────────────────────────────────────────
AGE_REF = _j("age_reference_matrix.json")           # per class: list of decade dicts
TIERS_V13 = _j("tier_breakpoints.json")
STAGE7_TIERS = [(t["tier_id"], t.get("a_score_range"), t.get("line_value")) for t in TIERS_V13["tier_system_v1_2"]["tiers"]]

# ── identity loci (gauge) ─────────────────────────────────────────────────────
_loci = _j("iamatlas_gauge_identity_loci_v1_0.json")
IDENTITY = {c: {"n_loci": len(v["loci"]), "H_min": v["H_min"], "H_min_beta": v["H_min_beta"], "band": v["band"]}
            for c, v in _loci.items() if not c.startswith("_")}
IDENTITY_PROV = _loci.get("_provenance", {})

# ── markers (separation) ──────────────────────────────────────────────────────
_mk = _j("iamatlas_celltype_markers_v0_2.json")
MARKERS_META = {k: v for k, v in _mk.items() if k.startswith("_") or k in ("meta", "version")}
MARKERS_SIZE_TRIAL = os.path.getsize(os.path.join(T, "iamatlas_celltype_markers_v0_2.json"))

# ── derived helpers ───────────────────────────────────────────────────────────
import math
def H(b):
    if b <= 0 or b >= 1: return 0.0
    return -b*math.log2(b) - (1-b)*math.log2(1-b)
def ceiling(cls, sub="methyl"): return 1.0 / H_MIN_TABLE[cls][SUB_ORDER.index(sub)]
def gauge_tier(A):
    if A < 1.01: return "NORMAL"
    if A < 1.05: return "MARGINAL"
    if A < 1.07: return "DETECTABLE"
    if A < 1.10: return "URGENT"
    return "FLOOR BREACH"
def age_band(cls, age):
    rows = AGE_REF[cls]; e = min(rows, key=lambda r: abs(r["age_midpoint"] - age)); return e

# ═══════════════════════════════════════════════════════════════════════════════
# RECONCILIATION — Issue 002 (April 2026) -> repo HEAD 66f37fe (2026-07-03)
# Each row: what 002 said / what HEAD says / why HEAD is current / evidence
# ═══════════════════════════════════════════════════════════════════════════════
RECON = [
 ("H1", "40-cell H_min table (8 classes x 5 substrates)", "as printed 002 pp.7,105",
  "byte-identical in cpg_gauge_engine.py H_MIN_TABLE", "unchanged", "G-002/G-003b freeze 2026-04-06; diffed 2026-09-19: 40/40 equal"),
 ("A1", "Where healthy sits", "A ~ 1.0 with an age curve (AGE_REF, 7 points/class)",
  "A = 1.0 is the COMMITMENT LINE, not healthy. Healthy = age_reference_matrix.json p10-p90 band (10 decades/class, n per decade, source per decade)",
  "engine docstring; Issue 002 §'NOT forced to 1.0'", "cpg_gauge_engine.py lines 26-29; commit 8cdf352 2026-07-01"),
 ("A2", "Age reference values", "AGE_REF immune 40y=0.946, 60y=0.966, 80y=0.992",
  "age_reference_matrix immune 44y=0.9477 [p10 .9021, p90 .9934], 64y=0.9652, 84y=0.9873",
  "HEAD carries percentiles + n + citation per decade; 002 carried means only", "age_reference_matrix.json; Hannum 2013, Horvath 2013, Alisch 2012"),
 ("O1", "Per-class input-scale offset", "not present", "explored (immune/methyl = 0.055, commit 143704d) then RETIRED",
  "it was reinventing the age band and forcing healthy onto 1.0", "cpg_gauge_engine.py lines 46-53"),
 ("T1", "Tier ladder (gauge)", "NORMAL<1.01 MARGINAL<1.05 DETECTABLE<1.07 URGENT<1.10 BREACH>=1.10",
  "identical", "unchanged", "cpg_gauge_engine.py lines 38-44; 002 tier_label()"),
 ("T2", "Tier ladder (Stage 7 customer, tier_breakpoints.json)", "not in 002",
  "SUPPRESSED<0.95 NORMAL<1.01 ELEVATED<1.07 [Warburg line 1.07] SIG_ELEVATED<1.10 BREACH>=1.10",
  "NORMAL->ELEVATED edge moved 1.04->1.01 to match gauge MARGINAL onset (last commit)", "commit 66f37fe 2026-07-03; tier_breakpoints.json _meta.marginal_onset_note"),
 ("T3", "Two tier vocabularies coexist", "one", "gauge says MARGINAL/DETECTABLE/URGENT; Stage 7 says ELEVATED/SIGNIFICANTLY_ELEVATED",
  "OPEN - same breakpoints except the gauge's 1.05 split; one vocabulary must be chosen for the report", "both files at HEAD"),
 ("S1 (LESSON-SCALE-01)", "Substrate floors for cfDNA", "cfDNA listed as a substrate with % contributions (CFDNA_PCT)",
  "'cfDNA is the frag/wps/nucl substrates - score it on those floors, NEVER the methyl floor'",
  "engine docstring", "cpg_gauge_engine.py lines 56-58"),
 ("I1", "Two instruments", "one A-score", "GAUGE = H(beta_mean)/H_min over per-class IDENTITY loci; SEPARATION = mean_i(H(beta_i))/H_min over per-cell DISCRIMINATIVE markers",
  "feeding the gauge discriminative markers produced the all-BREACH bug of 2026-06-11", "cpg_gauge_engine.py lines 16-25; iamatlas_a_scoring.py"),
 ("D1", "Deconvolver role", "not in 002", "Walther NNLS on IAMAtlasREBUILD -> class + cell-type fractions; PRESENCE gate: a cell/class below DETECT_FLOOR is absent and its A is not a reading",
  "cpg_conductor.py 2026-07 'Replaces the confusing walther_clinical.py'", "cpg_conductor.py DETECT_FLOOR = 0.01"),
 ("D2", "Presence floor value", "-", "1% (cpg_conductor.py) vs 3% (README_FOR_FUTURE_AI Mahalanobis adjudicator gate)",
  "OPEN - two floors give different answers on plasma terminal class", "both files at HEAD; run 2026-09-19"),
 ("D3", "NILC second deconvolver", "-", "CUT from Stage 2; Walther alone", "per flowchart", "commit c1be0c3 2026-07-02"),
 ("D4", "Deconvolver validation scope", "-", "N7: MAE 0.0076-0.0093 on SYNTHETIC Dirichlet mixtures; conformance MAE 0.0004 vs TEST_DATA_MANIFEST on 3 real EPIC tissue samples",
  "never validated against real tissue of KNOWN composition; GSE122126 in-vitro mixes are the available ground truth", "N7_OUTCOME.md; run 2026-09-19"),
 ("K1", "Atlas", "not in 002", f"IAMAtlasREBUILD: {N_CPGS:,} CpGs, {N_CELLTYPES} cell types -> 8 classes, built {ATLAS_BUILD}",
  "predecessor IAMAtlas.csv.xz retired (collapsed; flatness bug)", "IAMAtlasREBUILD_provenance.json"),
 ("K2", "Atlas CpG count stated", "-", f"provenance + file: {N_CPGS:,}; SOP v1.3.3 §28: 481,966", "OPEN - SOP figure is stale or counts a filtered set", "diffed 2026-09-19"),
 ("M1", "iamatlas_celltype_markers_v0_2.json", "-", "trial zip 200,311 B vs repo HEAD 237,545 B - DIFFERENT files, same name",
  "OPEN - which is current must be stated; version the filename", "sha256 diff 2026-09-19"),
 ("SOP", "Chain-of-custody SOP", "-", "SOP v1.3/1.3.3 is PRE-atlas; commit d7b0e1f says 'supersede SOP §41'; no post-atlas SOP exists",
  "the commits 2026-06-25..07-03 ARE the spec; this document is the first written one", "git log; README_FOR_FUTURE_AI.md"),
 ("SEC", "Sections 7-8 (detection trajectory, deployment readiness)", "present", "REPLACED by Substrate Characterization + Procedures",
  "user decision 2026-09-19: not a detection tool; the claim is the healthy range of the write process", "this issue"),
]

# ═══════════════════════════════════════════════════════════════════════════════
# RUNS — everything executed 2026-09-19 with the canonical files. Sources named.
# ═══════════════════════════════════════════════════════════════════════════════
CONFORMANCE = {  # PROC-DECON-01: WaltherIAMDeconvolver on betas_cache.pkl vs TEST_DATA_MANIFEST.md
 "GSM8772491": {"label": "high-grade colon adenoma, EPIC", "mae": 0.0004, "maxerr": 0.0005,
                "rows": [("cycling",0.3535,0.354),("immune",0.2530,0.253),("stem_pluri",0.1605,0.161),("secretory",0.1215,0.122),("terminal",0.1115,0.112)]},
 "GSM5065990": {"label": "CRC stage 1, EPIC", "mae": 0.0002, "maxerr": None, "rows": []},
 "GSM5065985": {"label": "CRC stage 4, EPIC", "mae": 0.0002, "maxerr": None, "rows": []},
}
WHOLE_BLOOD = [  # test-data IDATs, Stage-1 calibrated (betas_cache.pkl); deconvolver + immune gauge
 # gsm, label, age, epithelial frac, immune frac, progenitor frac, residual MAE, immune gauge A (methyl, identity loci)
 ("GSM2333901","healthy 58M",58,0.0000,0.8336,0.1280,0.0562,0.8066),
 ("GSM2333905","healthy 67F",67,0.0051,0.8881,0.0671,0.0522,0.8326),
 ("GSM2333950","healthy 43M",43,0.0111,0.9663,0.0000,0.0596,0.8739),
 ("GSM1051533","RA-study control",None,0.0000,0.9318,0.0515,0.0434,0.8924),
 ("GSM1051534","RA-study control",None,0.0000,0.8027,0.1740,0.0504,0.8673),
 ("GSM1051525","RA case",None,0.0000,0.8395,0.1163,0.0452,0.8789),
 ("GSM1051526","RA case",None,0.0000,0.8686,0.1147,0.0459,0.8743),
]
PLASMA_COMPOSITION = {  # GSE122126 (Moss 2018) EPIC plasma cfDNA; deconvolver class fractions; epithelial = cycling+secretory+terminal+stromal
 "healthy":       {"n":4,  "epi_mean":0.030, "resid":0.0575},
 "colon cancer":  {"n":4,  "epi_mean":0.463, "resid":0.155},
 "breast cancer": {"n":3,  "epi_mean":0.466, "resid":0.161},
 "lung cancer":   {"n":4,  "epi_mean":0.010, "resid":0.055},
 "CUP":           {"n":4,  "epi_mean":0.293, "resid":0.116},
 "sepsis":        {"n":22, "epi_median":0.040, "epi_p75":0.233, "epi_max":0.520, "resid":None},
}
PLASMA_NOTE = ("All plasma GAUGE A-scores computed 2026-09-19 were on the METHYL floor and are therefore out of spec "
               "(RECON S1). Only the COMPOSITION results are reported here. The GSE122126 in-vitro mixes (9+5 samples of "
               "known genomic-DNA proportion) are the available real-data ground truth for the deconvolver and have not yet been scored.")
CELL_ROUTING = ("Cell-level calls on shed epithelium route to gastric references (Fundus_diff, Antrum_diff) in colon tissue, "
                "colon plasma and breast plasma alike; Colon_epithelial_cells and Breast return 0.0000 despite being in the atlas. "
                "LESSON-DECONV-01 attributes this to reference mismatch, not solver conditioning; hierarchical refinement was tested, "
                "invents cells, and does not ship. Cell tier is INDICATIVE. No organ-level claim is supported by either channel today.")
CHK31 = [  # extreme-beta integrity check on Stage-1-calibrated test data: % <0.05 or >0.95, % in 0.4-0.6
 ("GSM2333901","whole blood 450K",33.8,4.5),("GSM1051533","whole blood 450K",27.3,5.0),
 ("GSM8772491","colon adenoma EPIC tissue",6.7,5.5),("GSM8772492","colon adenoma EPIC tissue",8.5,6.4),("GSM5065990","CRC EPIC tissue",23.6,11.7),
]

# ═══════════════════════════════════════════════════════════════════════════════
# LESSONS AS OPERATING RULES (dated; supersession stated)
# ═══════════════════════════════════════════════════════════════════════════════
RULES = [
 ("L-5", "Physics measures, cohorts only point.", "The A-score is intrinsic and self-calibrating. Cohort data enters only as a DIRECTION, never as a baseline. The moment a cohort mean/SD becomes the yardstick the model works only in its own cohort. Primary readout = absolute A per sample against the fixed reference and the age band; group statistics are supporting detail.", "LESSONS_collapse_and_KISS_v1", "current"),
 ("CCL-019/020", "Direction depends on (class, compartment), not disease.", "Specimen, class and panel are three independent dimensions and all three must match the clinical question. Discovered when VAL-061 predicted the wrong sign.", "LESSONS_LEARNED.md", "current"),
 ("CCL-039", "Marker-tile and full-genome A are two observables.", "They move in opposite directions in tumour vs adjacent normal (READ, COAD, COAD-VAL-099). Never compare one to the other.", "LESSONS_LEARNED.md", "current"),
 ("CCL-032 / CHK-3.1", "Extreme-beta integrity gate is substrate-dependent.", "Whole blood 450K reads 26-34% extreme; Stage-1-calibrated EPIC tissue reads 7-24%. The >30% criterion does not transfer to tissue. Thresholds must be stated per substrate.", "LESSONS_LEARNED.md; run 2026-09-19", "current, amended"),
 ("LESSON-DECONV-01", "Global NNLS is correct; sparsity is reference mismatch.", "Synthetic mixtures recover at MAE 0.000; real blood fits 2.6x worse. Hierarchical within-class refinement invents cells (MAE 0.271 on pure monocytes) and does not ship. Richer per-cell resolution is atlas work, not a solver swap.", "SOP v1.3.3 §0.9 (2026-06-21)", "current"),
 ("2026-06-11", "Never feed the gauge discriminative markers.", "They are bimodal; their mean beta collapses toward 0.5 and the gauge pins the ceiling (all-BREACH).", "cpg_gauge_engine.py", "current"),
 ("PRESENCE", "A class not in the sample has no A-score.", "Presence comes from the deconvolved fraction (DETECT_FLOOR). Scoring an absent class's identity loci produced spurious BREACH calls on stromal (fraction 0.0000) in 4/4 healthy plasma samples, 2026-09-19.", "cpg_conductor.py; run 2026-09-19", "current"),
 ("SUBSTRATE", "Every report states its substrate; every (class, substrate) has its own floor.", "Whole blood = immune-architecture readout; plasma cfDNA = shed-tissue readout on frag/wps/nucl floors; tissue = positive control.", "cpg_gauge_engine.py; TEST_DATA_MANIFEST.md", "current"),
 ("glioma-LL-002", "A high A on bulk tissue is a heterogeneity marker, not a tumour marker.", "Composition alone can move beta-bar. N-comp (report immune-class A alongside the target class) is the binding null for any tissue claim.", "LESSONS_LEARNED.md", "current"),
 ("PL-002", "The honest record is the best defense.", "Amendments after data are observed are labelled as such; outcomes are allowed to be RESTATE, FAIL or VOID.", "LESSONS_LEARNED.md", "current"),
]

# ═══════════════════════════════════════════════════════════════════════════════
# FALSIFICATION RECORD — outcomes that did not go the framework's way, kept on the record
# ═══════════════════════════════════════════════════════════════════════════════
FALSIFICATION = [
 ("VAL-004", "bimodality direction reversed vs prereg", "RESTATE"),
 ("VAL-006", "chr6 signal did not survive look-elsewhere correction", "FAIL"),
 ("VAL-061", "predicted wrong sign; produced CCL-019/020", "RESTATE"),
 ("VAL-102", "voided four minutes after sealing for post-hoc accommodation; original seal preserved", "VOID"),
 ("VAL-128", "failed opposite to prereg", "FAIL"),
 ("CPG-NEW-001 P1", "GSE48684 adenoma read above carcinoma; monotonic progression failed", "FAIL"),
 ("CPG-NEW-001 P4", "serrated vs conventional reversed sign between Spain and Finland", "FAIL"),
 ("CPG-NEW-001 N-comp", "cycling effect not larger than immune effect on adenoma contrasts", "composition-ambiguous"),
 ("2026-09-19 two-channel", "immune-A x epithelial-fraction did not beat epithelial alone (AUC .955 vs .961; LOO .793 vs .828)", "prediction FAILED"),
 ("2026-09-19 WB x immune", "3/3 healthy whole-blood donors read BELOW the age band (z = -2.07, -4.03, -3.49)", "OPEN - scale or biology"),
]

# ═══════════════════════════════════════════════════════════════════════════════
# SUBSTRATE x CLASS GRID — status of every cell of the grid
# ═══════════════════════════════════════════════════════════════════════════════
SUBSTRATES_GRID = ["whole blood","plasma cfDNA","tissue","urine","CSF","stool"]
GRID_STATUS = {  # (substrate, class): (status, note)
 ("whole blood","immune"):      ("MEASURED n=7", "3/3 healthy BELOW age band; RA (n=2) inside non-RA range on both channels"),
 ("whole blood","progenitor"):  ("MEASURED n=7", "fractions 0-0.17; CHANGELOG flags breach reads as suspected artifact"),
 ("whole blood","stem_adult"):  ("MEASURED n=7", "same artifact flag as progenitor"),
 ("whole blood","cycling"):     ("ABSENT by biology", "fraction 0.000-0.011"),
 ("whole blood","secretory"):   ("ABSENT by biology", "fraction ~0"),
 ("whole blood","terminal"):    ("ABSENT by biology", "fraction ~0"),
 ("whole blood","stromal"):     ("ABSENT by biology", "fraction 0.0000"),
 ("whole blood","stem_pluri"):  ("SATURATED", "methyl ceiling 1.018"),
 ("plasma cfDNA","immune"):     ("COMPOSITION ONLY", "gauge must use frag/wps/nucl floors (S1); methyl-floor A withdrawn"),
 ("plasma cfDNA","cycling"):    ("COMPOSITION ONLY", "epithelial sum 0.03 healthy -> 0.46 colon/breast; 0.01 lung"),
 ("plasma cfDNA","secretory"):  ("COMPOSITION ONLY", "as above"),
 ("plasma cfDNA","terminal"):   ("COMPOSITION ONLY", "present at 1-3% in 2/4 healthy - presence floor decides"),
 ("plasma cfDNA","stromal"):    ("ABSENT", "0.0000 in 33/33"),
 ("tissue","cycling"):          ("CONFORMANT", "MAE 0.0004 vs manifest"),
 ("tissue","secretory"):        ("CONFORMANT", "MAE 0.0004 vs manifest"),
 ("tissue","immune"):           ("CONFORMANT", "infiltrate; N-comp binding"),
 ("tissue","stromal"):          ("STRUCTURAL ZERO", "1,441/7,090 marker coverage, 5x thinner than other classes"),
 ("tissue","progenitor"):       ("STRUCTURAL ZERO", "0.0000 in 4/4 colorectal"),
 ("urine","*"):                 ("DATA IN HAND", "GSE119260: 4 prostate patients x tissue/plasma/urine sediment; not yet run"),
 ("CSF","*"):                   ("DATA LOCATED", "GSE292312 (24 CSF + 157 tumour), GSE269403 (39 CSF + 17 paired blood); supplementary files, not yet fetched"),
 ("stool","*"):                 ("NO PUBLIC DATA", "commercial assays are targeted PCR; requires primary collection"),
}

# ═══════════════════════════════════════════════════════════════════════════════
# PROC-PLASMA-MIX-01 — real-data ground truth, run 2026-09-19 after the first draft
# Moss 2018 Supplementary Data 1 Table 6 (9 genomic-DNA mixes into one donor's leukocytes) and Table 8
# (colon-cancer cfDNA CC2 spiked into healthy cfDNA). GEO titles in_vitro_mix_9..17 mapped to Mix1..9 by
# order; the mapping is corroborated by the max-terminal sample landing on Mix2 (10% neurons) and the
# max-cycling sample on Mix4 (10% colon) as Table 6 predicts.
# ═══════════════════════════════════════════════════════════════════════════════
MIX_T6 = [  # gsm, Mix, declared leuk, hep, lung, neur, colon | observed non-haem, secretory, terminal, cycling, resid
 ("GSM3455853",1,.865,.10,.035,0,0,     .207,.017,.035,.090,.0483),
 ("GSM3455854",2,.85,.05,0,.10,0,       .209,.000,.065,.073,.0467),
 ("GSM3455855",3,.915,.035,0,0,.05,     .164,.007,.024,.088,.0433),
 ("GSM3455856",4,.85,0,0,.05,.10,       .225,.020,.044,.103,.0481),
 ("GSM3455857",5,.915,0,.05,0,.035,     .158,.006,.021,.093,.0422),
 ("GSM3455858",6,.865,0,.10,.035,0,     .199,.013,.036,.097,.0451),
 ("GSM3455861",7,.94,0,0,0,.06,         .099,.000,.016,.053,.0406),
 ("GSM3455860",8,.92,0,.08,0,0,         .141,.000,.022,.087,.0428),
 ("GSM3455859",9,.96,.04,0,0,0,         .128,.004,.019,.078,.0428),
]
MIX_STATS = {"total_r":0.945,"total_p":0.000,"total_mae":0.0677,"total_bias":+0.0677,
             "neur_terminal_r":+0.945,"neur_terminal_p":0.000,"hep_secretory_r":+0.192,"hep_secretory_p":0.620,
             "colon_cycling_r":+0.101,"colon_cycling_p":0.796}
MIX_T8 = [("GSM3455845","cfDNA_mix_1","H2",0.10,0.130),("GSM3455847","cfDNA_mix_3","H2",0.03,0.079),
          ("GSM3455837","cfDNA_mix_5","H3",0.05,0.130),("GSM3455838","cfDNA_mix_6","H3",0.00,0.073),("GSM3455839","cfDNA_mix_7","H4",0.05,0.073)]
MIX_T8_R = (+0.725, 0.166)

# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE MAP — every stage of the running chain (flowchart_vKISS + CHANGELOG + file docstrings at HEAD)
# and whether THIS ISSUE covers it. Written so the reader can see what is not here.
# ═══════════════════════════════════════════════════════════════════════════════
ENGINE_MAP = [  # stage, files, status at HEAD, what it does (from source), covered in Issue 003?
 ("Stage 0 — Intake", "stage_0_intake.py, cpg_intake_form.html, questionnaire.json", "BUILT", "patient folder, questionnaire, substrate declaration", "NO"),
 ("Stage 1 — Calibration", "stage_1_idat_calibration.py, stage_1_calibration.py, idat_parse.py, idat_decoder_pure.py", "BUILT", "raw IDAT pair -> noob-normalised beta (methylprep 1.7.1; pure-python IDAT decoder shipped)", "PROC-CAL-01 written, NOT RUN"),
 ("Stage 2 — Deconvolution", "Walther_iam_deconvolver/walther_iam_deconvolver.py, IAM_Atlas/*", "BUILT (NILC cut 2026-07-02)", "NNLS class + cell fractions; composition/presence; gates no call", "YES — §2.4, §3.2, PROC-DECON-01, PROC-PLASMA-MIX-01"),
 ("Stage 3 — Foreground subtraction", "—", "NOT BUILT (flowchart)", "age / sex / smoking foregrounds; sex handled instead by chrX marker removal 2026-06-11; age by the reference band", "YES as a RECON row (F1); nothing to run"),
 ("Stage 4 — A-score gauge", "cpg_gauge_engine.py, Runtime Matrices/A_Scoring_Module/*, test_a_score_canonical.py", "BUILT", "H(beta_mean)/H_min over identity loci; placement vs age band; severity ladder; brightness CI", "YES — §3; brightness CI NOT described"),
 ("Stage 4.5 — Bidirectional decomposition", "Runtime Matrices/Directional Panel/bidirectional_decomposition.py, directional_panels_v1_0.json", "BUILT (wired 2026-06-27)", "composition-independent directional composite; AD detector (sealed VAL-051 Rule A, 7-CpG immune panel); gate composite > 0.40; AIBL-trained, does not transfer to GIFT", "NO"),
 ("Stage 4.6 — Patient CMB", "cpg_patient_cmb.py, Runtime Matrices/cpg healpix mapping/*", "BUILT (2026-06-29)", "per-class departure z=(beta-mu)/sd_class on a HEALPix sky; absent-tissue panels self-masked", "NO"),
 ("Stage 5 — Second chain", "stage_5_second_chain.py, Mahalanobis_healthy_reference/*, Literature_anchors_Report building/literature_anchors.json", "BUILT", "fires only on a flag; Mahalanobis adjudicator Option A (age-matched class gauge, presence gate >=3% AND outside [0.95,1.04)); RUN-everything residual matched-filter sweep (breast + immune-alarm maps; AD removed); literature anchors", "NO — only the 3% floor is mentioned (RECON D2)"),
 ("Stage 6 — Cellular age", "iam_cellular_age_scoring.py (in CPG_TRIAL_CODE; NOT in repo engine)", "BUILT 2026-06-30, calibration pending", "cellular age from class A vs age_reference_matrix", "NO — and the file is not in the repository (RECON F2)"),
 ("Stage 7 — Tier", "Runtime Matrices/Tier_breakpoints/tier_breakpoints.json", "BUILT (edge 1.04->1.01, 2026-07-03)", "continuous A -> customer tiers", "YES — §3.1, RECON T1-T3"),
 ("Stage 8 — Disease matching", "Disease Matrix/disease_cell_signature_matrix_v1_13.csv (81 rows, 49 VAL-anchored), disease_origin_cells.json, iamatlas_115_to_matrix_v0_2_mapping.json, Collinearity_Groups/*", "BUILT", "Route B directional concordance (weighted matcher over SIGNAL cells |d|>=0.20; STRONG needs dc>=0.70, coverage>=0.40, >=3 cells); Mode 2 cell-of-origin presence; Mode 3 systemic-stress wellness read (never a disease call); specificity gate (NLR axis = NON_SPECIFIC_GENERIC). Patient-match loop verified on 381 breast + 142 colon (VAL-093)", "NO"),
 ("Stage 9 — Report", "cpg_report_builder.py, cpg_report_builder_v2.py, build_dashboard_v1.py, cpg_gauge.py (Appendix A1 gauge), Crown Jewel and Patient Strawman/*", "BUILT", "clinician report; patient straw man on the eight-class grid; crown-jewel reference wall; dashboard", "NO"),
 ("Orchestration", "cpg_conductor.py (2026-07, replaces walther_clinical.py), walther_clinical.py (still the wired chain), run_batch.py, preflight.py, bootstrap.sh", "BOTH present", "conductor = Stage A pure functions; walther_clinical = the full wired chain the report builders still call", "PARTIAL — conductor's presence rule only. walther_clinical.py origin-gate fail-open (bare except disables specificity rule) NOT in this issue"),
 ("Test data", "TEST_DATA/TEST_DATA_MANIFEST.md, harness/*, N7", "present", "11 public IDATs with expected outputs; synthetic harness; N7 chain-integrity", "YES — §2.4, PROC-DECON-01"),
]
RECON_EXTRA = [
 ("F1", "Stage 3 foreground subtraction", "not in 002", "NOT BUILT (flowchart_vKISS). Sex: 131 chrX markers removed 2026-06-11 (derived invariance). Age: handled by the reference band, not subtraction. Smoking: nothing.",
  "no code exists; every A in this issue is un-subtracted for smoking", "flowchart_vKISS.html; iamatlas_celltype_markers_v0_2.json _sex_marker_removal"),
 ("F2", "Stage 6 cellular age module", "not in 002", "iam_cellular_age_scoring.py present in CPG_TRIAL_CODE.zip, ABSENT from repo engine directory",
  "OPEN — the canonical repo does not carry a file the trial bundle does", "sha256 listing 2026-09-19"),
 ("F3", "walther_clinical.py origin gate", "not in 002", "disease_origin_cells.json loaded inside a bare except; on failure the cell-of-origin specificity rule is silently disabled",
  "OPEN — patient-safety gate fails open; three-line fix", "walther_clinical.py ~lines 972-996 (CPG_first_read.md finding 3)"),
 ("F4", "Stale strings describing the retired mean-of-entropies gauge", "—", "cpg_conductor.py Stage A comment '(mean-of-per-CpG H/H_min)'; cpg_gauge.py line 158 axis label prints on the patient report figure",
  "OPEN — neither changes a number; both are how the 2026-06-11 bug gets reintroduced", "CPG_first_read.md finding 2"),
 ("F5", "Gauge carries no sign", "—", "A(beta)=A(1-beta); INVERSION below band is degenerate between hyper- and hypo-methylation. beta_mean is computed one line before A and not reported",
  "OPEN — report beta_mean or signed departure beside every A; no revalidation needed", "iamatlas_a_scoring.py; verified A(0.61)=A(0.39)=1.1501 on immune"),
 ("F6", "Two A-score definitions in the corpus", "002 §2.5 C1/C2/C3", "A = H/H_min (all numerical results) vs A = 1 + C3/H_actual (three-component decomposition) — equal only at A=1",
  "OPEN — one must be retired or the relation stated", "002 §2.5; IAM_for_physicists §4"),
 ("F7", "Immune H_min revision history", "0.838889", "calibration value 0.795000 -> 0.8389 +/- 0.0012 after MCMC reanalysis of six immune cell types; immune A moved down 0.055",
  "state the freeze date and provenance so 'frozen before disease testing' is checkable", "GAPE Issue 001/002 sources"),
]
RECON += RECON_EXTRA

# grid updates from PROC-PLASMA-MIX-01
GRID_STATUS[("plasma cfDNA","terminal")]  = ("MEASURED ground truth", "neuron spike recovered r=0.945, under-read (10% -> 6.5%)")
GRID_STATUS[("plasma cfDNA","secretory")] = ("FAILED ground truth", "10% hepatocyte spike -> 1.7% secretory; r=0.19")
GRID_STATUS[("plasma cfDNA","cycling")]   = ("FAILED ground truth", "colon spike not recovered (r=0.10); baseline cycling 5-10% in leukocyte-only mixes")
FALSIFICATION += [
 ("PROC-PLASMA-MIX-01 secretory", "10% hepatocyte genomic DNA in leukocytes reads 1.7% secretory (Moss Table 6 ground truth)", "FAIL"),
 ("PROC-PLASMA-MIX-01 cycling", "colon spike 0-10% not recovered; cycling reads 5-10% regardless (r=0.10)", "FAIL"),
]

# ═══════════════════════════════════════════════════════════════════════════════
# PROC-FORMULA-01 — the A-score formula conflict at HEAD, settled empirically 2026-09-19
#   iamatlas_a_scoring.py + test_a_score_canonical.py (2f758ba 06-30): A = mean_i H(beta_i)/H_min, "any other build is wrong by definition"
#   cpg_gauge_engine.py + walther_clinical.stage_4 (536c0e9/d7b0e1f 07-01): A = H(beta_mean)/H_min over IDENTITY loci, "supersedes SOP §41"
# All four (formula x loci) combinations computed on the 11 Stage-1-calibrated TEST_DATA samples.
# ═══════════════════════════════════════════════════════════════════════════════
_f2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Testing_and_Code", "PROC_data", "formula_2x2.json")  # PROC-FORMULA-01 data, in the repo
FORMULA_2X2 = json.load(open(_f2)) if os.path.exists(_f2) else {}
FORMULA_LABELS = {"GSM2333901":"WB healthy 58M","GSM2333905":"WB healthy 67F","GSM2333950":"WB healthy 43M","GSM1051525":"WB RA","GSM1051526":"WB RA",
     "GSM1051533":"WB RA-study ctrl","GSM1051534":"WB RA-study ctrl","GSM8772491":"tissue adenoma","GSM8772492":"tissue adenoma","GSM5065990":"tissue CRC st1","GSM5065985":"tissue CRC st4"}
FORMULA_FINDINGS = [
 ("Whole blood, immune class present - identity loci", "H(beta_mean) and mean-of-H agree in RANK exactly (Spearman +1.000 over 7 donors) with a constant Jensen offset of +0.029 +/- 0.002. On the declared substrate with the class present, the formula choice is a calibration convention, not a physics question."),
 ("Whole blood, cycling/secretory - classes ABSENT from blood", "offset +0.16: the absent class's identity loci sit locked/bimodal in immune DNA and H(beta_mean) inflates toward the ceiling. This is the SOP v1.4.0 s105 mechanism, and it is exactly what the presence gate (s3.2) prevents from being read."),
 ("Bulk tissue - a mixture", "offset +0.24 +/- 0.07 (immune). The 'disease ordering' first read off H(beta_mean) on tissue (adenoma 0.98-1.10, CRC-4 1.19) is COMPOSITION INFLATION, not architecture - glioma-LL-002 already says a high A on bulk tissue is a heterogeneity marker. This corrects the verdict written earlier on 2026-09-19."),
 ("age_reference_matrix.json", "A_mean = H(beta_mean)/H_min - exact to 5 decimals in all 80 (class x decade) cells; the beta_means are TYPED literature values (April HEALTHY_BASELINES; PROC-RECORD-03), not measured on any surface. The runtime gauge MUST use the same aggregation as the band it is read against; v1.4.0's mean-of-H against this band reads every patient ~0.03 low."),
 ("Discriminative markers, mean of H  [iamatlas_a_scoring; sealed-anchor formula]", "immune flat 0.60-0.70 across blood and tissue; cycling WB 0.35-0.41 vs tissue 0.57-0.62. Moves with PRESENCE of the class's DNA - the separation surface, which is why it reproduces the sealed GSE51032 anchor (d = +2.088) and why CCL-019 found its sign depends on compartment."),
]
FORMULA_VERDICT = ("SOP v1.4.0 (2026-06-30, current) says the A-score is NEVER H(beta_mean); the wired chain at HEAD (d7b0e1f, 2026-07-01) computes exactly H(beta_mean) over identity loci; SOP v1.3.3 carries the same date as v1.4.0 and the opposite formula. "
  "Measured: on the declared substrate with the class present the two aggregations differ by a constant (+0.029), so either is a valid gauge PROVIDED the age band is compiled the same way - and the band is H(beta_mean). "
  "On any mixed or absent-class panel H(beta_mean) inflates (s105 is right about the mechanism) and must never be read as architecture; the presence gate and glioma-LL-002 are the guards. The sealed anchors are separation-surface results and are reproduced by mean-of-H over discriminative markers, as v1.4.0 states. "
  "The decision was written the same day as RULING A3 (s1.5, 2026-09-19, under delegated authority; the author may overrule): one aggregation per surface, fixed by the reference each was built with - the gauge on identity loci is H(beta_mean)/H_min, matching H_min's own definition, and v1.4.0 s105's NEVER is scoped to mixed or absent panels; the separation surface keeps mean-of-H, as its sealed anchors require. Running one formula against the other's band remains the error. [The age_reference_matrix named here was later superseded by band_v2 + the three-layer reference, s3.5.]")
RECON += [
 ("A3", "A-score formula: SOP vs code", "A = H(beta)/H_min, beta = the cell's mean methylation; H_min_global = H(0.790)",
  "SOP v1.4.0 (2026-06-30, current per author): A = mean_i H(beta_i)/H_min, 'NEVER H(beta_mean)', s105 LESSON-ASCORE-02, guarded in iamatlas_a_scoring.py. SOP v1.3.3 (same date): GAUGE = H(beta_mean) over identity loci. Code at HEAD (d7b0e1f 2026-07-01): H(beta_mean) over identity loci via cpg_gauge_engine, 'supersedes s41'. age_reference_matrix compiled as H(beta_mean).",
  "OPEN - the current SOP and the running code disagree. Measured (PROC-FORMULA-01): on blood with class present the two differ by a constant +0.029 (rank identical); on mixed/absent panels H(beta_mean) inflates as s105 says. Decision needed: keep H(beta_mean) + scope s105, or adopt mean-of-H + recompile the band.", "SOP v1.4.0 s41-43, s105; git d7b0e1f; 2x2 run 2026-09-19"),
 ("M1b", "iamatlas_celltype_markers_v0_2.json - which is current", "-",
  "TRIAL file (11,369 markers; carries _sex_marker_removal: 131 chrX dropped 2026-06-11; 83/115 cells identical to HEAD) is the corrected one. REPO HEAD (11,500 markers) does NOT carry the sex-marker removal.",
  "the repository is stale on this file - the derived sex-invariance fix never reached it; commit the trial version", "diff 2026-09-19"),
]
FALSIFICATION += [
 ("Analyst verdict 2026-09-19 (first draft of PROC-FORMULA-01)", "'identity-loci H(beta_mean) carries the disease ordering healthy < adenoma < carcinoma' - the tissue ordering is composition inflation (+0.24 Jensen offset on a mixture), not architecture; retracted the same day when the whole-blood Spearman came back +1.000 for both formulas", "RETRACTED"),
 ("SOP v1.4.0 s105 universal claim", "'the A-score is NEVER H(beta_mean)' - on the declared substrate with the class present the two aggregations are rank-identical with a constant offset, and the shipped age band is compiled as H(beta_mean); the NEVER is correct for mixed/absent panels only", "OVER-SCOPED"),
]

# SOP v1.4.0 s105 read directly 2026-09-19 - the primary source for the aggregation conflict
S105 = {
 "one_sentence": "A = mean_i( H(beta_i)/H_min(class) ) - score each marker locus against its class floor, then average. Never H(beta_mean)/H_min.",
 "mechanism": "Jensen: H is concave so H(mean beta) >= mean H(beta); on a bimodal marker panel averaging beta first manufactures ~0.5 which entropy reads as maximal disorder. Worked guard GSM1235534 HSC panel: 11 loci <0.2, 24 >0.8, beta_mean 0.600 -> H(beta_mean)/0.874 = 1.111 false BREACH; mean_i H(beta_i)/H_min = 0.864 healthy.",
 "anchor": "GSE51032: 115/115 per-cell A vs sealed csv corr 1.0000 maxdiff 0.0000; 460/460 Mahalanobis; case(36) vs hc(424) d = +2.088, CI [1.502, 2.735] - reproduced with the vault mean-of-H module on the v0_2 DISCRIMINATIVE panel (healthy A ~ 0.52).",
 "open_decisions": [
   "1. Recipe s6.3 is the upstream source of the 'H of the mean is canonical' ruling and must be corrected at the vault source (Recipe is vault IP; not edited from the SOP).",
   "2. Marker convention is a separate baseline-placement choice: v0_2 discriminative (healthy A ~0.52, the +2.088 anchor) vs most-methylated loci (healthy A ~1.0). Either coherent under mean-of-H; tier breakpoints must match whichever is chosen; must be re-validated against +2.088 before production.",
   "3. s46.5 bidirectional recovery assumed the primary A cancels bidirectional signal - true only under H(beta_mean); needs re-derivation.",
 ],
 "what_code_did_next": "d7b0e1f (2026-07-01) chose a convention s105 does not list - IDENTITY loci (healthy immune A ~0.89 on the age band) - AND the H(beta_mean) aggregation s105 forbids, and compiled age_reference_matrix.json with it. None of s105's three open decisions is recorded as closed anywhere in the repo.",
 "tension": "s105 grounds mean-of-H in 'H_min is a per-locus floor'. The eight H_min values were calibrated (G-002) on per-CELL-TYPE mean beta from Roadmap/ENCODE (e.g. neuron 0.782, H_min_global = H(0.790) in Issue 002), i.e. on H of a mean beta. The per-locus reading of H_min is an interpretation added in s105, not the quantity the MCMC fit. Measured consequence: on identity loci (selected near H_min_beta, hence not bimodal) the two aggregations differ by a constant +0.029; on discriminative markers (bimodal by construction) they differ by ~0.35.",
}
RECON += [
 ("A3b", "SOP v1.4.0 s105 open decisions vs the code", "-",
  "s105 leaves three decisions open (Recipe s6.3 source; marker convention discriminative-vs-most-methylated; s46.5 re-derivation). The 07-01 code answered none of them and introduced a fourth convention (identity loci + H(beta_mean)) with the age band compiled to match.",
  "OPEN - the three s105 decisions plus the identity-loci convention need one written ruling; until then SOP and code cannot both be 'current'", "SOP v1.4.0 s105; d7b0e1f"),
]

# ═══════════════════════════════════════════════════════════════════════════════
# Content recovered by the line-by-line pass over eight source documents, 2026-09-19
# Sources: GAPE_EDEAR_Reproduction_Paper_v3.md (EDEAR), CPG_Chain_of_Custody_SOP_v1_4_0.md (SOP), The_Cellular_Margin.md,
# What_Is_Astro_Genetics.tex, CPG_Comprehensive_Report_v2.md (CR2), IAM_Hubble2GAPE_Alpha_Omega_4.tex (H2G), gape_derivation_tests.py
# ═══════════════════════════════════════════════════════════════════════════════
MCMC_HMIN = [  # G-002 / G-003b sampler settings (EDEAR s5)
 ("Sampler", "emcee affine-invariant ensemble"), ("N_WALKERS", "32"), ("N_BURNIN", "2,000 steps"), ("N_PRODUCTION", "5,000 steps"),
 ("N_CHAINS", "5 independent (for R-hat)"), ("THIN", "10"), ("PRIOR_SIGMA", "0.05 (prior width around the published central value; walkers initialised at mean + 0.1 x PRIOR_SIGMA x N(0,1))"), ("SIGMA_A", "0.03 (likelihood tolerance on (A - 1)^2)"), ("SEED_BASE", "20260420"),
 ("Acceptance window", "0.20-0.50 (step-size tuning; <0.1 too large, >0.7 too small)"), ("Autocorrelation time", "tau ~ 50 -> ~16,000 effective samples per class"),
 ("Posterior draws per class", "(32 x 5,000 x 5)/10 = 80,000"), ("Likelihood", "Gaussian on (A_i - 1.0)^2 / SIGMA_A^2 per reference cell"),
 ("Prior", "Gaussian centred on the published calibration value"), ("Cross-check", "leave-one-out bootstrap, n = 10,000 resamples; MCMC and bootstrap must agree within 5-10% or the cohort is audited. RUN in April for the 32 G-003b floors only (0.168%, 24/32 in CI); run for the 8 methylation floors on 2026-09-20 - PROC-HMIN-BOOT-01: 0.060% mean, 0.095% max, 8/8 in CI"),
 ("Convergence", "17 methylation chains R-hat < 1.001 (G-002); 5 x 32-walker ensembles per substrate (G-003b)"),
 ("Runtime", "G-002 29.7 s for 8 classes (laptop); G-003b ~24 min for 32 posteriors (desktop)"),
 ("Notable posterior shift", "immune: 0.795 (neutrophil-based calibration) -> 0.838889 after MCMC over six immune cell types, a 6.44 sigma move (EDEAR s4.6); every immune A in the database was revised downward by approximately 0.055 as a consequence (Issue 002, immune card)"),
 ("H_min_global", "0.756499 = H(0.782), frontal cortex neuron, Lister 2013 (E073) - the universal reference the class floors are read against"),
 ("Reference-cohort rule", "FACS-sorted or laser-microdissected only; never bulk tissue (cell-type purity)"),
]
ATLAS_BUILD_DETAIL = [  # EDEAR s5.4-5.6, SOP s26-29
 ("Source rows", "10,938,662 (iamatlas_mcmc_inputs.csv, 732 MB) from Moss 2018, Loyfer 2023, EpiDISH, Salas, Lister 2013 and others, harmonised to one cell-type label set"),
 ("Chains per cell type", "2"), ("SD tightness target", "posterior SD < 0.10 for >= 90% of CpGs per cell type"),
 ("Informative-CpG selection", "posterior SD <= 0.10 AND between-cell-type variance >= 0.02"),
 ("Array bridging", "only HM450-overlapping CpGs retained in production (483,092)"),
 ("Failure recorded", "stromal v0.1 chains failed to converge (R-hat 3.67) until source-specific labels were harmonised -> v0.2"),
 ("Deconvolver weighting", "posterior-SD-weighted NNLS: W = diag(1/max(posterior SD)), floor 1e-3; minimum 100 usable CpGs; dispersion penalty ceiling 0.20, high-dispersion flag 0.30; streaming 1K-marker chunks, re-solve every 10"),
 ("Marker selection", "one-vs-rest top-N (N = 100) on |target mean - mean(others)| per cell type (SOP s29); 131 chrX markers removed 2026-06-11 for derived sex-invariance"),
 ("Walther vs NILC", "substrate-level L1 disagreement of ~0.10-0.25 was EXPECTED and did not gate; inference-level disagreement on the disease-relevant class did (SOP s32-33). NILC cut 2026-07-02."),
 ("Versioning rule", "the atlas evolves by deepening (new source rows, new SHAs) not by product versions; every consumer reads H_min from IAMAtlasREBUILD_provenance.json (SOP s99) - hardcoding H_min refuses deployment"),
]
STAGE01_QC = [  # SOP s11-25
 ("Manifest", "raw IDAT pair keyed to demographics and array substrate before any processing (the Planck 'scan-ring' association); SHA-256 on every file"),
 ("Detection p-value", ">5% probes failed = HARD reject; 1-5% = SOFT flag (s15)"), ("Bead count", ">=3 per probe (Illumina default); >=5 for high-stakes (s16)"),
 ("Call rate", ">=0.98 for Stage 0 PROCEED (s19); <0.95 after QC = quarantine"), ("Sex check", "predicted vs declared sex; mismatch = sample swap or rare karyotype - halt (s18; Planck: pointing-direction sanity check)"),
 ("Dye bias", "noob (s20; Planck: bandpass correction)"), ("Probe-type normalisation", "funnorm default (s21)"), ("Batch", "ComBat requires >=3 samples per batch (s22)"),
 ("Bisulfite efficiency", "<0.95 HARD quarantine; 0.95-0.98 SOFT (s23); low BS efficiency + low intensity = degraded DNA"),
 ("Beta sanity", "range, bimodality coefficient 0.45-0.55 SOFT flag, cohort FLAG rate >5% = HARD halt (s25)"),
]
HULL = [  # SOP s48-50, CR2 s2.1
 ("What it is", "Mahalanobis distance of the patient's 115-cell A-vector from a healthy-control hull; Stage 5 adjudicator (Option A, age-matched class gauge)"),
 ("Hull v0_1", "n = 601 HC; fixed threshold d >= 2.0 - mathematically indefensible for 112-dimensional data (expected median under normality is sqrt(112) ~ 10.6)"),
 ("Hull v0_5 (production)", "n = 2,523 HC across 8 cohorts (CR2: 2,481); p95 default d >= 13.62; p99 strict d >= 18.43"),
 ("Anchor trade-off", "GSE51032 case-vs-HC Cohen's d +2.088 (v0_1) -> +1.450 (v0_5): the honest cost of a hull that represents more populations and platforms"),
 ("Transfer", "Han Chinese n = 42 (GSE141682) median d = 10.51, no systematic offset (small n); GIFT HC n = 193 median d = 12.18 - FTD-context HC selection broadens the envelope into case range"),
 ("Guards", "negative quadratic form = corrupted reference, halt; >15 imputed cell types = INSUFFICIENT_DATA (hard), 6-15 = PARTIAL_DATA (soft)"),
 ("Status in this issue", "NOT executed; specified here from SOP v1.4.0 as the specification for PROC-HULL-01 (open row: Stage 5 is driven by the stem_adult false alarm and waits on the lab zero; see CHAIN_COMMISSIONING)"),
]
NULLS = [  # SOP s80-91
 ("N1 label permutation", "observed statistic outside the 95% CI of the permuted distribution"), ("N2 age-stratified permutation", "shuffle within age decades; percentiles, not Gaussian"),
 ("N3 sex-stratified permutation", "as N2 by sex"), ("N4 cohort-split replication", ">=80% of 100 random splits PASS (same sign, CI overlap)"),
 ("N5 plate/position", "position-stratified permutation AND position-amplitude regression both p<0.05 - catches edge effects, scanner drift"),
 ("N6 injection-recovery", "recover an injected effect within +/-20%; recovery of opposite sign = HARD halt and full audit"),
 ("N7 end-to-end synthetic", "the gold standard - a chain that fails N7 is unverified; synthetic patient generator = the methylome's FFP10/NPIPE"),
 ("N8 search-space correction", "permutation FWE by default (Bonferroni over-conservative for correlated features); depends on N1"),
 ("Sealing", "SEALED (all pass) / RESTATE (one fails, scope limited) / RETRACT (two+ fail). PREREG modified after data observation = auto-RETRACT. Sealing is not transitive across artifact versions."),
]
CHAIN_RULES = [  # SOP s55-79 - downstream stages as specified
 ("Stage 7 tiering", "6 engine tiers (SUPPRESSED, NORMAL, ELEVATED, WARBURG_TRANSITION, SIGNIFICANTLY_ELEVATED, BREACH); DETECTABLE/URGENT/FLOOR_BREACH collapsed to one customer label deliberately; BORDERLINE_TIER when top-two tier probabilities differ by <0.20; per-class ceiling caps the reachable tier (stem_pluri blind to BREACH). Observed: all 1,174 EPIC-Italy patients saturate on >=1 class; only cycling in range for ~50%."),
 ("Stage 7 smoking bins (v1.2 interim)", "floor shifts current 1.10, former 0-5y 1.08, 5-15y 1.07, 15+y 1.05, never 1.04; recency scores 0/0.10/0.30/0.60/1.00"),
 ("Stage 8 matching", "two parallel paths (per-card Boolean rules + matrix lookup) on identical inputs; disease rows are time-phased templates (long_pre_dx, mid, mid_late, near_dx); no single tile fires a card; wrong-sign overlap downgrades, never matches; <80% residual-map coverage -> INSUFFICIENT, fall back to class tiers"),
 ("Stage 8 covariates", "smoking and BMI are within-card covariates (on the causal path) and adjust CONFIDENCE only, never the MATCH/NO_MATCH verdict; never subtracted globally"),
 ("Stage 8 prior", "cancer prior = US SEER age-stratified base rate, not personal risk: cycling 0.055, secretory 0.140, immune 0.020, terminal 0.008, stromal 0.005, stem_adult 0.008, progenitor 0.006"),
 ("Stage 9 anchors", "literature anchors are the ONLY external information in customer output; DOI-keyed, peer-reviewed, never paraphrase"),
 ("Stage 9 legal gate", "DRAFT report scanned against a CANNOT_SAY list (diagnostic determinism, directives, future-state claims, treatment specifics); >=1 hit halts shipping (s76)"),
 ("Audit", "per-patient append-only record; re-runs get new run_ids; any HARD anywhere halts; >=2 SOFT in one stage escalates (s79, s92-96)"),
 ("Foreground firewall (s104)", "the production chain subtracts NO age-, sex- or smoking-driven change - 'such change is the signal itself, not contamination'; age enters as the reference band, sex as chrX marker removal; the subtraction modules are test-only and off-by-default is insufficient - production config must not load them"),
 ("Pipeline rule (EDEAR)", "one IDAT-to-beta pipeline per patient at enrolment; cross-patient analytics use delta-A from personal baseline, never absolute A; a lab pipeline change triggers re-baseline (three readings) before drift detection resumes"),
]
XU538 = "Same Xu-538 panel, different IDAT-to-beta pipelines: absolute A baselines 0.38-0.62 across cohorts (within-EPIC-Italy ~0.44), direction preserved within pipeline (April 2026). Absolute A is pipeline-relative; this is the leading hypothesis for the 3/3 below-band healthy whole-blood donors (PROC-WB-IMMUNE-01)."
CANINE = ("H_min(canine)/H_min(human) = 1.00484 at T = 311.65 K via H_min(T) = H_min(37C) x (T/310.15)^2; canine Mahaffey number at 311.65 K = 54,000/(8.314 x 311.65) = 20.84 (scales as 1/T). A cross-species prediction that costs nothing to test on public canine arrays.")
CATALOGUE = [  # as recorded in EDEAR / CR2 / Issue 002; NOT re-verified in this issue
 ("Breast, >10 yr pre-diagnostic, EPIC-Italy", "VAL-047 / VAL-093", "d = -1.226 (p = 3e-4) matched filter; per-cell fan-out Baso d = +1.142 (VAL_001); pancreatic beta tile +1.020 off-tissue", "sign conventions differ by instrument - stated as recorded"),
 ("AML, blood at diagnosis", "VAL-082", "immune A d = +3.71 (largest single-cohort effect on record)", ""),
 ("DLBCL", "VAL-082", "A d = +3.20", ""),
 ("Glioma, plasma cfDNA", "VAL-090", "cortical-neuron fraction 1.09% (d = +1.96 vs healthy); GBM tumour 39.3% vs 62.4% non-tumour brain (d = -2.81)", "the shedding argument for CNS"),
 ("HCC without documented risk factors", "-", "secretory A d = +0.62, p = 0.0072", ""),
 ("NAFLD hepatocytes", "-", "A ~ 0.87, below the secretory floor: loss-of-identity, an INVERSION reading", ""),
 ("Pulmonary arterial hypertension, endothelium", "VAL-109/110/113", "d = +0.79 to +1.52 across three atlases (Caggiano +1.42, EpiSCORE -0.80, Loyfer +0.83)", "sign disagrees across atlases"),
 ("Bicuspid aortic valve, heart fibroblast", "VAL-110", "d = +2.10 (single-tile maximum, cardiovascular)", ""),
 ("Health ABC ageing, Hannum n = 656", "CPG_VAL_015", "r = -0.197, p = 3.7e-7; late-life acceleration rho = -0.854", ""),
 ("TCGA pan-cancer", "Issue 002", "A_tumour > A_normal in 27/28 types; independent set delta-A = 0.140 +/- 0.092, 14/15", "follows from global hypomethylation; not discriminating on its own"),
 ("DunedinPACE trajectory", "Issue 002", "E(a_bio) = exp(1 - 1/a_bio) reproduces rising pace, midlife peak (40.6 y), oldest-cohort deceleration", "asymptote, not survival bias - a claim"),
 ("NHANES 1999-2002 (external)", "Luo 2019", "top vs bottom decile age acceleration: cancer-mortality HR 2.14 (DunedinPoAm), >=1.58 for 3 of 4 clocks", "context for why cellular age matters"),
]
QAPE_SCAPE = [  # structure + published results only; inputs (Vienna n, Enigma) excluded by author decision
 ("QAPE", "A = -ln(1 - p_ref), p_ref = two-qubit gate error", "fault-tolerant anchor A < 1e-4 (100 ppm); NISQ-viable 1e-3", "floor set by the substrate's oxide TLS defect density: Al/AlOx many suboxides, Nb four oxide phases, Ta one (Ta2O5) - fewest interface defects; vacuum substrates (ions) have no TLS floor and QCCD connectivity scales ~log N", "best blind test: IonQ Forte -1.0% error"),
 ("SCAPE", "A = [TDP/(N_trans f)] / [k_B (T_op + 273.15) ln2]  (E per switch over the Landauer minimum, 2.81e-21 J at 75 C)", "three components: A_thermo = 1 + A_isa (ISA floor - 1) + A_node; ISA floors x86 ~70, Apple ARM ~15, Qualcomm ARM ~35, GPU ~250", "floor is temperature-set and cannot be engineered away; higher junction T lowers A", "P003 (Apple M5 third regression) NOT CONFIRMED - M5 65.8x vs M4 68.2x; P004 (Blackwell B200 step > 7.6%) CONFIRMED, +51.6%"),
]
CLINICIAN = {  # from The_Cellular_Margin.md and What_Is_Astro_Genetics.tex, with the two errata corrected
 "margin": "A cell spends one packet of ATP energy (about 54,000 J/mol) every time it does a unit of ordering work. Thermal noise at 37 C is worth RT = 8.314 x 310.15 J/mol. The ratio - how many ATP packets a cell has per packet of noise - is about 21. That is the cellular margin: healthy human cells run roughly twenty-one times above the bare thermal floor. It is a constant of the chemistry, the same for every patient; what varies between patients is how much of that margin the cell is actually using, which is what the A-score reads.",
 "floor": "The floor is not a statistical cutoff. It is the minimum disorder a cell of a given type must carry to stay that type - the Landauer cost, k_B T ln 2 per irreversible write, summed over the sites the cell is committed to. Eight cell architectures, eight floors, calculated once and frozen. A patient is read against the floor, not against other patients.",
 "reading": "A = 1.0 is the commitment line. Healthy cells sit on an age-dependent band near it (immune ~0.91 at age 4 rising to ~1.00 at 95). Frank cancer cells cluster at A ~ 1.28-1.32; established colon cancer reads ~1.15. Below the band is also a reading - a cell that has locked down (senescence, or the stem-cell reversion seen in seminoma).",
 "substrate": "The tube decides what can be read. Whole blood carries the immune architecture and nothing else - a healthy person has no epithelial DNA in it, and the instrument correctly says so. Plasma carries what tissues shed. Tissue reads everything present, mixed. Nothing in this document is a diagnosis; a departure is a flag for the clinician to place in context.",
 "errata": "Two earlier clinician-facing drafts stated the floor as kT (it is kT ln 2) and presented the Mahaffey number as a per-patient reading (it is a constant, 20.94; the per-patient quantity is A). Both are corrected here.",
 "life": "One way to say what the instrument measures: life is the continuous purchase of organised pattern against the second law, paid for in metabolic work. The A-score is the receipt.",
}
RULES += [
 ("PIPELINE", "One IDAT-to-beta pipeline per patient; absolute A is pipeline-relative.", XU538, "GAPE_EDEAR_Reproduction_Paper_v3 (T15, Rule 1-2)", "current"),
 ("FIREWALL", "Production subtracts no age/sex/smoking foreground.", "SOP s104: such change is the signal itself; age enters via the reference band, sex via chrX marker removal; subtraction modules are test-only and must not be loaded in production config.", "SOP v1.4.0 s104", "current"),
 ("SEAL", "A measurement that has not beaten the null suite is a candidate, not a claim.", "N1-N8; SEALED/RESTATE/RETRACT; PREREG edited after data = auto-RETRACT.", "SOP v1.4.0 s80-91", "current"),
]
# correct F1: it is a firewall decision, not an omission
RECON = [r if r[0]!="F1" else ("F1", "Stage 3 foreground subtraction", "not in 002",
  "NOT WIRED by decision (SOP s104 production foreground firewall): the production chain subtracts no age-, sex- or smoking-driven change. Sex: 131 chrX markers removed 2026-06-11. Age: the reference band. Smoking: Stage 7 bin floor shifts (v1.2 interim). Subtraction modules exist as test-only tooling.",
  "a stated design decision, not a gap; the flowchart's 'NOT BUILT' should read 'NOT WIRED - by s104'", "SOP v1.4.0 s104; flowchart_vKISS") for r in RECON]
ENGINE_MAP = [e if not e[0].startswith("Stage 3") else ("Stage 3 — Foreground subtraction", "age/sex/smoking modules (test-only)", "NOT WIRED by decision (SOP §104)", "production subtracts nothing: age via the reference band, sex via chrX marker removal, smoking via Stage-7 bin floors", "YES — RECON F1, RULES FIREWALL") for e in ENGINE_MAP]

# PROC-ANCHOR-01 — sealed GSE51032 anchor reproduced from raw GEO betas, 2026-09-19
ANCHOR = {
 "input": "GSE51032 series matrix (3,009 MB, EPIC, 845 samples) filtered to the 460 in the sealed foundation-cohort CSV (36 breast >10 yr pre-dx + 424 HC); iamatlas_celltype_markers_v0_2.json at repo HEAD and the TRIAL (chrX-removed) copy; H_min per class from iamatlas_gauge_identity_loci_v1_0.json",
 "operation": "for each of 115 cell types: per-cell A under (a) mean_i H(beta_i)/H_min and (b) H(beta_mean)/H_min over that cell's ~100 discriminative markers; Pearson r and max |diff| against GSE51032_115celltype_ascores.csv (sealed 2026-05-29)",
 "rows": [
  ("HEAD markers, mean-of-H",  "r = 1.00000", "max |diff| 0.00004", "112/115 cells r > 0.999", "REPRODUCES THE SEAL"),
  ("HEAD markers, H(beta_mean)","r = 0.450",   "max |diff| 1.048",   "0/115",                  "does not"),
  ("TRIAL markers, mean-of-H", "r = 0.99957", "max |diff| 0.060",   "89/115",                 "32 cells shift by up to 0.059 - the chrX removal"),
  ("TRIAL markers, H(beta_mean)","r = 0.449", "max |diff| 1.048",   "0/115",                  "does not"),
  ("GSE51057 (188: 11 case + 177 HC), HEAD markers, mean-of-H", "r = 1.00000", "max |diff| 0.00004", "115/115 cells r > 0.999", "REPRODUCES THE SEAL - both halves of the 648-sample foundation cohort"),
 ],
 "extras": "healthy per-cell A median 0.520 on this surface (s105 says ~0.52); per-cell case-vs-HC d median +0.32, 37/115 cells |d| > 0.5, largest Glia +1.10, stem_pluri +1.07, Baso +1.01. The seal was made with the REPO-HEAD markers (pre chrX removal): the TRIAL file changes 32 cells by up to 0.059 (largest: Mela) and would need re-sealing.",
 "verdict": "PASS for the separation surface: the sealed per-cell anchor is exactly mean-of-H over the v0_2 discriminative markers as SOP v1.4.0 s105 states, reproduced from raw GEO on a stranger's machine. This does NOT adjudicate the gauge (identity loci) - see PROC-FORMULA-01. It does establish that the current repo markers file, not the chrX-removed one, is what the seal rests on; adopting the chrX file re-opens the seal.",
}
RECON = [r if r[0]!="M1b" else ("M1b", "iamatlas_celltype_markers_v0_2.json - which is current", "-",
  "TRIAL file carries the 2026-06-11 chrX removal (131 markers, 32 cells affected by up to 0.059); REPO HEAD does not. PROC-ANCHOR-01 shows the sealed GSE51032 anchor was computed with the REPO-HEAD (pre-removal) markers.",
  "decision: adopt the chrX-removed file AND re-seal the anchor (expected: 89/115 cells unchanged, r = 0.9996), or keep HEAD and drop the sex-invariance fix. Either way the repo and the seal must agree.", "PROC-ANCHOR-01") for r in RECON]
GRID_STATUS[("whole blood","separation surface")] = ("CONFORMANT", "sealed GSE51032 per-cell anchor reproduced r=1.00000 from raw GEO")

# PROC-CAL-01 — Stage 1 executed from raw IDATs, 2026-09-19
_s1=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","handoff","stage1_conformance.json")
STAGE1 = json.load(open(_s1)) if os.path.exists(_s1) else []
CAL01 = {
 "input": "the eleven raw IDAT pairs in 10_TEST_DATA/idats (7 x 450K whole blood incl. 2 RA + 2 RA-study controls; 4 x EPIC colorectal tissue); stage_1_idat_calibration.py from the repo; methylprep 1.7.1 with numpy 1.26.4 / pandas 1.5.3 (methylprep uses DataFrame.append, removed in pandas 2 - the conflict the module's own docstring warns about); Illumina 450K and EPIC manifests fetched from methylprep's public bucket",
 "operation": "calibrate_idat_to_beta(Grn, Red) per sample: array type auto-detected from bead-address count, noob dye-bias + probe-type normalisation, per-sample, no cohort information; compare the resulting beta vector CpG-by-CpG to betas_cache.pkl (the Stage-1 output that produced PROC-DECON-01)",
 "expected": "identical betas (same algorithm, same manifest); any difference is a pipeline-version effect",
 "verdict": "PASS - 11/11 bit-identical: r = 1.000000, max |diff| = 0.000000, zero CpGs differing by more than 1e-4, on both array types (450K 415,863-418,055 CpGs, EPIC 490,390-757,003 CpGs); 24-48 s per sample. With PROC-DECON-01 and PROC-ANCHOR-01 this closes the chain: raw IDAT -> calibrated beta -> class fractions -> sealed per-cell anchor, every link reproduced from the repository on a machine that had never seen the project.",
 "consequence": "the 3/3 below-band healthy whole-blood donors in PROC-WB-IMMUNE-01 are NOT a Stage-1 artifact of this machine - the betas are the project's own. The remaining hypotheses are (a) the age band was compiled on a different IDAT-to-beta pipeline than noob (the Xu-538 pipeline-relativity effect, 0.38-0.62), or (b) the donors are below band. (a) is testable by locating the pipeline the band was compiled from.",
}
GRID_STATUS[("whole blood","Stage 1")] = ("CONFORMANT", "11/11 IDATs bit-identical to cache")
GRID_STATUS[("tissue","Stage 1")]      = ("CONFORMANT", "4/4 EPIC IDATs bit-identical to cache")

# ═══════════════════════════════════════════════════════════════════════════════
# RULINGS 2026-09-19 — author delegated the two open decisions to the analyst ("I trust you to make the
# most professional sound judgement now that you know the expectation for our project")
# ═══════════════════════════════════════════════════════════════════════════════
RULING_A3 = {
 "title": "RULING A3 - the A-score aggregation, one rule per surface",
 "rule": "A statistic is read only against a reference computed by the same aggregation. There are two surfaces and each keeps the aggregation its reference was built with.",
 "gauge": "CLASS GAUGE (8 classes, identity loci, Stage 4): A = H(beta_mean)/H_min(class). Because: (i) H_min is itself H of a mean beta - G-002 calibrated each class floor on per-cell-type mean methylation of sorted reference cells (H_min_global = H(0.790), Issue 002 s2.1), so H(beta_mean)/H(beta_ref) compares like with like; (ii) age_reference_matrix.json, the band every gauge reading is placed against, is compiled as H(beta_mean)/H_min to five decimals in 80/80 cells; (iii) identity loci are selected within 0.05 of H_min_beta and are therefore not bimodal - the Jensen inflation s105 describes is bounded, measured at +0.029 +/- 0.002 on the declared substrate (PROC-FORMULA-01); (iv) it is the formula of Issue 002 and of the chain wired at HEAD (d7b0e1f). Mean-of-H over identity loci would be a different quantity whose natural floor (mean per-locus entropy of the reference cell) was never computed.",
 "separation": "SEPARATION SURFACE (115 cell types, discriminative markers, Stage 4 per-cell and Stage 5 hull): A = mean_i H(beta_i)/H_min(class). Because: (i) discriminative markers are bimodal by construction, so H(beta_mean) manufactures ~0.5 and false BREACH - s105 is exactly right here (worked guard GSM1235534: 1.111 vs 0.864); (ii) it is what the sealed anchors are: PROC-ANCHOR-01 reproduces GSE51032 and GSE51057 at r = 1.00000 with this formula and not with the other.",
 "guards": "The gauge refuses to score when (a) the class is absent from the substrate (presence gate, DETECT_FLOOR) or (b) averaging beta first is not valid on the panel it actually receives, tested directly: the Jensen gap H(beta_mean) - mean_i H(beta_i) on that panel must not exceed 0.05. Calibrated 2026-09-19: identity loci in whole blood 0.033 (pass); identity loci in bulk tissue 0.20 (refused - a mixture, glioma-LL-002); s105's own HSC marker example 0.29 (refused); hepatocyte markers 0.78 (refused). This turns s105's 'never' into a measured condition enforced at runtime (kit/cpg_kit.py::gauge_A).",
 "sop": "SOP s105 is amended, not reversed: its one sentence becomes 'On any marker panel, score per locus then average; never average beta first. The class gauge on identity loci is the one surface where H(beta_mean) is the correct statistic, because H_min and the age band are both defined that way.' s105 open decision 1 (Recipe s6.3) is for the author to record in the vault; decision 3 (s46.5 bidirectional recovery assumed H(beta_mean)) stands as written because the gauge keeps H(beta_mean).",
 "status": "RULED 2026-09-19 by the analyst under delegated authority; recorded here and in RECON A3; the author may overrule.",
}
RULING_M1B = {
 "title": "RULING M1b - the markers file, and re-sealing the anchor",
 "rule": "Adopt iamatlas_celltype_markers_v0_2.json WITH the 2026-06-11 chrX removal (the TRIAL/user copy) as canonical; commit it to the repository; re-seal both foundation-cohort anchors against it.",
 "why": "The removal is a derived, documented sex-invariance correction (131 chrX markers; 'so sex cannot shift an A-score'). The original seal (2026-05-29) predates the fix (2026-06-11); a seal that rests on a known-uncorrected file is weaker than a re-seal that records the change. Keeping the pre-fix file to preserve the seal would be preserving the number over the correction.",
 "reseal": "Re-sealed values computed 2026-09-19 from the raw GEO matrices with the chrX-removed markers and mean-of-H (RULING A3): GSE51032 (n = 460) and GSE51057 (n = 188), 115 cells each. Relative to the 2026-05-29 seal: 32 of 115 cells change, largest shift 0.059 (Mela), r = 0.99957 / 0.99951, 83 cells unchanged. Files: kit/anchors_v2/GSE51032_115celltype_ascores_v2_chrXremoved.csv (sha256 212775a8...bcd9b), GSE51057_..._v2_chrXremoved.csv (sha256 b4b854a3...93597). The 2026-05-29 seal is retained as v1 and marked SUPERSEDED, not deleted.",
 "status": "RULED 2026-09-19 under delegated authority; the repository commit is the author's to make.",
}
RECON = [r if r[0]!="A3" else ("A3", "A-score aggregation: gauge vs separation", "A = H(beta)/H_min, beta = the cell's mean methylation; H_min_global = H(0.790)",
  "SOP v1.4.0 s105: mean-of-H, never H(beta_mean). Code at HEAD: H(beta_mean) over identity loci; age band compiled the same way.",
  "RULED (see RULING A3): gauge on identity loci = H(beta_mean)/H_min, matching H_min's own definition and the band; separation on discriminative markers = mean-of-H, matching s105 and the sealed anchors; s105 amended to scope its 'never' to marker panels; runtime bimodality guard added to the gauge.", "PROC-FORMULA-01; PROC-ANCHOR-01; RULING A3") for r in RECON]
RECON = [r if r[0]!="M1b" else ("M1b", "iamatlas_celltype_markers_v0_2.json - which is current", "-",
  "TRIAL file carries the 2026-06-11 chrX removal; REPO HEAD does not; the 2026-05-29 seal rests on the pre-removal file.",
  "RULED (see RULING M1b): adopt the chrX-removed file, commit it, re-seal both anchors (done 2026-09-19: 32/115 cells shift, max 0.059, r = 0.9996; v1 seal retained as SUPERSEDED).", "PROC-ANCHOR-01; kit/anchors_v2/RESEAL_REPORT.json") for r in RECON]
RULES += [
 ("SURFACE", "One aggregation per surface, fixed by its reference.", "Gauge (identity loci): H(beta_mean)/H_min. Separation (discriminative markers): mean_i H(beta_i)/H_min. Never read a statistic against a band compiled the other way.", "RULING A3", "current"),
]
RECON = [r if r[0]!="A3b" else ("A3b", "SOP v1.4.0 s105 open decisions vs the code", "-", r[3],
  "CLOSED by RULING A3 for decisions 2 (marker convention: identity loci for the gauge, discriminative for separation) and 3 (s46.5 stands, gauge keeps H(beta_mean)). Decision 1 (Recipe s6.3) remains the author's to record in the vault.", "RULING A3") for r in RECON]
RECON = [r if r[0]!="F7" else (r[0],r[1],r[2],r[3],r[4],"EDEAR s4.6 (0.795 -> 0.838889, 6.44 sigma); Issue 002 immune card ('revised downward by approximately 0.055')") for r in RECON]
RECON += [
 ("M2", "'Mahaffey number' - two definitions in the corpus", "glossary: 'metabolic sensitivity parameter n_bio = dG_ATP/(R T) = 20.94'",
  "IAM_Hubble2GAPE (l.2192): M = E_drive/(k_B T_local ln2); 'for chemical systems M ~ 30'. The Cellular Margin (~21) and the 002 glossary (20.94) omit the ln2. SCAPE's 117x and the transmon's exact M = 1 both carry the ln2.",
  "RECOMMENDED: M carries the ln2 everywhere (cell M = 30.21) so the three applications share one denominator and the transmon identity survives; n_bio = 20.94 keeps its own name, 'metabolic sensitivity parameter', exactly as the glossary calls it. Author to confirm. H_min is a different quantity in either case: the per-class floor in bits, 8 (40 with substrates); M is one energy ratio for the cell.", "002 glossary p251; H2G l.2192-2215; Cellular Margin l.14-28"),
]
GLOSSARY_NOTE_MAHAFFEY = ("[Issue 003 note, RECON M2] The entry above is Issue 002's and defines the name as n_bio = dG_ATP/(RT) = 20.94. IAM_Hubble2GAPE defines M = E_drive/(k_B T ln2), which gives 30.21 for the cell and is the form under which SCAPE reads 117x and the Al transmon reads exactly 1. "
  "Issue 003 s5A.1 uses the ln2 form. Neither is H_min: H_min is the per-class floor in bits; M is the cell's energy budget in units of one Landauer bit. Author to confirm which name attaches to which number.")

# AUTHOR RULING 2026-09-19 on the Mahaffey number (supersedes RECON M2 recommendation above)
MAHAFFEY = {
 "value": "20.94", "definition": "M = E_drive / (k_B T) for the cell: the ATP ordering budget per event over the thermal noise quantum at body temperature. Dimensionless, fixed by biochemistry, the same for every cell. M = dG_ATP/(R T) = 54,000/(8.314 x 310.15) = 20.94 at 37 C. Author 2026-09-19: one fixed number; the 30.21 (ln2) form was wrong.",
 "not_hmin": "H_min is a different quantity: the per-class entropy floor in bits (8 classes, 5 substrates each -> 40 floors), MCMC-confirmed, architecture-specific. M is one number for the cell; H_min is forty numbers for the classes. They are never interchangeable.",
 "n_bio": "n_bio was an early (QAPE-era) form of the similar ratio and is NO LONGER USED. Every occurrence of n_bio in Issue 002 (glossary, class cards 'n_bio = 20.94', canine 20.84) is superseded by the Mahaffey number.",
 "open": "Inputs: dG_ATP = 54,000 J/mol, R = 8.314 J/(mol K), T = 310.15 K (002 glossary, _gape_constants_private.py l.26, Cellular Margin l.28, Hubble2GAPE l.3504) - consistent across the corpus.",
 "ln2_note": "IAM_Hubble2GAPE l.2192 writes M = E_drive/(k_B T_local ln2) and gives ~30 for chemical systems; the author's ruling has no ln2 (20.94). That line of the tex should be brought into agreement. The earlier 003 draft's cross-substrate table (SCAPE 117x, transmon M = 1 exactly) was computed with the ln2 form and is WITHDRAWN from s5A.1; SCAPE and QAPE figures are quoted only as their own reports state them.",
}
RECON = [r if r[0]!="M2" else ("M2", "'Mahaffey number' - definition and value", "glossary: 'n_bio = dG_ATP/(RT) = 20.94'",
  "author 2026-09-19: M = 20.94, one fixed number, no ln2; n_bio is an early QAPE-era form, no longer used; H_min is a separate per-class quantity",
  "RULED BY AUTHOR. Open: Hubble2GAPE l.2192 ln2 form (~30) to be reconciled.", "author statement; 002 glossary p251; H2G l.2192, l.3504") for r in RECON]
GLOSSARY_NOTE_MAHAFFEY = ("[Issue 003 note, RECON M2] The entry above is Issue 002's and is superseded. The Mahaffey number of the cell is 20.94 (author, 2026-09-19), defined as the ATP ordering budget per event over k_B T at body temperature; "
  "n_bio was an early QAPE-era form of the ratio and is no longer used. H_min is not the Mahaffey number: H_min is the per-class entropy floor in bits, forty values across eight classes and five substrates.")
_GL_TAIL=("")
FALSIFICATION += [("Issue 003 draft v3-v8 s5A.1 claim", "'M_cell = n_bio/ln2 = 30.21; Al transmon M = 1 exactly; one statistic across three substrates' - computed with a ln2 denominator the author does not use; withdrawn", "WITHDRAWN")]

# APPENDIX V — VAL index built mechanically from the repository at 66f37fe (2026-09-19)
_vi=os.path.join(os.path.dirname(os.path.abspath(__file__)),"val_index.json")
VAL_INDEX = json.load(open(_vi)) if os.path.exists(_vi) else []
for _o in VAL_INDEX:
    _o.setdefault('status',''); _o.setdefault('result','')

VAL_INDEX_NOTE = ("Rebuilt 2026-09-21 from the record, not the tree (PROC-HISTORY-01): 175 rows with unique keys by series - 3 G-series calibrations, 119 pre-Atlas VAL identifiers (107 executed), the 15-cohort T-series of VAL-049, 22 post-Atlas CPG-VALs (21 executed), the five-version Mahalanobis hull lineage, L9 N7, and the September procedures. The 2026-09-19 index had counted 103: it keyed on the bare number (VAL-001 and CPG-VAL-001 collided) and counted only identifiers with a folder at HEAD. Verdicts are recorded from each OUTCOME, not re-verified.")
FOUR_SKIES_CAP = ("Four skies on one HEALPix grid (NSIDE 128, 196,608 pixels, Mollweide; CpGs in atlas row order chr1 -> chrY, the Plate 1 convention). "
  "(a) One realization of the microwave CMB from the Planck 2018 LCDM power spectrum (CAMB -> healpy.synfast). (b) The IAM Atlas immune-class posterior mean beta per CpG - the 'brilliance' sky, 483,092 CpGs, MCMC posterior. "
  "(c) The same class's posterior sd per CpG - the healthy-variance sky, which is what makes (d) possible. (d) Patient GSM1051533 (whole blood, 450K, Stage-1 calibrated, RA-study control) as z = (beta - mu)/sigma against (b, c), computed by the engine's cpg_patient_cmb.py. "
  "Median |z| 1.40; mean z +1.07 - the uniform red tint is the +0.05 reference-beta offset of s6a, not a patient finding. The engine's self-determined assessability flagged stromal as assessable from blood (median |z| 1.44); the deconvolver's presence gate says stromal = 0, and the presence gate is the correct one - a known weakness of the median-|z| criterion recorded here. "
  "The point of the figure is the method, not the patient: a reference map with per-pixel uncertainty and the residual of one observation against it is exactly the Planck workflow, and it is why the CMB toolkit transfers.")


# PROC-CHAIN-01 — first end-to-end run of cpg_conductor.run_full from the repository, 2026-09-19
_fc=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","..","..","..","handoff","fullchain_rows.json")
CHAIN01_ROWS = json.load(open(_fc)) if os.path.exists(_fc) else [["GSM1051525", "WB RA", 60, "1.0021", "IN_BAND", "1.1199 BREACH", "absent", 12.5], ["GSM1051526", "WB RA", 60, "1.0064", "IN_BAND", "1.1235 BREACH", "absent", 12.2], ["GSM1051533", "WB RA-ctrl", 60, "1.003", "IN_BAND", "1.1112 BREACH", "absent", 12.3], ["GSM1051534", "WB RA-ctrl", 60, "1.0046", "IN_BAND", "1.1327 BREACH", "absent", 12.2], ["GSM2333901", "WB healthy", 58, "0.9604", "IN_BAND", "1.1062 BREACH", "absent", 12.2], ["GSM2333905", "WB healthy", 67, "0.9539", "IN_BAND", "1.1014 BREACH", "absent", 12.1], ["GSM2333950", "WB healthy", 43, "0.9568", "IN_BAND", "absent", "absent", 12.2], ["GSM5065985", "tissue CRC st4", 60, "1.0408", "ABOVE_BAND", "absent", "1.0688 DETECTABLE", 12.7], ["GSM5065990", "tissue CRC st1", 60, "0.8964", "BELOW_BAND", "absent", "1.0078 NORMAL", 12.8], ["GSM8772491", "tissue adenoma", 60, "0.8155", "BELOW_BAND", "absent", "0.9887 NORMAL", 12.2], ["GSM8772492", "tissue adenoma", 60, "0.9321", "IN_BAND", "absent", "1.1076 BREACH", 12.6]]
CHAIN01 = {
 "input": "the eleven Stage-1 betas of PROC-CAL-01 (7 whole blood, 4 EPIC colorectal tissue); cpg_conductor.run_full at repo layout of 2026-09-19; IAM_Atlas/IAMAtlasREBUILD.csv; ages as documented, 60 where unknown",
 "operation": "Stage 2 deconvolution -> Stage 4 class gauge (identity loci, age band, tiers) -> Stage 4.5 bidirectional -> Stage 5 Mahalanobis Option A -> Stage 6 cellular age. ~12 s per sample",
 "expected": "healthy whole blood: immune IN_BAND / NORMAL (b40ffbd); tissue: epithelial classes present and elevated in neoplasia; classes absent from blood not scored",
 "findings": [
  ("healthy whole blood, immune", "IN_BAND / NORMAL in 7/7 (A 0.954-1.006). The +0.05..+0.09 identity-loci beta offset of s6a is REAL in the betas but the shipped gauge absorbs it through the age band; it is not a replication blocker. s6a is amended accordingly."),
  ("tissue, cycling / secretory", "adenoma GSM8772492 cycling 1.108 BREACH, secretory 1.066 DETECTABLE; CRC stage 4 cycling 1.069 DETECTABLE; CRC stage 1 cycling 1.008 NORMAL, adenoma GSM8772491 cycling 0.989 NORMAL. Direction as predicted; per-sample spread is the tissue-mixture story (glioma-LL-002)."),
  ("DEFECT 1 - stem_adult false alarm", "stem_adult reads BREACH (A 1.10-1.13, ABOVE_BAND) in 6/7 whole-blood samples - every sample where its deconvolved fraction cleared DETECT_FLOOR 0.01 (1.7-4.4%); in GSM2333950 the fraction was 0.55% and it was correctly not scored. 2 of 3 healthy donors flagged; three of the six would also pass the 3% adjudicator gate. Its age band is n=28 from one source (Adelman 2019; A_mean 0.956, p10-p90 0.922-0.991) - a cohort statistic, not a floor. A class that flags healthy donors whenever it is present is a broken BAND, not a finding. Until the band is rebuilt (Phase 1) stem_adult must be NOT_ASSESSABLE from whole blood."),
  ("DEFECT 2 - Mahalanobis key mismatch", "run_full emits departure as {distance, beyond, driver}; the conductor's own __main__ and cpg_report_builder read mahalanobis_distance / mahalanobis_beyond_band -> the report shows nothing. Distance 5.4-5.8 'beyond' on healthy donors is driven entirely by Defect 1."),
  ("DEFECT 3 - cellular age floors", "summary cellular age 4.0 yr for a 58-yr-old healthy donor on 5/6 classes (stem_pluri 84). The inversion is pinned at the curve's lower edge; Stage 6 stays 'built, calibration pending' and is not to be reported."),
  ("layout", "the conductor resolved every module and JSON as HERE/<file> (flat CPG_TRIAL_CODE layout) and had never run from the repo tree; a _find() over the Runtime Matrices subfolders was added. iam_cellular_age_scoring.py existed only in the trial bundle and is now in Runtime Matrices/Cellular_Age/."),
 ],
 "verdict": "CHAIN RUNS END TO END FROM THE REPO. Immune gauge on blood: CONFORMANT. Three defects recorded (band, key, age); none touches the immune result.",
}
RECON += [
 ("B1", "class bands are cohort statistics of unequal provenance", "age_reference_matrix.json: immune from hundreds of samples across decades; stem_adult n=28 single source; others in between",
  "the FLOOR (H_min, 40 values) is physics and is one number per class-substrate; the BAND (age_reference_matrix) is a cohort statistic and is only as good as its n and pipeline. Every 'healthy reads wrong' case of 2026-09-19 traces to a band, never to a floor.",
  "Phase 1: rebuild all eight class bands from ONE healthy cohort (GSE87571, n~730, ages 14-94) through ONE pipeline (Stage 1); record n/source/pipeline per band; classes without a defensible band from blood -> NOT_ASSESSABLE", "PROC-CHAIN-01"),
 ("V1", "what 'A' is in VAL-001/003/008 (the 27/28 claim)", "report: 'A = H(beta)/H_min(class)'",
  "numerator = COHORT-MEAN GENOME-WIDE beta (TCGA tumor / solid-tissue-normal means, Xu 2019; healthy from one Roadmap cell per class); divisor = H_min of the cancer's tissue-of-origin class (LGG->terminal, BRCA->secretory, COAD->cycling ...). Class-correct divisor, global-mean numerator, cohort-level - a different instrument from the identity-loci gauge, sharing only the divisor. The script itself notes a '~10% global entropy offset expected'.",
  "state this wherever 27/28 is quoted; it explains why direction holds while absolute A is offset", "GAPE_Evidence_Report_UPDATED VAL-003 script l.5208-5544"),
]

# PROC-STAGE0-01 — Stage 0 intake executed on raw IDATs for the first time, 2026-09-19
STAGE0_01 = {
 "input": "the eleven raw IDAT pairs (7 HM450K whole blood, 4 EPIC_v1 tissue), decompressed; a manifest per sample (sentrix id, array type, hashed patient id, substrate, sex, age); intensities read with methylprep IdatDataset for the QC steps that accept them",
 "operation": "step_0_1 arrival+header -> 0_3 SHA-256 integrity -> 0_5 detection-p -> 0_6 bead count -> 0_7 call rate -> 0_7b platform coverage -> 0_9 decision gate",
 "observed": [
  ("0.1", "STAGED 11/11; array type inferred from the IDAT header agreed with the declared type in every case (HM450K x7, EPIC_v1 x4)"),
  ("0.3", "INTEGRITY_OK 11/11; re-transmission detection exercised by the module's own self-test"),
  ("0.5", "DETECTION_BORDERLINE 11/11 - with a crude 2nd-percentile background in place of the negative-control probes; the verdict reflects the stand-in, not the arrays"),
  ("0.6", "PASS 11/11"), ("0.7 / 0.8", "DEFERRED - call rate and sex check need Stage 1's decoded per-probe data"),
  ("0.9", "PROCEED_WITH_PENALTY 11/11 (from the borderline detection)"),
 ],
 "defect_fixed": "the decision gate did not read Step 0.1's status, so QUARANTINE_INCOMPLETE_MANIFEST / QUARANTINE_MISSING_CHANNEL samples fell through to PROCEED. Guard added; module self-tests pass. Same fail-open class as the origin-gate finding of the first read.",
 "open": "the Stage 0 <-> Stage 1 intensity hand-off (control probes, detection-p from negative controls, call rate, X/Y for sex) is designed but not wired; until it is, those QCs report DEFERRED and Stage 0 is an integrity-and-manifest gate only.",
 "verdict": "RUNS. Every stage of the chain (0, 1, 2, 4, 4.5, 5, 6) has now executed from the repository on the same eleven samples.",
}

# GLOSSARY — CHAIN LINKS: one line per runtime file the conductor touches. Kind: FLOOR (physics, frozen) /
# RULER (where the gauge reads, derived from the Atlas, frozen) / BAND (cohort statistic) / CODE / DATA.
CHAIN_LINKS = [
 ("IAM_Atlas/IAMAtlasREBUILD.csv(.xz)", "DATA", "The reference sky: 483,092 CpGs x 115 cell types, per-class MCMC posterior mean, sd and CI per CpG. Every patient is read against it. 2026-05-28, frozen."),
 ("IAMAtlasREBUILD_celltype_to_class.json", "DATA", "115 cell types -> 8 architecture classes. The deconvolver's cell fractions are summed to class fractions through it."),
 ("cpg_gauge_engine.py  H_MIN_TABLE", "FLOOR", "The forty floors: 8 classes x 5 substrates (methyl, nucl, fuzz, wps, frag). Physics; one number each; frozen 2026-04-06, MCMC-confirmed. Never a cohort."),
 ("iamatlas_gauge_identity_loci_v1_0.json", "RULER", "Which CpGs the gauge reads, per class: the loci where healthy cells of that class sit within +/-0.05 of H_min_beta. Derived from the Atlas, frozen. Counts terminal 57,247 ... stromal 2,294."),
 ("iamatlas_celltype_markers_v0_2.json", "RULER", "The separation surface: ~100 one-vs-rest discriminative CpGs per cell type (bimodal by construction). Read with mean-of-per-CpG-H, never H(beta_mean) (s105). chrX markers removed (RULING M1b)."),
 ("age_reference_matrix.json", "BAND", "What healthy looks like at each age, per class: A and beta mean/sd/p10-p90 in 10 age bins. A cohort table compiled 2026-05-28 from nine papers; n is uneven (immune ~100/bin, stem_adult 28). The gauge's placement is read against this. Phase 1 rebuilds it from one cohort, one pipeline."),
 ("tier_breakpoints.json", "BAND", "The severity ladder: NORMAL 0.95-1.01, ELEVATED 1.01-1.07, SIGNIFICANTLY_ELEVATED 1.07-1.10, BREACH >= 1.10; Warburg line 1.07. Onset moved 1.04 -> 1.01 on 2026-07-03."),
 ("mahalanobis_healthy_reference_v2_0_age_matched_derived.json", "BAND", "Stage 5's reference: derived from the age band (mu = A_mean(class, age), sigma from p10). v0_5 / v1_0 retired and refused by the module."),
 ("walther_iam_deconvolver.py", "CODE", "Stage 2: NNLS composition against the Atlas. Returns cell and class fractions and a residual. Gates nothing; presence (fraction >= DETECT_FLOOR) decides which classes may be scored."),
 ("cpg_conductor.py", "CODE", "The orchestrator (2026-07; replaces walther_clinical.py). Pairs every per-cell A with its deconvolved fraction; a class below DETECT_FLOOR 0.01 is absent, not a reading. Runs Stages 2 -> 4 -> 4.5 -> 5 -> 6."),
 ("iamatlas_a_scoring.py", "CODE", "The separation statistic: mean_i H(beta_i)/H_min over each cell type's discriminative markers -> 115 per-cell A's for the disease matcher. Guarded by test_a_score_canonical.py (separation surface only)."),
 ("cpg_gauge_engine.py", "CODE", "Stage 4: A = H(beta_mean)/H_min over identity loci, placed in the age band (BELOW/IN/ABOVE) and on the severity ladder. Carries the FLOOR table and the SATURATION_MARGIN."),
 ("bidirectional_decomposition.py + directional_panels_v1_0.json", "CODE", "Stage 4.5, the fix for entropy's symmetry: H is symmetric about 0.5, so a disease pushing CpGs both ways leaves beta_mean unmoved and the gauge reads null (VAL-050 d=+0.08). Scores direction per CpG with the sealed VAL-051 composite (d=+0.62). Immune panel only in v1.0."),
 ("cpg_patient_cmb.py + IAM_Atlas/healpix_mapping/", "CODE", "Stage 4.6: the patient's per-CpG z = (beta - mu)/sigma against the Atlas posterior, on the HEALPix NSIDE-128 sky (Plate 05). The residual map of one observation against a reference map with per-pixel uncertainty - the Planck workflow."),
 ("iamatlas_mahalanobis_scoring.py", "CODE", "Stage 5, Option A: the eight class-gauge A's as a vector, distance from the age-band centroid with the band's covariance, n-adaptive chi-square alarm. Answers 'is the whole profile off' where any single class may be in band."),
 ("iam_cellular_age_scoring.py", "CODE", "Stage 6: the band run backwards - at what age does healthy beta_mean(class) equal this patient's? Third attempt: v1 was a trained clock (wrong), v2 inverted the wrong formula (Jensen). Built; calibration pending; reads 4 yr for adults today because the band curves are too flat to invert - not reportable."),
 ("stage_0_intake.py", "CODE", "Stage 0: manifest, IDAT header vs declared array, SHA-256 integrity with re-transmission detection, control-probe / detection-p / bead / call-rate / sex QC (the last four await the Stage 1 hand-off), PROCEED / PENALTY / QUARANTINE gate. Fail-open on 0.1 quarantines closed 2026-09-19."),
 ("stage_1_idat_calibration.py", "CODE", "Stage 1: raw IDAT pair -> noob-calibrated beta via methylprep (pandas < 2). Bit-identical to the project cache on 11/11 (PROC-CAL-01). Owns any raw-array alignment; no per-class input offset is applied downstream."),
 ("RETIRED/.../IAM_Cellular_Age/ (age_/sex_/smoking_axis_foreground.py, *_layer.csv)", "RETIRED", "Stage 3 foreground subtraction, built and deliberately not wired (SOP s104): a galactic foreground is a separate source; the methylome 'foreground' is the patient's own biology - annotated, never subtracted."),
]

# GLOSSARY — CMB AND CHAIN TERMS. Each definition is taken from the source file named; none is paraphrased from memory.
CHAIN_TERMS = [
 ("Cosmic Methylome Background (CMB)", "The patient's per-CpG map, named for its cosmological twin. The chain is built on the Planck pipeline stage for stage: raw intensities -> calibration -> an all-sky map -> component separation -> a statistic against a derived reference -> null tests -> sky rendering. The one deliberate departure is foreground subtraction (see Foreground). [Biological_Physics/README.md]"),
 ("Brilliance / brightness", "The Atlas's per-CpG posterior mean and sd for a class - a reference map with a per-pixel uncertainty, which is what makes the methylome a CMB problem. 'Brilliance' is the author's term for the posterior surface; Stage 4.6 (brightness comparison) reads a patient's residual against it. [cpg_patient_cmb.py; IAMAtlasREBUILD.csv *_mean / *_sd columns]"),
 ("z-departure", "z = (beta_patient - mu_class) / max(sigma_class, 0.005) per CpG: how many posterior standard deviations the patient sits from the healthy class at that locus. The residual map of one observation against the reference. Median |z| over a class was used for assessability; the deconvolver presence gate is the correct gate. [cpg_patient_cmb.py]"),
 ("HEALPix / NSIDE / Mollweide", "HEALPix: the equal-area sphere pixelisation used for CMB maps. NSIDE 128 -> 196,608 pixels. CpGs are laid on the sphere in genomic order (chr1 ... chrY, MAPINFO within chromosome), i-th CpG -> pixel floor(i x npix / n); ~2.46 CpGs per pixel, averaged. Mollweide is the equal-area full-sky projection every plate uses. Deterministic: same atlas + manifest -> byte-identical mapping. [IAM_Atlas/healpix_mapping/README]"),
 ("Component separation", "Cosmology's name for pulling a mixed signal apart into its sources; here, deconvolving a bulk sample into cell-type fractions (Stage 2). The Walther NNLS deconvolver is the one method in the chain; NILC (the Planck needlet method) was tried as a second and cut on 2026-07-02 after it collapsed on correlated blood mixtures. [walther_iam_deconvolver.py; commit c1be0c3]"),
 ("Foreground", "In cosmology, a separate physical source (the Galaxy) subtracted to reveal the CMB. In the methylome the 'foreground' - age, sex, smoking - is the patient's own biology, so it is annotated, not subtracted. The subtraction layers were built and deliberately left unwired (SOP s104). [RETIRED/.../IAM_Cellular_Age/]"),
 ("Matched filter", "Stage 5 second chain (SOP 8.2): the Pearson correlation between a patient's per-CpG departure and a disease's sealed residual template, with a bootstrap CI. Fires only when Stage 8 Route B flags; a template with no mean shift correctly does not match. [stage_5_second_chain.py]"),
 ("Mahalanobis departure (Option A)", "One number per patient: distance = sqrt( sum_c ((A_c - mu_c(age)) / sigma_c(age))^2 ) over the classes assessable in the substrate, where mu_c = A_mean(class, age) and sigma_c = (A_mean - A_p10)/1.2816 from the age band. Answers 'is the whole profile off' when no single class is. A healthy patient sums to ~0. Alarm by n-adaptive chi-square. Supersedes v1_0 (mu=1.0, sigma=0.02 on the 115 separation A's, which put healthy at z ~ -22). [iamatlas_mahalanobis_scoring.py]"),
 ("Null runner / the eight nulls", "The methylome's CMB null tests: no CPG-VAL is sealed until its declared nulls pass. N1 label permutation (1000x); N2 age-strata permutation; N3 sex-strata; N4 cohort split-half replication with consistent sign; N5 plate / array-position; N6 injection-recovery (inject a known signal into HC, recover it); N7 end-to-end simulation (synthetic patients with known truth through L1-L8); N8 look-elsewhere (Bonferroni/FDR). Required for every VAL: N1, N4; for end-to-end claims: N7. 'A VAL that doesn't declare its nulls is not a sealed VAL - it's a preliminary analysis.' [CPG_Engine/CPG_Null_Runner/cpg_null_runner.py]"),
 ("Synthetic patient", "A constructed beta vector (or Stage-4 dict) with known class fractions and known per-cell A's, everything else absent. The ground truth for N6/N7: the chain must recover what was put in. [report_builders/synthetic_patient_harness.py; TEST_DATA/harness/]"),
 ("PREREG / seal / RESTATE", "Every validation declares its prediction, direction, pass conditions and nulls in a PREREG before data are opened; the file is sealed by SHA-256. Outcomes are recorded as pre-registered codes (O1_PRIMARY_VALIDATED ... O3_INVERTED, NULL, DIRECTIONAL). A change after seeing data is an AMENDMENT, dated; a re-reading is RESTATE; a post-hoc accommodation VOIDs the VAL with the seal preserved (VAL-102). [SOP Part IV; Testing_and_Code/VAL_INDEX.csv]"),
 ("Chain of custody (L1-L9)", "The audit grading of the chain: nine links from raw IDAT (L1) through calibration (L2-3), component separation (L4), scoring (L5-6), nulls (L7), report (L8-9), each with its own evidence requirement. The SOP is organised by stage (operational order) and by link (audit order). [SOP s2, s3]"),
 ("Identity loci vs discriminative markers", "Two CpG sets, two surfaces. Identity loci: where a healthy class sits within +/-0.05 of H_min_beta (unimodal) - the GAUGE reads beta_mean here. Discriminative markers: ~100 one-vs-rest CpGs per cell type (bimodal by construction) - the SEPARATION statistic reads mean-of-per-CpG-H here. Using one set with the other's formula is the 2026-06-11 all-BREACH bug. [iamatlas_gauge_identity_loci_v1_0.json; iamatlas_celltype_markers_v0_2.json; s105-s106]"),
 ("Jensen gap", "H(beta_mean) - mean_i H(beta_i) >= 0 (Jensen's inequality, H concave). Near zero on a unimodal panel (identity loci in blood: 0.013-0.033); large on a bimodal one (marker panels: ~0.7) or a tissue mixture (~0.2). The gauge refuses any panel whose gap exceeds 0.05 - the runtime form of s105. [cpg_kit.gauge_A]"),
 ("Presence / DETECT_FLOOR", "A class is present in a sample when its deconvolved fraction >= DETECT_FLOOR (0.01 in the conductor; a 3% gate in the Mahalanobis adjudicator - RECON D2). An A-score for an absent class is entropy computed over cells that are not there and is not a reading. [cpg_conductor.py]"),
 ("Commitment line (A = 1.0)", "The point where the class's measured entropy equals its floor. It is NOT where healthy sits: healthy is the age-matched band (immune 0.906 at age 4 -> 1.000 at 95). 'A = 1.0 at the healthy floor' is the pre-band (Issue 002-era) definition. [cpg_gauge_engine.py docstring]"),
 ("Cellular age", "Stage 6: invert the age band's beta_mean(age) curve per class to the age at which healthy beta_mean equals the patient's. v1 was a Horvath-style trained clock (not physics); v2 inverted A with the wrong formula (Jensen). v3 is the Recipe's inversion; built, calibration pending, not reportable. [iam_cellular_age_scoring.py]"),
 ("Bidirectional decomposition", "Stage 4.5. Shannon H is symmetric about beta = 0.5, so a disease that pushes some CpGs up and others down leaves beta_mean unmoved and the pooled gauge reads null (VAL-050, d = +0.08). A directional per-CpG composite recovered d = +0.62 (VAL-051); Stage 4.5 runs that sealed formula per patient. [bidirectional_decomposition.py]"),
 ("Flatness (the Atlas lesson)", "The v0.1 rebuild lesson: the old MCMC script showed R-hat ~1.01 (looks perfect) while every cell type in a class came out identical - a flat, broken atlas; the fixed script shows R-hat 1.4-2.6 (looks alarming) while the cell types are distinct and the atlas is right. Convergence diagnostics cannot see the failure that matters; judge a rebuilt class by the distinctness test on per_celltype.csv, never by R-hat alone. [IAM_Atlas/IAMAtlas_FLATNESS_LESSON.md]"),
 ("Xu-538", "The 538-CpG breast-cancer panel of Xu et al. 2019 used as the per-patient scoring surface in VAL-047 (pre-Atlas); 'Xu-538 relativity' names the finding that a panel's absolute A depends on the pipeline that produced the betas, so bands compiled on one pipeline do not transfer to another without re-derivation. [VAL-047 record; s6a]"),
]

# APPENDIX VI / VII — the CMB->methylome translation map (author, pre-build; scored 2026-09-19) and the completion sprint, scored
_avi=os.path.join(os.path.dirname(os.path.abspath(__file__)),"appendix_vi_vii.json")
_A=json.load(open(_avi,encoding="utf-8")) if os.path.exists(_avi) else {"sections":[],"sprint":[],"lesson":""}
TRANSLATION_MAP=_A["sections"]; SPRINT_SCORED=_A["sprint"]; SPRINT_LESSON=_A["lesson"]
SPRINT_VERDICT="Too ambitious too quick. It ended up harming rather than helping. These should have been worked on long after the bones were trusted."
PART_II_OUTLINE=[
 ("Opening — the translation map", "Appendix VI as narrative: what the CMB gave us, row by row, and the two places we stopped taking it (a second deconvolver; de-aging)."),
 ("Three kinds of file", "FLOOR (H_min: physics, 40 numbers) / RULER (identity loci: where the gauge reads) / BAND (age reference: a cohort statistic). Every 'healthy reads wrong' case of 2026 traced to a band."),
 ("Stage 0 — intake", "Array type from the IDAT header; SHA-256 with re-transmission detection; the fail-open we closed."),
 ("Stage 1 — calibration", "noob via methylprep; bit-identical on 11/11; the 6-7% normalisation gain between atlas-source betas and Stage 1 output and why the band absorbs it."),
 ("Stage 2 — component separation", "Walther NNLS against the Atlas; presence, not gating; why NILC was cut."),
 ("Stage 3 — the foreground we refused to subtract", "Built age/sex/smoking layers; SOP s104; a galactic foreground is a separate source, the methylome's is the patient."),
 ("Stage 4 — the gauge", "H(beta_mean)/H_min on identity loci, placed in the band; two surfaces (s106); four formula changes in three weeks and the measurement that settled them."),
 ("Stage 4.5 — bidirectional", "Shannon H is symmetric about 0.5; VAL-050's null and VAL-051's recovery."),
 ("Stage 4.6 — the patient sky", "z = (beta - mu)/sigma on HEALPix; the residual map against a reference with per-pixel uncertainty; Plate 05."),
 ("Stage 5 — Mahalanobis Option A", "Eight coefficients as a vector; distance from the age-band centroid; why v1 put healthy at z ~ -22."),
 ("Stage 6 — cellular age", "The band run backwards; three attempts; why it does not calibrate yet."),
 ("The Atlas (several chapters)", "Per-class MCMC over 115 cell types; the flatness lesson (R-hat misled twice, in opposite directions); the brightness posterior; why a sphere."),
 ("What we built before the bones were trusted", "Appendix VII as narrative; the order that should have been."),
 ("For the bootstrapper", "MCMC vs resampling; what a posterior sd buys that a bootstrap CI does not; when a Mahalanobis distance is and is not a p-value."),
]

# FUTURE GOALS — from the scored translation map (Appendix VI) and the scored sprint (Appendix VII): only items not yet built,
# filtered to what the data and tools in hand can support, each with the gate it waits on. Order is the order.
FUTURE_GOALS = [
 # (gate, goal, from map row / sprint item, why realistic, what it needs)
 ("GATE 0 — before anything else", "Rebuild the eight class BANDS from one healthy cohort through one pipeline", "sprint lesson; RECON B1",
  "GSE87571 (n≈730 healthy whole blood, ages 14–94, raw 450K IDATs) through Stage 1 as shipped. Replaces a band compiled from nine papers on nine pipelines. Resolves the stem_adult false alarm and the identity-loci offset in one step.", "6 GB download; ~6 h of Stage 1; a sealed PREREG first"),
 ("GATE 0", "Wire the Stage 0 ↔ Stage 1 intensity hand-off", "map rows 10–11; PROC-STAGE0-01",
  "methylprep already exposes negative-control probes, per-probe detection p, bead counts and X/Y intensities; Stage 0 only has to receive them. Turns four DEFERRED QCs into real ones.", "half a day; no new data"),
 ("GATE 0", "Reproduce the breast pre-diagnostic anchor from raw IDATs", "sprint F1", "GSE51057 (329 IDATs) through the whole chain against the rebuilt band. The first result a reader will check; it should stand on the corrected reference.", "2.8 GB; ~3 h"),
 ("after GATE 0", "Cellular variance as cosmic variance", "map row 3",
  "A cfDNA sample carries a finite number of cell-equivalents; the variance floor that sets is computable from the deconvolved composition and the read depth. Names the fundamental limit on small plasma samples. Analytic; no new data.", "a derivation and one figure"),
 ("after GATE 0", "Transfer function of the chain", "map row 14; CCL-039",
  "Inject a known per-CpG signal into synthetic patients (N6) and measure what fraction survives each stage — the deconvolver 'explains away' composition-like signal and this quantifies how much. The synthetic generator and null runner already exist.", "N6/N7 runs; one table per stage"),
 ("after GATE 0", "Second, independent deconvolver — the right way this time", "map row 20; sprint B2",
  "Planck never trusted a single component-separation method. NILC failed on correlated blood mixtures; a parametric Bayesian (Commander-style) deconvolver on the same Atlas is the natural second, and it fails differently. Agreement within tolerance becomes a gate; disagreement becomes a flag, not a deletion.", "1–2 sessions; the Atlas; Moss 2018 mixtures as ground truth"),
 ("after GATE 0", "Nuisance marginalisation in the gauge", "map row 21",
  "Composition and age enter as point estimates today. Propagating the deconvolver's residual and the band's sd into the gauge A gives every reading an uncertainty — the posterior sd the MCMC atlas already carries but the chain drops at Stage 4.", "half a session; no new data"),
 ("after the anchor", "Banana degeneracy — 2D posterior shape for A-score pairs", "map row 38; sprint C3 ('I never got my banana degeneracy')",
  "For pairs such as immune × cycling or terminal × stem_pluri, map the 2D case-vs-HC distribution; the CIMP axis is the first known one. Cheap once bands are trusted; meaningless before.", "one session; the foundation cohort"),
 ("after the anchor", "C(d): two-point correlation of residuals vs genomic distance, per class", "map rows 17, 23, 25; sprint C1",
  "The methylome's power spectrum. Look for characteristic-scale features (the 'acoustic peaks' of the map), with MASTER-style correction for masked CpGs. Needs CHR/MAPINFO on every CpG (the manifest is in IAM_Atlas/external_manifests).", "one session; the Atlas + anchor cohort"),
 ("after the anchor", "Per-card likelihood, marginalised, with MCMC posteriors", "map rows 28, 37; sprint E2/E3",
  "Replace threshold-plus-band scoring with a proper posterior over per-card parameters, nuisance-marginalised. emcee is already in the toolchain from the Atlas build. This is what a Planck reader will expect L7 to be.", "1–2 sessions; everything above"),
 ("after the anchor", "Formal blinding for confirmation VALs", "map row 78", "Apply the pipeline before case labels are seen; unblind only after the seal. PREREG already does half of this; the other half is a procedural rule and a script flag.", "a checklist change"),
 ("substrates", "Urine, CSF, and the within-patient tissue/plasma/urine trio", "Issue 003 §7",
  "Cohorts already located and partly downloaded (GSE119260 four men × three substrates; GSE292312 and GSE269403 CSF). Each is one experiment with a declared prediction: the class present in the shed tissue should appear in its fluid and be absent from the same patient's blood.", "data in hand; PREREG each"),
 ("not now", "Bispectrum / trispectrum; Minkowski functionals; isotropy and alignment tests; spectral distortions", "map rows 45, 53–60, 69",
  "Real analogs, genuinely novel, and every one of them needs a trusted two-point function first. Listed so they are not forgotten; not scheduled.", "after C(d)"),
 ("not now", "Multi-substrate 5mC/5hmC (E/B separation); multi-omics cross-correlation", "map rows 4, 49, 70", "Tier 5. Needs oxBS or matched RNA-seq/ATAC on the same samples — data the project does not hold.", "new data"),
 ("does not translate", "Rees–Sciama; Rayleigh scattering", "map rows 63, 68", "The author's own ✗ rows; kept for completeness.", "—"),
]

# PROC-N7-01 — end-to-end synthetic simulation, 2026-09-19. THE FINDING OF THE DAY.
N7_01 = {
 "input": "synthetic_patient_generator.py (restored from RETIRED; patched to read IAM_Atlas/IAMAtlasREBUILD.csv; composition_alpha added): 16 healthy + 8 case, whole-blood composition (~immune 0.89, progenitor 0.08, stem_adult 0.03), disease panel 500 CpGs on the immune identity loci, signal 2.0. Each patient is a linear mix of the Atlas class posterior means plus age/sex/batch loadings and noise 0.03. Run through cpg_conductor.run_full.",
 "R1 composition": "PASS - deconvolver recovers every class: immune MAE 0.0095, progenitor 0.0145, stem_adult 0.0116, epithelial classes <= 0.0016 (r 0.79-0.98 on the legacy mix).",
 "R2 gauge": "FAIL - every synthetic patient, healthy or case, reads immune A 1.12-1.13 ABOVE_BAND / BREACH; case and healthy do not separate.",
 "root cause": "cpg_conductor.stage_b_classes computes H(beta_mean)/H_min over the class MARKER UNION (2,952 bimodal CpGs for immune; comment: 'PRODUCTION A-score (GAPE_WEB_v13 + Reproduction Paper v3) ... NOT identity loci'), while its own docstring, SOP s41/s106 and Issue 002 say identity loci (42,134 unimodal CpGs). Confirmed: ge.read on the marker-union beta_mean returns 1.1303, the conductor's number, to four decimals. The marker union's beta_mean depends on the SHAPE of the input (Jensen gap 0.257 vs 0.026 on identity loci): a smooth synthetic mixture pulls it to 0.63 (BREACH); real bimodal blood lands at 0.75 (NORMAL).",
 "same samples, both statistics": [
  ("synthetic healthy (pure mixture of healthy posteriors)", "identity 0.9875", "marker-union 1.1303 BREACH"),
  ("real WB healthy 43M", "identity 0.874", "marker-union 0.957 IN_BAND"),
  ("real WB healthy 58M", "identity 0.807", "marker-union 0.960 IN_BAND"),
  ("real WB RA", "identity 0.879", "marker-union 1.002 IN_BAND"),
  ("real tissue adenoma GSM8772492", "identity 1.096 (elevated)", "marker-union 0.932 IN_BAND (missed)"),
 ],
 "consequences": [
  "RULING A3 was right about the formula (H(beta_mean) on identity loci) and WRONG in asserting that the wired chain computes it. Corrected in s1.5.",
  "age_reference_matrix.json was compiled (GAPE_WEB_v13 _AGE_REFERENCE) on the marker-union statistic. PROC-CHAIN-01's '7/7 healthy IN_BAND' is the marker-union gauge agreeing with the marker-union band - self-consistency, not conformance. Withdrawn as evidence of a working gauge.",
  "s6a's 'below band' reading compared identity-loci A against a marker-union band: a statistic mismatch, not an offset. The +0.05-0.09 beta shift at identity loci vs the Atlas posterior IS real and remains Phase 1's business.",
  "The identity-loci gauge is the correct statistic - unimodal, passes N7, carries the adenoma signal - and has NO BAND. Phase 1 (rebuild the band on identity-loci H(beta_mean) from GSE87571 through Stage 1) is now a prerequisite for any gauge reading, not a refinement.",
  "The conductor is NOT switched today (a statistic without its band would read every healthy patient below band). Every class-gauge reading now carries gauge_surface = 'marker_union' and n_cpgs so it cannot be mistaken.",
 ],
 "verdict": "N7 did its job: the synthetic healthy patient exposed a gauge that real blood had been masking. Composition PASS; gauge as wired FAIL; identity-loci gauge PASS on N7 and blocked on its band.",
}
RULING_A3["status"] = ("RULING STANDS FOR THE FORMULA; CORRECTED ON THE CHAIN 2026-09-19 (PROC-N7-01): the wired conductor computes H(beta_mean) over the class MARKER UNION, not identity loci, "
    "and age_reference_matrix was compiled the same way. The ruling's target - identity-loci H(beta_mean) against an identity-loci band - does not exist yet; Phase 1 builds it. Until then the conductor's gauge is labelled gauge_surface='marker_union'.")
CHAIN01["findings"] = [(k, (v + " [WITHDRAWN as conformance by PROC-N7-01: this is the marker-union gauge agreeing with the marker-union band. See N7_01.]") if k=="healthy whole blood, immune" else v) for k,v in CHAIN01["findings"]]
FALSIFICATION += [
 ("PROC-CHAIN-01 'healthy blood IN_BAND 7/7 = the shipped chain absorbs the offset'", "the conductor's gauge and its band are both marker-union statistics; agreement between them is self-consistency, not validation (PROC-N7-01)", "WITHDRAWN"),
 ("RULING A3 'the wired chain computes H(beta_mean) on identity loci'", "the wired chain computes it on the marker union (cpg_conductor.stage_b_classes, comment citing GAPE_WEB_v13)", "CORRECTED"),
 ("s6a 'healthy whole blood reads below band'", "identity-loci A compared against a marker-union band - statistic mismatch; the beta shift at identity loci is real, the band conclusion was not", "RESTATED"),
]
RECON += [("A4", "which CpG set the production gauge reads", "docstring / SOP s41 / Issue 002: identity loci", "code: class marker union (iamatlas_celltype_markers_v0_2.json), with age_reference_matrix compiled on the same", "identity loci, once Phase 1 supplies their band; label until then", "PROC-N7-01")]

# ═══════════════════════════════════════════════════════════════════════════════
# WHAT THE COSMOLOGY TOOLS FOUND THAT COHORTS COULD NOT — the standing evidence ledger.
# Rule (RUNBOOK §10): every time a CMB-derived method surfaces something a cohort comparison could not have,
# it gets a row here, the same day. This is the pre-built answer to "your reasoning is circular".
# Columns: date, CMB method, why a cohort is blind to it, what was found, record.
# ═══════════════════════════════════════════════════════════════════════════════
COSMO_EVIDENCE = [
 ("2026-09-19", "End-to-end simulation with known truth (FFP / mock-sky discipline; null N7)",
  "A cohort supplies a comparison, never a truth. Two statistics that are wrong in the same way agree on every cohort.",
  "The production class gauge read H(beta_mean) over the bimodal MARKER UNION, not the identity loci; the age band had been compiled on the same statistic, so every real cohort for eleven weeks read 'healthy IN_BAND'. A synthetic healthy patient - a pure mixture of healthy Atlas posteriors - read BREACH (1.13) on the first run. Identity-loci gauge read 0.99 on the same input and 1.10 on real adenoma where the production gauge read NORMAL.", "PROC-N7-01; RECON A4"),
 ("2026-09-19", "Component separation validated on known mixtures (Planck FFP component maps; here Moss 2018 Table 6 in-vitro mixes)",
  "Bulk-tissue cohorts never expose the mixing fractions; a deconvolver can be systematically wrong and still order cohorts correctly.",
  "Neuron spikes recovered (r = +0.945, under-read 10% -> 6.5%); hepatocyte and colon spikes NOT recovered (r = +0.19, +0.10) - the atlas routes shed epithelium to gastric references. Tissue-of-origin claims withdrawn until the atlas recovers Table 6; composition departure itself confirmed real.", "PROC-PLASMA-MIX-01; LESSON-DECONV-01"),
 ("2026-09-19", "Transfer-function decomposition (what each pipeline stage does to a known input)",
  "On real data every stage's effect is confounded with biology; only a constructed input isolates the stage.",
  "Term-by-term decomposition of one synthetic patient (composition -> age loading -> sex -> batch -> noise -> clip) showed every generator term moved the identity-loci gauge by < 0.004; the +0.14 lived entirely inside Stage B. That is what located the marker-union code path in one step.", "PROC-N7-01 decomposition"),
 ("2026-05 -> 2026-09", "Half-mission x half-mission cross-check (two independent cohorts as two detector halves)",
  "A single cohort can carry a plate or preprocessing offset that looks like signal; only an independent half exposes it.",
  "GSE51032 x GSE51057 breast anchors: the effect replicates across both halves (d = +2.088 vs +2.097 in the sealed record); the whole 648-sample foundation cohort reproduced from raw GEO at r = 1.00000. Separately, GSE53740's healthy controls sat +2.3 SD above the 80-cell baseline - a cohort offset that a single-cohort analysis would have read as disease (CCL-004).", "PROC-ANCHOR-01; CCL-004"),
 ("2026-06", "Convergence diagnostics are not a truth test (the R-hat lesson from MCMC map-making)",
  "Cohort validation has no analogue of a chain diagnostic at all; it cannot see that a reference is flat.",
  "The first Atlas build showed R-hat ~1.01 (looks perfect) while every cell type in a class came out identical - a flat, useless atlas. The fixed build showed R-hat 1.4-2.6 (looks alarming) with distinct cell types. Judge a rebuilt class by the distinctness test, never by convergence alone.", "IAM_Atlas/IAMAtlas_FLATNESS_LESSON.md"),
 ("2026-06", "Component-separation cross-validation (Commander / NILC / SMICA discipline) - the reversal",
  "A second method fails differently from the first; with one method, the chain is 'talking to itself' (sprint sign-off question 3).",
  "NILC was built as the second deconvolver and CUT: it collapsed on correlated blood mixtures and deleted correct calls. Recorded as the one Planck principle the chain knowingly does not follow, with the reason. SUPERSEDED 2026-09-19: NILC was rerun as designed (PROC-NILC-01) and vindicated - its divergence marked the immune/progenitor/stem_adult indeterminacy - and the two-tool design (PROC-SEP-03) replaced the single solve. RUNBOOK s11: neither relay may be disabled again.", "commit c1be0c3; Appendix VI row 20; Appendix VII B2"),
 ("2026-05", "Look-elsewhere correction (null N8)",
  "A cohort scan over many features will always find one that separates; without the correction the finding is published.",
  "VAL-006's chr6 (MHC) signal died under look-elsewhere correction and was recorded as a null rather than a discovery.", "VAL-006; null N8"),
 ("2026-04 -> 2026-05", "Pre-registration with sealed prediction and direction (the CMB blinding culture)",
  "A cohort analysis performed after the labels are seen can accommodate any sign.",
  "VAL-061 predicted the wrong sign and was recorded as such, which produced CCL-019/020 (direction depends on class x compartment, not disease). VAL-102 was VOIDED four minutes after sealing for a post-hoc accommodation, seal preserved. VAL-128 failed opposite to its prereg and stands as a FAIL.", "CCL-019; CCL-020; VAL-102; VAL-128"),
 ("2026-09-19", "Foreground reasoning applied in reverse (delensing refused)",
  "Cohort clocks subtract age as a nuisance; that removes the patient's own biology along with it.",
  "Built age/sex/smoking subtraction layers were refused under SOP s104: a galactic foreground is a separate source, the methylome's 'foreground' is the patient. The band (annotate) replaced subtraction (de-age). Recorded as the second place the CMB analogy is deliberately broken - and why.", "Appendix VI row 47; SOP s104"),
]
COSMO_EVIDENCE_RULE = ("Every time a CMB-derived method surfaces something a cohort comparison could not have, it gets a row in this ledger the same day, with its PROC or VAL. "
    "This is not a list of successes; the NILC reversal and the withdrawn tissue-of-origin claim are here too. It is the record that answers the circularity objection: "
    "we do not ask cohorts to validate the instrument, because they cannot - we ask constructed truth, split halves, injection-recovery and pre-registration to do it, and here is what they found.")

RECON += [("D3", "why NILC and Walther never agreed", "flowchart: 'collapsed on correlated blood mixtures and deleted correct calls'",
  "NILC (RETIRED/NILC_Deconvolver_cut_from_chain_2026-07-02) solved UNCONSTRAINED generalized least squares on the CELL-TYPE MARKER POOL (iamatlas_celltype_markers_v0_1.json, bimodal one-vs-rest), then projected onto the simplex; Walther solves NNLS with a simplex constraint on its own top-600-per-class CLASS-discriminating CpGs. In blood the immune / progenitor / stem_adult reference columns are nearly parallel, so the unconstrained inverse is ill-conditioned: fractions swing negative, the projection zeroes them - 'deleted correct calls'. Not a second opinion; the unregularised version of the same inverse problem on a worse-conditioned CpG set.",
  "third instance of one lesson: the cell-type marker pool is a SEPARATION surface only (gauge -> PROC-N7-01; NILC basis -> here; the 2026-06-11 all-BREACH bug). A second deconvolver must fail DIFFERENTLY: parametric Bayesian on Walther's class-marker set with a simplex prior (Future Goal 6).", "NILC docstring; c1be0c3; PROC-N7-01")]
FUTURE_GOALS = [(g if g[1] != "Second, independent deconvolver — the right way this time" else
  (g[0], g[1], g[2], g[3] + " Lesson from the first attempt (RECON D3): NILC ran unconstrained GLS on the bimodal cell-type marker pool and was ill-conditioned on blood's near-parallel immune/progenitor/stem_adult columns; the replacement must solve on Walther's class-marker ruler with a simplex prior, so that it differs in principle rather than in stability.", g[4])) for g in FUTURE_GOALS]

# PROC-NILC-01 — the retired second deconvolver rerun as designed, 2026-09-19
NILC_01 = {
 "input": "Walther's own class-marker reference X (1,256 CpGs x 8 classes, built by the engine at load); the 11 Stage-1 test samples; NILC's stated algorithm (unconstrained least squares on X, then simplex projection) run on the SAME X for a fair comparison",
 "conditioning": "condition number of X: 47.0 (all 8); 30.6 for the blood sub-problem (immune, progenitor, stem_adult); 8.8 for the epithelial trio. Centred column correlations: progenitor-stem_adult +0.99, immune-progenitor +0.94, immune-stem_adult +0.93.",
 "result": "per-patient L1 disagreement 0.55-0.65 on every blood sample (design target < 0.05); NILC places 0.26-0.34 in stem_adult where Walther places 0.005-0.044, and deletes secretory and stromal; on tissue it drives progenitor negative (-0.16 to -0.23). Exactly the 'collapsed on correlated blood mixtures, deleted correct calls' of the cut.",
 "reading": "By NILC's own design rule, divergence marks where composition is 'genuinely ill-defined by the atlas reference'. It was: in blood the Atlas cannot separately determine immune, progenitor and stem_adult; Walther's 2-4% stem_adult is chosen by the simplex constraint, not by the data. That is the class that read false BREACH in 6/7 healthy donors (PROC-CHAIN-01). The second deconvolver diagnosed the stem_adult problem in June; it was cut for delivering the message.",
 "recommendation": "(1) Whole blood: report progenitor + stem_adult as ONE haematopoietic-progenitor component until the Atlas separates them (kappa of the blood sub-problem must fall below ~10 first). (2) Pre-flight: compute the substrate sub-problem's condition number alongside the Jensen gap. (3) A second deconvolver stays a goal, but its disagreement is a diagnostic to be read, not a gate to be failed.",
}
COSMO_EVIDENCE.insert(1, ("2026-09-19 (rerun of 2026-06)", "Cross-method disagreement read as a diagnostic (Commander / NILC / SMICA discipline)",
  "A single-method cohort pipeline cannot know which of its fractions the reference actually determines and which the solver's constraint chose.",
  "NILC rerun as designed on Walther's own reference: L1 disagreement 0.55-0.65 on every blood sample, mass moving between progenitor and stem_adult (reference columns r = +0.99, blood sub-problem condition number 30.6). The Atlas cannot separately determine those classes in blood; Walther's 2-4% stem_adult is constraint-chosen. That class produced the false BREACH alarms of PROC-CHAIN-01. The second method had diagnosed it in June and was cut for it.", "PROC-NILC-01; RECON D3"))
RECON = [r if r[0]!="D3" else (r[0], r[1], r[2], r[3] + " MEASURED (PROC-NILC-01): kappa(blood sub-problem) = 30.6, r(progenitor, stem_adult) = +0.99; NILC L1 vs Walther 0.55-0.65 on all blood.", "the divergence was the diagnostic NILC was designed to give: stem_adult/progenitor are not separately determined by the Atlas in blood. Report them jointly for whole blood until they are; a second deconvolver's disagreement is to be read, not failed.", r[5]) for r in RECON]
FUTURE_GOALS.insert(1, ("GATE 0", "Merge progenitor + stem_adult for whole blood; add condition number to pre-flight", "PROC-NILC-01",
  "The Atlas reference columns for the two classes are r = +0.99 at the deconvolver's markers; the split between them is chosen by the simplex constraint, and the stem_adult band then flags healthy donors. One joint haematopoietic-progenitor component removes the false alarm without touching any floor. kappa of the substrate sub-problem joins the Jensen gap as a pre-flight number.", "a config change in the conductor + the band rebuild of Phase 1"))

# stem_adult in the validation record (grep of every OUTCOME file, VAL_INDEX, disease cards, 2026-09-19) + the presence rule
STEM_ADULT_RECORD = [
 ("VAL-008 AD (AIBL), per cell type", "HSC d = -0.329 ***, third of eight class top-hits", "immune top hit Eosino d = -0.426 ***; progenitor L-MPP -0.385 (per-class top-hits table, CPG_VAL_008_OUTCOME.md)"),
 ("VAL-016 cross-disease, AIBL arm (n = 161 AD / 471 HC)", "class d = -0.329, ranked 2nd by |d|", "immune d = -0.364, 1st; progenitor -0.316, 3rd (per-cohort table, OUTCOME.md)"),
 ("VAL-011 AD age subtraction", "d = -0.004 -> -0.190 after age subtraction", "'interesting biology'; never a call"),
 ("VAL-015 / VAL-020 immune aging", "r = -0.103, 'weak aging signal'", "-"),
 ("VAL-018 menarche", "~0", "null"),
 ("VAL-022 smoking cessation", "the only class with true reversibility (former/never ratio 0.66)", "flagged 'worth investigating later' - the one distinctive stem_adult result"),
 ("VAL-135 (retired, off-scope)", "1.083 'ELEVATED - blood-stem-cell content'", "-"),
 ("disease cards (all)", "none names stem_adult as origin class", "Issue 002 targets: HSC-origin AML, CHIP, HSC aging - haematology, not the breast/CRC/AD work"),
 ("age band", "n = 4-32 per decade, sorted CD34+ HSC (Adelman 2019), two decades extrapolated", "purified stem cells, no whole-blood context"),
 ("fraction in blood", "2-4%, constraint-chosen (r = +0.99 with progenitor, PROC-NILC-01)", "false BREACH in 6/7 healthy donors (PROC-CHAIN-01)"),
]
PRESENCE_RULE = {
 "statement": "REPORTING RULE (extends the s3.2 presence gate, DETECT_FLOOR = 1%, which decides whether a class is in the sample; this decides whether its gauge is put on the report). A class gauge is REPORTED on a substrate only if the class is (1) DETERMINED - its fraction is separable from its neighbours in the Atlas on that substrate (condition number of the substrate sub-problem < ~10), or it is reported jointly with what it cannot be split from - and (2) PRESENT - its typical fraction on that substrate is >= 5% (above the conductor's 1% detection gate and the adjudicator's 3%), so that beta_mean over its loci is the class and not the background. Classes failing either condition are composition-only (present/absent, no A).",
 "whole blood": "REPORTED: immune. REPORTED JOINTLY: progenitor + stem_adult as one haematopoietic-progenitor component. COMPOSITION-ONLY: cycling, secretory, terminal, stromal, stem_pluri. Decided from kappa and fraction BEFORE any A is computed; written into the Phase 1 PREREG as a pass condition.",
 "rationale": "Risk over reward (author, 2026-09-19): a class that has never carried a finding, whose fraction the data does not determine, and whose band rests on n = 4-32 sorted cells, is a false-alarm source on the patient report and nothing else. Classes earn their way back onto a substrate's report as separability and presence are demonstrated there (plasma cfDNA is expected to qualify cycling and secretory).",
 "keep": "VAL-022's stem_adult reversibility (ratio 0.66) is retained as a note - the one distinctive result - testable once a joint component and a real band exist.",
}
FUTURE_GOALS = [(g if not g[1].startswith("Merge progenitor + stem_adult") else (g[0], "Presence rule: report a class gauge only where DETERMINED (kappa < ~10, or joint) and PRESENT (>= 5%); whole blood -> immune + one haematopoietic-progenitor component", "PROC-NILC-01; author's risk-over-reward rule", g[3] + " stem_adult has carried zero findings in nine VALs; its band is n = 4-32 sorted HSC.", g[4])) for g in FUTURE_GOALS]
RULES += [("L-10", "Reporting rule (extends the s3.2 presence gate)", "a class gauge is REPORTED on a substrate only where the class is DETERMINED (kappa of the substrate sub-problem < ~10, or reported jointly with what it cannot be split from) and PRESENT (typical fraction >= 5%); otherwise composition-only", "whole blood: immune reported; progenitor + stem_adult jointly as one haematopoietic-progenitor component; the other five composition-only. Decided from kappa and fraction BEFORE any A is computed", "author, 2026-09-19, after PROC-CHAIN-01 / PROC-NILC-01; SOP s108")]

PRESENCE_RULE["scope"] = ("GAUGE-LEVEL ONLY. The merge applies to the class gauge (the call) and to nothing else. The separation surface keeps HSC, MPP, CMP, GMP, erythroid progenitors, neutrophils and every other cell separate, "
  "so disease direction is read exactly as the disease matrix (v1.13) writes it.")
PRESENCE_RULE["myeloid check"] = ("The author asked where stem_adult is EXPECTED to matter; the matrix answers: CHIP->AML (HSC +0.3/+0.7; GMP/CMP/MPP +0.5/+1.0), CML (HSC +0.5/+1.0; neutrophils and GMP +1.0/+2.0), MDS (HSC +0.3/+0.7; GMP and neutrophils -0.5/-1.0), and AD (HSC -0.33 with immune -0.36 and progenitor -0.32 - the whole haematopoietic compartment moving together). "
  "In every row HSC is the smaller entry; none of the three myeloid rows is validated ('heme-epic v0.1 myeloid arm pending validation'). CML and CHIP: the joint gauge reads UP (GMP and HSC move together) and trips the call. "
  "MDS is the test case: HSC up, GMP down - opposite signs inside the merged group. By fraction the joint gauge still reads down and trips the call; the HSC-up / GMP-down divergence that separates MDS from AML lives on the separation surface, which the rule does not touch. MDS is why the rule must be gauge-only.")
PRESENCE_RULE["path back"] = ("stem_adult earns a separate gauge on whole blood through the myeloid arm: MDS / CML / CHIP cohorts (GSE62298 AML already cited; GSE63409 GMP d = -1.95 as sister case) through the full chain with the heme card, with the joint component and the separate stem_adult gauge both reported, sealed PREREG first. If the separate gauge adds a call the joint one misses, it comes back.")
FUTURE_GOALS.insert(3, ("after GATE 0", "Myeloid arm: MDS / CML / CHIP through the chain - the test of whether stem_adult earns its own gauge", "disease matrix v1.13; reporting rule",
  "The three diseases where HSC is expected to move are unvalidated ('heme-epic v0.1 pending'). Running them with the joint component AND the separate stem_adult gauge side by side decides the class's status on blood by measurement rather than by rule. MDS (HSC up, GMP down) is the discriminating case.", "GSE62298 + an MDS cohort; PREREG; the rebuilt band"))

# PROC-SEP-01 — is the HSC / progenitor information in the Atlas? (2026-09-19)
SEP_01 = {
 "question": "The Atlas cannot split stem_adult from progenitor in blood (kappa 30.6, r = +0.99). Is that because the information is absent (a reference-data problem) or unused (a marker-selection problem)?",
 "measurement": "Of 482,421 Atlas CpGs with both class means: 1,290 separate stem_adult from progenitor by > 0.2 beta (0.27%); 228 by > 0.3; 4,265 exceed 3x the combined posterior SD. For scale, 15,939 separate stem_adult from immune by > 0.2. Of Walther's 7,114 class markers, 129 split the pair by > 0.2.",
 "diagnosis": "The information exists and is barely used. Walther ranks each class's markers by separation from the FIELD; CpGs separating haematopoietic classes from epithelium are plentiful and win; the few hundred separating HSC from MPP/CMP/GMP mostly do not make the cut. Structural cause: stem_adult is ONE cell type (HSC) while progenitor is ELEVEN (CMP, GMP, MPP, L-MPP, MEP, erythrocyte progenitors, erythroblast, megakaryocyte, nRBC, NeuIm, OPC) - the class boundary is drawn one step down a continuous lineage, so the two class means are adjacent points on the same path. Contributing: HSC/MPP/CMP/GMP references come from the same few sorted-CD34+ studies (Roadmap E035, Adelman 2019), so study offsets inflate the agreement.",
 "remedy": [
  ("1 - contrast-specific markers (cheap; this week)", "for every class pair with kappa > 10, force in the top-N CpGs separating THAT pair (the 1,290 are already in the Atlas). Target: blood sub-problem kappa < 10. Test: N7 and PROC-NILC-01 rerun - if Walther and the unconstrained solve converge on stem_adult, the split has become data-determined."),
  ("2 - hierarchical composition (moderate)", "deconvolve haematopoietic as one component, then split it using contrast markers only - coarse-to-fine; removes the epithelial CpGs' vote on a question they cannot answer."),
  ("3 - independent HSC references (OUT OF SCOPE)", "a second sorted-HSC source would break the shared-donor coupling, but it is not required for the claim. Author, 2026-09-19: this project proves the MCMC atlas works; it does not join more atlases. Noted for whoever builds the next one."),
 ],
 "framing": "An ATLAS result, in the atlas's favour: the MCMC atlas already holds the HSC / progenitor split (1,290 CpGs; 4,265 above 3 sigma); its consumer was not asking for it. Steps 1-2 use the Atlas alone. If pairwise-forced markers bring the blood sub-problem from kappa 30.6 to single digits, that demonstrates the atlas resolves a lineage step its own deconvolver could not - which is exactly the claim this project makes. Goes in the ledger when step 1 lands.",
}
FUTURE_GOALS.insert(2, ("GATE 0", "Contrast-specific markers in Walther: split stem_adult from progenitor with the 1,290 CpGs the Atlas already holds", "PROC-SEP-01",
  "The information is in the Atlas (1,290 CpGs > 0.2 beta apart; 4,265 above 3 sigma) and the deconvolver uses ~129 of them because it ranks markers by separation from the field, not from the neighbour. A pairwise-forced marker step should bring the blood sub-problem from kappa 30.6 to single digits; N7 + PROC-NILC-01 are the test. Then hierarchical (coarse-to-fine) composition. No new atlases: the claim is that THIS atlas works.", "one session; Atlas only"))

# SCOPE STATEMENT — the author's positioning, 2026-09-19, placed before any table
SCOPE = {
 "not_claimed": [
  "the world's largest atlas, or a new MCMC atlas built to sell",
  "that one atlas is the right structure - a future builder may well use one per cell class, or a classification not yet conceived; that is fine",
  "detection of all diseases, or of any disease ten years early",
  "detection of the myeloid cancers (MDS, CML, CHIP) - those matrix rows are unvalidated and say so",
  "clinical readiness of any kind - nothing here is validated for patient care",
 ],
 "field": "This work is LANDAUER METROLOGY: the measurement of how far above the thermal noise quantum k_B T an information-writing process operates, against a fixed physical zero, in any substrate. Applied to the methylome it is the PHYSICS OF METHYLATION. Thermal noise is not the nuisance here; it is the unit: M = E_drive / k_B T. The Landauer bound is the peer-reviewed anchor (s0b); the metrology - calibration, transfer, absolute reading, reference interval - is what this document adds.",
 "claimed": ("A method biology did not know it needed: the physics of methylation - a per-class thermodynamic reference, an MCMC atlas as the posterior, and the CMB toolkit (end-to-end simulation, component separation, cross-method comparison, split-half replication) as the validation discipline. "
             "On a very limited public dataset with no funding or institutional support, this method already resolves things that cohort comparison structurally cannot (s1.6). For the immune class in whole blood there are reproducible signals worth investigating with proper support. That is the claim, and all of it."),
 "invitation": "The record is written so that a reader can break it: every constant loaded from a file, every procedure runnable from the kit, every reversal kept. The intended reader is the one who says 'cohorts, circular' first - and then pulls the repository.",
}

# PROC-SEP-02 — contrast-specific markers, tested (2026-09-19)
SEP_02 = {
 "change": "WaltherIAMDeconvolver gains contrast_pairs / n_contrast_markers_per_pair: for each named class pair the top-N CpGs by |mean_a - mean_b| are forced into the class reference. Default OFF; no sealed result changes.",
 "result": "baseline: 1,256 markers, kappa(all) 47.0, kappa(blood) 30.6, r(prog, stem_adult) +0.987, blood L1 Walther-vs-unconstrained 0.59. +300/pair: 1,335 markers, 32.8 / 22.0 / +0.974 / 0.47. +800/pair: 1,567 markers, 26.1 / 17.0 / +0.954 / 0.60 (unconstrained drives progenitor negative, stem_adult to ~0.3 on every blood sample).",
 "reading": "Conditioning improves monotonically - the contrast CpGs act on the reference exactly as intended - but kappa(blood) does not reach 10 and the two solvers do not converge. With the Atlas alone the HSC / progenitor split is not data-determined at the marker level. Walther's own stem_adult estimate in healthy blood falls from 0.017-0.044 to ~0.000 (6/7 at +300) once it has the contrast information: the 2-4% was the constraint's, not the blood's - the same thing NILC said in June.",
 "verdict": "Passing test of the METHOD (the Atlas holds the information and it moves kappa the right way); NEGATIVE result for the shortcut (marker selection alone does not make stem_adult reportable on blood). The reporting rule stands as the operating position: whole blood reports immune, plus progenitor + stem_adult as one haematopoietic-progenitor component (coarse-to-fine, which never asks the ill-conditioned question). Step 3 remains out of scope by the author's decision.",
 "ledger": "Second cross-method row: the disagreement metric, run under three marker configurations, located the limit of what this Atlas determines in blood - a measurement no cohort comparison can make.",
}
FUTURE_GOALS = [(("after GATE 0", "Coarse-to-fine composition for whole blood: haematopoietic as one component, split only where kappa allows", "PROC-SEP-02",
  "Contrast markers moved kappa(blood) 30.6 -> 17.0 and r(prog, stem_adult) 0.987 -> 0.954 but did not reach kappa < 10; the split is not data-determined at the marker level with this Atlas. The joint component is the operating position; the option stays in Walther (default off) for the next atlas.", "half a session; Atlas only")
  if g[1].startswith("Contrast-specific markers in Walther") else g) for g in FUTURE_GOALS]
COSMO_EVIDENCE.insert(2, ("2026-09-19", "Cross-method disagreement as a measuring instrument (three marker configurations)",
  "A cohort cannot tell which of a deconvolver's fractions the reference determines; only two solvers on the same reference can.",
  "Pairwise-contrast markers moved the blood sub-problem from kappa 30.6 to 17.0 and the two solvers still did not converge; Walther's own stem_adult estimate fell to ~0 once given the contrast information. Located the limit of what this Atlas determines in blood and confirmed the reporting rule's joint component as the correct operating position.", "PROC-SEP-02"))

# PROC-SEP-03 — the two-tool design: compartment deconvolver + lineage splitter (2026-09-19)
SEP_03 = {
 "design": "Tool A = WaltherIAMDeconvolver, unchanged: 'what is in the tube', field-ranked class markers, NNLS on the simplex. Tool B = CPG_Engine/Lineage_Splitter/lineage_splitter.py (new): takes ONE compartment Tool A found (default haematopoietic = progenitor + stem_adult) and asks only how it divides, using ONLY the CpGs where the member classes differ (|delta| >= 0.2 or 0.3), weighted by |delta|, after subtracting the other classes' contribution at those CpGs; solves a small NNLS on the compartment mass; carries its own condition-number check (kappa_max 10). Ill-conditioned -> COMPARTMENT_ONLY with the reason; it cannot manufacture information, it reports whether it is there.",
 "conditioning": "single solve on field markers: 1,256 CpGs, kappa 30.6, r(prog, stem_adult) +0.99. Lineage splitter |delta|>=0.2: 1,290 CpGs, kappa 5.5, r +0.43. |delta|>=0.3: 228 CpGs, kappa 4.2, r -0.18. Restricted to the CpGs where the classes differ, the columns stop being parallel and the split is well determined.",
 "result": "7/7 whole-blood test samples: SPLIT, kappa 3.9-5.2, progenitor = the whole compartment, stem_adult = 0.000. Compartment fractions per sample: 0.166, 0.107, 0.005 (GSM2333950, whose Tool A progenitor is 0.000 and stem_adult 0.005), 0.161, 0.131, 0.068, 0.197 - range 0.005-0.197. Tissue with no haematopoietic-progenitor mass: COMPARTMENT_ONLY, kappa = inf, 'nothing to split' (correct). CRC-1 tissue: SPLIT 0.010 / 0.004.",
 "reading": "Three independent routes now agree: NILC (June), Walther-with-contrast-markers (PROC-SEP-02), and the lineage splitter (this) all find stem_adult ~0 in healthy whole blood at array resolution; Tool A's 2-4% was the simplex constraint's choice. Biologically sensible (HSC ~0.01% of nucleated blood cells). The reporting rule's joint component is right for blood because the resolved stem_adult share is zero - a mechanism, not just an ill-conditioning excuse.",
 "myeloid": "MDS (HSC up, GMP down) now has its instrument: Tool B is what would see HSC content depart from zero on a well-conditioned problem with an uncertainty. Testable in the myeloid arm.",
 "verdict": "The Atlas resolves the lineage step when asked on the right vocabulary. The single solve reported an arbitrary split because 480,000 uninformative CpGs outvoted 1,290 informative ones. An ATLAS result, in the atlas's favour (Atlas only; no new data).",
}
COSMO_EVIDENCE.insert(3, ("2026-09-19", "Coarse-to-fine component separation (Planck's multi-scale / needlet discipline) - two tools, two questions",
  "A single solve has no way to know that most of its CpGs are uninformative for one particular split; a cohort has no way to know the split is arbitrary.",
  "A lineage splitter restricted to the 1,290 CpGs where HSC and progenitor differ brought the sub-problem from kappa 30.6 (r +0.99) to kappa 5.5 (r +0.43) and resolved the split on every blood sample: stem_adult = 0.000 in 7/7, agreeing with NILC and with contrast-marker Walther. Where there was no compartment it said so (kappa = inf) instead of inventing one.", "PROC-SEP-03"))
FUTURE_GOALS = [(("GATE 0", "Wire Tool B (lineage splitter) into the conductor behind the reporting rule; run it in the myeloid arm", "PROC-SEP-03",
  "Tool B resolves HSC vs progenitor at kappa ~4-5 on the contrast CpGs and reports stem_adult = 0 in healthy blood; it is the instrument that would see HSC content rise in MDS / CML / CHIP. Wire it so the joint component is reported by rule and the split is offered only when Tool B's kappa clears the bar, with its uncertainty.", "half a session + the myeloid cohorts")
  if g[1].startswith("Coarse-to-fine composition for whole blood") else g) for g in FUTURE_GOALS]

# PHASE 1 (2026-09-20) — identity-loci band from GSE87571 through Stage 1; sealed FAIL on P3/P4 with the cause measured
PHASE1 = {
 "verdict": "FAIL on P3 (synthetic healthy in band 0/16) and P4 (test WB in band 3/7) as sealed; P1 732/732; band built n=560 and kept (identity_band_v1.json). N-random 0/16 (pass). Spearman(age, immune A) = +0.52.",
 "cause": "Three beta scales on the same 42,024 immune identity loci: Roadmap/Atlas 0.737 (A 1.00, where G-002 calibrated H_min); GEO author-processed EPIC 0.774 (A 0.92; the anchor cohort); Stage-1 noob raw 450K 0.815 (A 0.82). The offset is additive (+0.066 beta).",
 "why_now": "Author's April 2026 record (VAL-003 output): 'G-002 H_min calibrated on Roadmap (GenomicStudio); TCGA uses sesame; ~10% offset expected; delta-A valid within-pipeline; absolute thresholds require a pipeline-matched healthy reference.' Every VAL since was within-pipeline and immune by design. Phase 1 is the first absolute reading of raw-IDAT beta against the floor with no cohort in the loop.",
 "mcmc": "UNCHANGED. The H_min floors are correct on the Roadmap scale; the analyst's recommendation to re-derive them on Stage 1 is WITHDRAWN. The fix is a per-pipeline affine map onto the Roadmap scale, fit on healthy blood - what the April note prescribed and Phase 1 finally built.",
 "after_map": "Swedish healthy immune A 0.990 (p10-p90 0.962-1.016): healthy sits at the floor. P3 15/15 (post-hoc; map fit on the band cohort, so a consistency check). P4 3/7 unchanged: the four out read ABOVE band by 0.01-0.02 A - a lab batch residual on top of the pipeline offset, the size of the band half-width; N-plate is its test.",
 "layers": "FLOOR (Roadmap scale, MCMC, physics) -> PIPELINE (+0.066 beta per pipeline, affine map) -> LAB (0.01-0.02 A per cohort). Cohort-relative statistics cancel layers 2-3 and never see them; the absolute gauge sees all three.",
 "next": "Phase 1c prereg: fix the map on GSE87571, test on an independent Stage-1 healthy cohort; N-plate from Sentrix IDs (done: PHASE 1c, band_v2); the sex-split band proposed here was WITHDRAWN the next day (sign flips between labs); P2 restated as immune + haem-progenitor >= 0.85 (adopted). Phase 1b (EPIC-Italy portability) proceeds.",
}
COSMO_EVIDENCE.insert(4, ("2026-09-20", "Absolute calibration against a fixed physical reference (the CMB's absolute-temperature discipline: FIRAS calibrated against a blackbody, not against another sky map)",
  "A cohort comparison subtracts the reference away, so a pipeline offset between the floor's scale and the patient's scale is invisible to it by construction.",
  "Reading 732 raw IDATs absolutely against the MCMC floor exposed a +0.066 beta pipeline offset the author's April note predicted and 200 within-pipeline VALs could not see; once mapped, healthy blood sits at A = 0.99 with the floor at 1.00.", "PHASE 1 OUTCOME + addendum; LESSON-SCALE-01. The April 2026 evidence report (Zenodo 10.5281/zenodo.19633499) had already stated that a systematic entropy offset between the calibration pipeline and others was expected and that absolute thresholds needed cross-pipeline validation - disclosed in April, lost by June, measured in September"))
RECON += [("S1", "beta scale of the gauge input vs the floor", "Issue 002: A read directly from any beta",
  "Three scales measured on the identity loci (Roadmap 0.737 / GEO-processed EPIC 0.774 / Stage-1 noob 0.815). H_min lives on the Roadmap scale. Patient beta must be mapped there before an absolute reading; cohort-relative statistics did not need this and so never showed it.",
  "per-pipeline affine map onto the Roadmap scale, fit on healthy blood; floors unchanged", "PHASE 1 OUTCOME, 2026-09-20"),]
FALSIFICATION += [("Analyst recommendation (2026-09-20 03:30) to re-derive H_min_beta on Stage-1 healthy blood", "would discard the G-002 MCMC confirmation and make the floor pipeline-dependent; the offset is in the input scale, not the floor", "WITHDRAWN same day"),]

PHASE1C = {
 "verdict": "P3 PASS (map transfers: GSE42861 controls n=315, independent lab, median mapped immune A 1.0097; unmapped 0.857). P4 FAIL (one-lab band: 53% in the GSE87571 band). Stage 1s COMMISSIONED.",
 "lab_layer": "GSE42861 - GSE87571 median per decade: +0.013, +0.016, +0.018, +0.016, +0.018 (25-74) - flat across age, ~+0.017 A. N-plate: 37 Sentrix chips, F=4.30, p=2.6e-12. The four TEST_DATA GSE42861 arrays sit inside their own cohort p5-p95 (4/4): Phase 1's 'above band' was the cohort, not the chips.",
 "n_random": "a mean-beta-matched random panel put 67% of controls IN band (must have been <50%): on healthy blood the identity loci set the LEVEL, but the band WIDTH is pipeline/chip variance any panel shares. A one-lab band is too narrow by the between-lab term.",
 "design": "healthy reference = pooled across >=2 labs AFTER the pipeline map, percentile bands on mapped A, lab/chip random effect estimated and disclosed, sex as a signed null (split only if the sign agrees across labs). Pool after mapping (the pipeline term must be removed first), never raw (Phase 1 was right about that).",
 "nulls": "N-sex SIGNED d(f-m): GSE42861 +0.28..+0.69 (women higher, 35-74) but GSE87571 -0.24..-0.44 (women LOWER, 35-64) - the sex effect flips sign between labs, so it is not a stable biological term; the sex-split band from Phase 1 is WITHDRAWN pending a signed test on pooled mapped data with a lab term. (The run reported |d| only; the direction first written here was unsupported and is corrected.) N-smoke: current 1.0108 / ex 1.0086 / never 1.0107 - no effect. RA arrays not opened.",
 "layers": "FLOOR (Roadmap, MCMC) -> PIPELINE (+0.066 beta; transfers; COMMISSIONED) -> LAB (+0.017 A between two labs; chips p~1e-12). SEX: sign flips between labs - not a layer until shown stable.",
}
COSMO_EVIDENCE.insert(5, ("2026-09-20", "Transfer test on an unseen calibrator (Planck's cross-frequency and cross-instrument consistency: a calibration is only accepted when it predicts data it was not fit on)",
  "A cohort-relative statistic has nothing to transfer: it is re-fit on every cohort by construction, so it cannot distinguish a pipeline constant from a lab offset.",
  "The pipeline map fit on one lab put an unseen lab's healthy blood at A = 1.010; the same test separated a constant lab offset (+0.017) and a chip effect (p = 1e-12) that no within-cohort analysis had ever resolved.", "PHASE 1c OUTCOME"))

FALSIFICATION += [("Phase 1c N-sex 'women higher every decade' (2026-09-20)", "the runner computed |d|; direction was asserted without support. Signed recomputation: women higher in GSE42861, LOWER in GSE87571 - sign flips between labs", "CORRECTED same day; sex-split band withdrawn"),]

BAND_V2 = {
 "verdict": "identity_band_v2 (pooled GSE87571 659 + GSE42861 315, mapped A, per-decade p10/p50/p90; adults >=45 at p50 1.00, 14-24 at 0.96) tested on GSE125105 Munich controls (n=201): P3 median 0.9645 FAIL (bar +/-0.02); P4 54.7% in band FAIL (bar 80%).",
 "three_cohorts": "mapped immune A offset vs Uppsala: Karolinska +0.018 (chips p~1e-12), Munich -0.030 (chips p=0.39) - each flat across age. The pipeline map moved Munich 0.79 -> 0.96 (shared term real; Stage 1s stays commissioned); the residual is a per-cohort constant the size of the band half-width.",
 "why_pooling_fails": "a band pooled over N labs is wider, but the next lab's constant is a new number - so ~50% of any new cohort lands in band regardless of N. Confirmed twice (Karolinska on a one-lab band 53%; Munich on a two-lab band 55%).",
 "design": "healthy reference = FLOOR (physics, universal) + PIPELINE MAP (universal per pipeline) + LAB ZERO (local, once per lab). This is CLSI EP28 practice for every clinical assay. Routes: (1) per-lab healthy-control panel of 20-30 arrays, offset disclosed on every report - recommended primary; (2) learn the offset from control-probe intensities (Stage 0 QCs, deferred) - Future Goal.",
 "nulls": "N-random REDESIGNED (size-matched, level test): random panel median 1.174, 0% in band -> the identity loci set the LEVEL; Phase 1c's mean-beta-matched null was circular for H(beta_bar) and is retired. N-sex unsigned as run (|d| 0.17-0.38) - no direction claimed. Stage 1 parallel: 210 arrays 37 min; per-sample IDAT fetch 1.7 GB in 6 min (tools/geo_fetch_idats.py).",
}
RECON += [("B2", "one universal healthy band", "Issue 002 / SOP: a single age band per class",
  "three Stage-1 cohorts on the Roadmap scale differ by per-cohort constants (0 / +0.018 / -0.030) flat across age, independent of chip effects; a pooled band cannot contain the next lab's constant",
  "FLOOR + PIPELINE MAP + LAB ZERO; the lab zero is set once per lab on healthy controls (or, future, from control probes) and printed on the report", "band_v2 OUTCOME 2026-09-20"),]
FUTURE_GOALS.insert(1, ("GATE 0b - lab zero", "Predict the per-lab offset from the array's own control-probe intensities (Stage 0 QCs)", "band_v2 OUTCOME; Stage 0 deferred QCs",
  "three cohorts with known offsets (0 / +0.018 / -0.030) are the training and test set; if control probes predict them, no healthy-control panel is needed per lab", "Stage 0 <-> Stage 1 intensity hand-off (unwired)"))

LABZERO_01 = {
 "question": "Can the per-lab offset be predicted from the 850 Illumina control probes on each array, so no per-lab healthy-control panel is needed?",
 "data": "1,277 raw IDATs (three cohorts already on disk), 33 features = log2 mean per Control_Type x channel + overall medians + G/R ratio; target mapped immune identity-loci A (1,173 gated). Ridge, leave-one-cohort-out.",
 "result": "held-out Karolinska: obs +0.024 pred +0.021 (err 0.002); Uppsala: obs -0.024 pred -0.026 (err 0.002); Munich: obs -0.021 pred -0.071 (err 0.050) - direction 3/3, size right for the Swedish labs, Munich overshoots 3x with no neighbouring lab in training. P1 FAIL as sealed.",
 "features": "cohort-separating features (range/sd ~2.6): NORM_T/A/C/G, BISULFITE CONVERSION II, SPECIFICITY I - red channel. The lab's chemistry is recorded on the array.",
 "within_cohort": "control probes explain r ~0.6 of a healthy donor's A inside a single cohort: about a third of the 'healthy spread' is per-array technical variance. Correcting it would narrow the band (LAB-ZERO-02).",
 "nulls": "the sealed permutation null was uninformative (shuffling cohort labels removes the offsets) and is recorded as such; the correct null is a fourth held-out cohort.",
 "verdict": "standard today = per-lab healthy-control panel (CLSI EP28), offset printed on the report. Control-probe zero = upgrade path, decided by a fourth Stage-1 healthy cohort (minutes to fetch with tools/geo_fetch_idats.py).",
}
FUTURE_GOALS[1] = ("GATE 0b - lab zero from control probes (PROMISING)", "Fourth Stage-1 healthy cohort as the held-out test; then LAB-ZERO-02: per-array technical correction to narrow the band", "LAB-ZERO-01",
  "direction 3/3, Swedish magnitude to 0.002, Munich overshoot with no neighbour; within-cohort r~0.6 says a third of band width is technical", "a fourth 450K healthy whole-blood cohort with raw IDATs")
COSMO_EVIDENCE.insert(6, ("2026-09-20", "Instrument self-calibration from housekeeping channels (Planck/WMAP used detector housekeeping - thermometry, gain monitors - to model systematics rather than fit them away on the sky)",
  "A cohort method has no housekeeping: it subtracts the lab offset with the cohort mean and never asks what caused it.",
  "The 850 control probes on every array carry the lab's chemistry signature; they predict two of three lab offsets to 0.002 and explain a third of the healthy within-cohort spread - a technical term cohort methods had always booked as biology.", "LAB-ZERO-01"))

# PRIOR ART — the door into the conversation (author's instruction 2026-09-20: cite, do not credit; independent route; agree, then carry further)
PRIOR_ART = {
 "citation": "Sanchez R, Mackenzie SA (2016). Information Thermodynamics of Cytosine DNA Methylation. PLoS ONE 11(3): e0150427. Sanchez R, Yang X, Maher T, Mackenzie SA (2019). Discrimination of DNA Methylation Signal from Background Variation for Clinical Diagnostics. Int J Mol Sci 20:5343. Software: MethylIT (R).",
 "independence": "This work did not build on, borrow from, or arrive via Sanchez & Mackenzie. The author reached the Landauer floor from the other end: cosmology (an entropy functional on the Friedmann background, 18 converged MGCAMB/CAMB chains, 2025-26), then particle physics, quantum computing (QAPE) and semiconductors (SCAPE), and only then the cell (GAPE, 2026). Their 2016 paper was first read on 2026-09-20, after every H_min value, the Atlas, the deconvolver and the validation record existed. It is cited here because it is peer-reviewed physics that the methylome obeys Landauer's bound, and because a reviewer who knows MethylIT would otherwise assume dependence.",
 "agreed": [
  "Landauer's principle - k_B T ln2 per irreversible logic operation - is the physical anchor for cytosine methylation as an information-writing process. (Sanchez & Mackenzie 2016 tested adherence to it on Arabidopsis ecotypes and 93 human tissues; IAM's H_min is the same bound expressed as a per-class entropy floor.)",
  "Methylation carries thermal background: fluctuations at operating temperature produce a distribution of beta that is not regulatory signal. (Their Weibull/generalized-gamma model of that background; IAM's k_B T denominator of the Mahaffey number.)",
  "Information-theoretic measures on beta (Shannon entropy; Hellinger / total-variation divergence) are the right vocabulary, not raw differences.",
  "Cohort-comparison methylome statistics have not translated to the clinic (their 2019 paper says so in its first paragraph; this document says so in s0 and s1.6).",
 ],
 "missing": [
  "A FIXED ZERO. MethylIT's reference is the centroid of the control group in the study at hand - a crowd-relative zero re-etched per cohort. IAM's reference is H_min: one per class per substrate, set by physics and frozen (G-002 MCMC, 2026-04), never re-fit on a cohort. Their instrument has no absolute zero; this one does.",
  "A SINGLE-SAMPLE ABSOLUTE READING. MethylIT asks 'which sites moved, relative to controls, beyond thermal background' - a group question. IAM asks 'how far is THIS sample from the healthy floor of its cell class' - a per-patient question, answerable with no other patient in the room (s3, the potassium analogy).",
  "COMPOSITION FIRST. Whole blood and tissue are mixtures; MethylIT scores the mixture. IAM deconvolves against the MCMC Atlas (115 cell types, 8 classes) and reports a class gauge only where the class is present and determined (s108).",
  "THE CMB TOOLKIT for navigating thermal fluctuations - end-to-end simulation with known truth (PROC-N7-01 found the wired gauge reading the wrong CpG set), cross-method disagreement as diagnostic not defect (PROC-NILC-01), absolute calibration against an unseen lab (PHASE 1c), instrument housekeeping from the array's own control probes (LAB-ZERO-01). None of these exist in a cohort-relative pipeline because a cohort-relative pipeline has nothing to calibrate against.",
 ],
 "one_line": "Both use k_B T ln2. Sanchez & Mackenzie use it to model the thermal background and FILTER it away to find regulatory sites. IAM uses it as the UNIT of measure and asks how far above it a healthy cell holds its pattern. One filters, one calibrates. The filter is a legitimate and complementary tool - a principled way to ask which identity loci carry regulatory rather than thermal signal - and is noted as a Future Goal.",
 "thermal": "Thermal noise is not a nuisance in IAM; it is the ruler. M = E_drive / k_B T: the M1 transistor pays ~117 thermal quanta per switch at 348 K, the cell nucleus 20.94 per ATP at 310 K, the aluminium transmon exactly 1 at its gap temperature - the saturated case qubit engineers reach by cooling until k_B T is the only scale left. The cell cannot cool; H_min is where a healthy cell holds its pattern against noise it cannot escape.",
}
FUTURE_GOALS.append(("GATE 2 - after the band", "Apply the Sanchez-Mackenzie thermal-background model (Weibull/gen-gamma Hellinger divergence) to the identity loci: which loci carry regulatory vs thermal signal? Compare to the empirical marker selection.", "PRIOR_ART",
  "the two uses of k_B T ln2 are complementary - their filter could sharpen our ruler's loci", "MethylIT on the Stage-1 healthy betas already cached (three cohorts)"))

LABZERO_02 = {
 "question": "Does a fourth lab (UCLA, USA) let the control probes predict the lab offset within 0.010, and does per-array correction narrow a healthy band by 20%?",
 "result": "P4 FAIL: UCLA predicted -0.020, observed -0.046 (err 0.026). P4b FAIL: LOO over four labs - Karolinska 0.004, Uppsala 0.018, UCLA 0.026, Munich 0.052 (1/4 within bar). P6 FAIL: within-cohort sd 0.0203 -> 0.0177 (12.6%, bar 20%).",
 "four_labs": "mapped immune A cohort constant vs Uppsala: Karolinska +0.024, Munich -0.021, UCLA -0.046 - each flat across age. The pipeline map (Stage 1s, commissioned) is common to all four; the constants are what it does not remove.",
 "reading": "control probes carry the DIRECTION of a lab's offset, not its size; adding a lab widened the error spread. The housekeeping channels record part of the chemistry (normalisation, conversion controls) but not the pre-analytical part (DNA input, bisulfite batch, storage). Honest end of the route for now.",
 "decision": "THE LAB ZERO IS THE HEALTHY-CONTROL PANEL (CLSI EP28). Panel size and form were then set by PROC-PANEL-01 → PROC-PANEL-03: 40 healthy arrays per lab through the same Stage 1, read against the reference age curve, offset printed on every report. Tried the ideal route first, fell back to the standard when it failed - the floor and the map are untouched.",
 "nulls": "N-plate 38 chips F 1.79 p 0.009. N-sex signed: women lower in every UCLA decade (as Uppsala, opposite Karolinska) - sign lab-dependent, no split. N-ethnicity Hispanic-Caucasian d -0.33 at n=18, descriptive only. P2 presence 86% in a median-age-70 cohort - the gate needs an age-aware look (commissioning note).",
}
RECON += [("B3 (LAB-ZERO-02)", "lab-zero mechanism", "LAB-ZERO-01: control probes promising", "fourth lab: direction 4/4, magnitude 1/4 within bar; within-cohort narrowing 13%", "LAB ZERO = per-lab healthy-control panel (CLSI EP28); control-probe route closed for now, revisit with >=6 labs", "LAB-ZERO-02 OUTCOME"),]
FALSIFICATION += [("LAB-ZERO-01 'promising - a fourth cohort decides' (2026-09-20)", "fourth cohort: P4 err 0.026, LOO 1/4, narrowing 13% - the control probes do not supply the lab zero", "DECIDED AGAINST same day (LAB-ZERO-02); panel standard adopted"),]
COSMO_EVIDENCE.insert(7, ("2026-09-20", "Housekeeping-channel systematics model tested on an unseen instrument (the Planck rule: a systematics model earns its place only by predicting a detector it was not fit on)",
  "A cohort method never asks whether a lab offset is predictable from the array itself; it subtracts it with the cohort mean and moves on.",
  "Four labs on one scale (0 / +0.024 / -0.021 / -0.046, each flat across age): the control probes predict every sign and only the Swedish pair's size. A negative result that fixed the design: the lab zero is a measured healthy panel, not a model.", "LAB-ZERO-02"))
FUTURE_GOALS[1] = ("GATE 0b - lab zero: CLOSED (panel standard)", "Per-lab healthy-control panel is the lab zero (LAB-ZERO-02). Revisit the control-probe route only with >= 6 labs and pre-analytical metadata; the 13% within-cohort narrowing is noted, not built.", "LAB-ZERO-01, LAB-ZERO-02",
  "direction 4/4, magnitude 1/4; the unrecorded pre-analytical term dominates", "closed")

HMIN_BOOT = {
 "question": "Were the eight methylation H_min values ever bootstrap cross-checked, as the record's 'all 40 values' sentence implies?",
 "finding": "No. bootstrap_vs_mcmc_comparison.tsv (commit 22749f0) has 32 rows = nucl/fuzz/wps/frag x 8 classes. The methylation eight had MCMC (R-hat < 1.001) and no bootstrap.",
 "run": "G-002 reference database (37 published reference cell methylomes, 4-6 per class) + bootstrap_h_min from gape_bootstrap_comparison.py, both from 22749f0; 10,000 resamples, seed 42, mean-over-cells H(beta); exact leave-one-out; compared with the frozen values at HEAD.",
 "result": "8/8 frozen values inside the bootstrap 95% CI; mean relative difference 0.060%, max 0.095% (immune). Tighter than the G-003b substrates. Frozen values unchanged.",
 "code_status": "The calibration scripts are NOT at HEAD (removed 2026-04-19, 538667d, 'commercial calibration layer'); they are in public git history at 22749f0, so the evidence report's links are dead while the files remain retrievable. Disclosure decision for the author.",
}
FALSIFICATION += [("'All 40 H_min values bootstrap cross-validated at 0.168%' (record, April 2026)", "the TSV holds the 32 non-methylation floors only; methylation had no bootstrap", "CORRECTED 2026-09-20; methylation bootstrap run (PROC-HMIN-BOOT-01): 8/8 in CI, 0.060%"),]

PANEL = {
 "question": "Define the per-lab healthy panel and test it on the four held cohorts: does a lab zeroed by its own panel read 1.00, and does a band built on three labs then hold the fourth?",
 "panel01": "k=25, flat zero. P3 (decisive) PASS: LOO lab-zeroed band holds 0.788/0.841/0.766/0.717 of a held-out lab (without lab zero 0.142/0.861/0.502/0.750). P2 FAIL as sealed (Uppsala 85% within 0.010 - panel size). P1: k=40 is the first size with SD(z)<=0.005 in every lab. P4 FAIL and a design error (two random panels differ by ~0.007 from sampling alone).",
 "panel02": "k=40, flat zero. P2' Uppsala 94.4% - FAIL by 0.6 points. P4' (deterministic): healthy immune A RISES WITH AGE within a lab - Uppsala teens -0.025 to eighties +0.019 about the cohort median. The between-lab offsets are parallel across age; the age curve itself is not flat. A flat panel median inherits the panel's age mix.",
 "panel03": "k=40, zero read against the reference healthy age curve built on the OTHER labs (LOO). P2'' 97.0/99.8/98.4/97.6 PASS. P4'' single-decade panels 96.0/99.9/100/99.5 PASS - a panel needs no age matching once read against the curve. P3'' LOO age-referenced band (width ~0.053) holds 0.823/0.839/0.767/0.753 (nominal 0.80) PASS.",
 "decision": "LAB ZERO COMMISSIONED: 40 healthy arrays per laboratory, any age mix, z = median[A - c(decade)] - 1; A'' = A - c(decade) - z. Closed in code: CPG_Engine/lab_zero.py + Runtime Matrices/A_Scoring_Module/reference_age_curve_v1.json (four labs, n=1,379); lab_zero=UNSET is not reportable; panels < 40 refused; kit test test_lab_zero.py. Supersedes the '20-30 arrays' wording of LAB-ZERO-02.",
 "residual": "The Sentrix-chip term within a lab (Karolinska p 8e-16, Uppsala 2e-19) is untouched by a constant and sits inside the band width - the next residual.",
}
RECON += [("B4 (PROC-PANEL-01 → PROC-PANEL-03)", "lab-zero panel specification", "LAB-ZERO-02: 20-30 healthy arrays, flat median", "k=25 fails P2 on the widest-age lab; healthy A rises ~0.045 teens->eighties within a lab; age-referenced 40-panel passes every test", "40 healthy arrays, any age mix, read against reference_age_curve_v1.json; CLOSED IN CODE (lab_zero.py)", "PROC-PANEL-03 OUTCOME"),]
FALSIFICATION += [("PROC-PANEL-01 prediction 'k=25 suffices' and 'offsets flat across age' read as 'A flat across age' (2026-09-20)", "k=40 needed for SD(z)<=0.005 in every lab; healthy immune A rises with age within a lab (between-lab offsets are parallel, the curve is not flat)", "CORRECTED same day by PROC-PANEL-02 and PROC-PANEL-03; lab zero is age-referenced"),]
COSMO_EVIDENCE.insert(8, ("2026-09-20", "Instrument zero set on a reference panel and read against a reference curve, tested leave-one-instrument-out (the Planck rule: calibrate each detector against the same sky, then test on a detector that did not enter the calibration)",
  "A cohort method sets its zero on each study's own controls and has no way to ask whether that zero transfers.",
  "Four labs, each zeroed by 40 of its own healthy arrays against an age curve built on the other three: a band built on three labs holds 75-84% of the fourth's healthy donors (nominal 80%). Without the lab zero: 14-86%. The healthy age curve (~+0.045 teens to eighties) was found because a flat zero failed on the one cohort wide enough in age to show it.", "PROC-PANEL-01 → PROC-PANEL-03"))

HISTORY = [  # THE COMPLETE VALIDATION HISTORY (PROC-HISTORY-01, 2026-09-21) - series, when, count, executed, what it was
 ("G-series", "April 2026, before VAL-001", "G-002 (8 methylation floors, 17 chains), G-003b (32 substrate floors), bootstrap of the 32; G-008, E_A,bio, n_bio", "3 sealed", "the H_min calibration; code restored to Hmin_Calibration/; the methylation bootstrap was first run 2026-09-20 (8/8 in CI)"),
 ("VAL-001 -> VAL-128 (pre-Atlas)", "April-May 2026", "119 identifiers in six families", "107 executed; 12 not run / gated / excluded / voided", "methylation 001-013; five substrates 014-033; drift cascade 037-046 (35/39 predictions); EDEAR disease cards 047-128 across 12 cards, SHA-sealed from VAL-050"),
 ("T1 -> T15 (VAL-049)", "April 2026", "15 cohorts, 6 populations", "12 executed; T4/T6/T7 dbGaP-gated", "frozen panel + frozen H_min transferred across US/AU/UY/UK/PL/CN-SG - the first cross-population test"),
 ("CPG-VAL-001 -> 022 (post-Atlas)", "29 May - 7 June 2026", "22 slots", "21 executed; 021 deferred", "breast 001-007 (Mahalanobis d +1.88/+2.10; 2 RESTATED), AD 008-014 (AD up / PSP-CBD down / FTD between), immune-aging 015-020 & 022; L9 nulls N1-N8 per VAL; PREREGs retrospective, so marked"),
 ("Mahalanobis HC hull v0_1 -> v0_5", "6 June 2026", "5 versions, 8 cohorts", "n_HC 601 -> 2,523", "four populations incl. Han Chinese (GSE141682, n=42, first Asian); fixed d>=2.0 shown invalid in 112-D, replaced by percentile-of-HC; anchor d fell honestly as the hull broadened"),
 ("L9 N7 chain integrity", "5 June 2026", "3 synthetic cohorts x 250", "R1 MAE 0.0076-0.0093", "synthetic truth through Walther -> A-scoring -> Mahalanobis; its September rerun (PROC-N7-01) found the gauge on the wrong loci"),
 ("September 2026 procedures", "18-21 September", "PROC-CAL/DECON/ANCHOR/FORMULA/N7/NILC/SEP/CHAIN/STAGE0/WB-IMMUNE/HMIN-BOOT/PANEL-01..03; PHASE 1/1c; band_v2; LAB-ZERO-01/02; CPG-NEW-001", "all sealed before data", "the rebuild from scratch: raw IDATs, absolute reading, the scale map, the age curve, the lab zero - this document"),
]
HISTORY_NOTE = ("What none of the April-June validations could see, and why this document exists: every one was a within-pipeline comparison or a reading against a control centroid. The pipeline-scale offset (predicted in the April caveats tab, measured in September) and the healthy age curve cancel in any difference. They were found by reading single arrays absolutely against the floor. The history is evidence of six months of consistent, sealed testing; it is not a set of results this document re-asserts.")
FALSIFICATION += [("'103 indexed pre-registered validations (81 pre-Atlas, 15 post-Atlas, 7 retired)' (Paper 1 draft and VAL_INDEX, 2026-09-19/20)", "the index keyed on bare numbers (VAL-001 collided with CPG-VAL-001) and counted folders, not the record: 119 pre-Atlas + 15 T-series + 22 post-Atlas + G-series", "CORRECTED 2026-09-21 by the author's reading; PROC-HISTORY-01; index rebuilt, 175 rows"),]
RECON += [("H1 (PROC-HISTORY-01)", "the validation count", "VAL_INDEX 2026-09-19: 103", "RETIRED inventory + v10 report + repo: 3 G + 119 VAL (107 run) + 15 T + 22 CPG-VAL (21 run) + 5 hull + N7 + Sept PROCs", "175-row index, unique keys by series; AD folders 008-014 moved to VAL_PostAtlas", "PROC-HISTORY-01 OUTCOME"),]

HISTORY_PROC = [("question","How many validations were run before this document, and how were they counted?"),
 ("finding","The author caught the undercount: 'we have 128 VALs plus VAL-049 includes 15 in our T series, the full G series before VAL-001, and post-Atlas we tested cpg_val_001-022, not 15.' The 2026-09-19 index keyed on the bare number (VAL-001 and CPG-VAL-001 collided, 22 times) and counted only identifiers with a folder at HEAD (51 pre-Atlas records live in the RETIRED report and Zenodo, not in folders). Seven AD folders were also mis-filed under VAL_PreAtlas."),
 ("correction","Index rebuilt from the record with unique keys by series: 175 rows. 3 G; 119 VAL-001..128 (107 executed, 12 not run for stated reasons); T1-T15 (12 executed); 22 CPG-VAL (21 executed); Mahalanobis hull v0_1-v0_5 (n=2,523, four populations incl. Han Chinese n=42); L9 N7; September PROCs. Paper 1 and this document corrected at source. Lesson: an index built by walking the tree counts folders, not the record.")]

# THE HEALTHY REFERENCE HAS THREE LAYERS - written 2026-09-21 at the author's request: "this is the sort of information that is necessary
# to recreate, and it gives confidence in our work." Canonical statement; every other mention of the lab zero points here.
REFERENCE_LAYERS = {
 "intro": "A single array is read absolutely against the floor. Between the raw IDAT and that reading stand three layers, each measured, each commissioned in a sealed procedure, each stated on the report. Nothing here is fitted to the patient.",
 "layers": [
  ("LAYER 1 - THE FLOOR", "H_min per class, from physics (G-002 MCMC on the 37 published reference cell methylomes listed in gape_mcmc_g002.py _RAW_DB - 'Database: 37 cells' in its own output; 4-6 per class; bootstrap 8/8, PROC-HMIN-BOOT-01). Universal. Frozen 2026-04-06; not refitted since. The commitment line A = 1.000 is the floor.", "one number per class, forever"),
  ("LAYER 2 - THE PIPELINE MAP", "The same blood read through two normalisation pipelines gives two beta scales; on the identity loci the offset between GenomicStudio-era Roadmap and noob-normalised 450K IDATs is +0.066 in A (LESSON-SCALE-01; predicted in the author's April 2026 caveats, measured 2026-09-20). One affine map per pipeline (beta_scale_maps_v1.json, stage1_noob_450K), fitted once on Uppsala and TRANSFERRED without refitting to Karolinska, Munich and UCLA (PHASE 1c, band_v2, LAB-ZERO-02). A reading without a map is labelled scale=UNMAPPED and is not reportable.", "one map per pipeline, transfers"),
  ("LAYER 3 - THE LABORATORY ZERO", "After the map, healthy blood from different laboratories still sits at different constants: Uppsala 0, Karolinska +0.024, Munich -0.021, UCLA -0.046 - each flat across age, so instrument, not biology. That constant is measured once per laboratory from a panel of 40 healthy arrays and subtracted from every reading from that laboratory. A reading without it is labelled lab_zero=UNSET and is not reportable as absolute.", "one constant per laboratory, measured once"),
 ],
 "what_a_lab_is": "\"Laboratory\" means one processing pipeline: the same scanner, reagent lots, technicians and normalisation. Two benches in one building can be two laboratories; one core facility serving a country is one. The constant belongs to the pipeline that scanned the array, and it is not written anywhere on the IDAT.",
 "two_routes": "Two routes to the constant were tested, in this order and by the author's choice. (1) The ideal: predict it from the array's own control probes, so a single IDAT would carry its zero. LAB-ZERO-01/02: the probes gave the sign of every laboratory's offset (4/4) and the size of only one (1/4); leave-one-lab-out error 0.004-0.052 against a bar of 0.010. CLOSED. (2) The clinical standard (CLSI EP28): measure it on a healthy panel. PROC-PANEL-01 -> 03: commissioned.",
 "panel": "The 40 arrays are healthy volunteers' blood run once through that laboratory's pipeline. Any age mix, because the panel is read against the reference age curve rather than against a flat 1.000 (see below). Why 40 and not the CLSI verification minimum of 20: the within-laboratory SD of healthy mapped A is 0.019-0.025, and the constant must be known to +/-0.005; the median of 25 reaches +/-0.006, of 40 reaches the bar in every laboratory we hold. In the commissioning test the panel was simulated by drawing 40 of each cohort's own healthy donors and asking whether the REST of that cohort then read 1.000: 97-100% of 1,000 draws within +/-0.010, every laboratory.",
 "age_curve": "Healthy immune A is not flat in age within a laboratory: it rises about 0.045 from the teens to the eighties (Uppsala, ages 14-94), even though the between-laboratory offsets are parallel across age. A flat panel median would therefore inherit the panel's age mix - the widest-age cohort failed on exactly this (PANEL-02). The zero is defined as the panel's median residual from the reference age curve, z_L = median_i[A_i - c(decade_i)] - 1, with c built on the OTHER laboratories only when tested (reference_age_curve_v1.json; four labs, n = 1,379). A patient reads A'' = A_mapped - c(decade) - z_L.",
 "test": "Leave-one-laboratory-out, 200 draws: a p10-p90 healthy band built on three laboratories, each zeroed by its own 40-panel, holds 82% (UCLA), 84% (Munich), 77% (Karolinska), 75% (Uppsala) of the fourth laboratory's healthy donors - nominal 80%. Without the laboratory zero the same test gives 14%, 86%, 50%, 75%. That spread is the whole reason the layer exists.",
 "one_idat": "THE SINGLE-ARRAY CASE. One IDAT from a laboratory we have not zeroed yields composition (Stage 2), a mapped identity-loci A, and lab_zero=UNSET: the reading carries an unknown offset of up to +/-0.05, the width of the healthy band, and is not reported as absolute. It becomes reportable three ways, cheapest first: (a) the laboratory already has healthy arrays in a public repository - its constant is computed from them with no action by the laboratory (this is how Uppsala, Karolinska, Munich and UCLA were zeroed); (b) the laboratory runs a 40-array healthy panel once - what a clinical laboratory already does for every assay it adopts; (c) serial sampling: the constant is constant, so two arrays from the same laboratory a year apart give a trajectory in which it cancels, with no panel at all. The chain can be run on a laptop from the repository; moving the arithmetic does not move the information - the constant is measured, not derived.",
 "residual": "What the three layers do NOT remove: the Sentrix-chip term within a laboratory (Karolinska ANOVA p = 8e-16, Uppsala 2e-19). A constant cannot touch it; it sits inside the band width today and is the next residual after the gauge switch. In code: CPG_Engine/lab_zero.py (panels under 40 refused; UNSET not reportable), Runtime Matrices/A_Scoring_Module/reference_age_curve_v1.json, beta_scale_maps_v1.json; kit test test_lab_zero.py recovers a synthetic -0.046 laboratory.",
}
CHAIN_TERMS += [
 ("Laboratory (as the chain uses the word)", "one processing pipeline - scanner, reagent lots, technicians, normalisation. The laboratory zero belongs to the pipeline that scanned the array, not to the company name."),
 ("Laboratory zero (lab zero)", "the constant by which healthy blood through one laboratory differs from the reference after the pipeline map; measured once on 40 healthy arrays read against the reference age curve; subtracted from every reading from that laboratory; printed on the report. Uppsala 0 / Karolinska +0.024 / Munich -0.021 / UCLA -0.046. UNSET = not reportable as absolute (s3.5)."),
 ("Pipeline map", "the affine beta-scale correction between a normalisation pipeline and the calibration scale, one per pipeline, fitted once and transferred (beta_scale_maps_v1.json). UNMAPPED = not reportable."),
 ("Reference age curve", "the per-decade healthy median of mapped immune A about the grand median, built on four laboratories each zeroed by its own median (reference_age_curve_v1.json); rises ~0.045 teens -> eighties. Panels and patients are read against it."),
]
CLINICIAN["your_sample"] = ("If you send one blood draw to a laboratory and receive one array, the chain will tell you what your blood is made of and where it sits on the identity loci - but it will not report an absolute number against the floor until it knows that laboratory's zero. The zero is a property of the laboratory's instrument, measured once on 40 healthy samples, exactly as a clinical laboratory establishes a reference interval for cholesterol. If the laboratory's healthy samples are already public, the zero is already known; if not, it is one plate of work, once. Two draws a year apart from the same laboratory give you your own trajectory with no zero needed at all, because the constant cancels. The report will say which of these applies; it will not print a number that means nothing.")

RECON += [("B5 (PROC-SWITCH-02)", "the atlas posterior's own zero", "assumed 0 (synthetic patients read as if on the floor's scale)", "G-002 floor A=1 at beta 0.7318 (37 reference cells); atlas immune identity-loci mean 0.7373; synthetic whole blood reads 0.985 with zero 0, 1.001 with its own 40-panel zero (-0.0146)", "the atlas is a fifth laboratory; every beta source is zeroed before it is read absolutely", "PROC-SWITCH-01 OUTCOME (S4 FAIL as sealed), PROC-SWITCH-02 OUTCOME"),]
FALSIFICATION += [("PROC-SWITCH-01 S4: 'synthetic healthy patients with lab zero 0 read 1.000 +/- 0.010'", "median 0.985 (97.5% in band); the atlas carries its own constant -0.0146", "FAIL as sealed; re-sealed PROC-SWITCH-02 with the synthetic cohort zeroed from a disjoint 40-panel: median 1.0009, 100% in band; marker-union on the same patients 1.125, 0% in band"),]
SWITCH_PROC = [("question","Does the conductor now REPORT the identity-loci gauge with the three-layer reference, and does the N7 defect disappear?"),
 ("finding","S1 7/7 identity_loci/MAPPED/reportable; S2 7/7 UNSET refuses; S3 4/5 healthy in band; S5 anchors r = 1.00000 (both), canonical A-score test PASS. S4 as sealed FAILED: 40 synthetic healthy read median 0.985 with zero 0 - not the generator's 2,745-locus subset (pure atlas immune reads 0.990 on both), not noise/age/batch (all off: 0.985), but the atlas posterior itself, which sits at beta 0.7373 where the G-002 floor puts A=1 at 0.7318. The analyst's first mixture arithmetic (weights summing to 0.989) was wrong and is recorded."),
 ("correction","PROC-SWITCH-02: the synthetic cohort read as a fifth laboratory. z_atlas = -0.0146 from a 40-panel; 40 disjoint patients read median 1.0009, 100% in identity_band_v3; the marker-union statistic on the same patients 1.125, 0% in band; UNSET refuses 80/80. ROW B COMMISSIONED. In code: cpg_conductor (classes = identity gauge; diagnostic_marker_union; pending_recalibration for Stages 5/6), identity_band_v3.json, Reproduction_Kit/test_gauge_switch.py."),]

FALSIFICATION += [("PROC-MAHA-01 M2: 'on the identity gauge, <= 7% of healthy donors read beyond p95 in every laboratory'", "pooled 6.5%; Karolinska 9.8% (p99 tail 5.1%) - the Sentrix-chip term: per-chip medians scatter with SD 0.020 there vs 0.012 elsewhere; chip-centring brings Karolinska to 4.1%", "FAIL as sealed on one laboratory; row 5 BUILT not commissioned; the chip is the named residual (row 5b); bar decision to the author"),]
RECON += [("C1 (PROC-MAHA-01)", "the residual after floor + map + lab zero", "assumed inside the band width", "per-chip median SD 0.012-0.020 by lab; chip-centring cuts every lab's within-lab SD to 0.017-0.018 and the p95 tail to 2-4%", "the chip is the next layer; a per-lab constant cannot touch it; needs a reference on the chip", "PROC-MAHA-01 OUTCOME, maha01_chip_diag.json"),]
MAHA_PROC = [("question","Does the departure (Stage 5) read correctly on the identity gauge, reach the report, refuse without a zero - and does healthy blood stay inside?"),
 ("finding","Re-based: z = (A'' - 1)/sigma, sigma 0.0204 from identity_band_v3; one banded axis on whole blood so distance = |z_immune|, thresholds 1.960/2.576; long + short keys; report renders. M1 5/5 healthy inside (z -0.75..+1.61), RA +0.5/+0.35; M3 40/40 synthetic inside; M4 render OK; M5 UNSET 7/7. M2 FAILED as sealed: Karolinska 9.8% beyond p95 (bar 7%; pooled 6.5%; others 4.4-6.5%)."),
 ("correction","Cause measured: the Sentrix chip. Per-chip median SD 0.020 at Karolinska vs 0.012 elsewhere; chip-centring -> Karolinska 4.1%, all labs 2-4%, within-lab SD 0.017-0.018 everywhere. A laboratory constant cannot remove it; a chip reference can. Row 5 BUILT, not commissioned; row 5b (chip term) opened; the acceptable false-alarm rate is the author's decision (PROC-MAHA-02).")]

RECON += [("C2 (PROC-MAHA-02)", "the false-alarm rate of the departure", "implicit 5% (chi2 p95)", "measured per laboratory on the arrays that set its zero: 4.4-9.8% at p95, 0.5-5.1% at p99; the excess over 5% is the chip", "printed on every report with the laboratory's own number, or the four-lab range if unmeasured (author decision: option 1 + 5b)", "PROC-MAHA-02 OUTCOME; identity_band_v3 _meta.cohorts"),]
MAHA2_PROC = [("question","With the p95 tail failing on one laboratory (the chip), what does the report say?"),
 ("finding","Author's decision: commission with the laboratory's own false-alarm rate on the report and open the chip term as row 5b. identity_band_v3 now carries each laboratory's p95/p99 healthy tail and chip-median SD (M6); the departure and the report print it - Karolinska: 'At this laboratory 10 of 100 healthy donors read beyond p95 on this axis (5 of 100 beyond p99); the excess over 5 is the chip term (row 5b)'; an unmeasured laboratory gets the four-lab range 4-10; M7 as sealed (exact == 0.098 against a stored 0.0984) FAILED and is recorded as such - the analyst's first 'PASS, the prereg says 3 dp' misattributed M6's qualifier and is withdrawn; MAHA-01 M1/M3/M5 unchanged (M8)."),
 ("correction","ROW 5 COMMISSIONED (whole blood, immune axis). ROW 5b OPEN with its bar: a reference on the chip brings every laboratory's p95 tail to <= 0.05 (chip-centring already reaches 2-4%). The number never travels without its false-alarm rate.")]

FALSIFICATION += [("PROC-MAHA-02 M7 as sealed: 'departure.lab_false_alarm_p95 == 0.098' (exact)", "stored 0.0984; the analyst first wrote PASS claiming the prereg said '3 dp' - that qualifier belongs to M6, not M7", "FAIL as sealed, recorded; the first write-up was a post-hoc loosening and is withdrawn (auditor catch 2026-09-21). Row 5 commissioned on M6/M8 with M7's failure on the record."),]

FALSIFICATION += [("PROC-AGE-01 A1: 'cellular age from the identity gauge places >= 80% of healthy adults within +/-10 yr'", "15.9% (median |delta| 35.6 yr); slope 0.47 mA/yr vs within-lab SD 0.0235 -> ~50 yr per array; the analyst's own prediction (30 +/- 8 yr) was too optimistic", "FAIL as sealed; per-patient Stage 6 CLOSED, NOT REPORTABLE at single-array resolution; the report prints the resolution. The population trajectory is reproduced (0.47 mA/yr = CPG-VAL-015), not retracted"),]
RECON += [("D1 (PROC-AGE-01)", "cellular age (Stage 6)", "inverted age_reference_matrix (marker union) with identity beta - pinned at 4 yr", "inverted reference_age_curve_v1 on the identity gauge, LOO: 15.9% within 10 yr, resolution ~50 yr, rho 0.27", "NOT REPORTABLE at single-array resolution; no reported path reads the marker-union statistic any more", "PROC-AGE-01 OUTCOME"),]
AGE_PROC = [("question","Can the identity gauge, whose healthy curve rises with age, be inverted to read a person's cellular age?"),
 ("finding","No. Leave-one-lab-out on 1,379 healthy donors: 15.9% within +/-10 yr (bar 80%), median |delta| 35.6 yr, Spearman 0.27. The curve moves 0.47 mA per year and the within-laboratory spread is 0.0235 - one array resolves age to ~50 years; where the curve must extrapolate it runs away (258 yr SD). A healthy 58-year-old inverts to 23; a healthy 43-year-old to '>85'. The analyst predicted ~30 yr before the run; it is 50 - recorded."),
 ("correction","Per-patient cellular age CLOSED as NOT REPORTABLE at single-array resolution: the report prints the resolution sentence instead of an age; the marker-union inversion is diagnostic only. With this, no reported number reads the marker-union statistic. THE AGING TRAJECTORY ITSELF STANDS AND IS REPRODUCED: 0.47 mA/yr monotone by decade on four labs is CPG-VAL-015's slope (~0.5 mA/yr on Hannum) - a population measurement, now the reference age curve every reading is corrected by. Sign differs by surface (RECON D2). The first write-up's 'the gauge measures fidelity, not time' overstated the closure and is withdrawn (author's challenge, same day).")]

RECON += [("D2 (PROC-AGE-01, author's challenge)", "the healthy aging trajectory", "CPG-VAL-015: A_immune FALLS with age, r=-0.197, ~0.5 mA/yr, on the marker-union surface (Hannum n=656)", "identity gauge RISES with age, 0.47 mA/yr, rho 0.27, monotone by decade, 1,379 donors / four labs", "the trajectory is REPRODUCED (same slope); the SIGN belongs to the surface - discriminative markers lose entropy with age, identity loci gain it; the mammalian lifespan rate (VAL-006 lineage) is to be re-derived on the identity gauge before both are quoted together", "PROC-AGE-01 OUTCOME (corrected); CPG-VAL-015 OUTCOME"),]
FUTURE_GOALS += [("GATE 1 - on the commissioned gauge", "Re-derive the class-average healthy drift rate and the cross-species lifespan scaling (mammalian paper) on the identity gauge", "PROC-AGE-01 D2", "the human rate is now measured on four labs (0.47 mA/yr); the dog cohorts (VAL-013, VAL-025-028) are public", "Stage 1 on the Wang 2020 Labrador arrays; identity loci mapped to the canine array"),]

RECON += [("R2 (PROC-RECORD-02)", "VAL-025..028 four-substrate aging trajectory", "PASS, r = 0.9998 human / 0.986 canine (DETAILED_VALIDATION_RECORD, VAL_INDEX)", "the script's age tables are typed literals from the literature's described direction, Monte-Carlo scored; Hannum and Wang 2020 carry no substrate data; Issue 002 already said 'modeled - prediction filed'", "reclassified MODELED PREDICTION in the index and the record; the methylation trajectory (VAL-006, CPG-VAL-015, PROC-AGE-01) is per-sample and stands", "PROC-RECORD-02 OUTCOME; val025_028_aging.py (Zenodo copy)"),]
FUTURE_GOALS += [("GATE 2 - needs data not in hand", "Test the four-substrate aging prediction (VAL-025..028) on real plasma cfDNA fragmentomics with donor ages", "PROC-RECORD-02", "the prediction is filed and monotone; WPS / fragment-size / nucleosome inference exist as methods on cfDNA", "a healthy-donor cfDNA WGS cohort with ages (Snyder 2016 n=36 public; Mouliere 2018 / Mathios 2022 controlled access)"),]

RECON += [("R3 (PROC-RECORD-03)", "the 80-cell age reference matrix (HEALTHY_BASELINES, April 18)", "described 2026-09-19 as 'compiled' by code on a gauge surface", "content-identical to the April table: typed beta_mean per decade with literature labels, A = H(beta_mean)/H_min in 80/80 cells, percentiles = A +/- 1.2816 sd in 80/80 (Gaussian around a typed number); no generating script anywhere", "a literature-informed constructed table; direction confirmed by measurement (PROC-AGE-01), slope ~2x the measured 0.47 mA/yr, level pre-scale-offset; superseded by reference_age_curve_v1; read by no reported path", "PROC-RECORD-03 OUTCOME; age_matrix_provenance_check.json"),
 ("R4 (PROC-RECORD-03)", "AD / breast 'cellular age in years'", "AD immune ~9 y younger than HC (d = -0.56); breast cycling ~5.5 y younger", "the years are dA read through the typed table's 1.0 mA/yr slope; on the measured curve the same dA is ~19 y", "years withdrawn as a unit; the finding stands as a group departure of the immune gauge from the healthy age curve, in A (d = -0.56); one-patient detectability is paper two's absolute question; 'tracks worse over years' was never tested (VAL-005 UNDERPOWERED)", "PROC-RECORD-03 OUTCOME; v10 evidence report"),]

SCOPE["what_it_reports"] = ("WHAT THE INSTRUMENT REPORTS (author, 2026-09-21). CPG reports whether a person's cellular write process is operating within the "
 "healthy range for their age, by architecture class, against a fixed physical zero. It is not a clock: a clock asks how old a person looks and attaches no action "
 "to the answer; this asks whether the healthy range is held, and - when it is not - which class of cells slipped. Each clause is a commissioned layer: the range is "
 "the floor + identity_band_v3 (four laboratories, 1,379 healthy); 'for their age' is reference_age_curve_v1 (measured; the slope CPG-VAL-015 found); 'sitting in it' "
 "is the Stage 5 departure with the laboratory's false-alarm rate beside it; 'which cells' is Walther's eight class fractions and the 115-cell atlas beneath them. "
 "The last clause is the stated goal, not yet a claim: today one axis (immune, whole blood) carries a band; per-class bands, disease cohorts read absolutely against "
 "them (paper two), and the cell-level layer are the gates in order.")
