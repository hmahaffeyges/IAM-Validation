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
 ("S1", "Substrate floors for cfDNA", "cfDNA listed as a substrate with % contributions (CFDNA_PCT)",
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
_f2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "handoff", "formula_2x2.json")
FORMULA_2X2 = json.load(open(_f2)) if os.path.exists(_f2) else {}
FORMULA_LABELS = {"GSM2333901":"WB healthy 58M","GSM2333905":"WB healthy 67F","GSM2333950":"WB healthy 43M","GSM1051525":"WB RA","GSM1051526":"WB RA",
     "GSM1051533":"WB RA-study ctrl","GSM1051534":"WB RA-study ctrl","GSM8772491":"tissue adenoma","GSM8772492":"tissue adenoma","GSM5065990":"tissue CRC st1","GSM5065985":"tissue CRC st4"}
FORMULA_FINDINGS = [
 ("Whole blood, immune class present - identity loci", "H(beta_mean) and mean-of-H agree in RANK exactly (Spearman +1.000 over 7 donors) with a constant Jensen offset of +0.029 +/- 0.002. On the declared substrate with the class present, the formula choice is a calibration convention, not a physics question."),
 ("Whole blood, cycling/secretory - classes ABSENT from blood", "offset +0.16: the absent class's identity loci sit locked/bimodal in immune DNA and H(beta_mean) inflates toward the ceiling. This is the SOP v1.4.0 s105 mechanism, and it is exactly what the presence gate (s3.2) prevents from being read."),
 ("Bulk tissue - a mixture", "offset +0.24 +/- 0.07 (immune). The 'disease ordering' first read off H(beta_mean) on tissue (adenoma 0.98-1.10, CRC-4 1.19) is COMPOSITION INFLATION, not architecture - glioma-LL-002 already says a high A on bulk tissue is a heterogeneity marker. This corrects the verdict written earlier on 2026-09-19."),
 ("age_reference_matrix.json", "compiled as A_mean = H(beta_mean)/H_min - exact to 5 decimals in all 80 (class x decade) cells. The runtime gauge MUST use the same aggregation as the band it is read against; v1.4.0's mean-of-H against this band reads every patient ~0.03 low."),
 ("Discriminative markers, mean of H  [iamatlas_a_scoring; sealed-anchor formula]", "immune flat 0.60-0.70 across blood and tissue; cycling WB 0.35-0.41 vs tissue 0.57-0.62. Moves with PRESENCE of the class's DNA - the separation surface, which is why it reproduces the sealed GSE51032 anchor (d = +2.088) and why CCL-019 found its sign depends on compartment."),
]
FORMULA_VERDICT = ("SOP v1.4.0 (2026-06-30, current) says the A-score is NEVER H(beta_mean); the wired chain at HEAD (d7b0e1f, 2026-07-01) computes exactly H(beta_mean) over identity loci; SOP v1.3.3 carries the same date as v1.4.0 and the opposite formula. "
  "Measured: on the declared substrate with the class present the two aggregations differ by a constant (+0.029), so either is a valid gauge PROVIDED the age band is compiled the same way - and the band is H(beta_mean). "
  "On any mixed or absent-class panel H(beta_mean) inflates (s105 is right about the mechanism) and must never be read as architecture; the presence gate and glioma-LL-002 are the guards. The sealed anchors are separation-surface results and are reproduced by mean-of-H over discriminative markers, as v1.4.0 states. "
  "REQUIRED DECISION (author): (i) keep H(beta_mean) in the gauge and amend v1.4.0 s105 to scope its NEVER to mixed/absent panels, or (ii) adopt mean-of-H in the gauge and recompile age_reference_matrix. Either is defensible; running one formula against the other's band is not. Until decided, this issue prints both.")
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
 ("Prior", "Gaussian centred on the published calibration value"), ("Cross-check", "leave-one-out bootstrap, n = 10,000 resamples; MCMC and bootstrap must agree within 5-10% or the cohort is audited"),
 ("Convergence", "17 methylation chains R-hat < 1.001 (G-002); 5 x 32-walker ensembles per substrate (G-003b)"),
 ("Runtime", "G-002 29.7 s for 8 classes (laptop); G-003b ~24 min for 32 posteriors (desktop)"),
 ("Notable posterior shift", "immune: 0.795 (neutrophil-based calibration) -> 0.838889 after MCMC over six immune cell types, a 6.44 sigma move (EDEAR s4.6); every immune A in the database was revised downward by approximately 0.055 as a consequence (Issue 002, immune card)"),
 ("H_min_global", "0.756499 = H(0.782), frontal cortex neuron, Lister 2013 (E073) - the universal reference the class floors are read against"),
 ("Reference-cohort rule", "FACS-sorted or laser-microdissected only; never bulk tissue (cell-type purity)"),
]
ATLAS_BUILD = [  # EDEAR s5.4-5.6, SOP s26-29
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
 ("Status in this issue", "NOT executed; specified here from SOP v1.4.0 so the next issue can write PROC-HULL-01"),
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
    if "VOID" in (_o.get("path","")+" ".join([])).upper() or "VOIDED" in json.dumps(_o).upper(): _o["decision"]="VOIDED (seal preserved)"
VAL_INDEX_NOTE = ("Built by walking the repository for every path matching VAL-###: 103 identifiers, 1,175 files. Title is the first heading of the outcome (or prereg) file; "
  "date, cohorts and SHA presence are read from that file. 'Decision' is filled ONLY where the file states an explicit outcome code or status (36 of 103); otherwise 'see file' - "
  "the older outcome files carry the verdict in prose and this index does not paraphrase it. 'Record' says what the repo holds: outcome (87), prereg only (0), files only (16). "
  "Seven identifiers exist only under RETIRED_Phase1_PreBuild_Cards. VAL-047, cited in Issue 002 s8.1 as the deployment-readiness validation, has scripts in the repo but no prereg and no outcome file. "
  "This index is the map to the evidence, not the evidence; the historical VALs were left to stand as their own record (Issue 003 s10) and none was re-run here except where a PROC says so.")
FOUR_SKIES_CAP = ("Four skies on one HEALPix grid (NSIDE 128, 196,608 pixels, Mollweide; CpGs in atlas row order chr1 -> chrY, the Plate 1 convention). "
  "(a) One realization of the microwave CMB from the Planck 2018 LCDM power spectrum (CAMB -> healpy.synfast). (b) The IAM Atlas immune-class posterior mean beta per CpG - the 'brilliance' sky, 483,092 CpGs, MCMC posterior. "
  "(c) The same class's posterior sd per CpG - the healthy-variance sky, which is what makes (d) possible. (d) Patient GSM1051533 (whole blood, 450K, Stage-1 calibrated, RA-study control) as z = (beta - mu)/sigma against (b, c), computed by the engine's cpg_patient_cmb.py. "
  "Median |z| 1.40; mean z +1.07 - the uniform red tint is the +0.05 reference-beta offset of s6a, not a patient finding. The engine's self-determined assessability flagged stromal as assessable from blood (median |z| 1.44); the deconvolver's presence gate says stromal = 0, and the presence gate is the correct one - a known weakness of the median-|z| criterion recorded here. "
  "The point of the figure is the method, not the patient: a reference map with per-pixel uncertainty and the residual of one observation against it is exactly the Planck workflow, and it is why the CMB toolkit transfers.")
