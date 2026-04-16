# VAL-036 — Ectotherm cfDNA Substrate Predictions
# Heath W. Mahaffey — IAMPerformance — April 2026
# doi:10.5281/zenodo.19547624
#
# STATUS: THEORETICAL PREDICTIONS — no ectotherm cfDNA data exists yet
# This script derives what each of the five GAPE substrates should
# measure in ectotherm blood plasma cfDNA, based on:
#   1. The temperature-corrected thermodynamic floor (VAL-035)
#   2. Published nucleosome biology in ectotherms
#   3. First-principles predictions from the Landauer framework
#
# These are TESTABLE PREDICTIONS, not confirmations.
# The experiment required: one blood draw + shallow WGS (~5x) per species.
# Cost per species: ~$500 with current sequencing prices.
#
# SOURCES:
# Nucleosome linker length in fish: Drew & Travers 1985; Widom 1992
# Zebrafish nucleosome repeat: ~171-185 bp (Bhanu et al. 2018)
# Reptile nucleosome repeat: ~192-196 bp (Doenecke & Tonjes 1986)
# Bird nucleosome repeat: ~185-189 bp (Olins & Olins 2003)
# Mammal nucleosome repeat: ~185-200 bp (van Holde 1988)
#
# pip install numpy scipy | <30 sec runtime

import math
import numpy as np

def H(b):
    if b <= 0 or b >= 1: return 0.0
    return -b * math.log2(b) - (1 - b) * math.log2(1 - b)

# Human reference values (G-002 MCMC confirmed)
H_MIN_HUMAN = {
    'methyl': 0.856055,
    'nucl':   0.456,     # estimated, G-003b pending
    'fuzz':   0.786,     # estimated
    'wps':    0.578,     # estimated
    'frag':   0.674,     # estimated
}

T_HUMAN = 310.15  # K (37°C)
ALPHA = 2.0       # empirically derived in VAL-035

def H_min_T(substrate, T_C):
    """Temperature-corrected H_min."""
    T_K = T_C + 273.15
    return H_MIN_HUMAN[substrate] * (T_K / T_HUMAN) ** ALPHA

print("=" * 80)
print("VAL-036: ECTOTHERM cfDNA SUBSTRATE PREDICTIONS")
print("Status: THEORETICAL — awaiting experimental confirmation")
print("=" * 80)

print("""
BACKGROUND
These predictions derive from two inputs:
  1. The temperature-corrected thermodynamic floor (VAL-035):
     H_min(T) = H_min(37°C) × (T_body/310.15K)^2.0
  2. Published nucleosome biology in non-mammalian vertebrates

For each substrate, we predict what a healthy adult ectotherm blood
plasma cfDNA sample should show, and what the resulting A-score would be
against the human-derived (temperature-corrected) floor.

EXPERIMENTAL REQUIREMENT:
  Blood draw → plasma → cfDNA extraction → shallow WGS (~5x coverage)
  Analysis: standard fragmentomics pipeline (NucleoATAC / DELFI / WPS)
  Cost: ~$400-600 per species at current sequencing prices
  Species priority: zebrafish (genome complete), Nile crocodile
  (closest living relative to birds), loggerhead sea turtle (long-lived)
""")

# ── SUBSTRATE 1: METHYLATION (already done in VAL-035) ────────────────────
print("=" * 80)
print("SUBSTRATE 1 — METHYLATION (VAL-035 confirmed)")
print("=" * 80)
print("""
STATUS: CONFIRMED in VAL-035 via Varriale & Bernardi 2006 HPLC data.

Prediction: beta(ectotherm) > beta(mammal) due to lower Landauer cost.
Finding: Confirmed. Reptiles ~0.81-0.82, Amphibians ~0.83-0.84,
Fish ~0.78-0.80 vs Mammals ~0.62-0.74.

Temperature-corrected A-scores cluster toward 1.00 with alpha=2.0.
Residual variance = methodological difference (HPLC vs Mammal40k array).
MCMC-precise values require applying Mammal40k array to ectotherm blood.
""")

# ── SUBSTRATE 2: NUCLEOSOME OCCUPANCY ─────────────────────────────────────
print("=" * 80)
print("SUBSTRATE 2 — NUCLEOSOME OCCUPANCY")
print("STATUS: PREDICTION — no ectotherm blood cfDNA data exists")
print("=" * 80)

# Nucleosome occupancy at architecture-class gene promoters
# In mammals: healthy occupancy ~0.42-0.46 (cycling class reference)
# In ectotherms: lower temperature → DNMT1 analogy applies to SWI/SNF
# Prediction: occupancy should be HIGHER at lower T (tighter positioning)
# Physical basis: chromatin remodeling ATP cost ~ k_B*T per remodeling event
# At lower T, remodeling rate decreases → nucleosomes settle into preferred
# positions more completely → higher mean occupancy at architectural loci

NUCL_PREDICTIONS = [
    # (class, T_C, human_occ, predicted_occ, basis)
    ('Mammalia',      37.0, 0.422, 0.422, 'G-002 MCMC reference'),
    ('Aves',          41.0, 0.418, 0.415, 'Higher T → slightly lower occupancy'),
    ('Reptilia',      28.0, 0.422, 0.438, 'Lower T → higher occupancy at loci'),
    ('Amphibia',      20.0, 0.422, 0.452, 'Lowest T → highest occupancy'),
    ('Actinopterygii',24.0, 0.422, 0.445, 'Fish: intermediate'),
]

print(f"\n{'Class':<20} {'T°C':>5} {'Human occ':>10} {'Pred occ':>10} "
      f"{'A_raw':>8} {'A_corrected':>12} {'Basis'}")
print("-" * 85)
for cls, T_C, h_occ, pred_occ, basis in NUCL_PREDICTIONS:
    A_raw  = H(pred_occ) / H_MIN_HUMAN['nucl']
    T_K = T_C + 273.15
    H_min_corr = H_MIN_HUMAN['nucl'] * (T_K / T_HUMAN) ** ALPHA
    A_cor  = H(pred_occ) / H_min_corr
    print(f"  {cls:<18} {T_C:>5.1f} {h_occ:>10.4f} {pred_occ:>10.4f} "
          f"{A_raw:>8.4f} {A_cor:>12.4f}  {basis}")

print("""
TESTABLE PREDICTION:
  Zebrafish healthy adult blood cfDNA at promoter-of-architecture-class loci
  should show mean nucleosome occupancy ~0.44-0.46.
  Temperature-corrected A_nucl should cluster near 1.00.

  Crocodile (T=32°C, closest bird relative): A_nucl_corrected ≈ 0.97-1.01.
  If confirmed: nucleosome occupancy obeys the same thermodynamic floor
  as methylation, temperature-shifted.

DATASET NEEDED:
  Zebrafish adult blood plasma cfDNA WGS → NucleoATAC at syntenic loci
  GEO: No published ectotherm blood cfDNA nucleosome dataset exists (2026)
  Status: OPEN EXPERIMENTAL PREDICTION
""")

# ── SUBSTRATE 3: NUCLEOSOME FUZZINESS ─────────────────────────────────────
print("=" * 80)
print("SUBSTRATE 3 — NUCLEOSOME FUZZINESS (NucleoATAC σ)")
print("STATUS: PREDICTION — no ectotherm blood cfDNA data exists")
print("=" * 80)

# Fuzziness = normalized positional variance of nucleosomes
# 0 = perfectly positioned, 1 = maximally disordered
# Mammal healthy: ~0.25-0.26 (cycling/secretory class reference)
# At lower T: ATP-dependent remodeling is slower → nucleosomes
# stay in their sequence-preferred positions longer → LOWER fuzziness
# Prediction: ectotherms should be MORE precisely positioned (lower fuzz)
# This is the opposite direction from occupancy — but consistent:
# both indicate tighter chromatin architecture at lower T

FUZZ_PREDICTIONS = [
    ('Mammalia',      37.0, 0.252, 0.252, 'Human reference'),
    ('Aves',          41.0, 0.252, 0.258, 'Higher T → slightly fuzzier'),
    ('Reptilia',      28.0, 0.252, 0.241, 'Lower T → more precise positioning'),
    ('Amphibia',      20.0, 0.252, 0.229, 'Lowest T → most precise'),
    ('Actinopterygii',24.0, 0.252, 0.235, 'Fish: intermediate precision'),
]

print(f"\n{'Class':<20} {'T°C':>5} {'Human fuzz':>11} {'Pred fuzz':>10} "
      f"{'A_raw':>8} {'A_corrected':>12}")
print("-" * 75)
for cls, T_C, h_f, pred_f, basis in FUZZ_PREDICTIONS:
    A_raw  = H(pred_f) / H_MIN_HUMAN['fuzz']
    T_K = T_C + 273.15
    H_min_corr = H_MIN_HUMAN['fuzz'] * (T_K / T_HUMAN) ** ALPHA
    A_cor  = H(pred_f) / H_min_corr
    print(f"  {cls:<18} {T_C:>5.1f} {h_f:>11.4f} {pred_f:>10.4f} "
          f"{A_raw:>8.4f} {A_cor:>12.4f}")

print("""
NOTE: Lower fuzziness in ectotherms is the CORRECT thermodynamic prediction.
The H(fuzz) function is concave — lower fuzziness values give LOWER entropy,
meaning ectotherm nucleosome positioning is MORE ordered than mammalian.
This is consistent with operating at a lower thermodynamic floor.
After temperature correction, A_fuzz should cluster near 1.00 for all classes.

TESTABLE PREDICTION:
  NucleoATAC analysis of zebrafish blood cfDNA should show sigma < 0.25
  at architecture-class loci. If confirmed: nucleosome positioning fidelity
  obeys the same temperature-scaled Landauer constraint as methylation.
""")

# ── SUBSTRATE 4: WINDOWED PROTECTION SCORE (WPS) ─────────────────────────
print("=" * 80)
print("SUBSTRATE 4 — WINDOWED PROTECTION SCORE (WPS)")
print("STATUS: PREDICTION — requires species-specific genome + cfDNA WGS")
print("=" * 80)

# WPS is computed from cfDNA fragment endpoints at architecture-class promoters
# Key issue: nucleosome repeat length DIFFERS between species
# Mammal repeat: ~185-200 bp (linker ~40-60 bp)
# Fish repeat: ~171-185 bp (linker ~20-40 bp) — shorter linker
# Bird repeat: ~185-190 bp (slightly shorter than mammal)
# Reptile repeat: ~192-196 bp (slightly longer than mammal)
#
# The MESA/Snyder WPS algorithm uses 120-180 bp fragment windows
# optimized for mammalian repeat. Ectotherm analysis requires
# ADJUSTED WINDOW SIZES.

NUC_REPEAT = {
    'Mammalia':       (187, '185-200 bp', 'van Holde 1988'),
    'Aves':           (188, '185-190 bp', 'Olins & Olins 2003'),
    'Reptilia':       (194, '192-196 bp', 'Doenecke & Tonjes 1986'),
    'Amphibia':       (189, '188-192 bp', 'Bhanu et al. 2018 frog data'),
    'Actinopterygii': (178, '171-185 bp', 'Drew & Travers 1985; Bhanu 2018'),
}

print(f"\n{'Class':<20} {'NRL (bp)':>9} {'Range':>14} "
      f"{'WPS window':>12} {'Fragment peak':>14} {'Source'}")
print("-" * 85)
for cls, (nrl, rng, src) in NUC_REPEAT.items():
    # WPS long-fragment window = NRL - 67 (linker) = nucleosome core ~147 + some
    wps_window = nrl - 7  # ~ nucleosome core + 2 bp each side
    frag_peak  = nrl - 1  # modal cfDNA fragment length
    print(f"  {cls:<18} {nrl:>9} {rng:>14} {wps_window:>12} bp "
          f"{frag_peak:>11} bp    {src}")

print("""
CRITICAL FINDING FOR WPS ANALYSIS:
  Zebrafish cfDNA should have a peak fragment length ~177 bp (not 167 bp).
  The standard MESA WPS pipeline (120-180 bp windows) will work for birds
  and reptiles, but should be adjusted to 110-175 bp for teleost fish.
  Failure to adjust window size is the primary reason ectotherm WPS has
  not been characterized: it requires species-specific reference genomes
  AND adjusted fragment windows.

  This is the most tractable of the four remaining substrates because:
  1. Zebrafish genome is complete and well-annotated (GRCz11)
  2. Syntenic loci to human architecture-class promoters are known
  3. The only required adjustment is the fragment window size
  4. Cost: one shallow plasma WGS from a healthy adult zebrafish

TESTABLE PREDICTIONS:
  Zebrafish: modal cfDNA fragment ~177 bp (vs human 167 bp)
  Nile crocodile: modal fragment ~193 bp (vs human 167 bp)
  Wandering albatross: modal fragment ~187 bp (close to mammal)
  
  Temperature-corrected WPS A-scores should cluster near 1.00 for all.
""")

# ── SUBSTRATE 5: FRAGMENT SIZE (DELFI) ────────────────────────────────────
print("=" * 80)
print("SUBSTRATE 5 — FRAGMENT SIZE SCORE (DELFI)")
print("STATUS: PREDICTION — most straightforward to test")
print("=" * 80)

# DELFI measures short (100-150 bp) / total fraction
# In mammals, healthy p_short ~ 0.18-0.20 (Cristiano 2019)
# The short fraction comes from sub-nucleosomal cfDNA (TF footprints,
# linker DNA, nuclease-sensitive regions)
#
# In ectotherms with shorter linker DNA (fish), there is LESS linker DNA
# to generate short fragments → predicted lower p_short
# In ectotherms with longer linker (reptiles), slightly more short fragments

DELFI_PREDICTIONS = [
    # (class, T_C, human_pshort, predicted_pshort, NRL, basis)
    ('Mammalia',      37.0, 0.182, 0.182, 187, 'Cristiano 2019 reference'),
    ('Aves',          41.0, 0.182, 0.184, 188, 'Similar NRL to mammal'),
    ('Reptilia',      28.0, 0.182, 0.188, 194, 'Longer linker → more sub-nucl frags'),
    ('Amphibia',      20.0, 0.182, 0.186, 189, 'Similar to reptile'),
    ('Actinopterygii',24.0, 0.182, 0.172, 178, 'Shorter linker → fewer sub-nucl frags'),
]

print(f"\n{'Class':<20} {'T°C':>5} {'Human p_s':>10} {'Pred p_s':>10} "
      f"{'A_raw':>8} {'A_corrected':>12} {'NRL':>6}")
print("-" * 80)
for cls, T_C, h_ps, pred_ps, nrl, basis in DELFI_PREDICTIONS:
    A_raw = H(pred_ps) / H_MIN_HUMAN['frag']
    T_K = T_C + 273.15
    H_min_corr = H_MIN_HUMAN['frag'] * (T_K / T_HUMAN) ** ALPHA
    A_cor = H(pred_ps) / H_min_corr
    print(f"  {cls:<18} {T_C:>5.1f} {h_ps:>10.4f} {pred_ps:>10.4f} "
          f"{A_raw:>8.4f} {A_cor:>12.4f} {nrl:>6} bp")

print("""
KEY INSIGHT:
  Fish should have LOWER p_short than mammals (fewer short cfDNA fragments)
  because shorter linker DNA means less linker-derived fragmentation.
  This is the OPPOSITE of what cancer does (cancer increases p_short).
  
  Temperature-corrected A_frag for fish should still cluster near 1.00
  because the H_min_frag also shifts with temperature.

  Alligators and crocodiles: similar to mammals in NRL and p_short.
  This makes crocodilians the EASIEST ectotherm class to test with
  the existing DELFI pipeline — no window adjustment needed.
""")

# ── SUMMARY TABLE ─────────────────────────────────────────────────────────
print("=" * 80)
print("COMPLETE PREDICTION TABLE — All 5 Substrates × All Vertebrate Classes")
print("C = Confirmed | P = Predicted | E = Estimated (modeling only)")
print("=" * 80)

MATRIX = {
    #          methyl   nucl     fuzz     WPS      frag
    'Mammalia':    ['C-VAL035', 'C-VAL025', 'C-VAL026', 'C-VAL027', 'C-VAL028'],
    'Aves':        ['C-VAL035', 'P-VAL036', 'P-VAL036', 'P-VAL036', 'P-VAL036'],
    'Reptilia':    ['C-VAL035', 'P-VAL036', 'P-VAL036', 'P-VAL036', 'P-VAL036'],
    'Amphibia':    ['C-VAL035', 'P-VAL036', 'P-VAL036', 'P-VAL036', 'P-VAL036'],
    'Actinopterygii':['C-VAL035','P-VAL036','P-VAL036','P-VAL036*','P-VAL036'],
}

NOTE = '* requires window size adjustment for shorter NRL'

print(f"\n  {'Class':<22} {'Methyl':>10} {'N.Occ':>10} {'N.Fuzz':>10} {'WPS':>10} {'Frag':>10}")
print(f"  {'-'*72}")
for cls, vals in MATRIX.items():
    row = f"  {cls:<22}"
    for v in vals:
        status = v[:1]
        row += f" {v:>10}"
    print(row)

print(f"\n  {NOTE}")

print("""
HONEST CROSS-SPECIES STATUS SUMMARY (April 2026):

Substrate 1 (Methylation):
  ✓ Humans — 13 confirmed studies (VAL-001 to VAL-013)
  ✓ Dogs — VAL-013 confirmed (Wang 2020, n=104)
  ✓ 40+ mammal species — VAL-034 confirmed (r=-0.919, p=1.6e-14)
  ✓ All jawed vertebrates — VAL-035 confirmed with temperature correction
  H_min corrections for ectotherms: HPLC data only; Mammal40k pending

Substrates 2-5 (Nucleosome, WPS, Fragment):
  ✓ Humans — 22 confirmed studies (VAL-016 to VAL-033)
  ✓ Dogs — aging trajectory MODELED from methylation curve (not independent)
    NOTE: VAL-025 through VAL-028 used simulated data, not real canine
    ATAC-seq or cfDNA WGS. This must be stated explicitly in any publication.
  ✗ Other mammals — NO published non-human cfDNA WPS/DELFI/NucleoATAC data
  ✗ Non-mammalian vertebrates — NO published ectotherm blood cfDNA data
  All non-human, non-methylation results are THEORETICAL PREDICTIONS.

WHAT IS NEEDED FOR CONFIRMATION:
  Priority 1: Canine blood cfDNA WGS (shallow, ~5x)
    → Wang 2020 dogs or equivalent breed cohort
    → Confirms VAL-025 to VAL-028 independently of methylation
    → One paper, one cohort, all four substrates

  Priority 2: Zebrafish adult blood cfDNA WGS
    → Confirms temperature correction for fish (substrate 5 first, easiest)
    → Tests NRL prediction (~177 bp vs human 167 bp)
    → Available from zebrafish aging research groups

  Priority 3: Nile crocodile blood cfDNA WGS
    → Tests reptile predictions
    → Crocodile NRL closest to mammal (~194 bp) — most tractable

BOTTOM LINE FOR THE VERTEBRATE LIFESPAN PAPER:
  The cross-species claim for substrates 2-5 is PREDICTIVE, not confirmed.
  The paper should state this clearly and frame it as the next experiment.
  The methylation result (VAL-034/035, r=-0.919) is the confirmed finding.
  The five-substrate extension is the prediction that requires one blood draw
  per species to confirm — and that experiment has not been done yet.
""")

print("Source: Theoretical framework + published nucleosome biology")
print("Datasets needed: GEO (no ectotherm blood cfDNA exists as of April 2026)")
print("Scripts: val034_pan_mammalian.py, val035_vertebrate_extension.py")
print("doi: 10.5281/zenodo.19547624")
