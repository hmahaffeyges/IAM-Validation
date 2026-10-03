# DEV_SPECIES_AGEING_01 — within-species ageing on the consortium blood arrays

Development note, 2026-10-03. Development reading, not a test. Every species is read against its own young adults; no species is compared with another, and no statistic is pooled across species.

Question (author): does each vertebrate species, read against itself, lose methylation fidelity with age, and do species reach a similar reading at a similar fraction of their own maximum lifespan? "Age should move the needle and each one can only get so old before no longer maintaining fidelity."

Data: GEO GSE223748 (Mammalian Methylation Consortium, HorvathMammalMethylChip40), "Blood" arrays only, the per-array values and probe sets from the species rebuild package (`species_mmc_rebuild.zip`, built 2026-10-03), AnAge Build 15. All 30 species that qualify are mammals; the array has no non-mammal vertebrates.

## 1. Choices set before any reading was looked at (written 2026-10-03 00:07 local, before Met-A values were plotted or summarised)

Status labels: **SET** = fixed by this note before reading; **INHERITED** = taken unchanged from the species rebuild package; **TASK** = fixed by the task brief.

| # | Choice | Status |
|---|---|---|
| C1 | Arrays: GSE223748, GEO source name "Blood" only; "DifferentiatedBloodCells" excluded. | TASK / INHERITED |
| C2 | Species kept if >= 10 blood arrays with a recorded age AND (oldest age - youngest age) >= 30 % of the species' AnAge Build 15 maximum longevity. Counted on all aged arrays (juveniles included), as the brief states it. | TASK |
| C3 | Adult = age >= AnAge sexual maturity for that animal's sex (female maturity for females, male maturity for males; if the animal's sex is unrecorded, the larger of the two; if one sex's value is missing, the other's). Fallback to the consortium's age-at-maturity column if AnAge gives neither (not needed: all 30 species have AnAge values). | SET |
| C4 | Young reference = the youngest quarter of that species' adults by age, rounded up, minimum 3 animals. Ties broken by GSM id. | SET |
| C5 | Juveniles (below maturity) get a Met-A reading and are plotted (grey, hollow) but are excluded from the reference, from the oldest-decile group and from the rank correlation. | SET |
| C6 | Per-animal quantity: mean over the panel probes of H(beta), H = binary entropy in bits, beta values with detection p >= 0.05 on that array dropped. Taken from the package's per-array table (`sample_stats_final.csv`), and recomputed on the box from the raw beta matrix as a check. | INHERITED |
| C7 | Met-A = that per-animal mean H / median of the same quantity over the species' own young reference. Nothing fitted. Computed on (a) the species' primary probe set and (b) the conserved panel (3,980 probes; each species uses the subset it detects). | TASK |
| C8 | Age axis: age / AnAge Build 15 maximum longevity of the species. Values above 1 are kept and flagged (recorded age above the AnAge record). | TASK |
| C9 | Oldest decile = the oldest 10 % of that species' adults by fractional age, rounded up, minimum 2. Also reported: animals with fractional age >= 0.90 (often none). | SET |
| C10 | "Rises with age" is described by Spearman rank correlation of Met-A against fractional age within the species' adults. Description only: no p-value, no threshold, no pooled statistic across species. | SET |
| C11 | Overlay: all species on one fractional-age axis, coloured by AnAge order; per species a running median (window = 15 % of lifespan) drawn to help the eye, not a fit. No common curve. | SET |
| C12 | Mahaffey number M = E_drive / (k_B T), E_drive = one ATP = 54 kJ/mol (per molecule 54,000 / N_A J), T = AnAge "Temperature (K)"; missing if AnAge has none. | TASK |
| C13 | Blood-composition check (describe only): lineage CpGs taken from human sorted blood cells (GSE110554, Salas et al. 2018, EPIC array) that are also on the mammalian array. Myeloid-low set: CpGs whose mean beta in neutrophils+monocytes is >= 0.40 lower than in CD4 T, CD8 T, B and NK cells (153 CpGs); lymphoid-low set: the reverse (21); T-low set: CD4+CD8 T mean >= 0.40 below the other four (70). Per animal: mean beta over those CpGs that the species detects. Assumes the human cell-type pattern holds in the other species; not shown. | SET |
| C14 | Probe-level variance check (describe only): per probe, SD of beta across animals in the young reference and in the oldest-decile group; median over probes. Also the share of panel probes whose mean H is higher in the oldest-decile group than in the reference. | SET |
| C15 | Sex, number of animals, chip (array barcode prefix) count, age-estimate confidence < 100, and probe basis (own assembly / congener / detection only) reported per species. | SET |

30 species meet C2.


## 2. What was run

1. `scripts/species_ageing_readings.py --stage groups`: species selection, adult/juvenile split, young reference, oldest decile (`groups.csv`, `species_qualification.csv`); lineage CpGs from the GSE110554 series matrix (`immune_marker_probes.csv`).
2. `data/DEV_SPECIES_AGEING_01/scripts/probe_level_confounds.py` on the box (methylphys-cpu-01) against the 3.3 GB beta matrix and the IDAT detection p-values: recomputed per-array mean H on both panels, lineage-CpG means, probe-level SDs (`box_outputs/`). The recomputed mean H matches the package's per-array values on all 3,890 arrays × 2 panels (largest difference 0.0).
3. `scripts/species_ageing_readings.py --stage readings`: Met-A per animal, per-species summary, figures.

## 3. Which species qualify

30 of 126 blood species (3,890 aged blood arrays; 3,111 adults, 779 juveniles). Orders: Carnivora 6, Artiodactyla 5, Cetacea 4, Primates 4, Rodentia 4, Perissodactyla 3, Proboscidea 2, Diprotodontia 2. The 96 that do not qualify, with their number of aged blood arrays (and, where the span was too short, the age span as % of max lifespan):

- **age span < 30% of max lifespan** (4): *Ovis aries* 168 (18%), *Sus scrofa domesticus* 98 (23%), *Sus scrofa* 60 (19%), *Apodemus sylvaticus* 13 (14%)
- **fewer than 10 aged arrays** (89): *Globicephala macrorhynchus* 9, *Tragelaphus strepsiceros* 9, *Trichechus manatus* 9, *Nanger dama* 8, *Panthera tigris* 8, *Equus asinus somalicus* 7, *Litocranius walleri* 7, *Panthera leo* 7, *Tragelaphus imberbis* 7, *Crocuta crocuta* 6, *Gorilla gorilla* 6, *Hippotragus equinus* 6, *Steno bredanensis* 6, *Chrysocyon brachyurus* 5, *Diceros bicornis* 5, *Equus grevyi* 5, *Hippotragus niger* 5, *Macropus fuliginosus* 5, *Notamacropus rufogriseus* 5, *Oryctolagus cuniculus* 5, *Tachyglossus aculeatus* 5, *Tragelaphus angasii* 5, *Tragelaphus eurycerus* 5, *Vicugna pacos* 5, *Choloepus hoffmanni* 4, *Eudorcas thomsonii* 4, *Nanger soemmerringii* 4, *Orycteropus afer* 4, *Phascolarctos cinereus* 4, *Rhinoceros unicornis* 4, *Varecia variegata* 4, *Daubentonia madagascariensis* 3, *Delphinus delphis* 3, *Enhydra lutris* 3, *Eulemur flavifrons* 3, *Eulemur fulvus collaris* 3, *Eulemur macaco* 3, *Eulemur mongoz* 3, *Eulemur rubriventer* 3, *Eulemur rufus* 3, *Eulemur sanfordi* 3, *Gazella leptoceros* 3, *Hapalemur griseus* 3, *Lemur catta* 3, *Nanger granti* 3, *Notamacropus agilis* 3, *Osphranter robustus* 3, *Otolemur crassicaudatus* 3, *Propithecus coquereli* 3, *Propithecus tattersalli* 3, *Tragelaphus spekii* 3, *Varecia rubra* 3, *Xanthonycticebus pygmaeus* 3, *Antidorcas marsupialis* 2, *Cavia porcellus* 2, *Cheirogaleus medius* 2, *Eulemur albifrons* 2, *Eulemur coronatus* 2, *Eulemur fulvus* 2, *Galago moholi* 2, *Kobus megaceros* 2, *Microcebus murinus* 2, *Mirza zaza* 2, *Muntiacus vaginalis* 2, *Mustela putorius furo* 2, *Nycticebus coucang* 2, *Ornithorhynchus anatinus* 2, *Pan troglodytes* 2, *Propithecus diadema* 2, *Addax nasomaculatus* 1, *Aepyceros melampus* 1, *Callithrix geoffroyi* 1, *Choloepus didactylus* 1, *Cryptomys hottentotus* 1, *Didelphis virginiana* 1, *Giraffa camelopardalis* 1, *Hystrix cristata* 1, *Loris tardigradus* 1, *Okapia johnstoni* 1, *Perodicticus potto* 1, *Phocoena phocoena* 1, *Pongo pygmaeus* 1, *Procavia capensis* 1, *Puma concolor* 1, *Tragelaphus oryx* 1, *Ursus americanus* 1, *Vombatus ursinus* 1, *Vulpes vulpes* 1, *Lynx rufus* 0
- **no AnAge max longevity** (3): *Cephalorhynchus commersonii* 0, *Odobenus rosmarus divergens* 0, *Phoca groenlandica* 0

## 4. Readings as measured

Met-A = 1 is the median of the species' own young adults. Above 1 = higher mean per-probe entropy than the young.

| Species | Common name | Order | Adults | Ref n (ages, y) | Ref Met-A range | Oldest decile frac. age | Met-A oldest decile, median (species set / conserved) | ρ adults (species set / conserved) | M | Probe basis |
|---|---|---|---|---|---|---|---|---|---|---|
| *Bos taurus* | Cattle | Artiodactyla | 250 | 63 (1.5–2.7) | 0.931–1.061 | 0.50–0.72 | 1.069 / 1.061 | +0.61 / +0.62 | 20.87 | own assembly |
| *Capreolus capreolus* | Roe deer | Artiodactyla | 123 | 31 (1.7–3.0) | 0.904–1.093 | 0.55–0.78 | 1.035 / 1.022 | +0.39 / +0.35 | 20.87 | detection only |
| *Cervus canadensis* | Wapity elk | Artiodactyla | 24 | 6 (2.7–4.7) | 0.994–1.003 | 0.37–0.43 | 1.004 / 1.004 | +0.30 / +0.10 | 21.06 | detection only |
| *Cervus elaphus* | Red deer | Artiodactyla | 32 | 8 (2.8–3.9) | 0.988–1.008 | 0.31–0.34 | 0.998 / 0.998 | -0.09 / -0.14 | 21.06 | detection only |
| *Connochaetes taurinus albojubatus* | White-bearded gnu | Artiodactyla | 10 | 3 (7.0–8.2) | 0.992–1.137 | 0.71–0.78 | 1.090 / 1.088 | +0.59 / +0.46 | 20.87 | detection only |
| *Acinonyx jubatus* | Cheetah | Carnivora | 11 | 3 (5.0–6.7) | 0.991–1.014 | 0.49–0.61 | 0.992 / 0.991 | -0.20 / -0.15 | 20.81 | own assembly |
| *Aonyx cinereus* | Asian small-clawed otter | Carnivora | 12 | 3 (7.6–7.6) | 0.988–1.068 | 0.78–0.79 | 1.037 / 1.049 | +0.11 / +0.18 | missing | detection only |
| *Canis lupus familiaris* | Dog | Carnivora | 632 | 158 (1.4–3.8) | 0.952–1.084 | 0.47–0.65 | 1.048 / 1.043 | +0.47 / +0.50 | missing | own assembly |
| *Felis catus* | Cat | Carnivora | 93 | 24 (0.8–4.8) | 0.951–1.078 | 0.57–0.70 | 0.999 / 1.009 | +0.13 / +0.12 | 20.87 | own assembly |
| *Phoca vitulina* | Harbor seal | Carnivora | 35 | 9 (3.6–8.2) | 0.966–1.063 | 0.65–1.01 | 1.021 / 1.026 | +0.19 / +0.27 | 20.96 | own assembly |
| *Zalophus californianus* | California sea lion | Carnivora | 31 | 8 (3.0–6.6) | 0.957–1.073 | 0.74–0.85 | 0.976 / 0.977 | -0.07 / -0.09 | 20.87 | own assembly |
| *Aethalodelphis obliquidens* | Pacific white-sided dolphin | Cetacea | 9 | 3 (12.2–19.0) | 0.971–1.016 | 0.76–0.86 | 1.052 / 1.057 | +0.54 / +0.55 | missing | own assembly |
| *Delphinapterus leucas* | Beluga whale | Cetacea | 46 | 12 (4.7–10.8) | 0.958–1.033 | 0.97–1.22 | 1.033 / 1.037 | +0.47 / +0.50 | 21.02 | own assembly |
| *Orcinus orca* | Killer whale | Cetacea | 28 | 7 (10.7–16.2) | 0.949–1.070 | 0.58–0.60 | 0.989 / 0.986 | -0.12 / -0.22 | 21.01 | own assembly |
| *Tursiops truncatus* | Bottlenose dolphin | Cetacea | 110 | 28 (8.3–13.5) | 0.938–1.103 | 0.78–1.12 | 1.031 / 1.049 | +0.16 / +0.19 | missing | own assembly |
| *Macropus giganteus* | Eastern grey kangaroo | Diprotodontia | 8 | 3 (1.8–1.9) | 0.968–1.016 | 0.49–0.53 | 1.063 / 1.055 | +0.53 / +0.53 | 20.99 | detection only |
| *Osphranter rufus* | Red kangaroo | Diprotodontia | 32 | 8 (2.2–6.7) | 0.924–1.062 | 0.43–0.47 | 1.030 / 1.033 | +0.21 / +0.22 | 21.02 | detection only |
| *Ceratotherium simum simum* | White rhino | Perissodactyla | 14 | 4 (7.5–9.0) | 0.971–1.029 | 0.56–0.66 | 1.063 / 1.049 | +0.59 / +0.41 | missing | own assembly |
| *Equus caballus* | Horse | Perissodactyla | 185 | 47 (3.0–7.0) | 0.926–1.174 | 0.37–0.49 | 1.058 / 1.057 | +0.38 / +0.36 | 20.85 | own assembly |
| *Equus quagga* | Zebra | Perissodactyla | 46 | 12 (2.6–3.8) | 0.958–1.058 | 0.39–0.53 | 1.066 / 1.060 | +0.55 / +0.54 | missing | congener |
| *Callithrix jacchus* | Marmoset | Primates | 83 | 21 (1.4–3.2) | 0.966–1.106 | 0.54–0.68 | 1.030 / 1.013 | +0.20 / +0.18 | 21.01 | own assembly |
| *Chlorocebus sabaeus* | Vervet | Primates | 99 | 25 (3.1–10.0) | 0.949–1.130 | 0.67–0.78 | 1.119 / 1.109 | +0.45 / +0.42 | 20.95 | own assembly |
| *Homo sapiens* | Human | Primates | 240 | 60 (13.0–40.4) | 0.899–1.098 | 0.66–0.75 | 0.973 / 0.991 | -0.38 / -0.24 | 20.94 | own assembly |
| *Macaca mulatta* | Rhesus macaque | Primates | 192 | 48 (6.6–11.7) | 0.941–1.057 | 0.69–1.05 | 1.010 / 1.013 | +0.34 / +0.36 | 20.92 | own assembly |
| *Elephas maximus* | Asian elephant | Proboscidea | 79 | 20 (9.1–24.2) | 0.971–1.054 | 0.74–0.92 | 1.025 / 1.015 | +0.36 / +0.30 | 21.02 | own assembly |
| *Loxodonta africana* | Savanna elephant | Proboscidea | 43 | 11 (10.3–24.2) | 0.981–1.035 | 0.62–0.75 | 1.024 / 1.023 | +0.37 / +0.38 | 20.99 | own assembly |
| *Heterocephalus glaber* | Naked mole rat | Rodentia | 92 | 23 (0.7–1.8) | 0.969–1.030 | 0.33–0.84 | 1.062 / 1.057 | +0.56 / +0.58 | 21.28 | own assembly |
| *Marmota flaviventris* | Yellow-bellied marmot | Rodentia | 123 | 31 (2.0–4.0) | 0.933–1.061 | 0.47–0.57 | 1.022 / 1.030 | +0.36 / +0.38 | 20.97 | own assembly |
| *Mus musculus* | Mouse | Rodentia | 366 | 92 (0.1–0.7) | 0.914–1.182 | 0.55–0.68 | 1.076 / 1.082 | +0.50 / +0.49 | 20.95 | own assembly |
| *Rattus norvegicus* | Rat | Rodentia | 153 | 39 (0.2–0.6) | 0.919–1.230 | 0.64–0.64 | 1.122 / 1.113 | +0.60 / +0.58 | 20.93 | own assembly |

Ref Met-A range is on the species-specific probe set; the conserved-panel ranges are in `per_species_summary.csv`. M = Mahaffey number (§5).

Plain description:

- **Direction within species.** In 25 of 30 species Met-A rises with fractional age among adults (ρ > 0 on both panels; ρ from +0.11 to +0.61 on the species-specific set). In 5 it does not: human (ρ −0.38 / −0.24), killer whale (−0.12 / −0.22), cheetah (−0.20 / −0.15; 11 adults), red deer (−0.09 / −0.14; detection-only probes) and California sea lion (−0.07 / −0.09). The two panels give the same sign in every species.
- **Size of the move.** Small. The oldest-decile median runs from 0.973 (human) to 1.122 (rat) on the species-specific set. In 7 species it sits above the top of the young reference range (cow, wapiti, Pacific white-sided dolphin, eastern grey kangaroo, white rhinoceros, plains zebra, naked mole-rat); in the other 23 it is inside the spread of the young. The largest within-species moves are rat, vervet monkey (1.119), wildebeest (1.090, 10 animals), mouse, cow, plains zebra, white rhinoceros, eastern grey kangaroo and naked mole-rat (1.06–1.08).
- **End of life is mostly not sampled.** The oldest decile starts at a median of 0.57 of AnAge maximum lifespan (range 0.31–0.97). Only 5 species have any adult at or above 0.9: harbour seal (1 animal, Met-A 1.016), beluga (5, 1.033), bottlenose dolphin (4, 1.100), rhesus macaque (2, 0.978), Asian elephant (1, 1.027). Six animals have recorded ages above the AnAge record (frac > 1; seal 1, beluga 3, dolphin 1, macaque 1).
- **Do the species line up on one fractional-age axis?** See `fig_overlay_fractional_age_*`. Up to ~0.5 of lifespan most running medians climb together between 1.00 and ~1.06. Past that the curves fan out (rat and vervet ~1.10–1.12; mouse and cow ~1.07; dog ~1.05; elephants and most cetaceans ~1.02–1.05; human below 1). With the end of life sampled in 5 species only, whether readings converge near the end of life cannot be read off these data.

## 5. Mahaffey number

M = E_drive / (k_B T), E_drive = one ATP = 54 kJ/mol = 8.97 × 10⁻²⁰ J per molecule; no ln 2. From AnAge body temperature: M = 20.81 (cheetah, 312.15 K) to 21.28 (naked mole-rat, 305.25 K); human 20.94 (310.15 K). Missing (no AnAge temperature): dog, small-clawed otter, Pacific white-sided dolphin, bottlenose dolphin, plains zebra, white rhinoceros. **Across these mammals M differs only through body temperature**, because the drive is the same ATP in every species; the spread is 2.3 %. M is reported per species, not set against the readings.

## 6. Confounds (reported, not removed)

- **Blood cell composition.** No purified-cell reference exists for these species. Proxy: human sorted-cell lineage CpGs present on the array and detected in the species (myeloid-low 32–136 CpGs per species, T-low 6–65, lymphoid-low 2–21). Among adults, the myeloid-low CpGs lose methylation with age in 24 of 30 species (median ρ -0.31), which is what a shift toward myeloid cells would look like; the T-low CpGs gain methylation with age in 22 of 30 (median ρ +0.10). Met-A moves with these proxies within species (T-low CpG methylation vs Met-A: ρ < 0 in 29 of 30). So composition change is present and tracks Met-A; these arrays cannot say how much of the Met-A rise is composition and how much is within-cell change. The proxy assumes the human cell-type pattern holds in each species; not shown.
- **Probe-level spread.** Median probe SD across animals is higher in the oldest-decile group than in the young reference in only 11 of 30 species (ratio 0.45–1.92; the oldest group is small, so this is noisy). The share of panel probes whose mean H is higher in the oldest group than in the reference runs 0.37–0.82 (median 0.58); low-β probes gaining and high-β probes losing methylation both contribute (median shares 0.59 and 0.56).
- **Batch / study, confounded with age.** Arrays from one species often come from different submissions run on different chips at different ages. Share of fractional-age variance lying between chips (η²) > 0.5 in 15 of 30 species; share of Met-A variance between chips > 0.5 in 12. Human is the clearest case: the young reference (60 animals, ages 13.0–40.4; median 14.4) contains 55 animals aged 13–15 on chips 203877…/204027…/204018…, which hold only 12–15-year-olds; the older adults sit on chips 203203… (ages 26–76) and 205600… (ages 34–92, median 68), and the whole oldest decile (24 animals, ages 81.3–92.0) is on 205600…; η²(age by chip) = 0.87. The negative human reading cannot be separated from study batch on these data.
- **Detection rate.** Per-array detection rate moves with age in some species (ρ −0.54 to +0.65), i.e. DNA quality or batch varies with age.
- **Sex.** In 22 of 26 species with both sexes, the adult female median Met-A is above the male median (difference median +0.015, range −0.046 to +0.066). Reference and oldest groups differ in female share by > 0.3 in 11 species (e.g. rat 0.46 → 1.00, human 0.70 → 0.33, both elephants → 1.00). Not adjusted.
- **Few animals.** Five species have ≤ 12 adults (Pacific white-sided dolphin 9, eastern grey kangaroo 8, wildebeest 10, cheetah 11, otter 12); their reference is 3 animals.
- **Probe binding.** 7 species rest on detection-only probe sets (no own or congener assembly; † in figures): roe deer, wapiti, red deer, wildebeest, small-clawed otter, eastern grey kangaroo, red kangaroo. Plains zebra uses a congener assembly.
- **Age.** 415 of 3,890 arrays have consortium age confidence < 100 (wild animals with estimated ages); kept.
- **Lifespan record.** AnAge maximum longevity is a single record, often from captivity; fractional age depends on it. Six animals exceed it.

## 7. Single-molecule data (next step only)

The per-molecule reading needs reads, not array β. Public bisulfite-sequencing blood data with ages found in this search (provisional; not downloaded, coverage and age range not checked):

| Species | Data | Accession / source |
|---|---|---|
| Dog | RRBS, blood, 46 dogs (+ 62 grey wolves) | Thompson et al. 2017, Aging 9:1055 |
| Dog | RRBS, PBMC, 71 dogs | PRJNA1049514 (Jin et al. 2024, Aging Cell) |
| Dog | WGBS ~70×, whole blood, 19 dogs aged 3–14 y | PMC11566899 (2024) |
| Dog | Targeted bisulfite (SyBS), blood, 104 Labradors 0.1–16 y | Wang et al. 2020, Cell Systems |
| Rhesus macaque | RRBS, whole blood, 573 samples, free-ranging Cayo Santiago | PMC12269644 (2025) |
| House mouse | RRBS, blood, ~230 mice across lifespan | Petkovich et al. 2017, Cell Metab |
| Norway rat | RRBS, whole blood, F344, 1–27 months | Levine et al. 2020 (PMC7661040) |
| Human | Many WGBS/long-read blood sets; sorted-cell WGBS atlas (Loyfer 2023) already on the box | to be chosen for age span |

Not searched yet: the other 22 species. RRBS and targeted reads are short; whether they satisfy the single-molecule requirement (reads spanning enough CpGs) is open.

## 8. What is open

- Composition: needs either purified-cell arrays per species or single-molecule reads where cell type can be called per read.
- Batch: a within-chip or within-submission reading (age range inside one submission) for species where submissions span age, e.g. dog, mouse, cow, horse.
- End of life: only 5 species reach 0.9 of lifespan; the "each can only get so old" part needs older animals, especially of the short-lived species, whose oldest arrays stop at ~0.6–0.7.
- The human negative reading: re-read inside one submission (the 205600… chips, ages 34–92) before drawing anything from it.

## Files

- `per_animal_readings.csv` — one row per aged blood array of the 30 species (Met-A on both panels, role, fractional age, sex, chip, lineage-CpG means, M).
- `per_species_summary.csv` — one row per species (reference spread, oldest decile, ρ, confounds, M).
- `species_qualification.csv` — all 126 blood species with counts and reason.
- `groups.csv`, `groups_full.csv`, `immune_marker_probes.csv` — inputs to the box script.
- `box_outputs/` — box job outputs. `figures/` — small multiples (both panels) and overlay (both panels), PNG + PDF.
- `scripts/` — `species_ageing_readings.py` (stages `groups`, `readings`), `probe_level_confounds.py` (box).

Sources: GEO GSE223748 / GPL28271; GEO GSE110554 (Salas et al. 2018, Genome Biol 19:64); AnAge Build 15 (Tacutu et al. 2013).
