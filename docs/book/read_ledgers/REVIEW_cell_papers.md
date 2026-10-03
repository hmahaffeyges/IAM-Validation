# REVIEW — two early cellular papers (proposals only; nothing merged)

Repo checked: `github.com/hmahaffeyges/IAM-Validation` at HEAD `06aabc5` (sparse: `docs/book`, `CANON`).
Book files read in full before proposing: `part4/p4_01_bridge.tex` (181 lines), `part4/p4_02_landauer.tex` (271), `part3/p3_08_one_gauge.tex` (157), `part4/p4_10_temperature.tex` (73), `appendices/app_B2_errata_cells.tex` (27). Every other "ALREADY IN BOOK" call below rests on a grep plus the lines cited.

Every number below marked with a tag in brackets is recomputed by `docs/verification/scripts/verify_cell_papers.py` (output: `verify_cell_papers_output.txt`, 35 checks pass, plus INFO lines that give recomputed values beside printed ones).

## Bottom line

- **Vertebrate lifespan paper:** 742 lines read (1–742), plus `refs_vertebrate.bib` (247) and `make_figures.py` (435). Six items are worth keeping, all with changes: the opening question and the literature it cites, the species picture, the scope of where the gauge can apply, the bird/reptile temperature prediction, an in-vitro enzyme test, and the warning about HPLC values. Everything resting on the class floor, the 1.05/1.10 tiers, cross-species correlations, the α exponent, or the five substrates is out of date.
- **Cell thermodynamics paper:** 1923 lines read (1–1923), plus `fig_thermodynamic_validation.py` (493). The physics that is still valid (kT ln 2, the copy floor, ATP, binary entropy) is **already in the book**, in corrected form. One paragraph (its intellectual lineage) is worth adding. Everything else is built on class floors, tiers, cohort means or clinical claims and is out of date.
- **Species figure:** redrawn from our own script and data (`fig_p4_species_lifespan.pdf/.png`). It looks the same. The changes are listed in F1–F8. **One data gap:** none of the 34 species β values has a recorded per-species source, in the script, the paper or the repo. The paper names Lowe 2018, Lu 2023 and Wang 2020 as sources. Lowe 2018 covers six species and does not publish a mean blood β per species. The Wang 2020 entry in the bib points to an unrelated paper. Until the values are traced, the figure can be printed only as "first compilation, sources not recorded" (proposed caption in D2).

## Key to verdicts
KEEP AS WRITTEN · KEEP WITH A CHANGE (original and proposed text given below the table) · ALREADY IN BOOK (file:line) · OUT OF DATE (one-line reason)

---

## 1. Vertebrate lifespan paper (`iam_vertebrate_lifespan.tex`, 742 lines)

| row | lines | section / result | verdict | reason |
|---|---|---|---|---|
| V-01 | 37–64, 697–716 | Abstract, Conclusions: r = −0.90 lifespan law, A ≈ 0.98–1.16, α = 2, "law of jawed vertebrates" | OUT OF DATE | Rests on the immune class floor, on cross-species statistics and on a fitted α. The n and r values contradict each other within the paper (V-07). |
| V-02 | 68–71 | Why do mammals live so differently (mouse vs bowhead, 80 % of genes) | KEEP WITH A CHANGE → D1 | Waterston 2002 compares mouse with human, not with bowhead. |
| V-03 | 73–86 | Clocks track lifespan; clocks are statistical, not physical | KEEP WITH A CHANGE → D1 | Lowe 2018 is six species; the 42-species scaling is Crofts 2024. Haghani 2023 is 348 species and 15,456 profiles, not 167. Lu 2023 is checked and correct. [K5] |
| V-04 | 88–104 | IAM derives H_min per class; companion-paper results (MCMC, 27/28 TCGA) | OUT OF DATE | Class floors, cohort cancer results and a self-citation. |
| V-05 | 108–124 | A = H(β)/H_min(class), immune 0.838889 ± 0.000681, 17 chains | OUT OF DATE | No class floors. The errata in app_B2_errata_cells.tex:13 already records that the 17 chains and the ±0.000681 are wrong. H(β) itself is ALREADY IN BOOK (p4_03_surface.tex; p3_08_one_gauge.tex:66–69). |
| V-06 | 126–136 | Mammalian dataset: published mean blood β for 40 species | OUT OF DATE as a method; data used in the figure with caveats | No per-species source exists in the script or the repo; repo tree searched, no VAL-034/035 script. Lowe 2018 is six species with no per-species mean table. The `Wang2020` bib DOI resolves to a mammary-cell paper; the 104-Labrador study is Wang et al. 2020, *Cell Systems* 11:176, doi 10.1016/j.cels.2020.06.006. Our own VAL_043 script uses dog β = 0.764; this script uses 0.695. |
| V-07 | 168–203 | Lifespan vs A across mammals; order table; "taxonomically coherent" | KEEP WITH A CHANGE → D2 (picture only) | Kept as a picture of the first compilation, with no fit and no r/p. The paper's numbers are inconsistent: text 40 species, r = −0.9018; abstract 43 species; caption 34 species, r = −0.919. The script has 34 species in 12 orders, not 14 [K4a, K4b, V1]. The order table does not match the species list: Cetacea has 3 entries in the list and 5 in the table; Carnivora 6 vs 9 [V3]. |
| V-08 | 203–211, 239–252, 467–473 | A = 1.05 "cancer threshold" separates long- from short-lived "with complete accuracy" | OUT OF DATE (and false in our own data) | Tiers are retired. In the script's 34 species the 20-year split has 23 long-lived species, not 17, and three of them read ≥ 1.05 (naked mole rat, beaver, dog). The 35-year split has six short-lived species below 1.05 [V2]. Cohen's d is hard-coded as 1.99; from the paper's own means and SDs it is 8.55 (already in app_B2:11) [V2c]. |
| V-09 | 213–225, 423–443 | Bat anomaly; "bats are the right test case" | OUT OF DATE as an argument | The argument rests on the untraceable β values. The bat longevity fact (Wilkinson 2021) is kept in D2. |
| V-10 | 227–237, 682–685 | Naked mole rat: temperature lowers A, "approaches 1.13" | OUT OF DATE (mistake) | With α = 2 at 32 °C the correction *raises* A, from 1.123 to 1.160 [V4]; already in app_B2:12. |
| V-11 | 138–157, 254–285, 583–642 | Temperature correction H_min(T) ∝ (T/310.15)^α, α = 2 "derived", 41 % variance cut | OUT OF DATE | Replaced by the fixed-holding-energy floor, which has no exponent to choose (p4_10_temperature.tex:10–19). Several numbers are also wrong. α was fitted, not derived. On the script's own 29 species the variance minimum is at α = 3.95, not 2.0 [V8]. The variance cut at α = 2 is 46 %, not 41 % [V7]. In the "derivation", 40–80 kJ/mol is 15.5–31.0 kT₀, not 4.8–9.6. The paper's own formula then gives α ≈ 5.8, not 1.8 [V9]. |
| V-12 | 294–312 | Scope: applies where DNMT1 maintenance is the identity mechanism; not Drosophila, C. elegans, honey bee | KEEP WITH A CHANGE → D5 | The scope statement is valid and useful. "Framework" becomes "gauge". The unverified species count (≈ 50,000) and the timescales (450 vs 500 Myr, both used in the paper) are dropped. |
| V-13 | 314–330, 676–679 | Coral/anemone anomaly; "law of the jawed vertebrate lineage … conserved since" | KEEP WITH A CHANGE → D5 (open problem only) | The `Dixon2016` citation is a 2015 coral heat-tolerance paper with no methylation data, so it is dropped together with its numbers (18–25 %, > 100 yr). The "law … conserved 450 Myr" sentence goes: only human neutrophils are commissioned. |
| V-14 | 332–389 | Figures 1–3 and captions | Fig 1 → redrawn (F1–F8); Fig 2 → option B (left panel only); Fig 3 → OUT OF DATE | Fig 3 is tiers plus a t-test, and its "All 23/23 A < 1.05" annotation is false (three of the 23 read ≥ 1.05) [V2]. Fig 2's right panel is α (V-11). |
| V-15 | 286–290 | Birds: warmer, so less methylated than same-lifespan mammals | KEEP WITH A CHANGE → D3 | Restated with the book's Eq. eps0T: floor ×1.025 at 40 °C, ×1.041 at 42 °C; reptiles ×0.902–0.959 at 25–32 °C [K1]. The albatross/killer whale β comparison is dropped (untraceable values). |
| V-16 | 445–466, 638–642 | Prediction: DNMT1 fidelity measurable vs temperature in vitro, 15–42 °C | KEEP WITH A CHANGE → D3 | The test is valid and is not yet in the book: p3_08:57–58 asks for the selectivity, but not for a temperature series. The (T₂/T₁)² law is replaced by the fixed-energy relation S(T) = S₃₇^(310.15/T): 7-fold → 8.1 at 15 °C and 6.8 at 42 °C; 80-fold → 112 and 75 [K3]. |
| V-17 | 393–421 | r- vs K-selection: short-lived species run at A ≈ 1.12–1.16, cancer at "ceiling 1.10" | OUT OF DATE | Tiers and the class floor. |
| V-18 | 467–489 | Aging implications, interventions, "Lu 2023 A-score deviations correlate with mortality" | OUT OF DATE (misattribution) | Lu 2023 reports *age* deviations, not A-score deviations. Intervention claims and a self-cited senolytic result: out. |
| V-19 | 491–534 | Five substrates; fish nucleosome repeat 178 bp; cfDNA experiment | OUT OF DATE | Five substrates are retired (app_B2:19). The `Bhanu2018` DOI does not resolve, and `Drew1985` is about DNA bending, not repeat length. |
| V-20 | 536–581 | Methodological note: HPLC 5mC/C is not an array β; path: array on ectotherms | KEEP WITH A CHANGE → D4 | The definitional point is valid. The claim that ectotherm genomes hold more transposable elements is not checked against Canapa 2015 and is not needed, so it is dropped. The (ii) "convergent conclusions" paragraph defends α: out. |
| V-21 | 644–695 | Open problems: canine cfDNA, G-003b MCMC, cnidarians, NMR, prospective, Lu 2023 on "gaming PC" | OUT OF DATE except cnidarians (→ D5) | Retired substrates and pipelines. The prospective newborn-lifespan test is a cohort design. |
| V-22 | 159–164, 718–742 | Statistics; data availability; COI/patents; acknowledgments | OUT OF DATE | Self-citations and patents; no scripts with sources exist at the stated path. |

### Citation errors found (vertebrate bib)
`Wang2020`: wrong paper; correct is *Cell Systems* 11:176 (doi 10.1016/j.cels.2020.06.006). `Dixon2016`: actually 2015, heat tolerance, no methylation. `Bashtrykov2014`: the DOI resolves to an obituary in *Leukemia*; it gives no DNMT1 activation energy. `Klimasauskas1994`: HhaI base flipping, not DNMT1 activation energy. `Bhanu2018`: DOI does not resolve. `Lyko2018`: the DOI in the file (nrg.2017.81) resolves to a GRID-seq highlight; the correct DOI is 10.1038/nrg.2017.80. `Haghani2023` count wrong (V-03); `Lowe2018` count wrong (V-03).

### Changed wording, side by side (vertebrate)

**V-02** (lines 68–71)
- Original: "The house mouse and the bowhead whale share approximately 80\% of their protein-coding genes~\citep{Waterston2002}. Their cellular machinery is nearly identical at the molecular level. Yet one lives four years; the other, two centuries."
- Proposed: "The house mouse and the human share most of their genes: about 80~\% of mouse genes have a single identifiable orthologue in the human genome~\cite{Waterston2002}. Their cellular machinery is nearly identical at the molecular level. Yet a house mouse lives four years and a bowhead whale two centuries~\cite{Tacutu2018}."
- Why: the cited source compares mouse with human. The lifespans (4 yr, 211 yr) are the script values [K6]; AnAge is cited for them.

**V-03** (lines 73–80)
- Original: "\citet{Lowe2018} showed that the \emph{rate} of methylation change at conserved age-related CpG sites scales with maximum lifespan across 42 mammalian species. \citet{Lu2023} demonstrated that universal pan-mammalian clocks trained on 11,754 arrays from 185 species achieve $r > 0.96$ accuracy … \citet{Haghani2023} characterized the co-methylation networks underlying mammalian traits across 167 eutherian species."
- Proposed: "Lowe and co-workers showed that the \emph{rate} of methylation change at age-associated CpG sites falls with maximum lifespan across six mammalian species~\cite{Lowe2018}, and Crofts and co-workers found the same scaling at conserved age-related sites across 42 species~\cite{Crofts2024}. Universal pan-mammalian clocks built on 11,754 arrays from 185 species estimate age with $r>0.96$~\cite{Lu2023}, and the co-methylation networks underlying mammalian traits have been mapped across 15,456 profiles from 348 species~\cite{Haghani2023}."
- Why: the counts are taken from each paper's abstract (Lowe: six mammals; Crofts: 42; Haghani: 15,456 profiles, 348 species). Lines 82–86 ("What has been missing … why it happens") are kept as written, plus one new linking sentence, marked `% NEW` in D1.

**V-07** (lines 168–171, 198–203)
- Original: "Figure~\ref{fig:lifespan} shows the relationship between maximum lifespan and $\Ascore$ across 40 mammalian species spanning 14 taxonomic orders. The correlation is strong and highly significant: Pearson $r = -0.9018$ …" and "Long-lived K-selected species --- whales, elephants, great apes, horses --- cluster near $\Ascore = 1.00$, at the thermodynamic floor. Short-lived r-selected species --- rodents, shrews, rabbits --- cluster at $\Ascore \approx 1.10$--$1.16$, significantly above it."
- Proposed: "Figure~\ref{fig:p4_species} shows the first compilation we made of mean blood methylation for 34 mammalian species in twelve orders, against maximum lifespan. The pattern is taxonomically coherent. Long-lived species --- whales, elephants, great apes, horses --- sit at the lowest entropy; short-lived species --- rodents, shrews, rabbits --- sit highest." Three caveats follow (D2).
- Why: the count is what the script holds [K4a, K4b]. The r/p is a population statistic and does not match the script [V1]. The "floor" and "above it" wording is the retired class floor and tiers. **Meaning change, listed:** the paper said lifespan *predicts* A. The draft says only that the compiled values sort by order and lifespan, and that this is not a reading or a test.

**V-12 / V-13** (lines 296–330)
- Original: "The IAM $\Hmin$ framework applies wherever DNMT1-mediated CpG maintenance methylation is the primary mechanism of epigenomic identity preservation. … All jawed vertebrates … approximately 50,000 species spanning 450 million years … The precise scope definition is itself a result: the thermodynamic entropy floor is a law of the \emph{jawed vertebrate} lineage … conserved ever since."
- Proposed: D5 ("The gauge applies wherever … because what it reads is the error of that copy … Corals and sea anemones are the open case … no floor is proposed for them. \openprob{} The scope is itself a statement about the instrument: a floor read from a copy error exists only where a copy is made. \interp").
- Why: there is no class floor (the "IAM H_min framework" is now the gauge). The species count and the timescale are unverified, and the paper itself gives two different ones. **Meaning change, listed:** "a law of the jawed vertebrate lineage" becomes a statement of where the instrument can apply. Only human neutrophils are commissioned, so no law across a clade is claimed.

**V-15 / V-16** (lines 286–290, 445–466)
- Original: "The ratio of error rates should scale approximately as $(T_2/T_1)^2$, consistent with the empirically derived $\alpha = 2.0$." and "Bird DNMT1, operating at 40--42$^\circ$C, should show higher per-site error rates than mammalian DNMT1."
- Proposed: D3: floor ×1.025 at 40 °C and ×1.041 at 42 °C; reptiles ×0.902–0.959 at 25–32 °C [K1]; enzyme preference S₃₇ → S₃₇^(310.15/T) under a fixed discrimination energy; the alternative is error fixed in kT units [K3]. Status \prediction.
- Why: α is retired, and p4_10:10–19 replaces it with Eq. eps0T. The direction of the paper's prediction (warmer means more error) survives; its size does not.

**V-20** (lines 539–549, 568–576)
- Original: "This is a different quantity from the Mammal40k array mean beta used in the mammalian dataset … Ectotherm genomes contain a higher fraction of TEs than mammalian genomes~\citep{Canapa2015} …"
- Proposed: D4: "It is a different quantity from a per-site $\beta$ on the sites that make a cell what it is, and from the mean of per-site entropies that the gauge reads, so no such value is placed on the gauge."
- Why: the gauge's statistic is the mean of per-site entropies on identity sites (p4_03:46–62; Jensen). The transposable-element sentence was not verified and is not needed.

---

## 2. Cell thermodynamics paper (source `.tex`, 1923 lines)

| row | lines | section / result | verdict | reason |
|---|---|---|---|---|
| T-01 | 91–148 | Title, abstract (8 classes, MCMC, 27/28 TCGA, AD/T2D tiers) | OUT OF DATE | Class floors, tiers, cohorts, clinical claims. |
| T-02 | 151–160 | Data availability; "access to the analytical engine" on request | OUT OF DATE | The engine is not proprietary in the book. |
| T-03 | 172–180 | Methylation is the most stable heritable information layer; "no first-principles framework has derived …" | first half ALREADY IN BOOK (p4_02_landauer.tex:3–5); second half OUT OF DATE | Sanchez & Mackenzie 2016 applied Landauer to methylation first (p4_02:202). |
| T-04 | 182–194 | Epigenetic clocks are statistical, not physical | ALREADY IN BOOK (p3_09_reach.tex:60) | The vertebrate paper's version is carried in D1. |
| T-05 | 196–200 | The floor is physical: set by temperature, genome size, irreversibility | ALREADY IN BOOK in corrected form (p4_01_bridge.tex:102–105; p3_08:54–63) | The floor's height is now measured from the holding energy, not set by genome size. |
| T-06 | 202–225 | Commitment spectrum; four "cosmology-style" validations | OUT OF DATE | Class floors; retired parameter. |
| T-07 | 236–253 | Landauer bound; kT ln 2 = 2.97 × 10⁻²¹ J | ALREADY IN BOOK (p4_01:41–46; p4_02:9–16, 2.968 × 10⁻²¹ J) | [T1] |
| T-08 | 241–247 | A hemimethylated CpG is a binary decision; each correct copy restores one bit | ALREADY IN BOOK (p4_02:133–136) | |
| T-09 | 255–264 | N_CpG = 19.6 × 10⁶, E_floor = 5.82 × 10⁻¹⁴ J | OUT OF DATE | The book uses hg19 N = 28,217,448 → 8.37 × 10⁻¹⁴ J (p4_02:136–143) [T2, T3]. |
| T-10 | 266–268 | ≈ 10⁶ ATP per copy; ΔG_ATP 54 kJ/mol → 9 × 10⁻²⁰ J | ALREADY IN BOOK (p4_02:143, 9.3 × 10⁵) | The paper's own floor gives 6.5 × 10⁵, not 10⁶ [T4, T5]. |
| T-11 | 268–274 | DNMT1 error ≈ 10⁻⁶ per CpG per division | OUT OF DATE (mistake) | Genereux 2005, the paper's own citation, gives a maintenance efficiency of 0.90–0.98 (p4_02:172–177). |
| T-12 | 279–290 | Binary entropy H(β): maximum at 0.5, symmetric | ALREADY IN BOOK (p4_03_surface.tex; p3_08:66–69; app_F glossary) | |
| T-13 | 290–293, 1192–1194 | H of the mean genome-wide β as the cell's statistic | OUT OF DATE | The book uses the mean of per-site entropies; the entropy of a mean over both channels is meaningless (p4_03:46–62). |
| T-14 | 295–323 | A = H(β)/H_min(class); four tiers 1.00/1.05/1.07/1.10 | OUT OF DATE | No class floors, no tiers. |
| T-15 | 326–356 | Global floor H(0.782) = 0.7565, neurons; commitment ordering | OUT OF DATE | Retired. The arithmetic is right [T8], but app_B2:22 records that the reference β values are not genome-wide means of the cited epigenomes. |
| T-16 | 358–376 | ATP drive over RT = 20.94; per-class values | the 20.94 is ALREADY IN BOOK as M (p4_02:28–29) [T6]; per-class parameter OUT OF DATE | The per-class parameter is marked superseded in app_B2:18. |
| T-17 | 378–415 | Three-component decomposition C1/C2/C3; cancer amplifier | OUT OF DATE | app_B2:20: not part of the instrument. |
| T-18 | 417–439 | Biological activation E(a) = exp(1 − 1/a); pace peaks at t_max/2 | OUT OF DATE | Fitted to cohort means of a trained clock (a population statistic). The algebra is right: the peak is at a = ½ [T13]. |
| T-19 | 445–482 | Reference database and the 8-class MCMC | OUT OF DATE | Class floors; reference β values not genome-wide means (app_B2:22). |
| T-20 | 484–498, 882–955 | TCGA 27/28 direction test; table of 28 cancers | OUT OF DATE | Cohort means, class floors, clinical framing. The TGCT row is 0.430/0.250 in the table but 0.745/0.720 in the figure script. |
| T-21 | 500–517, 993–1008 | Metabolic ordering test, ρ = 0.905 | OUT OF DATE | Retired parameter. |
| T-22 | 519–528, 1010–1026 | Clock fit, t_max = 120.3 ± 7.1 yr | OUT OF DATE | Population statistic. The text says 8 points; the figure script plots 6 hand-entered values. |
| T-23 | 542–581 (figure + script) | Four-panel validation figure | OUT OF DATE | Every panel shows retired items. This paper has no species figure. |
| T-24 | 583–722 | Alzheimer's, type 2 diabetes, DCIS on tiers | OUT OF DATE (and arithmetic wrong) | Clinical framing, cohort means, tiers. Printed entropies are wrong: H(0.775) = 0.7692, not 0.8058; H(0.764) = 0.7883, not 0.8215; H(0.660) = 0.9248, not 0.929 [T9, T12]. |
| T-25 | 724–777 | Posterior table; immune "6.44σ correction" | OUT OF DATE (internally inconsistent) | The neutrophil reference has H(0.760) = 0.7950, below the immune "floor" 0.8389 it is said to define [T11]. Roadmap IDs disagree (E030 vs E034 for neutrophils). |
| T-26 | 779–880 | Class specification sheet; terminal and pluripotent narratives | OUT OF DATE | Class floors, retired parameter, clinical applications. |
| T-27 | 957–991 | Three "structural confirmations"; GBM non-linearity "validates Shannon" | OUT OF DATE (mistake) | H is symmetric: H(0.40) = H(0.60) = 0.971 bits, so a tumour at 0.40 does not carry "more entropy than one at 0.60" [T7]. |
| T-28 | 1032–1060 | What the framework adds; physical vs population reference | the reference point is ALREADY IN BOOK (p4_01:111, 165–170); decomposition OUT | |
| T-29 | 1062–1145 | Applications: detection tiers, cfDNA tiers, DCIS triage, drugs, iPSC | OUT OF DATE | Clinical claims, tiers, class floors. |
| T-30 | 1147–1173 | Limitations | OUT OF DATE | All of it concerns retired items. |
| T-31 | 1175–1196 | Lineage: Schrödinger, Adami, England, Friston; "requires only" Landauer | KEEP WITH A CHANGE → D6 | Valid and not in the book (only Schrödinger, p4_01:32, and England, p5_05c:133, appear). The "first" claim is dropped (T-03); the second assumption (entropy of the mean β) is dropped (T-13). |
| T-32 | 1199–1286 | Patents; conclusion; acknowledgments; contributions; competing interests | OUT OF DATE | Patents and the engine-access clause. |
| T-33 | 1288–1769 | Bibliography | see errors below | |
| T-34 | 1788–1870 | Table S1, 37 reference cells | OUT OF DATE (arithmetic wrong) | 16 of 37 printed H(β) values are off by more than 0.001, e.g. 0.780 → 0.7602, not 0.7951 [T10] (first example already in app_B2:10). |
| T-35 | 1872–1922 | Chain specifications; ±5 % sensitivity | OUT OF DATE | Retired items. |

### Citation errors found (cell thermodynamics bib)
`Genereux2005` fourth author printed as "Bhaudeau, C."; it should be Laird, C. D. (the book's iam.bib entry, line 1310, is already correct). `Hata2020` lists "Bhatt, D.L." twice; not traceable. `TCGA_UVM2017` carries the melanoma title; the real title is *Integrative analysis identifies four molecular and clinical subsets in uveal melanoma*. `TCGA_THYM2018` carries the title of a pan-cancer DNA-repair paper. `Edelman2018` key on a 2012 paper. `Movassagh2011` cited as *NEJM* in Table S1 but *Circulation* in the bib. None of these is needed by the KEEP items.

### Changed wording, side by side (cell thermodynamics)

**T-31** (lines 1177–1196)
- Original: "The present work sits in a lineage that includes Schr\"{o}dinger's ``What is Life?''~\cite{Schrodinger1944} (…), Adami's work on information content of genetic sequences~\cite{Adami2002}, England's derivation of thermodynamic constraints on self-replicating systems~\cite{England2013}, and Friston's free energy principle~\cite{Friston2010}. The specific contribution here is the application of Landauer's principle to a concrete, measurable, genome-wide information maintenance process (DNMT1-mediated methylation copying), yielding quantitative class-specific floors … It requires only the following: (1) that Landauer's principle applies to DNA methylation maintenance …, and (2) that Shannon entropy computed from mean genome-wide methylation $\beta$ is a meaningful summary statistic of epigenomic disorder."
- Proposed: D6. The first sentence is kept as written (citation key mapped to the book's `SchrodingerWhatIsLife`), followed by "The framework does not require agreement with any of these predecessors … It requires only that Landauer's principle applies to DNA methylation maintenance, a point already made for methylation by Sanchez and Mackenzie (Chapter~\ref{ch:landauer}). \interp"
- Why: "the specific contribution … class-specific floors" claims a first that p4_02:202 attributes to Sanchez and Mackenzie, and it names retired class floors. Assumption (2) contradicts p4_03:46–62. **Meaning change, listed:** the framework no longer claims to be the first to apply Landauer to methylation.

---

## 3. Species figure (`fig_p4_species_lifespan.pdf/.png`; script `docs/book/figscripts/fig_p4_species_lifespan.py`)

Source: Fig. 1 of the vertebrate paper (`make_figures.py`, identical in the repo apart from the private-calibration import). The data lists `MAMMALS`, `VERTEBRATES` and `ORDER_DATA` and the colours are copied unchanged. Same figure size, fonts, colours, markers, legend, log axis and frame.

| fix | what changed | why |
|---|---|---|
| F1 | Y axis is now H(β̄) in bits, not A = H(β̄)/0.838889 | Class floors are retired. Every point sits where it was: the axis is rescaled and the frame is the old one × 0.838889. |
| F2 | Removed the 1.00 / 1.05 / 1.10 lines and their labels | Tiers are retired; 1.00 is not a floor on the current gauge. The "1.00 thermodynamic floor" text also covered the chimpanzee label. |
| F3 | Removed the long/short-lived shaded zones and their texts | Tier zones. The script's own data contradict the split (V-08). The "short-lived" text sat on the shrew point; the "long-lived" text was hidden behind the legend. |
| F4 | Removed the regression line and the r/p from the title (`--with-fit` redraws the original line for comparison) | No population statistics, and the values are untraceable. |
| F5 | Title: "Methylation entropy and maximum lifespan across mammals / 34 species, mean blood β as first compiled" | The old title "Lifespan predicts …" claims a result the gauge does not make. |
| F6 | Labels placed by screen offset | Human, bowhead and killer whale labels sat on their markers, because a 4-year offset is invisible on a log axis above 100 years. |
| F7 | "Insectivora" shown as "Eulipotyphla (shrews)" | Insectivora is no longer a recognised order; the data key is unchanged. |
| F8 | Output to `figures/part4/fig_p4_species_lifespan.{pdf,png}` (PNG 200 dpi) | Book layout; the old path was `/home/claude/`. |

**Option B** (`fig_p4_species_temperature.pdf/.png`): only if "the figure with all the species" meant the all-classes Fig. 2. It is the left panel only (29 vertebrates, H(β̄) against body temperature), with F1, F2 and F4 applied. The two annotations that fell outside the axes or named tiers are removed. The α-corrected right panel is not redrawn (V-11). **Not recommended for the book:** its ectotherm values are converted HPLC measurements, a different quantity (D4).

**Data missing:**
1. A per-species source for every β: 34 mammals and 29 vertebrates.
2. The 6 species behind "40" and the 9 behind "43", which are not in the script.
3. A per-species check of the lifespans against an AnAge release (not done here).

The dog value disagrees between our scripts (0.695 here, 0.764 in VAL_043). The shrew lifespan is 2.5 yr in one list and 2 yr in the other, which is where "105-fold" comes from; with 2.5 yr the ratio is 84 [V5].

---

## 4. What we propose (DRAFT_cell_papers_keep.tex — DRAFT FOR AUTHOR REVIEW)

| block | goes to | content | status labels |
|---|---|---|---|
| D1 | p4_10_temperature.tex, new `\section{Other species}` before line 68 | Opening question and clock literature (V-02, V-03) | \observed, \interp |
| D2 | same, after D1 | Species figure plus three caveats (V-07) | \observed (picture), \openprob (sources) |
| D3 | p4_10, after line 27 | Birds, reptiles, enzyme temperature test (V-15, V-16) | \calc, \prediction |
| D4 | p4_10, after line 47 | HPLC values are a different quantity (V-20) | \observed, \derived |
| D5 | p4_10, end of file | Where the gauge can apply; cnidarians open (V-12, V-13) | \observed, \openprob, \interp |
| D6 | p4_01_bridge.tex, after line 32 | Lineage paragraph (T-31) | \interp |

New citation keys are in `bib_cell_papers_DRAFT.bib` (13 entries, every DOI resolved on CrossRef). Static checks: braces balance; every `\ref`/`\eqref` target exists in the book; every `\cite` key is in iam.bib or the draft bib. No LaTeX compiler was available here, so the blocks are not compile-tested.

## 5. Not done
- The species β values were not traced to primary data. That needs the per-species sources, or recomputation from the public mammalian-array data (Lu 2023 / Haghani 2023 deposits).
- The lifespans were not checked against AnAge.
- The book was not compiled with the blocks inserted.
