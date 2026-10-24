# DEV-ALL-INFO-01: the published information-theory study of paediatric leukaemia, replicated through the chain (development)

**The instrument's job:** the chain reads each leukaemia and each control by itself, against its own reference and the physics floor, and
the readings are set beside what the study found by comparing groups.

**The study:** an information-theory analysis of whole-genome bisulfite sequencing in paediatric pre-B acute lymphoblastic leukaemia
(Nature Biomedical Engineering 2021). Each 3-kb region was modelled with a one-dimensional Ising model fitted to the reads (informME), giving
a mean methylation level, a normalized methylation entropy (0 ordered to 1 random) and a methylation sensitivity index, and leukaemias were
compared with normal controls (sorted pre-B2 precursors from an earlier study, and cord-blood CD19+ B cells) by the Jensen-Shannon distance.

**What it found, written here before the chain reads any of its samples:**
1. Leukaemia epigenomes are more stochastic than the controls: higher normalized methylation entropy at specific regulatory regions and
   genes.
2. Gains in mean methylation, gains in entropy and losses of methylation sensitivity inside the bivalent domains of normal B cells, across
   four cytogenetic subtypes.
3. Higher promoter entropy goes with lower gene expression and with higher expression variability.
4. The genes with discordant methylation converge on genes involved in the leukaemia's translocations, with UHRF1 as a central node.
5. Knocking out UHRF1 in the Reh line lost methylation and raised entropy genome-wide; in gene-poor heterochromatic B domains, where
   methylation went almost to zero, the entropy fell instead.

**Data spent (author: "Spend the whole series"):** GEO GSE116229, all 36 WGBS samples with reads (31 leukaemias, the Reh control and UHRF1 knockout,
3 cord-blood CD19+ samples; `data/DEV_ALL_INFO_01/all_samples.csv`). The study's pre-B2 controls come from an earlier study and are to be
found. Its RNA sequencing is not read.

**Pass 1:** IAM-A's count on every sample at the eraser recount's fixed 10 million read pairs (`erase_gh.sh`, unchanged), each through
`run_sample.py`. Compared with findings 1 and 5: do the leukaemias read beyond the cord-blood controls on the held error and loose share,
and does the knockout. **Pass 2:** full depth through a split alignment, then the chain's regional readings (Met-A on sequencing, the
territories, the globe, the distance to the ceiling) set region by region beside findings 2 to 5.
