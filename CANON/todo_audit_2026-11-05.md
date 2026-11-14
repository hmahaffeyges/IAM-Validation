# To-do audit: the author's two files, line by line

Every non-blank line of the author's two files checked against CANON/plan.json and the committed records. Lines that are plan items show the item's
current status. Lines with no plan entry were each confirmed against the plan text and the records before anything was added; added items and
corrections are listed first. A line marked 'status report' was read and holds no action of its own (a run report, a result already recorded, an
explanation).

Counts: {'report': 591, 'in plan': 301, 'not pursued': 9, 'other': 81, 'added': 24, 'private': 47}

## Added or changed in the plan

- **New section, DOWNTIME (laptop work only):** 61.1 workflow document; 61.2 data search; 61.3 sequencing atlas planned against built; 61.4 book and website for the atlases; 61.5 prior-art search; 61.6 one-button intake; 61.7 the fourth C-Score; 61.8 the Decoder's open findings.
- **Machine items added to step 1:** 60.17 planted methylome through the atlas build (M05); 60.19 held-error check on every cell column; 60.20 the floor as a bound on v2.1 (a check, not a refit); 60.21 BLUEPRINT columns for cells the sequencing atlas lacks.
- **Not pursued, with the reason:** 60.18 MCMC across laboratories (one laboratory per cell; reopens with a second laboratory); 60.6 GSE55763 (no raw IDATs; reopens with a 450K set that has them).
- **Status corrected from the checklist:** 35.30.2 and 35.30.3 built; 35.30.1 and 35.30.4 next steps rewritten (A1 partly built; A5 not measurable inside the atlas; D12 not possible: no oxidative data).
- **Found while checking:** a run with no --specimen is read as whole blood (the command's default), not refused; the workflow draft and 61.6 say so.

## Every line

| File | Line | Verdict | Text |
|---|---|---|---|
| 1 | 1 | status report, result or explanation; no action of its own | 11-5-26 updated to do list: |
| 1 | 3 | in plan (a step of the to-do list) | Plan |
| 1 | 4 | status report, result or explanation; no action of its own | 1. Score B and T at full depth once all 12 samples are merged. |
| 1 | 5 | status report, result or explanation; no action of its own | 2. Score the Johns Hopkins test (bar: tared median Normal). If both bars are met, the same-lab tare is commissioned for own-cell Met-A on sequencing. |
| 1 | 6 | status report, result or explanation; no action of its own | 3. Run the disease side on the Johns Hopkins tumor and normal pairs, written down before reading. |
| 1 | 7 | status report, result or explanation; no action of its own | 4. Fix the two pipeline faults (no silent chunk successes, retry the index download), after these readings finish so they all run on one chain version. |
| 1 | 8 | status report, result or explanation; no action of its own | 5. Draft the workflow document's settled boxes. |
| 1 | 9 | status report, result or explanation; no action of its own | 6. Then plasma, and a chosen test for each instrument. |
| 1 | 14 | status report, result or explanation; no action of its own | Updated to do list 11-4-26 |
| 1 | 16 | status report, result or explanation; no action of its own | The step-by-step list. It's now step 1 of the commissioning plan, written in the plan database. A step closes only with its date and its committed record, so th |
| 1 | 17 | status report, result or explanation; no action of its own | Standing, every hour: GitHub's 20 and Google's 8 machines kept full. Watchers on every run, plus a direct machine check every 30 minutes. The A-score database u |
| 1 | 18 | status report, result or explanation; no action of its own | A. Commission on the data we set aside |
| 1 | 19 | status report, result or explanation; no action of its own | 1. Second outside lab's healthy CD8 T cells (GSE245832, 12 sequenced libraries). Running on 3 Toronto machines. Tells us whether BLUEPRINT's low reading belongs |
| 1 | 20 | status report, result or explanation; no action of its own | 2. B and T sequencing commissioning test (GSE134664, GSE98837: 12 healthy B and CD8 T cells, written 10-29). Only 3 have landed so far. Finish the other 9, scor |
| 1 | 21 | status report, result or explanation; no action of its own | 3. Fresh outside lab, the real sequencing test: GSE144691 (13 healthy CD8 subsets, unread). Written after step 1 lands. |
| 1 | 22 | GSE55763 has no raw IDATs: 60.6 marked not pursued as written, with the reason | 4. B and T on arrays. First explain why the CD4 central-memory and memory-B columns read low. Then read the held confirmation (35 arrays from 3 labs) and commis |
| 1 | 23 | status report, result or explanation; no action of its own | 5. Monocytes on arrays. Explain why bar 2 missed, then find and read another independent lab. |
| 1 | 24 | status report, result or explanation; no action of its own | 6. 450K whole blood: GSE55763 (2,711 bloods, unspent, with technical replicates) as the healthy commissioning test for Met-A, the C-Score and Group-A on 450K. |
| 1 | 25 | status report, result or explanation; no action of its own | 7. Reserved BLUEPRINT neutrophils, read as their test's confirmation once the outside-lab floor is understood. |
| 1 | 26 | status report, result or explanation; no action of its own | 8. One table of every cell (79 + 57) against every instrument: what's commissioned, what data could commission it, and which test. Worked cell by cell. |
| 1 | 27 | in plan (a step of the to-do list) | B. Plasma, healthy first |
| 1 | 28 | status report, result or explanation; no action of its own | 1. Plasma IAM-A reads low: test the next physical cause, synthetic first. |
| 1 | 29 | status report, result or explanation; no action of its own | 2. The domains follow the erythroid share: explain it. |
| 1 | 30 | status report, result or explanation; no action of its own | 3. Once healthy plasma reads Normal: commission on the held healthy plasmas (248 + 23), then the matched blood halves (GSE186888). |
| 1 | 31 | status report, result or explanation; no action of its own | 4. Plasma cfDNA arrays (lung, liver, prostate over time) read and flagged, with A scores captured. |
| 1 | 32 | in plan (a step of the to-do list) | C. The atlases, globes and spheres |
| 1 | 33 | status report, result or explanation; no action of its own | 1. The sequencing atlas's commissioning bars (like v2.1's three sealed bars), read on outside healthy samples. |
| 1 | 34 | status report, result or explanation; no action of its own | 2. Its partly built checklist rows finished. |
| 1 | 35 | status report, result or explanation; no action of its own | 3. The globe's neuron change count explained. |
| 1 | 36 | status report, result or explanation; no action of its own | 4. The information sphere and the distance to the ceiling, each given a job, a synthetic and bars. |
| 1 | 37 | status report, result or explanation; no action of its own | D. A chosen test for each score: one per instrument, on unspent data, written before reading, showing what that score sees that nothing else does. |
| 1 | 38 | status report, result or explanation; no action of its own | E. Every old test replayed: the 150+ VALs and the 119-test retest plan through the full chain, on all cells, compared with the earlier results. R5 liver and R9  |
| 1 | 39 | status report, result or explanation; no action of its own | F. The A-score lines: from the whole database, where healthy ends and where each cancer type and stage begins, per instrument. Internal only. |
| 1 | 40 | status report, result or explanation; no action of its own | G. Last: the paper, the package and contact with the researchers. |
| 1 | 41 | status report, result or explanation; no action of its own | Running right now: |
| 1 | 42 | status report, result or explanation; no action of its own | * GitHub: R5 and R9 on all cells (20 machines). |
| 1 | 43 | status report, result or explanation; no action of its own | * Google: step A1 (3 machines). |
| 1 | 44 | status report, result or explanation; no action of its own | * One watcher covers both platforms. |
| 1 | 45 | status report, result or explanation; no action of its own | Next: step A2, finding which B and T samples have landed and re-sending the rest. |
| 1 | 50 | status report, result or explanation; no action of its own | What the R replays are. Each R number is an old public study that IAM's earlier engine tested, now re-read by today's full chain. They are reference only. Nothi |
| 1 | 51 | status report, result or explanation; no action of its own | #	Study	Comparison |
| 1 | 52 | status report, result or explanation; no action of its own | R5	GSE281691, blood	liver cancer cases vs controls |
| 1 | 53 | status report, result or explanation; no action of its own | R6	GSE141682, healthy blood	aging (plus a stroke set) |
| 1 | 54 | status report, result or explanation; no action of its own | R9	GSE180683, blood of brain-tumor patients	by tumor grade (below) |
| 1 | 55 | status report, result or explanation; no action of its own | R10	GSE99511, cervical self-samples	precancer vs normal |
| 1 | 56 | status report, result or explanation; no action of its own | R11	GSE122126, plasma on arrays	sepsis, cancer vs normal |
| 1 | 57 | status report, result or explanation; no action of its own | R12	GSE51057, Italian blood drawn years before diagnosis	later breast cancer vs none |
| 1 | 58 | status report, result or explanation; no action of its own | R13	GSE287994, cervical cytology	precancer/cancer vs benign (sent tonight) |
| 1 | 59 | status report, result or explanation; no action of its own | R14	GSE256092, lung tumors	stage 3–4 vs stage 1 (sent tonight) |
| 1 | 61 | status report, result or explanation; no action of its own | IAM to-do list, October 25, 2026, 06:30 Pacific |
| 1 | 63 | status report, result or explanation; no action of its own | Agreed. It fits the law, too: IAM-A is already commissioned on CD8 T cells, because lymphocytes hold their marks the most firmly, they live for years, and every |
| 1 | 64 | status report, result or explanation; no action of its own | The order I'll follow: |
| 1 | 65 | status report, result or explanation; no action of its own | 1. B cells first, because their study needs them. Sequencing Met-A against the atlas's B cells, tested on healthy B-cell sequencing the atlas doesn't hold. |
| 1 | 66 | status report, result or explanation; no action of its own | 2. T cells next: CD4 and CD8. The atlas has several T-cell subsets, so each one gets its own test. |
| 1 | 67 | status report, result or explanation; no action of its own | 3. Both on arrays too: the atlas-curve version, using unread sorted B- and T-cell EPIC series, once monocytes confirm the method on the other two labs. |
| 1 | 68 | status report, result or explanation; no action of its own | Each test is written before anything is read. B and T cells become the first cells commissioned on the atlas as the ruler, ahead of monocytes, which just proved |
| 1 | 69 | status report, result or explanation; no action of its own | I'll put this order into the plan database and memory, then start the search for healthy B-cell sequencing. |
| 1 | 71 | status report, result or explanation; no action of its own | To-do list, 2026-10-25 |
| 1 | 72 | status report, result or explanation; no action of its own | Generated from the plan database (CANON/plan.json); every item is a plan task or branch, nothing typed by hand. Order set by the author: the Johns Hopkins study |
| 1 | 73 | status report, result or explanation; no action of its own | 1. The Johns Hopkins study (their dataset through the chain, and the paper) |
| 1 | 74 | in plan (a step of the to-do list) | Replicate the published information-theory study of pediatric leukemia through the chain, for its authors Done when: every finding of the study set beside the c |
| 1 | 75 | in plan: 49.2, now [built] | * [built] 49.2 The split alignment: each sample's reads spread over machines, one index; Next: built and fixed (the molecule step on the prebuilt index, the mer |
| 1 | 76 | in plan: 49.3, now [running] | * [running] 49.3 Pass 2 at full depth: Met-A on sequencing, the territories, the globe and the distance to the ceiling, region by region beside the study's five |
| 1 | 77 | in plan: 49.5, now [running] | * [open] 49.5 What we learned from the study, each added only after its own synthetic test through the one command, with a Decoder entry: Next: synthetic first |
| 1 | 78 | in plan: 49.6, now [open] | * [open] 49.6 A research package for this group, in this order : Next: after 49.3 |
| 1 | 79 | in plan: 49.7, now [open] | * [open] 49.7 The package for the study's authors, built inside this repository like all other work : Next: build the package folder and its generator here; |
| 1 | 80 | in plan: 49.8, now [open] | * [open] 49.8 Order : (1) finish the chain and the derived baselines (the send gate); Next: order : the leukemia study first, then the firm rate (14.1), then Me |
| 1 | 81 | status report, result or explanation; no action of its own | The paper : IAM's Law and the chain introduced in the format of the published information-theory study of pediatric leukemia, built on it, tested on its data Do |
| 1 | 82 | in plan: 50.1, now [built] | * [built] 50.1 Outline and figure plan in the journal's format (abstract, introduction, Results under finding headings, Discussion, Methods with equations, Data |
| 1 | 83 | in plan: 50.2, now [built] | * [built] 50.2 Methods drafted from the Decoder and the records: Next: resolve every [check] against its record; |
| 1 | 84 | in plan: 50.3, now [open] | * [open] 50.3 Results filled from the canon only as each gated test lands (plan 49 gate); Next: after 49 |
| 1 | 85 | in plan: 50.4, now [open] | * [open] 50.4 An independent reproducibility audit by a second author from a clean clone, following only the written instructions Next: when the package is expo |
| 1 | 86 | in plan: 50.5, now [open] | * [open] 50.5 bioRxiv preprint; then the study's authors asked to read it critically and to collaborate Next: after 50.3 |
| 1 | 87 | status report, result or explanation; no action of its own | 2. The physics derivation (the firm rate and the derived baselines) |
| 1 | 88 | in plan (a step of the to-do list) | IAM-A's firm rate, derived again on the commissioned counter: Done when: the recount scored once and recorded; a sealed prediction written or the derivation rec |
| 1 | 89 | in plan: 14.1, now [running] | * [running] 14.1 THE TOP PHYSICS ITEM : with the firm rate derived, the healthy disorder at each site follows from IAM's Law (M = 20.94, E_hold = phi M), so hea |
| 1 | 90 | in plan: 14.2, now [open] | * [open] 14.2 If one human sample needs more than GitHub's 6 hours: Next: timing test: 10 million pairs align in about 4 hours on one machine (index half of it) |
| 1 | 91 | in plan: 14.5, now [running] | * [open] 14.5 GSM3618718 in the firm-rate recount stopped at GitHub's 6-hour limit while downloading and aligning (run 39994316396, nothing saved): Next: re-run |
| 1 | 92 | in plan: 14.8, now [open] | * [open] 14.8 Layer 2: the held error region by region, from each region's own loose share (shares the work of IAM-A's C-Score version 2, plan 41.1) Next: after |
| 1 | 93 | in plan: 14.9, now [open] | * [open] 14.9 Stage Q2's 2,000-CpG tiles (stage_q2_selfref.hg19_tiles, built from HG19_CPGS) do not line up with the genome dictionary's chromosome index ranges |
| 1 | 94 | in plan: 14.11, now [open] | * [open] 14.11 Cross-check against informME on the same files: Next: after 14.10, on the same spent files |
| 1 | 95 | in plan: 14.12, now [running] | * [open] 14.12 The C-Scores: their value for scattered departures (1) is derived by construction; Next: written after 14.11 |
| 1 | 96 | in plan: 14.13, now [open] | * [open] 14.13 Rename pass : 'self-reference line', 'reference line', 'healthy line', 'healthy middle' (174 live uses) become the baseline with its status; Next |
| 1 | 97 | status report, result or explanation; no action of its own | 3. Met-A with the atlas as ruler |
| 1 | 98 | in plan: 14.14, now [running] | * [open] 14.14 A Met-A version whose identity sites are chosen on IAM-Atlas v2.1 and tested on arrays the atlas does not hold (after step 3: Next: read the cand |
| 1 | 99 | in plan: 33.2, now [running] | * [running] 33.2 The sealed test of the same-slide tare Next: read once on GitHub (series run); score; if all three bars are met, commission Met-A against the a |
| 1 | 100 | status report, result or explanation; no action of its own | 4. Replays of the earlier work, if there is room |
| 1 | 101 | in plan (a step of the to-do list) | Tune the chain: replay the earlier work through the one command until every signal is found again, every reading wired, the Decoder, globes and spheres right; D |
| 1 | 102 | in plan: 0.1, now [running] | * [running] 0.1 R1: the earlier CPG engine's 11 test arrays (blood and colon tissue) Next: read both runs; then EPIC-Italy and the retired report's raw-array se |
| 1 | 103 | in plan: 0.2, now [open] | * [open] 0.2 The atlas ruler, so healthy reads 1 on the whole-sample Met-A and the territories (healthy 1.113 untared) Next: the straight-line ruler overcorrect |
| 1 | 104 | in plan: 0.3, now [running] | * [running] 0.3 Speed: jobs hit GitHub's 6-hour limit after the one command gained every reading (Group-A pass 2, 11 shards; Next: read the profile; speed up th |
| 1 | 105 | in plan: 0.4, now [built] | * [built] 0.4 Distance to the ceiling read by the chain on every atlas reading (saturation, distance, turnover share), in every report (CAN-005; Next: collect i |
| 1 | 106 | status report, result or explanation; no action of its own | 5. Everything else open, in the plan's order |
| 1 | 107 | status report, result or explanation; no action of its own | * EVERY TOOL FIRES, the priority . The whole chain on every public blood we can get: Done when: every spent pre-diagnosis and plasma series read this way, the p |
| 1 | 108 | in plan: 42.1, now [running] | * [running] 42.1 GSE51032 (EPIC-Italy, 845 whole bloods, 450K IDATs, breast and colorectal cases 0.02 to 14.4 years before diagnosis; Next: smoke run 2 (450K fl |
| 1 | 109 | in plan: 42.2, now [open] | * [open] 42.2 Plasma cfDNA on EPIC with raw IDATs, at diagnosis (common enough to explore): Next: after GSE122126's 58 plasma arrays are recorded: |
| 1 | 110 | in plan: 42.4, now [open] | * [open] 42.4 Stool . Metadata search: the only public human stool DNA methylation set found is GSE212997 (newborns, necrotizing enterocolitis; Next: the one co |
| 1 | 111 | in plan: 42.5, now [open] | * [open] 42.5 GEO search for buffy coat, matched blood and plasma, cfDNA arrays and the same people over time, IDATs only: Next: choose for each commissioning t |
| 1 | 112 | in plan: 42.6, now [open] | * [open] 42.6 EPIC-Italy first reading: the breast shift repeats the earlier pattern, but the case arrays are lower quality; Next: why the case samples carry we |
| 1 | 113 | in plan (a step of the to-do list) | * Blind studies: published methylation studies read by the chain before their findings are opened, then compared Done when: every wave-1 study has our reading c |
| 1 | 114 | in plan: 43.1, now [built] | * [built] 43.1 Protocol, the ledger (CANON/blind_studies.json, BLIND_STUDIES.md) and blind_prep.py; Next: enter each study's reading as it lands, then its publi |
| 1 | 115 | in plan: 43.2, now [running] | * [running] 43.2 Tissue and unnamed specimens accepted on the full-atlas route, flagged (stage_0_intake.is_tissue); Next: the series run on GSE149282, pass 1 |
| 1 | 116 | in plan: 43.3, now [built] | * [built] 43.3 blind_unblind.py: after the readings are committed, fetch the labels from GEO, compute each reading's AUC between the study's groups, and open th |
| 1 | 117 | in plan: 43.4, now [open] | * [open] 43.4 A researcher's own blind run: coded IDATs, our readings committed (and deposited on Zenodo) before they unblind Next: after wave 1; needs the outr |
| 1 | 118 | in plan: 43.7, now [open] | * [open] 43.7 Each blind study's readings deposited on Zenodo before unblinding, so the date they were stated is public and cannot change Next: after GSE149282' |
| 1 | 119 | status report, result or explanation; no action of its own | * Reading cancer's own territories . The commissioned Met-A, the C-Score and Group-A read the 6,000 identity sites of a cell (healthy 0.05-0.25 or 0.75-0.95), s |
| 1 | 120 | in plan: 48.1, now [built] | * [built] 48.1 Island gain reading: mean H(beta) at the 0-of-79 islands over the atlas expectation in the person's mix; Next: the atlas ruler (35.14) first: hea |
| 1 | 121 | in plan: 48.2, now [built] | * [built] 48.2 Domain loss reading: at methylated sites in tiles loose in any healthy cell type (the loose-prone domains), the same form Next: the atlas ruler ( |
| 1 | 122 | in plan: 48.3, now [open] | * [open] 48.3 Whiteboard check, the healthy offset at the unmarked islands: Next: the gain-side count on Stage Q2's molecules (unmethylated molecules, isolated  |
| 1 | 123 | in plan: 48.4, now [open] | * [open] 48.4 Cancer's territories on the information sphere . Next: (a) after 35.10's function labels; (b) after 48.1 and 48.2, on spent series only |
| 1 | 124 | in plan: 48.5, now [built] | * [built] 48.5 Met-A as a two-level instrument : the whole-sample reading over every atlas site (read_whole) and each cell's identity-site reading, with the ter |
| 1 | 125 | in plan (a step of the to-do list) | * Sky-map additions: the posterior noise maps into the chain and the report, the 3D sphere, the guidebook / 2026-10-21 : Done when: each built and in the releas |
| 1 | 126 | in plan (a step of the to-do list) | * Every major blood cell scored in whole blood : Done when: a whole-blood report prints a share and a labeled score for every panel cell; |
| 1 | 127 | in plan (a step of the to-do list) | * Documents updated for Group-A, every blood cell scored and the firm-rate result : Done when: every listed document describes the three results as recorded; |
| 1 | 128 | status report, result or explanation; no action of its own | * The two chain documents : the SOP becomes the Official IAM Chain Operator Guide (how to use our chain as built: Done when: every chain file and function has a |
| 1 | 129 | status report, result or explanation; no action of its own | * 450K arrays read by the chain . After DEV-SELFTARE-03: Done when: a 450K whole-blood array runs through the one command and reads on every instrument commissi |
| 1 | 130 | in plan: 27.1, now [open] | * [open] 27.1 On a 450K array the atlas readers run, flagged; Next: the first real 450K IDATs (EPIC-Italy smoke run 2) through the one command end to end; |
| 1 | 131 | status report, result or explanation; no action of its own | * The genome on a globe with layers . (1) layers and highlighting on the 813,951-site sky sphere (CpG islands, promoters, gene bodies, a cell's identity sites,  |
| 1 | 132 | in plan (a step of the to-do list) | * The IAM-A globe: Stage Q2's counter keeps its held-error and loose-share counts per atlas block (not only totals), tested to give the same totals to every dig |
| 1 | 133 | in plan (a step of the to-do list) | * What the panel cannot explain: the Deconvolver's leftover at each site after the fit (measured minus the fitted mix) as a globe layer and one report number; D |
| 1 | 134 | status report, result or explanation; no action of its own | * A Globe Explorer chapter in the book , written when the Explorer is complete: Done when: the chapter builds in the book; every number in it traces to a commit |
| 1 | 135 | in plan (a step of the to-do list) | * The information sphere, masked (DEV-SPHERE-01 next): Done when: the construction and its check committed before the run; |
| 1 | 136 | in plan: 35.2, now [built] | * [built] 35.2 One person's information sphere against their own healthy cells, wired into the one command Next: sample test 1 (35.4) on GitHub; then the null o |
| 1 | 137 | in plan: 35.3, now [built] | * [built] 35.3 The information sphere as a picture, by the one command (run_sample.py --build info-globe): Next: the browser check on GitHub (globe_audit.py) |
| 1 | 138 | in plan: 35.4, now [running] | * [running] 35.4 The five uses of the information sphere, simulated through the one command (sample_sphere_01.py): Next: the three other cells' runs; then the t |
| 1 | 139 | in plan: 35.5, now [open] | * [open] 35.5 The seven microwave-sky tools on the information sphere (masked power spectrum, cross-spectrum of two readings, component separation with cell typ |
| 1 | 140 | in plan: 35.6, now [open] | * [open] 35.6 The atlas as the one ruler: healthy bloods from many laboratories read on the information sphere with no same-lab references stay dark ; Next: syn |
| 1 | 141 | in plan: 35.7, now [open] | * [open] 35.7 Which instruments catch which genome-wide change: Next: add a domain plant to --dev-plant; run on the healthy example blood |
| 1 | 142 | in plan: 35.8, now [open] | * [open] 35.8 The address globe and the information sphere linked side by side in the report: Next: after 35.4 and the null on real healthy bloods |
| 1 | 143 | in plan: 35.9, now [built] | * [built] 35.9 A book chapter for the information sphere (what it is, the math, how it differs from the address globe, the sky tools, the old mathematics it bui |
| 1 | 144 | in plan: 35.10, now [open] | * [open] 35.10 Gene function on the information sphere: each site labeled by its gene's function (public gene sets: Next: synthetic first: a small loss planted  |
| 1 | 145 | in plan: 35.11, now [open] | * [open] 35.11 The information sphere on EPIC-Italy (DEV-FIRES-01): Next: add --dev-info-sphere to the series run's flags once sample test 1 passes |
| 1 | 146 | in plan: 35.12, now [open] | * [open] 35.12 GSE287331 (breast: healthy donors, the patient's other breast, other quadrant, adjacent, tumor; Next: after sample test 1 (35.4) and the many-lab |
| 1 | 147 | in plan: 35.13, now [open] | * [open] 35.13 The sphere's territory reading misses a loss across a whole cell program (sample test 1, CD4: Next: write it and its synthetic test (the same pla |
| 1 | 148 | in plan: 35.14, now [open] | * [open] 35.14 IAM-Atlas v2.1 as the ruler for every array : Next: synthetic first; then the same person's buffy coat on EPIC v1 and EPIC v2 (AK0005) must read  |
| 1 | 149 | in plan: 35.15, now [open] | * [open] 35.15 Anchors cancer does not move . Early cancer methylation changes are known to concentrate in two places: Next: exploration on spent tumor and norm |
| 1 | 150 | in plan: 35.16, now [open] | * [open] 35.16 The lever simulator : in the browser, every site of the information sphere runs IAM's write and erase balance (writer rate from the methyl supply |
| 1 | 151 | in plan: 35.17, now [open] | * [open] 35.17 The sites the information sphere cannot hold : Next: build the layer through --build atlas-globe (the one command), then read it on the planted a |
| 1 | 152 | in plan (a step of the to-do list) | * Chain only: every reading, reproduction, picture and globe comes from the one command, chain/MethylPhys_Interface/run_sample.py . Done when: every reproductio |
| 1 | 153 | in plan: 37.1, now [running] | * [running] 37.1 Group-A's 278 bloods through the one command Next: pass 1 shard 17 rerun (attempt 2, 2026-10-22 10:30 PM); |
| 1 | 154 | in plan: 37.3, now [open] | * [open] 37.3 10 scripts still outside the one command (REPLICATE.md) Next: move each, as Group-A and the band were |
| 1 | 155 | status report, result or explanation; no action of its own | * Commission the IAM-Globe : the instrument's job: Done when: the sealed test meets every bar and the globe is listed as commissioned with its scope; |
| 1 | 156 | in plan: 38.3, now [open] | * [open] 38.3 Globe commissioning on 278 healthy bloods Next: read the globe counts from Group-A's rerun (37.1), then synthetic plants, then a sealed test |
| 1 | 157 | in plan: 38.5, now [open] | * [open] 38.5 The C-Score family : (a) the commissioned C-Score at Met-A's neutrophil sites; Next: write the family into the catalogue (39.13); |
| 1 | 158 | in plan: 39.14, now [open] | * [open] 39.14 One source per fact : (1) INTRODUCTION.md's per-instrument 'Commissioned' lines generated from CANON/catalogue.json into a marked block, checked  |
| 1 | 159 | in plan: 39.16, now [built] | * [built] 39.16 One master source, in code : (1) CANON/build_all.py rebuilds all 28 generated documents in data-flow order and runs the cross-document checks; N |
| 1 | 160 | in plan (a step of the to-do list) | * Plasma cell-free DNA and every cell in a mix, through the one command . Done when: a plasma sample is read by run_sample.py with each reading commissioned or  |
| 1 | 161 | in plan: 39.6, now [open] | * [open] 39.6 The person's own buffy coat as the reference for their plasma (hypothesis) Next: explore in run_draw on 40.6's data; synthetic first |
| 1 | 162 | in plan: 39.8, now [open] | * [open] 39.8 Met-A's site-rule window stops at 0.95 and 0.05; Next: test windows reaching 1 and 0 against the current ones, by the same planted losses through  |
| 1 | 163 | in plan: 39.10, now [running] | * [running] 39.10 The every-site weighted reading as a whole-sample disorder reading (very sensitive, not cell-specific) Next: exploration 8 (series run on 56 o |
| 1 | 164 | in plan: 39.12, now [open] | * [open] 39.12 The Deconvolver's resampled range holds the known amount in 10 of 15 (too narrow for a 95 % range) Next: a widening rule written before it runs ( |
| 1 | 165 | in plan: 39.17, now [open] | * [open] 39.17 Placing a loss across a whole pattern on the right cell: Next: write it, synthetic first (plants in one cell, then two at once), through the one  |
| 1 | 166 | status report, result or explanation; no action of its own | * ONE DRAW, the priority : 'the full chain running end to end, EVERY instrument scoring on one whole blood sample that is separated between Buffy coat and cfDNA |
| 1 | 167 | in plan: 40.1, now [built] | * [built] 40.1 One-draw runner run_draw.py and the one_draw workflow (buffy coat and plasma, one joint record) Next: first GitHub run on a real draw, once 40.6  |
| 1 | 168 | in plan: 40.4, now [open] | * [open] 40.4 Depth test on real plasma (short fragments, low input, bisulfite damage) Next: plasma molecules still needed for IAM-A's depth test; |
| 1 | 169 | in plan: 40.5, now [open] | * [open] 40.5 Smallest share each instrument reads at each depth Next: 10, 30 and 60 reads recorded: below about 0.02 small shares move with the draw at every d |
| 1 | 170 | in plan: 40.6, now [open] | * [open] 40.6 Open, matched buffy coat and plasma sequencing from the same person Next: GSE186888 found (40.10, every instrument but IAM-A); |
| 1 | 171 | in plan: 40.7, now [running] | * [running] 40.7 58 plasma arrays (GSE122126, spent) through the chain on the full atlas panel Next: record the outcome in DEV_ONE_DRAW_01.md; |
| 1 | 172 | in plan: 40.9, now [open] | * [open] 40.9 What a departure predicts: GSE51032 is spent (usable for testing, not commissioning) Next: find another series of blood drawn before diagnosis, ke |
| 1 | 173 | in plan: 40.11, now [running] | * [open] 40.11 Met-A on sequencing: the chemistry offset (0.757 at full depth against the array-built atlas) and the loss of sensitivity at few reads. Next: a s |
| 1 | 174 | in plan: 40.14, now [open] | * [open] 40.14 IAM-A by cell of origin in plasma: assign each sequenced fragment to a cell type by its identity pattern (fragment-level assignment is published) |
| 1 | 175 | in plan: 40.15, now [open] | * [open] 40.15 Two independent indicators in one plasma : Next: after 40.13 and 40.14: constructed plasma with both changes planted, then the reserved plasma se |
| 1 | 176 | in plan: 40.16, now [built] | * [built] 40.16 Each cell's Met-A for a cell the IAM-Deconvolver did not find (below 1 % of the sample) is flagged as not a reading of that cell (stage_metaa_at |
| 1 | 177 | in plan: 40.17, now [open] | * [open] 40.17 A colorectal cancer plasma set with whole-genome bisulfite reads (none found yet); Next: search SRA and ENA for colorectal cancer cfDNA bisulfite |
| 1 | 178 | in plan: 40.18, now [running] | * [running] 40.18 Provisional reference columns beside IAM-Atlas v2.1 for cells it lacks : Next: versions 1 and 2 not met on donor 2; ploidy does not separate; |
| 1 | 179 | in plan: 40.19, now [open] | * [open] 40.19 Provisional columns for liver sinusoidal endothelium and Kupffer cells : Next: search public purified methylation data for both |
| 1 | 180 | status report, result or explanation; no action of its own | * The book, before outreach : after commissioning and the chain settle, one pass over the whole book: Done when: every chapter reviewed line by line and its cor |
| 1 | 181 | in plan: 45.1, now [open] | * [open] 45.1 A chapter on how every instrument references itself with no healthy cohort . Next: write once testing and the derivations are done (DEV-METAA-HEAL |
| 1 | 182 | in plan: 45.2, now [open] | * [open] 45.2 The atlas chapter: how central IAM-Atlas v2.1 has become, and why . Next: after the five-laboratory test (DEV-METAA-HEALTHY-01 bar 2): |
| 1 | 183 | status report, result or explanation; no action of its own | * The C-Score family commissioned . The instrument's job, for each: Done when: each version met its sealed bars, or is recorded as development with the reason |
| 1 | 184 | in plan: 41.1, now [open] | * [open] 41.1 IAM-A's C-Score, version 2: the held error of each 1,000-CpG region predicted from that region's OWN loose share by the commissioned self-referenc |
| 1 | 185 | in plan: 41.2, now [built] | * [built] 41.2 The whole-genome C-Score against IAM-Atlas v2.1 on arrays: Next: synthetic C-Score test 1 through the one command (cscore_cell_01.py); |
| 1 | 186 | in plan: 41.3, now [built] | * [built] 41.3 Each cell's C-Score in a mix: the same rule on each cell's own identity sites and its globe layer (exploration 5 shows the layer works) Next: syn |
| 1 | 187 | in plan (a step of the to-do list) | * The readiness map: every specimen (whole blood, buffy coat, plasma cfDNA, tissue, isolated cells, unnamed) against every platform (EPIC v1, EPIC v2, 450K, seq |
| 1 | 188 | in plan: 45.1, now [open] | * [open] 45.1 readiness.yml: one fixed public file of each kind (EPIC v1 whole blood, buffy coat, isolated neutrophils, plasma, tissue, unnamed, 450K, EPIC v2,  |
| 1 | 189 | in plan (a step of the to-do list) | * The Researcher Challenge, the last step of the researcher campaign : Done when: the website section, the IAM-Chain mirror and its Zenodo DOI, and the scoreboa |
| 1 | 190 | in plan: 43.5, now [open] | * [open] 43.5 The Researcher Challenge : three routes that never put a patient's arrays in public (a private copy of the repository with my_samples.yml; Next: t |
| 1 | 191 | in plan: 43.6, now [open] | * [open] 43.6 IAM-Chain, a small mirror of only what the chain needs (chain, IAM-Atlas v2.1, runtime matrices, requirements, the researcher workflow; Next: afte |
| 1 | 192 | in plan: 43.8, now [open] | * [open] 43.8 Each IAM-Chain release archived on Zenodo with a DOI (GitHub release linked to Zenodo), so the exact instrument a result used can be cited and fet |
| 1 | 193 | status report, result or explanation; no action of its own | * The wide-angle matrix : every system as its own column on the same few rows, generated from one source (CANON/systems.json), every value from the book or a co |
| 1 | 194 | in plan: 47.1, now [open] | * [open] 47.1 Mark which comparisons are true tests and which hold by construction: Next: write CANON/systems.json from Part V's tables (tab:qp_which, tab:platf |
| 1 | 195 | in plan (a step of the to-do list) | * Whole-genome atlas and globe, built together : |
| 1 | 196 | status report, result or explanation; no action of its own | 6. Waiting on something outside the chain |
| 1 | 197 | in plan (a step of the to-do list) | * The book's Zenodo record, after steps 9 and 10 Done when: the draft uploaded by the box job with every book check passing; |
| 1 | 198 | in plan (a step of the to-do list) | * The researcher package's Zenodo record (the biology folder), after step 11 Done when: the draft uploaded by the box job after the archive passes the release c |
| 1 | 199 | in plan (a step of the to-do list) | * When AWS returns: clean-clone check of the chain after the report's American-spelling change (gray), and the example reports rebuilt Done when: a clean clone  |
| 1 | 200 | in plan (a step of the to-do list) | * When AWS returns: delete CANON/billing_unreadable.json and fill the pending computing-and-cost blocks from billing (day_compute.py, compute_snapshot.py) Done  |
| 1 | 201 | in plan (a step of the to-do list) | * When AWS returns: remove idle standing charges (the unattached public IP address, about 1 TB of machine disks and snapshots kept between jobs) after checking  |
| 1 | 202 | status report, result or explanation; no action of its own | * Independence from AWS : (1) a committed data register, one line per input any test or instrument reads: Done when: the register has no input whose only home i |
| 1 | 203 | status report, result or explanation; no action of its own | * [parked] Salmon: IAM-A on hatchery and wild coho : |
| 1 | 204 | status report, result or explanation; no action of its own | 7. Completed 2026-10-24 and 2026-10-25 |
| 1 | 205 | in plan: 49.1, now [done] | * [done] 49.1 Pass 1: IAM-A's count on all 36 samples at 10 million read pairs |
| 1 | 206 | in plan: 49.2, now [built] | * [built] 49.2 The split alignment: each sample's reads spread over machines, one index; |
| 1 | 207 | in plan: 49.4, now [done] | * [done] 49.4 The study's pre-B2 controls are controlled-access and are not used (public data only, author). |
| 1 | 208 | in plan: 50.1, now [built] | * [built] 50.1 Outline and figure plan in the journal's format (abstract, introduction, Results under finding headings, Discussion, Methods with equations, Data |
| 1 | 209 | in plan: 50.2, now [built] | * [built] 50.2 Methods drafted from the Decoder and the records: |
| 1 | 210 | in plan: 0.8, now [done] | * [done] 0.8 R5 to R9: Crohn's and colitis (GSE87650), stroke (GSE69138), prostate (GSE269244), pancreas, brain and aorta tissues (GSE49149, GSE60274, GSE74071, |
| 1 | 211 | in plan: 14.3, now [read] | * [read] 14.3 Held error and replication error are two routes with different physics . |
| 1 | 212 | in plan: 14.4, now [read] | * [read] 14.4 Met-A's healthy number from physics and the atlas, the independent test: |
| 1 | 213 | in plan: 14.7, now [done] | * [done] 14.7 Layer 1: the atlas's healthy levels predicted from each cell type's loss-side and gain-side held errors; |
| 1 | 214 | in plan: 14.10, now [done] | * [done] 14.10 A maintenance knockout as a test of the floor: |
| 1 | 215 | in plan: 40.16, now [built] | * [built] 40.16 Each cell's Met-A for a cell the IAM-Deconvolver did not find (below 1 % of the sample) is flagged as not a reading of that cell (stage_metaa_at |
| 1 | 217 | status report, result or explanation; no action of its own | Your validation list, line by line. The list has 119 tests: VAL-001 to VAL-128, CPG_VAL_001 to 022, and the 10 open predictions. Each row gives the condition, t |
| 1 | 218 | status report, result or explanation; no action of its own | Where the 119 tests stand	Tests |
| 1 | 219 | status report, result or explanation; no action of its own | Already replayed (R4 to R9)	14 |
| 1 | 220 | status report, result or explanation; no action of its own | Data public and already spent: replay now	52 |
| 1 | 221 | replay run order: every dataset is in the retest plan (plan 0.9, val_retest_by_dataset.csv); runs in Phase E (60.13) | Data public but unread: replaying spends it	7 |
| 1 | 222 | replay run order: every dataset is in the retest plan (plan 0.9, val_retest_by_dataset.csv); runs in Phase E (60.13) | Dataset not found yet	16 |
| 1 | 223 | replay run order: every dataset is in the retest plan (plan 0.9, val_retest_by_dataset.csv); runs in Phase E (60.13) | Old fitted method, nothing to replay (panels, Mahalanobis hulls, principal components, age subtraction, other atlases)	23 |
| 1 | 224 | replay run order: every dataset is in the retest plan (plan 0.9, val_retest_by_dataset.csv); runs in Phase E (60.13) | Controlled access, excluded	4 |
| 1 | 225 | replay run order: every dataset is in the retest plan (plan 0.9, val_retest_by_dataset.csv); runs in Phase E (60.13) | Not human, so the atlas can't read it (dogs, the 348-species chip)	3 |
| 1 | 226 | replay run order: every dataset is in the retest plan (plan 0.9, val_retest_by_dataset.csv); runs in Phase E (60.13) | Run order by dataset, on machines the higher priorities leave free: |
| 1 | 227 | replay run order: every dataset is in the retest plan (plan 0.9, val_retest_by_dataset.csv); runs in Phase E (60.13) | * A. Spent, with raw array files: |
| 1 | 228 | replay run order: every dataset is in the retest plan (plan 0.9, val_retest_by_dataset.csv); runs in Phase E (60.13) | * GSE51057, EPIC-Italy's second series, which feeds 11 tests |
| 1 | 229 | replay run order: every dataset is in the retest plan (plan 0.9, val_retest_by_dataset.csv); runs in Phase E (60.13) | * GSE122126, the published plasma arrays |
| 1 | 230 | replay run order: every dataset is in the retest plan (plan 0.9, val_retest_by_dataset.csv); runs in Phase E (60.13) | * GSE256092, GSE287994, GSE99511, GSE141682 and GSE281691 |
| 1 | 231 | replay run order: every dataset is in the retest plan (plan 0.9, val_retest_by_dataset.csv); runs in Phase E (60.13) | * B. Only processed values on GEO: the earlier map marked these "not readable", but the values route built yesterday reads them, flagged. This opens: |
| 1 | 232 | replay run order: every dataset is in the retest plan (plan 0.9, val_retest_by_dataset.csv); runs in Phase E (60.13) | * GSE144858 (AddNeuroMed) |
| 1 | 233 | replay run order: every dataset is in the retest plan (plan 0.9, val_retest_by_dataset.csv); runs in Phase E (60.13) | * GSE53740 (the PSP/FTD cohort) |
| 1 | 234 | replay run order: every dataset is in the retest plan (plan 0.9, val_retest_by_dataset.csv); runs in Phase E (60.13) | * GSE40279 (Hannum) |
| 1 | 235 | replay run order: every dataset is in the retest plan (plan 0.9, val_retest_by_dataset.csv); runs in Phase E (60.13) | * GSE50660 (smoking cessation) |
| 1 | 236 | replay run order: every dataset is in the retest plan (plan 0.9, val_retest_by_dataset.csv); runs in Phase E (60.13) | * GSE62298 (AML) |
| 1 | 237 | replay run order: every dataset is in the retest plan (plan 0.9, val_retest_by_dataset.csv); runs in Phase E (60.13) | * three cervical/breast sets and GSE69914 |
| 1 | 238 | replay run order: every dataset is in the retest plan (plan 0.9, val_retest_by_dataset.csv); runs in Phase E (60.13) | * Each format gets checked on the laptop first. That's downtime work. |
| 1 | 239 | status report, result or explanation; no action of its own | * C. TCGA tissue: 12 projects, raw files from the NCI data portal, one project at a time. This is the heaviest machine load. |
| 1 | 240 | status report, result or explanation; no action of its own | * D. Unread: TCGA-PRAD, TCGA-TGCT, GSE119260, GSE298812 and GSE147436. I recommend the replay reads a fixed half of each and keeps the other half unread for a b |
| 1 | 247 | status report, result or explanation; no action of its own | IAM to-do list, October 23, 2026, 07:00 Pacific |
| 1 | 250 | one-button intake: was missing; added as 61.6 | On priorities: your goal is a one-button chain that works out what it's looking at. That's mostly an integration problem, and most of the parts exist. The criti |
| 1 | 251 | one-button intake: was missing; added as 61.6 | 1. Auto-detection at intake. The chain already reads the platform from the IDAT header (450K, EPIC v1, EPIC v2). What it still needs: |
| 1 | 252 | one-button intake: was missing; added as 61.6 | * telling the input type from the files themselves (IDAT pairs, value tables, sequencing molecules); |
| 1 | 253 | readiness map: in plan (45.1) | * making the specimen optional, so the Deconvolver says what the sample is ("unknown" now works for tissue); |
| 1 | 254 | commissioning order of 10-23: superseded by step 1's phases (60.x) | * one folder-and-button workflow for all of it, built on the existing researcher workflow. |
| 1 | 255 | commissioning order of 10-23: superseded by step 1's phases (60.x) | 2. The readiness map as the acceptance test. One real file of each kind through the button, every instrument's verdict recorded. Every blank cell is a gap, and  |
| 1 | 256 | commissioning order of 10-23: superseded by step 1's phases (60.x) | 3. Commission what's closest, in this order: |
| 1 | 257 | commissioning order of 10-23: superseded by step 1's phases (60.x) | * Met-A against the atlas (the second seal is waiting on its synthetic run); |
| 1 | 258 | commissioning order of 10-23: superseded by step 1's phases (60.x) | * each C-Score (the family is built, its synthetic tests are written); |
| 1 | 259 | commissioning order of 10-23: superseded by step 1's phases (60.x) | * full deconvolution (the known-mixture runs are queued). |
| 1 | 260 | commissioning order of 10-23: superseded by step 1's phases (60.x) | 4. Then the routes that need more: |
| 1 | 261 | commissioning order of 10-23: superseded by step 1's phases (60.x) | * plasma and buffy coat: a matched public pair to test on; |
| 1 | 262 | commissioning order of 10-23: superseded by step 1's phases (60.x) | * tissue: the blind study now running is its first real test; |
| 1 | 263 | status report, result or explanation; no action of its own | * stool: the sequencing route plus the gut cells in the atlas, so it comes last; the data is already in S3. |
| 1 | 264 | status report, result or explanation; no action of its own | 5. Physics derivation, in parallel, at the laptop, no machines. |
| 1 | 265 | status report, result or explanation; no action of its own | Machine allocation from now on: |
| 1 | 266 | status report, result or explanation; no action of its own | * First: the readiness run and the commissioning steps above. |
| 1 | 267 | status report, result or explanation; no action of its own | * Blind studies: finish GSE149282, which is also the tissue route's first real test. The other 13 wait until the readiness map is green. |
| 1 | 270 | status report, result or explanation; no action of its own | IAM to-do list, 2026-10-23 07:21 AM PDT |
| 1 | 271 | status report, result or explanation; no action of its own | Built from the repository's plan (CANON/plan.json) and GitHub's live run list. Step numbers are the plan's current order. Branch numbers (like 42.1) were given  |
| 1 | 272 | status report, result or explanation; no action of its own | Running now on GitHub |
| 1 | 273 | status report, result or explanation; no action of its own | Run	Machines |
| 1 | 274 | status report, result or explanation; no action of its own | C-Score synthetic test: lung + neurons mixture	1 in progress |
| 1 | 275 | status report, result or explanation; no action of its own | C-Score synthetic test: colon + neurons mixture	1 in progress |
| 1 | 276 | status report, result or explanation; no action of its own | C-Score synthetic test: hepatocytes + lung mixture	1 in progress |
| 1 | 277 | status report, result or explanation; no action of its own | C-Score synthetic test: healthy blood (41.2, 41.3)	1 in progress |
| 1 | 278 | status report, result or explanation; no action of its own | Second Met-A seal: synthetic test, rebuilt (33.2)	1 in progress |
| 1 | 279 | in plan (a step of the to-do list) | Group-A's 278 bloods through the one command (37.1)	28 success, 12 in progress |
| 1 | 280 | status report, result or explanation; no action of its own | EPIC-Italy smoke run: 4 arrays, full chain (42.1)	10 queued |
| 1 | 281 | status report, result or explanation; no action of its own | Exploration 8: each cell's healthy spread, 56 bloods (39.5)	8 queued, 2 in progress |
| 1 | 282 | status report, result or explanation; no action of its own | 58 plasma arrays, last shard (40.7)	1 cancelled, 9 success, 1 queued, 1 skipped |
| 1 | 283 | status report, result or explanation; no action of its own | Also queued: the repository's own push checks and website build. |
| 1 | 284 | status report, result or explanation; no action of its own | Open steps, in priority order |
| 1 | 285 | in plan (a step of the to-do list) | Step 1: EVERY TOOL FIRES, the priority (author 2026-10-23: 'find the cancer before the tumor crystalizes ... The early signals are in the blood/immune cells ... |
| 1 | 286 | in plan (a step of the to-do list) | Done when: every spent pre-diagnosis and plasma series read this way, the per-person readings delivered as one table per series, and each rare set's sealed test |
| 1 | 287 | in plan: 42.1, now [running] | * [running] 42.1 GSE51032 (EPIC-Italy, 845 whole bloods, 450K IDATs, breast and colorectal cases 0.02 to 14.4 years before diagnosis; spent): smoke run of 4 arr |
| 1 | 288 | in plan: 42.2, now [open] | * 42.2 Plasma cfDNA on EPIC with raw IDATs, at diagnosis (common enough to explore): GSE277078 (185, lung), GSE298812 (245, liver, with cirrhosis, fibrosis and  |
| 1 | 289 | in plan: 42.3, now [parked] | * [parked] 42.3 RARE, kept for sealed tests, not explored: GSE130748 (the same 20 people at baseline and year 6, Health ABC, EPIC, with later cancer diagnoses:  |
| 1 | 290 | in plan: 42.4, now [open] | * 42.4 Stool (author: 'Stool samples? That is what I would like to run through the chain just to see what is possible'). Metadata search: the only public human  |
| 1 | 291 | in plan (a step of the to-do list) | Step 12: The book's Zenodo record, after steps 9 and 10 |
| 1 | 292 | in plan (a step of the to-do list) | Done when: the draft uploaded by the box job with every book check passing; IAMPerformance publishes |
| 1 | 293 | in plan (a step of the to-do list) | Step 13: The researcher package's Zenodo record (the biology folder), after step 11 |
| 1 | 294 | in plan (a step of the to-do list) | Done when: the draft uploaded by the box job after the archive passes the release check; IAMPerformance publishes |
| 1 | 295 | in plan (a step of the to-do list) | Step 14: Sky-map additions: the posterior noise maps into the chain and the report, the 3D sphere, the guidebook / 2026-10-21 (author): (a) every place the glob |
| 1 | 296 | in plan (a step of the to-do list) | Done when: each built and in the release check, the report and the page |
| 1 | 297 | in plan (a step of the to-do list) | Step 15: IAM-A's firm rate, derived again on the commissioned counter: (1) recount the five DEV-ERASE-01 primary units (eraser-removed vs same-lab control molec |
| 1 | 298 | in plan (a step of the to-do list) | Done when: the recount scored once and recorded; a sealed prediction written or the derivation recorded as not supported |
| 1 | 299 | in plan: 14.1, now [running] | * 14.1 THE TOP PHYSICS ITEM (author 2026-10-22: 'the firm rate derivation is a bit more important than I realized. That could be the thing that makes MET-A not  |
| 1 | 300 | in plan: 14.2, now [open] | * 14.2 If one human sample needs more than GitHub's 6 hours: split it as the mouse run was split, each machine aligning a part of the reads, then one job mergin |
| 1 | 301 | in plan (a step of the to-do list) | Step 17: When AWS returns: clean-clone check of the chain after the report's American-spelling change (gray), and the example reports rebuilt |
| 1 | 302 | in plan (a step of the to-do list) | Done when: a clean clone passes every release check; the example reports print gray |
| 1 | 303 | in plan (a step of the to-do list) | Step 18: When AWS returns: delete CANON/billing_unreadable.json and fill the pending computing-and-cost blocks from billing (day_compute.py, compute_snapshot.py |
| 1 | 304 | in plan (a step of the to-do list) | Done when: no day log carries a pending block; costs_fresh.py passes without the declaration |
| 1 | 305 | in plan (a step of the to-do list) | Step 19: When AWS returns: remove idle standing charges (the unattached public IP address, about 1 TB of machine disks and snapshots kept between jobs) after ch |
| 1 | 306 | in plan (a step of the to-do list) | Done when: nothing idle billed; each removal recorded |
| 1 | 307 | in plan (a step of the to-do list) | Step 20: Independence from AWS (author 2026-10-20, after the account suspension: 'pull what we need to replicate or test our physics from Zenodo and not have to |
| 1 | 308 | in plan (a step of the to-do list) | Done when: the register has no input whose only home is S3; a fresh clone rebuilds every test from public archives and Zenodo with AWS credentials absent |
| 1 | 309 | in plan (a step of the to-do list) | Step 22: Every major blood cell scored in whole blood (author 2026-10-20: 'to be able to filter and find each cell in the blood and score each one it finds is w |
| 1 | 310 | in plan (a step of the to-do list) | Done when: a whole-blood report prints a share and a labeled score for every panel cell; release check and clean clone pass |
| 1 | 311 | in plan (a step of the to-do list) | Step 23: Documents updated for Group-A, every blood cell scored and the firm-rate result (author 2026-10-20: 'Lots of updating as the last thing'): the operatio |
| 1 | 312 | in plan (a step of the to-do list) | Done when: every listed document describes the three results as recorded; all push checks pass; the site and book build |
| 1 | 313 | in plan (a step of the to-do list) | Step 26: The two chain documents (author 2026-10-20: 'researchers are not going to just want to trust us ... they are gonna want to dissect it and try and break |
| 1 | 314 | in plan (a step of the to-do list) | Done when: every chain file and function has a Decoder section; the worked example reproduces the chain's numbers by hand to stated precision; the coverage chec |
| 1 | 315 | in plan (a step of the to-do list) | Step 28: 450K arrays read by the chain (author 2026-10-21: 'What would it take for us to get the 450K working? ... after the self tare test'). After DEV-SELFTAR |
| 1 | 316 | in plan (a step of the to-do list) | Done when: a 450K whole-blood array runs through the one command and reads on every instrument commissioned for 450K; release check passes |
| 1 | 317 | in plan: 27.1, now [open] | * [built] 27.1 On a 450K array the atlas readers run, flagged; the commissioned Met-A and C-Score stay refused Next: a real spent 450K array with raw IDATs thro |
| 1 | 318 | in plan (a step of the to-do list) | Step 29: The genome on a globe with layers (author 2026-10-21: 'map the entire genome that way ... change the colors depending on what a researcher is looking f |
| 1 | 319 | in plan (a step of the to-do list) | Done when: a researcher can turn the globe, pick a layer, and see the highlighted sites with their shuffled null; then the whole-genome tiles |
| 1 | 320 | in plan (a step of the to-do list) | Step 30: The IAM-A globe: Stage Q2's counter keeps its held-error and loose-share counts per atlas block (not only totals), tested to give the same totals to ev |
| 1 | 321 | in plan (a step of the to-do list) | Done when: the counter's per-block counts sum to the committed totals on the release check's BAM and on one grand-test sample; the IAM-A layer is drawn by the o |
| 1 | 322 | in plan (a step of the to-do list) | Step 31: What the panel cannot explain: the Deconvolver's leftover at each site after the fit (measured minus the fitted mix) as a globe layer and one report nu |
| 1 | 323 | in plan (a step of the to-do list) | Done when: synthetic: a planted missing cell raises the leftover at its own marker sites and is named; real healthy bloods: the leftover reported per sample |
| 1 | 324 | in plan (a step of the to-do list) | Step 33: A Globe Explorer chapter in the book (author 2026-10-21: 'we are going to have a whole new chapter to write'), written when the Explorer is complete: w |
| 1 | 325 | in plan (a step of the to-do list) | Done when: the chapter builds in the book; every number in it traces to a committed output; its figures are screenshots of the chain's own pages |
| 1 | 326 | in plan (a step of the to-do list) | Step 34: Met-A against IAM-Atlas v2.1, sealed (author 2026-10-22 'when do we do this?'): the instrument's job: a healthy purified cell, read against its atlas c |
| 1 | 327 | in plan (a step of the to-do list) | Done when: the bars scored once; Met-A against the atlas commissioned for that cell type, or the shortfall recorded |
| 1 | 328 | in plan: 33.1, now [done] | * 33.1 Exploration 8 through the one command: same-slide tare 100 % Normal in both series (1 % loss 4.09 and 3.73 spreads); laboratory tare 90.3 % and 97.0 % (1 |
| 1 | 329 | in plan: 33.2, now [running] | * [running] 33.2 The sealed test of the same-slide tare Next: read once on GitHub (series run); score; if all three bars are met, commission Met-A against the a |
| 1 | 330 | in plan (a step of the to-do list) | Step 36: The information sphere, masked (DEV-SPHERE-01 next): v1's distance-faithful projection with the empty sky masked and apodized as microwave-sky maps are |
| 1 | 331 | in plan (a step of the to-do list) | Done when: the construction and its check committed before the run; planted test 6 scored once |
| 1 | 332 | in plan: 35.1, now [done] | * 35.1 Test 8 (real atlas cells) met C1 and C3 and failed C2: a site's level dominates its place. Construction v5 takes the level out / Test 11 PASSED on the re |
| 1 | 333 | in plan: 35.2, now [built] | * 35.2 One person's information sphere against their own healthy cells, wired into the one command (author: 'the blind spot we didnt know we had') Next: write i |
| 1 | 334 | in plan (a step of the to-do list) | Step 38: Chain only: every reading, reproduction, picture and globe comes from the one command, chain/MethylPhys_Interface/run_sample.py (author 2026-10-22: 'th |
| 1 | 335 | in plan (a step of the to-do list) | Done when: every reproduction and every published globe is produced by run_sample.py on GitHub and matches; the gate covers every reproduction step |
| 1 | 336 | in plan: 37.1, now [running] | * [running] 37.1 Group-A's 278 bloods through the one command Next: pass 1 shard 17 rerun (attempt 2, 2026-10-22 10:30 PM); then pass 2 and score_groupa_chain.p |
| 1 | 337 | in plan: 37.2, now [done] | * 37.2 The C-Score band through the one command: MET (0.9545 not above, bar 0.95), against the committed 0.9565. One array fewer: GSM5583015 is now stopped by S |
| 1 | 338 | in plan: 37.3, now [open] | * 37.3 10 scripts still outside the one command (REPLICATE.md) Next: move each, as Group-A and the band were |
| 1 | 339 | in plan (a step of the to-do list) | Step 39: Commission the IAM-Globe (author 2026-10-22: 'hurry up and get the chain commissioned so we can use it'): the instrument's job: a healthy sample's glob |
| 1 | 340 | in plan (a step of the to-do list) | Done when: the sealed test meets every bar and the globe is listed as commissioned with its scope; or it misses a bar and the outcome is recorded |
| 1 | 341 | in plan: 38.1, now [done] | * 38.1 A change layer for each cell on the globe (healthy example blood: 0-5 of ~590 blocks per cell) |
| 1 | 342 | in plan: 38.2, now [done] | * 38.2 The report's flat sky pictures: their reference was right (Met-A's 6,000 neutrophil sites against healthy neutrophils, the C-Score's own sites); the titl |
| 1 | 343 | in plan: 38.3, now [open] | * 38.3 Globe commissioning on 278 healthy bloods Next: read the globe counts from Group-A's rerun (37.1), then synthetic plants, then a sealed test |
| 1 | 344 | in plan: 38.4, now [done] | * [built] 38.4 Two rules for 'changed' in one report: the flat picture at Met-A's 6,000 sites (report_figs_v4.changes_sky_png) marks a pixel beyond +/-2 of the  |
| 1 | 345 | in plan: 38.5, now [open] | * 38.5 The C-Score family (author 2026-10-22: 'it might be the star of the show'): (a) the commissioned C-Score at Met-A's neutrophil sites; (b) the C-Score on  |
| 1 | 346 | in plan: 39.13, now [done] | * 39.13 Instrument catalogue, line by line (author 2026-10-22): update stale entries; review every candidate instrument (some started, in development or commiss |
| 1 | 347 | in plan: 39.14, now [open] | * 39.14 One source per fact (author 2026-10-23: 'we are not hand updating things like this because that is how we end up with all kinds of stale information'):  |
| 1 | 348 | in plan: 39.15, now [done] | * 39.15 No dates in public prose (author 2026-10-23: 'dates belong nowhere except the logs ... NEVER in the book or webpage or anywhere in prose'): CANON/no_dat |
| 1 | 349 | in plan: 38.6, now [done] | * [built] 38.6 Changed blocks drawn as patches on the globe (author: 'Those patches are exactly what should be seen, not the tiny specs'): Met-A's and each cell |
| 1 | 350 | in plan: 39.16, now [built] | * [built] 39.16 One master source, in code (author: 'All of this could be avoided by using code and keeping the master data source in ONE place'): (1) CANON/bui |
| 1 | 351 | in plan (a step of the to-do list) | Step 40: Plasma cell-free DNA and every cell in a mix, through the one command (author 2026-10-22: 'why else did we create a 79 cell atlas? To read 6 cells??? . |
| 1 | 352 | in plan (a step of the to-do list) | Done when: a plasma sample is read by run_sample.py with each reading commissioned or flagged; the healthy plasma reference committed with its sources |
| 1 | 353 | in plan: 39.1, now [done] | * 39.1 --cell-panel atlas: the Deconvolver on all 79 cell types; Met-A for each cell found (site-dropping fault fixed) |
| 1 | 354 | in plan: 39.2, now [done] | * 39.2 Which share estimate to keep: the best fit (mean absolute error 0.0057 against 0.0079 for the resampled middle on the 9 known mixtures) |
| 1 | 355 | in plan: 39.3, now [done] | * 39.3 Met-A for each cell, synthetic test 2 through the chain only: planted cell moves 92-100 % of its predicted shift; spillover up to 26 % |
| 1 | 356 | in plan: 39.4, now [done] | * 39.4 Exploration 4: identity sites tell which cell; every site is far more sensitive but cannot say which cell; identity sites kept |
| 1 | 357 | in plan: 39.5, now [done] | * [running] 39.5 Each cell's healthy spread in whole blood (its detection limit) Next: exploration 8 (series run on 56 of Group-A's bloods); record the spreads |
| 1 | 358 | in plan: 39.6, now [open] | * 39.6 The person's own buffy coat as the reference for their plasma (hypothesis) Next: explore in run_draw on 40.6's data; synthetic first |
| 1 | 359 | in plan: 39.7, now [done] | * 39.7 IAM-A states its own counting error on every reading (finding F38) |
| 1 | 360 | in plan: 39.8, now [open] | * 39.8 Met-A's site-rule window stops at 0.95 and 0.05; no reason is recorded with the rule (operational) Next: test windows reaching 1 and 0 against the curren |
| 1 | 361 | in plan: 39.9, now [done] | * 39.9 Identity sites specific to each cell (held by the cell and not by its relatives), for attribution in a mix |
| 1 | 362 | in plan: 39.10, now [running] | * [running] 39.10 The every-site weighted reading as a whole-sample disorder reading (very sensitive, not cell-specific) Next: exploration 8 (series run on 56 o |
| 1 | 363 | in plan: 39.11, now [done] | * 39.11 Write why Met-A reads these sites in the book's Met-A chapter and INSTRUMENTS.md (the Decoder section is written) |
| 1 | 364 | in plan: 39.12, now [open] | * 39.12 The Deconvolver's resampled range holds the known amount in 10 of 15 (too narrow for a 95 % range) Next: a widening rule written before it runs (for exa |
| 1 | 365 | in plan: 39.17, now [open] | * 39.17 Placing a loss across a whole pattern on the right cell: identity sites spread it over relatives (exploration 4, 7) and cell-specific sites lose the sig |
| 1 | 366 | in plan (a step of the to-do list) | Step 41: ONE DRAW, the priority (author 2026-10-22): 'the full chain running end to end, EVERY instrument scoring on one whole blood sample that is separated be |
| 1 | 367 | in plan (a step of the to-do list) | Done when: one button reads a real draw's buffy coat and plasma end to end, every reading commissioned or flagged, and a fresh fork reproduces it |
| 1 | 368 | in plan: 40.1, now [built] | * [built] 40.1 One-draw runner run_draw.py and the one_draw workflow (buffy coat and plasma, one joint record) Next: first GitHub run on a real draw, once 40.6  |
| 1 | 369 | in plan: 40.2, now [done] | * 40.2 Sequencing values (--dev-seq-values) and the depth test: IAM-A Normal down to 1 % of the molecules (~3.7 M), below at 0.3 %; Deconvolver reads CD8 down t |
| 1 | 370 | in plan: 40.3, now [done] | * 40.3 Met-A on sequencing: the k-of-n expectation (expected_H_counts); depth test 2: no downward slide (0.757, 0.798, 0.897 at 31, 9, 3 reads; before 0.673, 0. |
| 1 | 371 | in plan: 40.4, now [open] | * 40.4 Depth test on real plasma (short fragments, low input, bisulfite damage) Next: plasma molecules still needed for IAM-A's depth test; GSE186888 (40.10) gi |
| 1 | 372 | in plan: 40.5, now [open] | * 40.5 Smallest share each instrument reads at each depth Next: 10, 30 and 60 reads recorded: below about 0.02 small shares move with the draw at every depth; n |
| 1 | 373 | in plan: 40.6, now [open] | * 40.6 Open, matched buffy coat and plasma sequencing from the same person Next: GSE186888 found (40.10, every instrument but IAM-A); keep searching for matched |
| 1 | 374 | in plan: 40.7, now [running] | * [running] 40.7 58 plasma arrays (GSE122126, spent) through the chain on the full atlas panel Next: record the outcome in DEV_ONE_DRAW_01.md; the islet-transpl |
| 1 | 375 | in plan: 40.8, now [done] | * 40.8 9 known mixtures: best fit mean absolute error 0.0057; tissues not put in at most 0.0030; resampled range holds the truth 10 of 15 |
| 1 | 376 | in plan: 40.9, now [open] | * 40.9 What a departure predicts: GSE51032 is spent (usable for testing, not commissioning) Next: find another series of blood drawn before diagnosis, kept unre |
| 1 | 377 | in plan: 40.10, now [done] | * [parked] 40.10 GSE186888: whole-genome bisulfite sequencing (85x) of white blood cells and matched plasma from the same 23 healthy donors (per-CpG counts, no  |
| 1 | 378 | in plan: 40.11, now [running] | * 40.11 Met-A on sequencing: the chemistry offset (0.757 at full depth against the array-built atlas) and the loss of sensitivity at few reads Next: 3, 10, 30 a |
| 1 | 379 | in plan (a step of the to-do list) | Step 42: The book, before outreach (author: 'the book is going to need some special attention once we get everything commissioned and the chain fully operationa |
| 1 | 380 | in plan (a step of the to-do list) | Done when: every chapter reviewed line by line and its corrections pushed; build_all, no_dates, status_claims and the book build pass from a fresh clone |
| 1 | 381 | in plan (a step of the to-do list) | Step 43: The C-Score family commissioned (author 2026-10-23: 'We have more C-Scores to commission'). The instrument's job, for each: a healthy sample reads insi |
| 1 | 382 | in plan (a step of the to-do list) | Done when: each version met its sealed bars, or is recorded as development with the reason |
| 1 | 383 | in plan: 41.1, now [open] | * 41.1 IAM-A's C-Score, version 2: the held error of each 1,000-CpG region predicted from that region's OWN loose share by the commissioned self-reference line. |
| 1 | 384 | in plan: 41.2, now [built] | * [built] 41.2 The whole-genome C-Score against IAM-Atlas v2.1 on arrays: the globe's block departures (every atlas site, blocks of 10 neighbors) against the sa |
| 1 | 385 | in plan: 41.3, now [built] | * [built] 41.3 Each cell's C-Score in a mix: the same rule on each cell's own identity sites and its globe layer (exploration 5 shows the layer works) Next: syn |
| 1 | 386 | status report, result or explanation; no action of its own | Parked |
| 1 | 387 | in plan (a step of the to-do list) | * Step 37: Salmon: IAM-A on hatchery and wild coho (author 2026-10-22, for the fisheries biologists): public RRBS of 39 coho salmon smolts, white muscle, 20 hat |
| 1 | 388 | in plan (a step of the to-do list) | * Branch 42.3: RARE, kept for sealed tests, not explored: GSE130748 (the same 20 people at baseline and year 6, Health ABC, EPIC, with later cancer diagnoses: t |
| 1 | 389 | in plan (a step of the to-do list) | * Branch 40.10: GSE186888: whole-genome bisulfite sequencing (85x) of white blood cells and matched plasma from the same 23 healthy donors (per-CpG counts, no m |
| 1 | 390 | status report, result or explanation; no action of its own | Completed steps |
| 1 | 391 | in plan (a step of the to-do list) | * Step 2: Showcase scenario 4: the three healthy colon samples (GSE151732: GSM4589141, GSM4589144, GSM4589145) through IAM-Atlas v2.1's commissioned tissue scor |
| 1 | 392 | in plan (a step of the to-do list) | * Step 3: Showcase README: scenarios 2 and 5 stated as covered by the example reports, with scenario 5's reason (no fresh public whole-genome set of healthy res |
| 1 | 393 | in plan (a step of the to-do list) | * Step 4: Mouse reproduction (run 13): read the verdict and record it |
| 1 | 394 | in plan (a step of the to-do list) | * Step 5: Manifest: the 14 record names that do not exist (TODO.md in 56 rows, old_val_datasets.csv in 52, candidate_datasets_2026-10-13.csv in 12, and 11 singl |
| 1 | 395 | in plan (a step of the to-do list) | * Step 6: Manifest: the 5 status wordings ('stays development', 'in development') and the 5 running rows naming no machine job, each read and corrected against  |
| 1 | 396 | in plan (a step of the to-do list) | * Step 7: The 16 finished or running rows with a pending next step: each put on this plan, reopened, or its follow-up recorded done (5 placed on plan items, 5 p |
| 1 | 397 | in plan (a step of the to-do list) | * Step 8: Instrument catalogue (INSTRUMENTS.md): each instrument entry cites its commissioning note and runtime matrices, and every number it prints is found in |
| 1 | 398 | in plan (a step of the to-do list) | * Step 9: Every number printed in the public documents (README, REFERENCE, Q&A, package README, commissioning report, SOP, reproduce README, atlas v2.1 README,  |
| 1 | 399 | in plan (a step of the to-do list) | * Step 10: Every other table audited the way FOUR_DOMAINS was: bit tables, COMMISSIONING_REPORT, STATUS, EARLIER_RESULTS, and the book's tables (COMMISSIONING_R |
| 1 | 400 | in plan (a step of the to-do list) | * Step 11: The 88 book checks whose printed number their chapter no longer prints: the cell ones fixed; the cosmology, quantum and chip ones shown to IAMPerform |
| 1 | 401 | in plan (a step of the to-do list) | * Step 16: When AWS returns: the mouse reproduction's per-number detail copied from its verdict file in S3 (results/REPRODUCE_V1/run13/out/mice/) into reproduce |
| 1 | 402 | in plan (a step of the to-do list) | * Step 21: Commission Group-A (whole blood, EPIC v1, groups within one series): synthetic first done 2026-10-20; next: a metadata search for one unspent healthy |
| 1 | 403 | in plan (a step of the to-do list) | * Step 24: Stage 0 control-probe limits set from healthy arrays (PROC-STAGE0-04): the bisulfite-conversion minimum (2026-10-20: the low readings were the metric |
| 1 | 404 | in plan (a step of the to-do list) | * Step 25: Every block explained (author rule 2026-10-20: 'they should indicate exactly what flagged it ... Imagine being a researcher and not even understandin |
| 1 | 405 | in plan (a step of the to-do list) | * Step 27: Reproductions on GitHub's machines (author 2026-10-20: GitHub first): (a) metaa_rebuild, already verified PASS from a fresh clone on AWS on 2026-10-1 |
| 1 | 406 | in plan (a step of the to-do list) | * Step 32: The information globe, DEV-SPHERE-01 construction v4: one fixed filling rule written before running, the planted test rerun with the coverage and dis |
| 1 | 407 | in plan (a step of the to-do list) | * Step 35: The two cell-mix readers compared (author list 2026-10-20: 'Do your two cell-mix readers agree?'): Stage A's neutrophil share against the IAM-Deconvo |
| 1 | 408 | status report, result or explanation; no action of its own | Completed branches |
| 1 | 409 | in plan: 33.1, now [done] | * 33.1 Exploration 8 through the one command: same-slide tare 100 % Normal in both series (1 % loss 4.09 and 3.73 spreads); laboratory tare 90.3 % and 97.0 % (1 |
| 1 | 410 | in plan: 35.1, now [done] | * 35.1 Test 8 (real atlas cells) met C1 and C3 and failed C2: a site's level dominates its place. Construction v5 takes the level out / Test 11 PASSED on the re |
| 1 | 411 | in plan: 37.2, now [done] | * 37.2 The C-Score band through the one command: MET (0.9545 not above, bar 0.95), against the committed 0.9565. One array fewer: GSM5583015 is now stopped by S |
| 1 | 412 | in plan: 38.1, now [done] | * 38.1 A change layer for each cell on the globe (healthy example blood: 0-5 of ~590 blocks per cell) |
| 1 | 413 | in plan: 38.2, now [done] | * 38.2 The report's flat sky pictures: their reference was right (Met-A's 6,000 neutrophil sites against healthy neutrophils, the C-Score's own sites); the titl |
| 1 | 414 | in plan: 39.13, now [done] | * 39.13 Instrument catalogue, line by line (author 2026-10-22): update stale entries; review every candidate instrument (some started, in development or commiss |
| 1 | 415 | in plan: 39.15, now [done] | * 39.15 No dates in public prose (author 2026-10-23: 'dates belong nowhere except the logs ... NEVER in the book or webpage or anywhere in prose'): CANON/no_dat |
| 1 | 416 | in plan: 39.1, now [done] | * 39.1 --cell-panel atlas: the Deconvolver on all 79 cell types; Met-A for each cell found (site-dropping fault fixed) |
| 1 | 417 | in plan: 39.2, now [done] | * 39.2 Which share estimate to keep: the best fit (mean absolute error 0.0057 against 0.0079 for the resampled middle on the 9 known mixtures) |
| 1 | 418 | in plan: 39.3, now [done] | * 39.3 Met-A for each cell, synthetic test 2 through the chain only: planted cell moves 92-100 % of its predicted shift; spillover up to 26 % |
| 1 | 419 | in plan: 39.4, now [done] | * 39.4 Exploration 4: identity sites tell which cell; every site is far more sensitive but cannot say which cell; identity sites kept |
| 1 | 420 | in plan: 39.7, now [done] | * 39.7 IAM-A states its own counting error on every reading (finding F38) |
| 1 | 421 | in plan: 39.9, now [done] | * 39.9 Identity sites specific to each cell (held by the cell and not by its relatives), for attribution in a mix |
| 1 | 422 | in plan: 39.11, now [done] | * 39.11 Write why Met-A reads these sites in the book's Met-A chapter and INSTRUMENTS.md (the Decoder section is written) |
| 1 | 423 | in plan: 40.2, now [done] | * 40.2 Sequencing values (--dev-seq-values) and the depth test: IAM-A Normal down to 1 % of the molecules (~3.7 M), below at 0.3 %; Deconvolver reads CD8 down t |
| 1 | 424 | in plan: 40.3, now [done] | * 40.3 Met-A on sequencing: the k-of-n expectation (expected_H_counts); depth test 2: no downward slide (0.757, 0.798, 0.897 at 31, 9, 3 reads; before 0.673, 0. |
| 1 | 425 | in plan: 40.8, now [done] | * 40.8 9 known mixtures: best fit mean absolute error 0.0057; tissues not put in at most 0.0030; resampled range holds the truth 10 of 15 |
| 1 | 426 | status report, result or explanation; no action of its own | Legend: [ ] open, [running] on GitHub now, [built] built and waiting on its test or data, [x] done, [parked] set aside on purpose. |
| 1 | 428 | status report, result or explanation; no action of its own | COMPLETED 10-23-26 07:00 |
| 1 | 430 | status report, result or explanation; no action of its own | I checked the logs against all 159 commits from the last 30 hours, every record and every finished to-do item. Everything you listed was missing from the develo |
| 1 | 431 | status report, result or explanation; no action of its own | Now in the logs: |
| 1 | 432 | status report, result or explanation; no action of its own | * Information sphere, tests 5 to 11: |
| 1 | 433 | status report, result or explanation; no action of its own | * Test 8, on the real atlas, failed one check: a site's methylation level was setting its place (88 of 100). |
| 1 | 434 | status report, result or explanation; no action of its own | * Test 9, the version with the level taken out, passed on made-up data. |
| 1 | 435 | status report, result or explanation; no action of its own | * Test 10, on the real atlas, came much closer (29 of 100). |
| 1 | 436 | status report, result or explanation; no action of its own | * Test 11 is written and not yet run. |
| 1 | 437 | in plan (a step of the to-do list) | * Depth: |
| 1 | 438 | status report, result or explanation; no action of its own | * IAM-A reads Normal down to 1% of the molecules (about 3.7 million). The Deconvolver reads correctly down to 3%. |
| 1 | 439 | status report, result or explanation; no action of its own | * The read-count fix removed Met-A's downward slide. |
| 1 | 440 | status report, result or explanation; no action of its own | * A 5% loss in CD4 T cells stands out at 3 reads per site (4.6 times its error), 10 reads (6.2) and 30 reads (8.7). |
| 1 | 441 | status report, result or explanation; no action of its own | * Mixtures and draws: |
| 1 | 442 | status report, result or explanation; no action of its own | * The 9 known mixtures were found with an average error of 0.0057. |
| 1 | 443 | status report, result or explanation; no action of its own | * In the constructed draws, liver DNA is found from 0.5% of the plasma, colon only from 5%. Below about 2%, the limit is the split into cells, not the depth. |
| 1 | 444 | status report, result or explanation; no action of its own | * Every cell: the Deconvolver on all 79 cells, Met-A for each cell, each cell's globe layer, and explorations 3 to 8. |
| 1 | 445 | status report, result or explanation; no action of its own | * Met-A against the atlas: |
| 1 | 446 | status report, result or explanation; no action of its own | * The first sealed test was not commissioned (2 of 3 bars). |
| 1 | 447 | status report, result or explanation; no action of its own | * Both monocyte series read Normal through the one command with the same-slide tare. |
| 1 | 448 | status report, result or explanation; no action of its own | * The second sealed test is written. |
| 1 | 449 | status report, result or explanation; no action of its own | * Exploration 7 never ran. Its script skipped the one command, which now always applies the self-tare it was meant to test. It's recorded as replaced. |
| 1 | 450 | status report, result or explanation; no action of its own | * Comparisons and reproductions: ten of ten reproductions pass on GitHub. The C-Score band holds again. Stage A and the Deconvolver agree on the 278 bloods (r 0 |
| 1 | 451 | status report, result or explanation; no action of its own | * The C-Score family: the plan, including IAM-A's C-Score version 2 as a test of the law region by region. The C-Score for every cell and for the whole genome i |
| 1 | 452 | status report, result or explanation; no action of its own | * Early detection: EPIC-Italy, the public-data survey, the plasma and one-draw build, and stool. |
| 1 | 453 | status report, result or explanation; no action of its own | * Also logged: the chain-only rule and its lessons, one source for every document, the globe patches, the documents, and today's mistakes. |
| 1 | 454 | status report, result or explanation; no action of its own | Still running on GitHub: |
| 1 | 455 | status report, result or explanation; no action of its own | * the EPIC-Italy smoke run |
| 1 | 456 | status report, result or explanation; no action of its own | * each cell's healthy spread (exploration 8) |
| 1 | 457 | status report, result or explanation; no action of its own | * the C-Score synthetic test on four arrays |
| 1 | 458 | status report, result or explanation; no action of its own | * the second Met-A seal's synthetic test |
| 1 | 459 | status report, result or explanation; no action of its own | * the 58 plasma arrays |
| 1 | 460 | status report, result or explanation; no action of its own | * Group-A's last shard |
| 1 | 461 | status report, result or explanation; no action of its own | Next: the globe quality you noticed. I'll draw one sample both ways, the current 6,000-site layers and the full atlas background with the change patches on top, |
| 1 | 463 | status report, result or explanation; no action of its own | Still running. I'll report each result as it lands, starting with the EPIC-Italy smoke run. |
| 1 | 467 | status report, result or explanation; no action of its own | IAM to-do list, October 22, 2026, 11:10 PM Pacific |
| 1 | 471 | status report, result or explanation; no action of its own | Running on GitHub: |
| 1 | 472 | status report, result or explanation; no action of its own | Branch	What |
| 1 | 473 | in plan: 37.1, now [running] | 37.1	Group-A's 278 bloods: lost shard rerunning, then the comparison with the committed result |
| 1 | 474 | in plan: 37.2, now [done] | 37.2	The C-Score band: last shard |
| 1 | 475 | in plan: 33.1, now [done] | 33.1	Met-A on the atlas, children's monocytes: last shard |
| 1 | 476 | in plan: 39.9, now [done] | 39.9	Sites held by one cell alone |
| 1 | 477 | in plan: 40.5, now [open] | 40.5	Constructed draws, 3 depths (3 machines) |
| 1 | 478 | in plan: 40.11, now [running] | 40.11	Planted losses on sequencing, 4 depths (4 machines) |
| 1 | 479 | in plan: 40.7, now [running] | 40.7	58 plasma arrays |
| 1 | 480 | in plan: 35.1, now [done] | 35.1	Information sphere test 10 on real atlas cells |
| 1 | 481 | in plan: 38.1, now [done] | 38.1	Clustered plants on each cell's globe |
| 1 | 482 | status report, result or explanation; no action of its own | Where the list stands: 17 steps done and 23 open. The 33 forks are 10 done, 8 running, 3 built, 11 open and 1 parked (GSE186888, held unread). |
| 1 | 483 | status report, result or explanation; no action of its own | New results since your last check: |
| 1 | 484 | status report, result or explanation; no action of its own | * Information sphere test 9 passed once each site's level is taken out, so test 10 on the real cells is running. |
| 1 | 485 | status report, result or explanation; no action of its own | * The 9 known mixtures: tissue DNA at 3.5–10% in blood was found at the right size, off by 0.0057 on average, and nothing that wasn't put in showed up above 0.0 |
| 1 | 486 | status report, result or explanation; no action of its own | * Met-A against the atlas on adult monocytes, through the one command: with healthy arrays from the same slide as the reference, every array read Normal and a 1 |
| 1 | 487 | status report, result or explanation; no action of its own | The list file has every step and fork, grouped done, running and open, with the open ones in priority order, humans first. |
| 1 | 489 | status report, result or explanation; no action of its own | IAM to-do list, October 22, 2026, 11:10 PM Pacific |
| 1 | 490 | status report, result or explanation; no action of its own | Built from the repo's plan (CANON/plan.json, 40 steps and every branch). Numbers like 40.3 are branches: forks a step grew. The public page with the same list i |
| 1 | 491 | status report, result or explanation; no action of its own | Count: 17 steps done, 23 open. Of the 33 branches: 10 done, 8 running, 3 built (one of them, 38.1, also has its test running), 11 open, 1 parked. |
| 1 | 493 | status report, result or explanation; no action of its own | Running on GitHub right now |
| 1 | 494 | status report, result or explanation; no action of its own | 14 of 20 machines busy, 6 open. |
| 1 | 495 | status report, result or explanation; no action of its own | Branch	What it is |
| 1 | 496 | in plan: 37.1, now [running] | 37.1	Group-A's 278 bloods through the one command: the lost shard is rerunning, then pass 2 and the number-by-number comparison with the committed result |
| 1 | 497 | in plan: 37.2, now [done] | 37.2	The C-Score band through the one command: last shard of pass 2 |
| 1 | 498 | in plan: 33.1, now [done] | 33.1	Met-A against the atlas, children's monocytes (GSE168779): last shard. The adult monocytes are done |
| 1 | 499 | in plan: 39.9, now [done] | 39.9	Exploration 7: sites held by one cell alone, for a loss across a cell's whole pattern |
| 1 | 500 | in plan: 40.5, now [open] | 40.5	Constructed draws: liver, vessel and colon DNA at 0 to 5 % of the plasma, at 10, 30 and 60 reads per site (3 machines) |
| 1 | 501 | in plan: 40.11, now [running] | 40.11	Planted losses read on sequencing at 3, 10, 30 and 60 reads per site (4 machines) |
| 1 | 502 | in plan: 40.7, now [running] | 40.7	58 plasma arrays on the full atlas |
| 1 | 503 | in plan: 35.1, now [done] | 35.1	Information sphere test 10: construction v5 on IAM-Atlas v2.1's real healthy cells (test 9 passed) |
| 1 | 504 | in plan: 38.1, now [done] | 38.1	Clustered plants on each cell's globe layer |
| 1 | 505 | status report, result or explanation; no action of its own | Done today (October 22) |
| 1 | 506 | status report, result or explanation; no action of its own | * Mouse reproduction on GitHub: PASS (step 15). All 10 of 10 reproductions now run on GitHub. |
| 1 | 507 | status report, result or explanation; no action of its own | * The one-click reproduction workflow for researchers (reproduce_all). |
| 1 | 508 | status report, result or explanation; no action of its own | * The two cell-mix readers agree on 278 bloods (step 34; correlation 0.960). |
| 1 | 509 | status report, result or explanation; no action of its own | * Every cell in a mix: the Deconvolver reads all 79 atlas cell types; Met-A reads each cell found (39.1, 39.3). Identity sites kept for Met-A for each cell (39. |
| 1 | 510 | status report, result or explanation; no action of its own | * Known mixtures: tissue DNA at 3.5 to 10 % in blood, found at the right size (mean error 0.0057), nothing false above 0.0030 (40.8). The best fit is the estima |
| 1 | 511 | status report, result or explanation; no action of its own | * Sequencing: IAM-A reads Normal down to 1 % of a sample's molecules (40.2) and states its own counting error (39.7). Met-A's read-count correction stops the sl |
| 1 | 512 | status report, result or explanation; no action of its own | * Met-A against the atlas, adult monocytes, through the one command: the same-slide tare is 100 % Normal and a 1 % loss stands out at 4.09 spreads; the laborato |
| 1 | 513 | status report, result or explanation; no action of its own | * 450K arrays: the atlas readers run on them, flagged (27.1, code path only). |
| 1 | 514 | status report, result or explanation; no action of its own | * The flat sky pictures' titles now name their sites (38.2). |
| 1 | 515 | status report, result or explanation; no action of its own | * Information sphere test 9 PASSED: with each site's level taken out (construction v5), all three checks met on the synthetic test (35.1). |
| 1 | 516 | status report, result or explanation; no action of its own | * New chain tools, development: the one-draw runner (40.1), constructed draws with a known answer, count files from plasma sequencing, plants made by the chain, |
| 1 | 517 | status report, result or explanation; no action of its own | * The plain-language computer-and-qubit key for M, φ and ε in FOUR_DOMAINS.md. |
| 1 | 518 | status report, result or explanation; no action of its own | * The push gate now refuses any Python file that does not compile (lesson C36). |
| 1 | 519 | status report, result or explanation; no action of its own | * Breadcrumbs: every branch is tracked in the plan with its next action. |
| 1 | 520 | status report, result or explanation; no action of its own | Done before today (steps 1 to 10, 20, 23, 24, 26, 31): the showcase, the manifest and number audits, the instrument catalogue, the book's number checks, Group-A |
| 1 | 522 | status report, result or explanation; no action of its own | Open, in priority order (humans first) |
| 1 | 523 | status report, result or explanation; no action of its own | A. Commissioning and the chain |
| 1 | 524 | in plan (a step of the to-do list) | 1. Step 37: everything through the one command. 37.1 and 37.2 running; 37.3: 10 scripts still outside the one command (listed in REPLICATE.md). |
| 1 | 525 | status report, result or explanation; no action of its own | 2. Step 33: Met-A against IAM-Atlas v2.1, sealed. After 33.1: find a series with at least 3 healthy arrays on most slides and seal the same-slide tare. |
| 1 | 526 | status report, result or explanation; no action of its own | 3. Step 38: commission the IAM-Globe. 38.1 running; 38.3: the globe counts from Group-A's rerun, then synthetic plants, then a sealed test. |
| 1 | 527 | in plan (a step of the to-do list) | 4. Step 21: every major blood cell scored in whole blood. 39.5: each cell's healthy spread from Group-A's 278 bloods, after 37.1. |
| 1 | 528 | status report, result or explanation; no action of its own | 5. Step 27: 450K arrays. 27.1: a real spent 450K array with raw IDATs. |
| 1 | 529 | in plan (a step of the to-do list) | 6. Step 30: what the panel cannot explain (the Deconvolver's leftover as a globe layer). Not started. |
| 1 | 530 | status report, result or explanation; no action of its own | B. One blood draw (step 40) and every cell in a mix (step 39) |
| 1 | 531 | status report, result or explanation; no action of its own | 7. 40.5, 40.7, 40.11: running. |
| 1 | 532 | status report, result or explanation; no action of its own | 8. 40.4: the depth test on real plasma. 40.6: open matched buffy coat and plasma with molecules, for IAM-A. 40.9: a fresh unread series of blood drawn before di |
| 1 | 533 | status report, result or explanation; no action of its own | 9. 40.10 parked: GSE186888 (23 healthy donors, matched white cells and plasma) stays wholly unread for the one-draw test that counts. |
| 1 | 534 | in plan (a step of the to-do list) | 10. 39.6: the person's own buffy coat as the reference for their plasma (hypothesis). 39.8: test Met-A's site windows reaching 1 and 0. 39.10: the every-site re |
| 1 | 535 | status report, result or explanation; no action of its own | C. Physics |
| 1 | 536 | status report, result or explanation; no action of its own | 11. Step 14 / 14.1, the firm rate (top physics item). The GitHub route is ready for mouse samples: one sample took 4.25 hours on one machine. Two things before  |
| 1 | 537 | status report, result or explanation; no action of its own | 12. Steps 35, 29: the information sphere and the IAM-A globe. Test 10 running; then the sky tools (cross-spectrum of two readings, the dipole). The whole-genome |
| 1 | 538 | status report, result or explanation; no action of its own | D. Documents |
| 1 | 539 | status report, result or explanation; no action of its own | 13. Step 25: the two chain documents. Findings still open: F14, F23, F36. |
| 1 | 540 | status report, result or explanation; no action of its own | 14. Step 22: documents for Group-A, every blood cell scored and the firm rate. "Lots of updating as the last thing." |
| 1 | 541 | status report, result or explanation; no action of its own | 15. Step 32: the Globe Explorer chapter (drafted; goes into the book when the Explorer is complete). |
| 1 | 542 | status report, result or explanation; no action of its own | 16. Step 13: the guidebook for the sky maps. |
| 1 | 543 | status report, result or explanation; no action of its own | 17. Steps 11 and 12: the book's and the researcher package's Zenodo records. |
| 1 | 544 | status report, result or explanation; no action of its own | E. Machine downtime only |
| 1 | 545 | salmon: in plan (parked) | 18. Step 36: salmon (coho hatchery and wild, PRJNA389610). |
| 1 | 546 | status report, result or explanation; no action of its own | F. Waiting on AWS |
| 1 | 547 | status report, result or explanation; no action of its own | 19. Steps 16, 17, 18: the clean-clone check after the spelling change, the billing blocks, and removing idle charges. |
| 1 | 548 | status report, result or explanation; no action of its own | 20. Step 19: independence from AWS (most of it done; a data register exists). |
| 1 | 550 | status report, result or explanation; no action of its own | Private (not on the public plan) |
| 1 | 551 | private whiteboard: private, correctly off the public plan | Private ideas stay on the private whiteboard. |
| 1 | 555 | status report, result or explanation; no action of its own | (The firm-rate derivation, the lever we talked about: 16 machines, about 5 hours. |
| 1 | 556 | status report, result or explanation; no action of its own | * What it is: recounting the five sets of eraser-removed and matched control samples, 16 samples in all, with the chain's commissioned counter. Then the sealed  |
| 1 | 557 | status report, result or explanation; no action of its own | * The timing test already ran on GitHub: one mouse sample, 10 million read pairs, 2 threads, finished in 4.25 hours with nothing failing. That includes 1.7 hour |
| 1 | 558 | status report, result or explanation; no action of its own | * So the full run is one machine per sample, all at once: 16 machines for about 4 to 5 hours, then one short job to score. |
| 1 | 559 | status report, result or explanation; no action of its own | * One check first: four of the samples are human (hg19), and the timing test was mouse. A one-sample human timing test should come first so nothing hits the 6-h |
| 1 | 560 | status report, result or explanation; no action of its own | Because it takes 16 of the 20 machines at once, I'd run it at night, when nothing else needs them. You set it as a fallback for if AWS doesn't come back, but wi |
| 1 | 569 | status report, result or explanation; no action of its own | IAM to-do list, 2026-10-22, 7:05 am |
| 1 | 572 | status report, result or explanation; no action of its own | The FULL CHAIN EXPLORATION BUFFY COAT/cfDNA: |
| 1 | 574 | status report, result or explanation; no action of its own | cfDNA comes mostly from the body's own cells dying. Every day billions of cells die and are cleared, and their DNA is cut into short pieces that float in the bl |
| 1 | 575 | status report, result or explanation; no action of its own | * Where it comes from in a healthy person: mostly blood cells (neutrophils, monocytes, lymphocytes, the cells that make platelets and red cells), then the linin |
| 1 | 576 | status report, result or explanation; no action of its own | * In a person with cancer: some of it comes from the tumor (circulating tumor DNA), usually a small share. Other organs under strain add their own. |
| 1 | 577 | status report, result or explanation; no action of its own | What plasma is: the liquid part of blood after the cells are spun out: water, proteins, salts, hormones, and the cfDNA pieces. A good plasma sample has no whole |
| 1 | 578 | status report, result or explanation; no action of its own | Yes, a sample that gives both already exists: one ordinary blood tube. cfDNA studies routinely spin one tube into two parts: |
| 1 | 579 | status report, result or explanation; no action of its own | 1. Plasma: the cfDNA pieces. |
| 1 | 580 | status report, result or explanation; no action of its own | 2. Buffy coat: the thin white layer of white cells between the plasma and the red cells. It gives long, intact DNA from the person's own living blood cells. |
| 1 | 581 | status report, result or explanation; no action of its own | Which instruments can read each part: |
| 1 | 582 | status report, result or explanation; no action of its own | Part of the tube	Run on	Met-A (and Deconvolver, C-Score, globes)	IAM-A |
| 1 | 583 | status report, result or explanation; no action of its own | Buffy coat (whole cells)	array	yes	no (an array gives no molecules) |
| 1 | 584 | status report, result or explanation; no action of its own | Buffy coat	sequencing	yes	yes: long molecules, its best case |
| 1 | 585 | status report, result or explanation; no action of its own | Plasma (cfDNA)	sequencing	yes, with enough depth	limited by the short pieces; the instrument will tell us how far |
| 1 | 586 | status report, result or explanation; no action of its own | So one draw, split and sequenced, can give every score: Met-A and IAM-A on the cells, and the same on the plasma. |
| 1 | 587 | buffy coat as the plasma's own reference: in plan (39.6) | That split may also be worth more than the scores themselves, and it fits your self-reference idea. Most of a person's cfDNA comes from their own blood cells, a |
| 1 | 588 | status report, result or explanation; no action of its own | Next I'll search for public data where the same people have both buffy coat and plasma sequenced, which would let us explore this on real samples without spendi |
| 1 | 591 | status report, result or explanation; no action of its own | Every instrument on one blood draw, run with one click on raw public data, would make researchers notice, because they can rerun it and check it themselves. I'v |
| 1 | 592 | status report, result or explanation; no action of its own | What it would look like: one tube of blood, spun into buffy coat and plasma, both sequenced. One GitHub workflow then runs the one command on both halves: |
| 1 | 593 | status report, result or explanation; no action of its own | 1. Buffy coat: the Deconvolver splits it across the whole atlas. Met-A scores every cell with enough share, the C-Score reads how the changes are laid out, IAM- |
| 1 | 594 | status report, result or explanation; no action of its own | 2. Plasma: split the same way, then read against the person's own buffy coat. Whatever their own blood cells can't explain is matched to the atlas's tissues and |
| 1 | 595 | status report, result or explanation; no action of its own | 3. One report for the draw, with every reading either commissioned or flagged. |
| 1 | 596 | status report, result or explanation; no action of its own | What's built: |
| 1 | 597 | status report, result or explanation; no action of its own | * Cells: the Deconvolver on the full atlas panel (added just now), and Met-A for each cell. |
| 1 | 598 | status report, result or explanation; no action of its own | * Spatial and molecular readings: the C-Score and IAM-A. |
| 1 | 599 | status report, result or explanation; no action of its own | * Globes: the globe for arrays, the IAM-A globe for molecules. |
| 1 | 600 | status report, result or explanation; no action of its own | * Test tooling: planting a loss inside the chain itself. |
| 1 | 601 | status report, result or explanation; no action of its own | What's missing, in the order I'll build it: |
| 1 | 602 | status report, result or explanation; no action of its own | 1. Sequencing reads to site values. The one command reads sequencing data only for IAM-A today, so Met-A and the Deconvolver can't yet read a sequenced buffy co |
| 1 | 603 | status report, result or explanation; no action of its own | 2. Plasma at intake, read against the person's own buffy coat. |
| 1 | 604 | status report, result or explanation; no action of its own | 3. IAM-A's depth rule (finding F38): every fragment long enough is read, and the instrument reports its own counting error. |
| 1 | 605 | status report, result or explanation; no action of its own | 4. The one-draw workflow: one button for both halves and the joint report. |
| 1 | 606 | status report, result or explanation; no action of its own | 5. The data: an open public set where the same person's buffy coat and plasma were sequenced genome-wide. That may be the hardest part. The large matched sets I |
| 1 | 607 | status report, result or explanation; no action of its own | What could make it impossible, and only the instruments can tell us: whether plasma at public sequencing depths carries enough reads per site, and enough long f |
| 1 | 611 | status report, result or explanation; no action of its own | Today's branches, all on the plan now: |
| 1 | 612 | status report, result or explanation; no action of its own | Step	Open branches |
| 1 | 613 | status report, result or explanation; no action of its own | 40, one draw	40.1 the one-draw runner (built; waits on a real draw). 40.3 Met-A on sequencing: the read-count correction. 40.4 the depth test on real plasma. 40 |
| 1 | 614 | in plan (a step of the to-do list) | 39, every cell in a mix	39.2 which share estimate to keep, after the mixtures. 39.4 the 6,000 identity sites against all sites (running), then writing the ratio |
| 1 | 615 | status report, result or explanation; no action of its own | 38, the globe	38.1 the per-cell layer, tested with a clustered plant. 38.2 the flat picture's wrong title. 38.3 globe commissioning on the 278 bloods. |
| 1 | 616 | status report, result or explanation; no action of its own | 37, chain only	37.1 Group-A (running; one lost shard to rerun). 37.2 the C-Score band (running). 37.3 the 10 scripts still outside the chain. |
| 1 | 617 | status report, result or explanation; no action of its own | 33, sealed Met-A	33.1 the second pass on both monocyte series (running). |
| 1 | 618 | status report, result or explanation; no action of its own | 27, 450K	27.1 a real 450K array with raw IDATs. |
| 1 | 619 | status report, result or explanation; no action of its own | Branches already done are on the plan too, with their records: the depth test, the full-atlas panel, Met-A for each cell (test 2), and IAM-A's counting error. |
| 1 | 620 | status report, result or explanation; no action of its own | The private ideas stay on the private whiteboard, not the public plan. |
| 1 | 629 | status report, result or explanation; no action of its own | IAM-A Globe "only changes too big to be chance" |
| 1 | 632 | status report, result or explanation; no action of its own | Read from CANON/plan.json (commit 82d6c6bb). 34 steps: 14 done, 20 open. Steps 33 and 34 were added this morning from your earlier lists. |
| 1 | 633 | status report, result or explanation; no action of its own | Running now on GitHub |
| 1 | 634 | status report, result or explanation; no action of its own | What	Step	Where it stands |
| 1 | 635 | status report, result or explanation; no action of its own | The mouse reproduction, all 38 read sets	15	16 of 19 alignment jobs done, all successful; 3 still running, then the comparison with the committed numbers |
| 1 | 636 | status report, result or explanation; no action of its own | Open, can be done now on GitHub, in the order I'll work them |
| 1 | 637 | status report, result or explanation; no action of its own | Order	Step	What	Where it stands |
| 1 | 638 | status report, result or explanation; no action of its own | 1	33	Met-A against IAM-Atlas v2.1, sealed test	Not started. Exploration 6 gave the rule: tare on healthy cells of the same type on the same slide. Next: find an |
| 1 | 639 | pair globe on the Explorer; direction of change; own two normals: in plan (step 'The genome on a globe with layers') | 2	28	Tumor against own normal in the Globe Explorer	The chain draws it (pair mode) and 191 pairs are scored as pictures (DEV-TISSUE-GLOBE-01), but it is not on  |
| 1 | 640 | status report, result or explanation; no action of its own | 3	34	The two cell-mix readers compared (Stage A vs Deconvolver, Group-A's 263 people)	Not started. From your 10-20 list. |
| 1 | 641 | status report, result or explanation; no action of its own | 4	31	The information globe, construction v4 (synthetic)	Not started. v3 met 4 of 6 bars: covered 46.9 % of the globe (bar 90 %), distances 0.567 (bar 0.7). |
| 1 | 642 | status report, result or explanation; no action of its own | 5	28	Globe layers with highlighting and shuffled nulls; then the whole-genome globe (about 28 million CpGs, from the sequencing samples)	Not started. |
| 1 | 643 | status report, result or explanation; no action of its own | 6	29	The IAM-A globe	Not started. The counter must keep per-block counts first. |
| 1 | 644 | status report, result or explanation; no action of its own | 7	30	What the panel can't explain (Deconvolver leftover)	Not started. Synthetic first. |
| 1 | 645 | in plan (a step of the to-do list) | 8	21	Every major blood cell scored in whole blood	Met-A on the atlas now works for sorted monocytes (exploration 6). Left: the per-cell stage in the chain and e |
| 1 | 646 | Decoder findings: added as 61.8 | 9	25	Operator Guide and Decoder: the last items	Every one of 286 functions has a Decoder entry (gated). Left: findings F14, F23, F36 open and F7, F15, F17, F24, |
| 1 | 647 | status report, result or explanation; no action of its own | 10	27	450K arrays read by the chain	Waits for a 450K self-tare rule, then a sealed test. |
| 1 | 648 | status report, result or explanation; no action of its own | 11	32	The Globe Explorer chapter in the book	Not written. Written once steps 28 to 31 are on the Explorer, so it describes the finished thing. |
| 1 | 649 | status report, result or explanation; no action of its own | 12	22	Documentation pass: book, page, Q&A, catalogue, commissioning report	Partly done as we go. Done last, after 21 and 33. |
| 1 | 650 | status report, result or explanation; no action of its own | 13	11, 12	Zenodo records for the book and the researcher package	After step 22. |
| 1 | 651 | status report, result or explanation; no action of its own | Open, waiting for AWS |
| 1 | 652 | status report, result or explanation; no action of its own | Step	What |
| 1 | 653 | in plan (a step of the to-do list) | 14	IAM-A's firm rate, derived again on the commissioned counter |
| 1 | 654 | status report, result or explanation; no action of its own | 16	When AWS returns |
| 1 | 655 | status report, result or explanation; no action of its own | 17	When AWS returns |
| 1 | 656 | status report, result or explanation; no action of its own | 18	When AWS returns |
| 1 | 657 | status report, result or explanation; no action of its own | 19	Independence from AWS |
| 1 | 658 | S3-only files to Zenodo: in plan ('Independence from AWS'); replication is Zenodo and GitHub only | Step 15 (mouse detail from S3) closes without AWS if the GitHub run passes. Step 19's part 1 is done; the 71 GB of our own work files in S3 still need depositin |
| 1 | 659 | status report, result or explanation; no action of its own | Done |
| 1 | 660 | status report, result or explanation; no action of its own | Step	What	Done |
| 1 | 661 | status report, result or explanation; no action of its own | 1	Showcase scenario 4	2026-10-19 |
| 1 | 662 | status report, result or explanation; no action of its own | 2	Showcase README	2026-10-19 |
| 1 | 663 | status report, result or explanation; no action of its own | 3	Mouse reproduction (run 13)	2026-10-19 |
| 1 | 664 | status report, result or explanation; no action of its own | 4	Manifest	2026-10-19 |
| 1 | 665 | status report, result or explanation; no action of its own | 5	Manifest	2026-10-19 |
| 1 | 666 | in plan (a step of the to-do list) | 6	The 16 finished or running rows with a pending next step	2026-10-19 |
| 1 | 667 | status report, result or explanation; no action of its own | 7	Instrument catalogue (INSTRUMENTS.md)	2026-10-19 |
| 1 | 668 | in plan (a step of the to-do list) | 8	Every number printed in the public documents (README, REFERENCE, Q&A, package README, commissioning report, SO	2026-10-19 |
| 1 | 669 | in plan (a step of the to-do list) | 9	Every other table audited the way FOUR_DOMAINS was	2026-10-19 |
| 1 | 670 | in plan (a step of the to-do list) | 10	The 88 book checks whose printed number their chapter no longer prints	2026-10-19 |
| 1 | 671 | in plan (a step of the to-do list) | 20	Commission Group-A (whole blood, EPIC v1, groups within one series)	2026-10-20 |
| 1 | 672 | in plan (a step of the to-do list) | 23	Stage 0 control-probe limits set from healthy arrays (PROC-STAGE0-04)	2026-10-21 |
| 1 | 673 | status report, result or explanation; no action of its own | 24	Every block explained	2026-10-20 |
| 1 | 674 | status report, result or explanation; no action of its own | 26	Reproductions on GitHub's machines	2026-10-21 |
| 1 | 675 | status report, result or explanation; no action of its own | Done overnight (inside the steps above) |
| 1 | 676 | status report, result or explanation; no action of its own | * Every layer of every globe has a working slider: 217 layers checked in a browser on the live site. |
| 1 | 677 | status report, result or explanation; no action of its own | * Release check: 34 of 34 pass, including V13 on every globe layer. |
| 1 | 678 | status report, result or explanation; no action of its own | * Met-A against the atlas, exploration 6: 67 healthy children's monocytes from an independent laboratory read 100 % Normal against the atlas's adult monocytes w |
| 1 | 679 | status report, result or explanation; no action of its own | * X and Y in IAM-A's held error (finding F37): IAM-A moves by -0.32 % on average; the line and band stay. |
| 1 | 680 | status report, result or explanation; no action of its own | * Colon controls matched on the slide light as much as across slides. In colon the person's own normal is the reference that means something. |
| 1 | 681 | status report, result or explanation; no action of its own | * Ruled out as the cause of the sorted cells' spread: sorting purity, and site weighting. |
| 1 | 682 | status report, result or explanation; no action of its own | Your earlier lists, line by line |
| 1 | 683 | status report, result or explanation; no action of its own | From your lists	Now |
| 1 | 684 | status report, result or explanation; no action of its own | Met-A on 109 sorted cells (GSE184269)	Done; spread too wide when tared across slides; led to exploration 6 |
| 1 | 685 | status report, result or explanation; no action of its own | Tumor globes: 3 colon, 3 breast, a control	Done as pictures (then all 191 pairs); not on the Explorer yet (order 2) |
| 1 | 686 | status report, result or explanation; no action of its own | Mouse memory test	Done: 1 thread fits, counts identical; full run running |
| 1 | 687 | status report, result or explanation; no action of its own | Ideas: direction of change at identity sites; leaving out the X; a person's own two normals	Leaving out the X: done (run 2). The other two: open, step 28 |
| 1 | 688 | status report, result or explanation; no action of its own | Met-A atlas: tighter sites; same-slide references; naive vs broad types; donor age; lab-specific sites	Tighter sites: tested, no effect. Same-slide references:  |
| 1 | 689 | status report, result or explanation; no action of its own | Full-genome globes, address and information versions	Open: steps 28 and 31 |
| 1 | 690 | status report, result or explanation; no action of its own | Reproductions on GitHub (9 of 10)	Done; the 10th (mouse) running |
| 1 | 691 | status report, result or explanation; no action of its own | Noise maps	Done, on the Explorer |
| 1 | 692 | status report, result or explanation; no action of its own | Stage 0 bisulfite and hybridization fixes; limits from healthy arrays	Done (step 23) |
| 1 | 695 | IAM-A globe design: in plan (step 'The IAM-A globe'); design carried in this audit | No, the IAM-A globe isn't built yet. It's step 29, not started. The counter only keeps genome-wide totals right now, so it first has to be changed to keep count |
| 1 | 696 | IAM-A globe design: in plan (step 'The IAM-A globe'); design carried in this audit | How I'd draw it. I haven't settled the design yet. This is my proposal, worked out from how IAM-A itself measures: |
| 1 | 697 | IAM-A globe design: in plan (step 'The IAM-A globe'); design carried in this audit | 1. The healthy map comes from the sample alone. IAM-A takes the sample's loose share, and the self-reference line turns that into the held error the sample shou |
| 1 | 698 | IAM-A globe design: in plan (step 'The IAM-A globe'); design carried in this audit | 2. The measured map. For each region of the genome, take the sample's isolated errors on held molecules, divided by its chances to make one. That's the same cou |
| 1 | 699 | IAM-A globe design: in plan (step 'The IAM-A globe'); design carried in this audit | 3. The departures. Each region's measured error is compared with the predicted one, scaled by its counting noise. With few molecules, a region's count wobbles b |
| 1 | 700 | IAM-A globe design: in plan (step 'The IAM-A globe'); design carried in this audit | 4. The slider. It fades only the healthy regions, the same as on the other globes, so the departures stand alone and their count never moves. |
| 1 | 701 | IAM-A globe design: in plan (step 'The IAM-A globe'); design carried in this audit | 5. A second layer. The sample's loose-share map, region by region, so you can see where the order is held loosely. |
| 1 | 702 | IAM-A globe design: in plan (step 'The IAM-A globe'); design carried in this audit | Why this way: it is IAM-A's own measurement drawn region by region, rather than anything new. No other person enters it, and nothing is fitted. |
| 1 | 703 | IAM-A globe design: in plan (step 'The IAM-A globe'); design carried in this audit | Two things to settle first: |
| 1 | 704 | IAM-A globe design: in plan (step 'The IAM-A globe'); design carried in this audit | * Where each region goes on the globe. The sequencing data uses a different genome numbering than the arrays the globe was laid out for. The sequencing regions  |
| 1 | 705 | IAM-A globe design: in plan (step 'The IAM-A globe'); design carried in this audit | * Synthetic first: plant a held-error rise in one stretch of genome in a healthy sample, and check that the globe lights there and stays dark on the unplanted s |
| 1 | 714 | status report, result or explanation; no action of its own | OLD BELOW: |
| 1 | 717 | status report, result or explanation; no action of its own | IAM CHAIN IN ORDER: 10-21-26 |
| 1 | 723 | status report, result or explanation; no action of its own | IAM to-do list |
| 1 | 724 | status report, result or explanation; no action of its own | Read from CANON/plan.json in the repository (commit 227ce8b1), 2026-10-21, 3:20 PM Pacific. 32 steps: 14 done, 18 open. |
| 1 | 725 | status report, result or explanation; no action of its own | Running now on GitHub |
| 1 | 726 | status report, result or explanation; no action of its own | What	Step |
| 1 | 727 | status report, result or explanation; no action of its own | Met-A against IAM-Atlas v2.1 on 109 sorted cells from an independent laboratory	21 |
| 1 | 728 | status report, result or explanation; no action of its own | The tumor-against-own-normal globes: 3 colon pairs, 3 breast pairs, a two-person control, each screenshotted in a browser	13, 28 |
| 1 | 729 | status report, result or explanation; no action of its own | The mouse memory test (decides how the last reproduction splits across GitHub machines)	15 |
| 1 | 730 | status report, result or explanation; no action of its own | Open, can be done now |
| 1 | 731 | status report, result or explanation; no action of its own | #	Step	Where it stands |
| 1 | 732 | status report, result or explanation; no action of its own | 11	The book's Zenodo record	Waits for step 22 (the documentation pass). |
| 1 | 733 | status report, result or explanation; no action of its own | 12	The researcher package's Zenodo record	Waits for step 11. |
| 1 | 734 | status report, result or explanation; no action of its own | 13	Sky-map additions: noise maps, the 3D globe, report globe, guidebook	Done: the 3D atlas globe (79 cells), the noise globe, the report's globe with its fixed  |
| 1 | 735 | status report, result or explanation; no action of its own | 19	Independence from AWS (every input has a home outside our S3)	Done: part 1, the data register in the push gate. Left: parts 2 and 3 need AWS (deposit our S3- |
| 1 | 736 | in plan (a step of the to-do list) | 21	Every major blood cell scored in whole blood	Running now: Met-A against the atlas on 109 sorted cells from an independent laboratory (synthetic test 1 met bo |
| 1 | 737 | status report, result or explanation; no action of its own | 22	Documents updated for Group-A, per-cell scoring and the firm rate	Waits for steps 14 and 21; partly done as we go (glossary, Q&A, book Part 6, Introduction). |
| 1 | 738 | status report, result or explanation; no action of its own | 25	The Operator Guide and the Operations Manual with its Chain Decoder	Done: both documents renamed and built; every one of 279 chain functions has a Decoder en |
| 1 | 739 | status report, result or explanation; no action of its own | 27	450K arrays read by the chain	Waits: the self-tare rule for 450K, then a sealed test. |
| 1 | 740 | status report, result or explanation; no action of its own | 28	The genome on a globe with layers (and the whole genome)	Done: the noise globe, the Explorer, two reference globes. Left: layers with highlighting and shuffl |
| 1 | 741 | status report, result or explanation; no action of its own | 29	The IAM-A globe	Not started. The counter must keep per-region counts first. |
| 1 | 742 | status report, result or explanation; no action of its own | 30	What the panel can't explain (cells the panel leaves out)	Not started. Synthetic first. |
| 1 | 743 | status report, result or explanation; no action of its own | 31	The information globe, construction v4	Not started. |
| 1 | 744 | status report, result or explanation; no action of its own | 32	A Globe Explorer chapter in the book	When the Explorer is complete. |
| 1 | 745 | status report, result or explanation; no action of its own | Open, waiting for AWS |
| 1 | 746 | status report, result or explanation; no action of its own | #	Step	Where it stands |
| 1 | 747 | status report, result or explanation; no action of its own | 14	IAM-A firm rate recounted on the commissioned counter	Needs AWS: the molecules are in our S3. |
| 1 | 748 | status report, result or explanation; no action of its own | 15	Mouse reproduction detail copied from S3	Needs AWS. The memory test for a GitHub version is running now. |
| 1 | 749 | status report, result or explanation; no action of its own | 16	Clean-clone check after the spelling change	Needs AWS for the clean-clone machine. |
| 1 | 750 | status report, result or explanation; no action of its own | 17	Billing records filled in from AWS	Needs AWS billing. |
| 1 | 751 | status report, result or explanation; no action of its own | 18	Remove idle AWS charges	Needs AWS. |
| 1 | 752 | status report, result or explanation; no action of its own | Done |
| 1 | 753 | status report, result or explanation; no action of its own | #	Step	Done |
| 1 | 754 | status report, result or explanation; no action of its own | 1	Showcase: three healthy colon samples through the atlas's tissue scorer	2026-10-19 |
| 1 | 755 | status report, result or explanation; no action of its own | 2	Showcase README states what the example reports cover	2026-10-19 |
| 1 | 756 | status report, result or explanation; no action of its own | 3	Mouse reproduction verdict recorded	2026-10-19 |
| 1 | 757 | status report, result or explanation; no action of its own | 4	Manifest: every record it names exists, or it says it doesn't	2026-10-19 |
| 1 | 758 | status report, result or explanation; no action of its own | 5	Manifest: status wordings and running rows checked	2026-10-19 |
| 1 | 759 | status report, result or explanation; no action of its own | 6	Rows with a pending next step placed on this plan	2026-10-19 |
| 1 | 760 | status report, result or explanation; no action of its own | 7	Instrument catalogue: every number traced to its record	2026-10-19 |
| 1 | 761 | status report, result or explanation; no action of its own | 8	Every number in the public documents traced to a measurement	2026-10-19 |
| 1 | 762 | status report, result or explanation; no action of its own | 9	Every table audited (book, status, commissioning report)	2026-10-19 |
| 1 | 763 | status report, result or explanation; no action of its own | 10	The book's checks whose numbers had changed, fixed	2026-10-19 |
| 1 | 764 | status report, result or explanation; no action of its own | 20	Commission Group-A	2026-10-20 |
| 1 | 765 | in plan (a step of the to-do list) | 23	Stage 0 control-probe limits set from healthy arrays	2026-10-21 |
| 1 | 766 | status report, result or explanation; no action of its own | 24	Every block explained	2026-10-20 |
| 1 | 767 | status report, result or explanation; no action of its own | 26	Reproductions on GitHub	2026-10-21 |
| 1 | 768 | status report, result or explanation; no action of its own | Also done today, inside the steps above |
| 1 | 769 | status report, result or explanation; no action of its own | * Nine of ten reproductions run on GitHub with every number identical (step 26), including IAM-A's line split across 13 machines. |
| 1 | 770 | status report, result or explanation; no action of its own | * The globe's change is fixed by the measurement, read in the C-Score's blocks of 10 sites; a healthy blood goes dark at the right end, a planted clustered loss |
| 1 | 771 | status report, result or explanation; no action of its own | * The noise globe, built from the public posterior draws on Zenodo. |
| 1 | 772 | status report, result or explanation; no action of its own | * Met-A against the atlas: the development stage in the chain and synthetic test 1 (both bars met). |
| 1 | 773 | status report, result or explanation; no action of its own | * Colon reproduced on GitHub, 6 of 6 identical. |
| 1 | 776 | status report, result or explanation; no action of its own | One result from the tissue-pair globes you should know about. The chain drew each tumor against the same person's own normal tissue, and a browser screenshotted |
| 1 | 777 | status report, result or explanation; no action of its own | Pair	Changed blocks |
| 1 | 778 | status report, result or explanation; no action of its own | 3 colon tumors	298, 1,109, 1,289 |
| 1 | 779 | status report, result or explanation; no action of its own | 3 breast tumors	912, 1,134, 1,570 |
| 1 | 780 | status report, result or explanation; no action of its own | Control: two different people's normal colon	741 |
| 1 | 781 | status report, result or explanation; no action of its own | The pictures look right: black background, with the change lit in red and blue clumps. But the control sits inside the tumors' range. So counting changed blocks |
| 1 | 782 | status report, result or explanation; no action of its own | Ideas to test next, not conclusions: |
| 1 | 783 | ideas: in plan (step 'The genome on a globe with layers'); leaving out the X done | * the direction of change at a cell's identity sites; |
| 1 | 784 | ideas: in plan (step 'The genome on a globe with layers'); leaving out the X done | * leaving out the X; |
| 1 | 785 | ideas: in plan (step 'The genome on a globe with layers'); leaving out the X done | * a person's own two normals as the control. |
| 1 | 786 | status report, result or explanation; no action of its own | This is recorded in the repo as DEV-TISSUE-GLOBE-01 (development) and is going into the development log now. |
| 1 | 791 | status report, result or explanation; no action of its own | We're waiting on three GitHub runs: |
| 1 | 792 | status report, result or explanation; no action of its own | 1. The release check on the new color keys and the X/Y counts. It includes the browser test, which fails if a healthy blood doesn't go dark. |
| 1 | 793 | status report, result or explanation; no action of its own | 2. The mouse test with 1 aligner thread. It decides whether the last reproduction fits on GitHub or has to wait for AWS. |
| 1 | 794 | status report, result or explanation; no action of its own | 3. Colon run 3, which compares colon controls only within the same slide. |
| 1 | 795 | status report, result or explanation; no action of its own | I'll report each as it lands. |
| 1 | 796 | status report, result or explanation; no action of its own | Met-A against the atlas: not ready yet, but not dead. It's still a development idea and has no effect on anything commissioned. |
| 1 | 797 | status report, result or explanation; no action of its own | * What worked: on real arrays from a laboratory the atlas never saw, every one of the 109 sorted cells was read, through the chain. The atlas's offset (readings |
| 1 | 798 | status report, result or explanation; no action of its own | * What didn't: healthy people of the same cell type spread too widely. After the tare only 37–75% read Normal, against close to all for the commissioned Met-A o |
| 1 | 799 | status report, result or explanation; no action of its own | * Why, so far: |
| 1 | 800 | status report, result or explanation; no action of its own | * It's not the X. Each cell's 6,000 sites hold at most 27 on the X. |
| 1 | 801 | status report, result or explanation; no action of its own | * About a third of the spread is the slide. Each donor's five cells sit on one slide and read high or low together. But here a slide is one donor, so correcting |
| 1 | 802 | status report, result or explanation; no action of its own | * Not yet tested: |
| 1 | 803 | naive vs broad types and laboratory-specific sites: the laboratory term is now the same-laboratory tare (60.3); naive vs broad is the BLUEPRINT share finding (DEV_ATLAS_V21_01.md) | * these are naive T and B cells, while the atlas cells are broader types; |
| 1 | 804 | naive vs broad types and laboratory-specific sites: the laboratory term is now the same-laboratory tare (60.3); naive vs broad is the BLUEPRINT share finding (DEV_ATLAS_V21_01.md) | * donor age; |
| 1 | 805 | naive vs broad types and laboratory-specific sites: the laboratory term is now the same-laboratory tare (60.3); naive vs broad is the BLUEPRINT share finding (DEV_ATLAS_V21_01.md) | * sites where this laboratory's arrays read differently from the atlas. |
| 1 | 806 | status report, result or explanation; no action of its own | What it means. The commissioned Met-A stays exactly as it is, reading against purified arrays of the same cell. For other cells, the atlas route needs a tighter |
| 1 | 807 | status report, result or explanation; no action of its own | It's all recorded in the repo (Met-A atlas plan, GRP-014, development log, commit 395c0f7f). |
| 1 | 823 | status report, result or explanation; no action of its own | Status of everything open: |
| 1 | 824 | status report, result or explanation; no action of its own | Item	Where it stands |
| 1 | 825 | status report, result or explanation; no action of its own | Reproductions on GitHub	8 of 10 with every number identical. Colon just passed, 6 of 6 checks in 3 hours. |
| 1 | 826 | status report, result or explanation; no action of its own | Split IAM-A run	12 of 13 jobs finished OK; the last counting job is still running, then the comparison job. |
| 1 | 827 | status report, result or explanation; no action of its own | Mouse run	The first test was cut off by GitHub about 100 minutes in, and its memory log was lost with it. The rerun started just now, logging memory live into G |
| 1 | 828 | status report, result or explanation; no action of its own | Noise maps	Done and live: the 5th globe in the Explorer, "The atlas's noise, beaten down by averaging" (https://hmahaffeyges.github.io/IAM-Validation/atlas/sphe |
| 1 | 829 | status report, result or explanation; no action of its own | Globe fix	The banded colors plus the new release check V13 are running now. V13 opens the chain's own globe pages in a real browser on GitHub, slides them as yo |
| 1 | 830 | status report, result or explanation; no action of its own | Site rebuild	Queued behind the latest push. |
| 1 | 831 | status report, result or explanation; no action of its own | Met-A on IAM-Atlas v2.1	Not tested yet. Only the plan is written. Under the chain-only rule the synthetic test has to run through the chain, so I'm building it  |
| 1 | 832 | full-genome globes, address and information: built on the sequencing atlas (checklist C1, C2); on the Explorer | Full Genome globes both address version and information version remain as well |
| 1 | 835 | status report, result or explanation; no action of its own | Group-A's second reading matched the sealed result exactly, so I've recorded it as the 10th reproduction from public data. 12 of the 26 to-do steps are done. Th |
| 1 | 836 | status report, result or explanation; no action of its own | Group-A re-read. It ran from a later version of the code than the sealed reading and gave the same numbers: |
| 1 | 837 | status report, result or explanation; no action of its own | * 263 of 278 people read. |
| 1 | 838 | status report, result or explanation; no action of its own | * Women and men: −0.36% apart, 1.206 standard errors. |
| 1 | 839 | status report, result or explanation; no action of its own | * Planted 1% loss: +2.00%, 7.253 standard errors. |
| 1 | 840 | status report, result or explanation; no action of its own | The book's reproduced-results chapter now lists it. |
| 1 | 841 | status report, result or explanation; no action of its own | The new SOP and OM. Both are built under their new names: |
| 1 | 842 | status report, result or explanation; no action of its own | * Official IAM Chain Operator Guide: how to run the chain. |
| 1 | 843 | status report, result or explanation; no action of its own | * Official IAM Operations Manual: Instrument Diagnostics and Chain Decoder: all 260 chain functions explained in the order the chain runs them. A push is refuse |
| 1 | 844 | status report, result or explanation; no action of its own | Until this morning the webpage still listed them under the old names. That's fixed, and the page rebuilds on GitHub. The manual's PDF file still has its old fil |
| 1 | 845 | status report, result or explanation; no action of its own | Researcher package, book and webpage. |
| 1 | 846 | status report, result or explanation; no action of its own | * Group-A is in: the book chapter, introduction, Q&A, instrument catalogue and webpage, all done when it was commissioned. |
| 1 | 847 | status report, result or explanation; no action of its own | * Not in yet: the Decoder and the overnight Stage 0 fixes are not in the book. That is step 22, the final documentation pass. |
| 1 | 848 | status report, result or explanation; no action of its own | Done (12): |
| 1 | 849 | in plan (a step of the to-do list) | #	Step |
| 1 | 850 | status report, result or explanation; no action of its own | 1–3	Showcase scenario 4, showcase README, mouse reproduction recorded |
| 1 | 851 | status report, result or explanation; no action of its own | 4–7	Manifest record names, status wordings, pending next steps, instrument catalogue |
| 1 | 852 | status report, result or explanation; no action of its own | 8–10	Every public number traced, every table audited, the 88 stale book checks |
| 1 | 853 | status report, result or explanation; no action of its own | 20	Group-A commissioned and reproduced |
| 1 | 854 | status report, result or explanation; no action of its own | 24	Every chain stop says what it measured, the limit, its source, what it means and what to do; release check passed 32 of 32 |
| 1 | 855 | status report, result or explanation; no action of its own | Open, doable now on GitHub (8): |
| 1 | 856 | status report, result or explanation; no action of its own | #	Step	Where it stands |
| 1 | 857 | Stage 0 limits: done (intake_thresholds_v2.json, 337 healthy arrays of three laboratories) | 23	Stage 0 control-probe limits	Two real bugs found and fixed overnight (below). Next: re-read the 731 healthy arrays with the fixed code and set the limit by t |
| 1 | 858 | Decoder findings: added as 61.8 | 25	Operator Guide and Decoder	Done and gated. 10 of the 32 code problems the Decoder turned up are still not fully fixed (small ones: old names in frozen files, |
| 1 | 859 | status report, result or explanation; no action of its own | 21	Every blood cell scored in whole blood	Not started |
| 1 | 860 | status report, result or explanation; no action of its own | 22	Documentation pass: book, page, Q&A, manual	Last, after 21 and 23 |
| 1 | 861 | status report, result or explanation; no action of its own | 26	Reproductions on GitHub	Group-A done. Still to do: the blood-reference rebuild (it already passed on AWS) |
| 1 | 862 | status report, result or explanation; no action of its own | 13	Sky-map additions	Not started |
| 1 | 863 | status report, result or explanation; no action of its own | 11, 12	Zenodo records for the book and the researcher package	Can go after step 22 |
| 1 | 864 | status report, result or explanation; no action of its own | Open, waiting for AWS (6): |
| 1 | 865 | status report, result or explanation; no action of its own | * 14: IAM-A firm-rate recount; its data is in our S3. |
| 1 | 866 | status report, result or explanation; no action of its own | * 15–18: the mouse detail copy, a clean-clone check, the billing blocks, and removing idle charges. |
| 1 | 867 | status report, result or explanation; no action of its own | * 19: independence from AWS. Part 1 is done; the rest needs the account. |
| 1 | 868 | status report, result or explanation; no action of its own | Found overnight in Stage 0, the intake checks: |
| 1 | 869 | status report, result or explanation; no action of its own | * Bisulfite check: the chain read 3 of the 5 control pairs in the wrong color channel. That is why every healthy array failed the 0.95 line. Read correctly, hea |
| 1 | 870 | status report, result or explanation; no action of its own | * Hybridization check: the two "failed" slides were not failing. Their high and low controls were swapped. Both checks are fixed, and none of the 59 healthy tes |
| 2 | 1 | status report, result or explanation; no action of its own | WHOLE GENOME ATLAS |
| 2 | 3 | status report, result or explanation; no action of its own | The method we are using to determine healthy for the atlas, for both of them, are we using IAM's law in some way? Does the atlas we are creating now, the sequen |
| 2 | 4 | status report, result or explanation; no action of its own | The short answer: neither atlas has IAM's law built into it, and that's deliberate. The law enters in the instruments that read against the atlas, not in the at |
| 2 | 5 | status report, result or explanation; no action of its own | How "healthy" is decided for both atlases |
| 2 | 6 | status report, result or explanation; no action of its own | Entirely by what the sources state. A sample goes in if it comes from a donor the study describes as healthy and from purified or sorted primary cells (not cell |
| 2 | 7 | status report, result or explanation; no action of its own | What each atlas actually is |
| 2 | 8 | status report, result or explanation; no action of its own | * IAM-Atlas v2.1 (arrays): each cell type's healthy methylation level at each array site, with its uncertainty. It's fitted jointly across labs using methods ta |
| 2 | 9 | status report, result or explanation; no action of its own | * The sequencing atlas: the same idea at every CpG, from healthy donors' read counts, with the same cosmology methods for noise and uncertainty. |
| 2 | 10 | status report, result or explanation; no action of its own | Where the law does come in: what we chose to measure and store |
| 2 | 11 | status report, result or explanation; no action of its own | 1. Each donor's measured disorder at every site. The law is about how much disorder a cell shows against the floor physics allows. So the atlas stores healthy d |
| 2 | 12 | status report, result or explanation; no action of its own | 2. Each donor's read patterns, so held error (IAM-A) and the coupling have a healthy reference. |
| 2 | 13 | status report, result or explanation; no action of its own | 3. Every value's uncertainty and noise maps, so an instrument can tell a real departure from counting scatter. |
| 2 | 14 | status report, result or explanation; no action of its own | The instruments then apply the law: Met-A compares a sample's disorder with the floor its own healthy cell type sets, IAM-A does the same for held error, and th |
| 2 | 15 | status report, result or explanation; no action of its own | Why keeping the law out of the atlas matters to a researcher |
| 2 | 16 | status report, result or explanation; no action of its own | * It makes the tests not circular. If the atlas had been built to agree with the law, every result read against it would partly confirm itself. Because it's bui |
| 2 | 17 | status report, result or explanation; no action of its own | * Anyone can check it. A researcher who doesn't accept the law can still inspect, rebuild and reuse the atlas, because it's a measurement product. |
| 2 | 18 | status report, result or explanation; no action of its own | Could others build the same thing without IAM? Why haven't they? |
| 2 | 19 | status report, result or explanation; no action of its own | The atlas part, yes. Reference atlases of healthy cell methylation already exist, including the published sequencing atlas both of ours draw from. What I believ |
| 2 | 20 | status report, result or explanation; no action of its own | * per-donor healthy disorder and uncertainty stored at every CpG; |
| 2 | 21 | status report, result or explanation; no action of its own | * noise maps and half-splits taken over from cosmology; |
| 2 | 22 | status report, result or explanation; no action of its own | * a design built to serve as a ruler for a score against a floor. |
| 2 | 23 | prior-art check: was missing; added as 61.5 | Most existing atlases are built to estimate which cells a sample came from, so they store average levels at marker sites. Measures of methylation disorder (entr |
| 2 | 24 | status report, result or explanation; no action of its own | In one line: the atlas measures what healthy looks like, and the law says how to read a sample against it. |
| 2 | 27 | status report, result or explanation; no action of its own | My estimate is about 2 to 3 weeks to finish it with everything we can add now. Almost all of that is writing and testing the code. The machines' share is about  |
| 2 | 28 | status report, result or explanation; no action of its own | Step by step: |
| 2 | 29 | status report, result or explanation; no action of its own | Step	Writing and testing (synthetic first)	Machine time |
| 2 | 30 | sequencing-atlas build steps: 35.30.1 to 35.30.4 (status corrected from the checklist); step 7 MCMC: 60.18 not pursued, with the reason | 1. Core counting tables from the 153 (each donor's counts and disorder at every CpG, 16-bit)	1 to 2 days	a few hours |
| 2 | 31 | sequencing-atlas build steps: 35.30.1 to 35.30.4 (status corrected from the checklist); step 7 MCMC: 60.18 not pursued, with the reason | 2. Open sources: hand check of donors, format converters, hg38 to hg19 conversion	1 to 2 days	a few hours (about 770 files to download) |
| 2 | 32 | sequencing-atlas build steps: 35.30.1 to 35.30.4 (status corrected from the checklist); step 7 MCMC: 60.18 not pursued, with the reason | 3. The two noise maps (donor halves, read halves), masks, 95 % ranges	1 day	hours (the same counting, rerun) |
| 2 | 33 | sequencing-atlas build steps: 35.30.1 to 35.30.4 (status corrected from the checklist); step 7 MCMC: 60.18 not pursued, with the reason | 4. Healthy references for every sequencing instrument	1 to 2 days	hours |
| 2 | 34 | sequencing-atlas build steps: 35.30.1 to 35.30.4 (status corrected from the checklist); step 7 MCMC: 60.18 not pursued, with the reason | 5. Sky sphere and information sphere at whole-genome resolution, with their noise maps	2 days	hours |
| 2 | 35 | sequencing-atlas build steps: 35.30.1 to 35.30.4 (status corrected from the checklist); step 7 MCMC: 60.18 not pursued, with the reason | 6. Spectra, covariance, cosmic variance, transfer function, null tests	2 to 3 days	hours |
| 2 | 36 | sequencing-atlas build steps: 35.30.1 to 35.30.4 (status corrected from the checklist); step 7 MCMC: 60.18 not pursued, with the reason | 7. MCMC across labs on the instruments' sites, with the convergence check and the cross-check against the counts	1 day	1 to 2 days |
| 2 | 37 | sequencing-atlas build steps: 35.30.1 to 35.30.4 (status corrected from the checklist); step 7 MCMC: 60.18 not pursued, with the reason | 8. Commissioning tests on sealed held-out data, cell by cell	2 to 3 days	1 to 2 days |
| 2 | 38 | status report, result or explanation; no action of its own | Why the writing takes longer than the computing: each step gets a synthetic test through the chain before it's used, then a record and a push. That's what makes |
| 2 | 39 | status report, result or explanation; no action of its own | What could make it longer: |
| 2 | 40 | status report, result or explanation; no action of its own | * The instruments' own commissioning comes first (your priority order), and it shares the 20 machines. |
| 2 | 41 | status report, result or explanation; no action of its own | * A failed synthetic means a fix and a rerun, as with Met-A this week. |
| 2 | 42 | status report, result or explanation; no action of its own | * Whole-genome spheres are new at this size: 28 million sites on the sphere hasn't been done yet. |
| 2 | 43 | status report, result or explanation; no action of its own | What could make it shorter: |
| 2 | 44 | status report, result or explanation; no action of its own | * Steps 3 to 6 can be written while step 1 computes. |
| 2 | 45 | status report, result or explanation; no action of its own | * More machines: Google's 8 now, more if the CPU limit is raised. |
| 2 | 46 | status report, result or explanation; no action of its own | Cost: under about $30 on Google spot machines, if we use them at all. GitHub's machines are free. |
| 2 | 47 | status report, result or explanation; no action of its own | One correction to keep in mind: no single sample reads all 28 million CpGs. A healthy sample typically covers most of them with some reads, and the atlas stores |
| 2 | 54 | status report, result or explanation; no action of its own | The whole-genome atlas: what it takes, and why plasma is the right target. |
| 2 | 55 | status report, result or explanation; no action of its own | Why plasma: your feeling has a concrete basis. A plasma draw sequenced at shallow depth lands its reads on random CpGs. IAM-Atlas v2.1 only knows the 814,000 ar |
| 2 | 56 | status report, result or explanation; no action of its own | What it must carry from v2.1: |
| 2 | 57 | carried from v2.1: posterior width (counted ranges, D4), donor spread (A3), laboratory terms (not measurable inside the atlas; same-laboratory tare 60.3), seeds and spent sealing (D15) | * every cell's level with its posterior width |
| 2 | 58 | carried from v2.1: posterior width (counted ranges, D4), donor spread (A3), laboratory terms (not measurable inside the atlas; same-laboratory tare 60.3), seeds and spent sealing (D15) | * the donor spread |
| 2 | 59 | carried from v2.1: posterior width (counted ranges, D4), donor spread (A3), laboratory terms (not measurable inside the atlas; same-laboratory tare 60.3), seeds and spent sealing (D15) | * the lab terms (the second labs come from the IHEC list) |
| 2 | 60 | carried from v2.1: posterior width (counted ranges, D4), donor spread (A3), laboratory terms (not measurable inside the atlas; same-laboratory tare 60.3), seeds and spent sealing (D15) | * the same seeds rule |
| 2 | 61 | carried from v2.1: posterior width (counted ranges, D4), donor spread (A3), laboratory terms (not measurable inside the atlas; same-laboratory tare 60.3), seeds and spent sealing (D15) | * the same sealing of which data is spent |
| 2 | 62 | status report, result or explanation; no action of its own | What has to be decided at build time, because it can't be added afterwards (from your sphere and CMB list and the v2.1 checklist's "after" items): |
| 2 | 63 | build-time items: half-atlases built (D1), coverage (A4), neighbor covariance (D8), neighbor pairs (coupling-A, B2), HEALPix address (C1); posterior draws not needed (counted, not fitted); M05 was missing: added as 60.17 | 1. Two half-atlases (F07 and F08): each cell's donors split in half and sealed before the fit, both fitted the same way. They give the half-difference noise map |
| 2 | 64 | build-time items: half-atlases built (D1), coverage (A4), neighbor covariance (D8), neighbor pairs (coupling-A, B2), HEALPix address (C1); posterior draws not needed (counted, not fitted); M05 was missing: added as 60.17 | 2. Posterior draws kept for noise maps (M03). At 28 million sites, 100 draws per cell would be more than 500 GB. The plan: keep each site's mean and width, a fe |
| 2 | 65 | build-time items: half-atlases built (D1), coverage (A4), neighbor covariance (D8), neighbor pairs (coupling-A, B2), HEALPix address (C1); posterior draws not needed (counted, not fitted); M05 was missing: added as 60.17 | 3. Coverage recorded per site and per cell, so the masks and their tapered edges (M02, APODMASK) can be built later. |
| 2 | 66 | build-time items: half-atlases built (D1), coverage (A4), neighbor covariance (D8), neighbor pairs (coupling-A, B2), HEALPix address (C1); posterior draws not needed (counted, not fitted); M05 was missing: added as 60.17 | 4. Healthy covariance between neighboring sites (M07), computed while the donors' values are in memory. It's cheap then and expensive later, and it's also what  |
| 2 | 67 | build-time items: half-atlases built (D1), coverage (A4), neighbor covariance (D8), neighbor pairs (coupling-A, B2), HEALPix address (C1); posterior draws not needed (counted, not fitted); M05 was missing: added as 60.17 | 5. Neighbor-pair counts on each molecule. This feeds the Johns Hopkins coupling item and the ceiling turnover. |
| 2 | 68 | build-time items: half-atlases built (D1), coverage (A4), neighbor covariance (D8), neighbor pairs (coupling-A, B2), HEALPix address (C1); posterior draws not needed (counted, not fitted); M05 was missing: added as 60.17 | 6. A fixed address for every CpG on the sphere (HEALPix), so every map, mask and spectrum lines up. |
| 2 | 69 | build-time items: half-atlases built (D1), coverage (A4), neighbor covariance (D8), neighbor pairs (coupling-A, B2), HEALPix address (C1); posterior draws not needed (counted, not fitted); M05 was missing: added as 60.17 | 7. A planted synthetic methylome run through the same pipeline (M05), which measures what the build does to a known signal. |
| 2 | 70 | status report, result or explanation; no action of its own | What it costs. These are rough figures, scaled from v2.1. Under your rule, the real number comes from timing one block first. |
| 2 | 71 | status report, result or explanation; no action of its own | v2.1 (actual)	Whole genome (scaled, about 34× the sites) |
| 2 | 72 | status report, result or explanation; no action of its own | Machine time	about 21 hours on 128 cores, about 2,700 core-hours	about 90,000 core-hours |
| 2 | 73 | status report, result or explanation; no action of its own | Cost if paid	$150 to $170 (AWS on demand)	Hetzner about €3,000; AWS spot about $2,000 to $3,000; double either with the two half-atlases |
| 2 | 74 | status report, result or explanation; no action of its own | Free on GitHub	no (one 128-core run)	possible if a block fits 16 GB and 6 hours: about 7 weeks on 20 machines, longer with the half-atlases |
| 2 | 75 | status report, result or explanation; no action of its own | Two ways to bring it down without leaving anything useful out: |
| 2 | 76 | status report, result or explanation; no action of its own | * Fit neighboring CpGs that move together as one block. That matches how molecules are read anyway, and likely cuts the work 4 to 5 times. |
| 2 | 77 | status report, result or explanation; no action of its own | * Fit only the cell types on the ranked list, by priority: about 50 columns of the 105. |
| 2 | 78 | status report, result or explanation; no action of its own | Proposed order: |
| 2 | 79 | status report, result or explanation; no action of its own | 1. Finish the ranked list with you. |
| 2 | 80 | status report, result or explanation; no action of its own | 2. Write the build plan into the plan database, as branch 35.30: the seven build-time items, the block rule and the sealed donor halves. |
| 2 | 81 | status report, result or explanation; no action of its own | 3. Time one real block on GitHub (free) and on one small paid machine (a dollar or two), so the cost is measured, not guessed. |
| 2 | 82 | status report, result or explanation; no action of its own | 4. Then you decide where it runs and what you're willing to spend. |
| 2 | 85 | status report, result or explanation; no action of its own | Building the whole-genome atlas and globe together: agreed, they belong together. |
| 2 | 86 | whole-genome globe layer: built on sequencing (C2, flagged on neurons); per-cell packages 35.30.13 | * What the globe's whole-genome layer does now: it already covers the genome in blocks (77,355 of them). Outside the atlas's 813,951 sites, though, it can only  |
| 2 | 87 | whole-genome globe layer: built on sequencing (C2, flagged on neurons); per-cell packages 35.30.13 | * What a whole-genome atlas adds: every block would get its own healthy reference in the person's own cell mix. The globe could then show departures from the ex |
| 2 | 88 | whole-genome globe layer: built on sequencing (C2, flagged on neurons); per-cell packages 35.30.13 | * Shallow plasma gains the most: the fragments we now discard would count. |
| 2 | 89 | whole-genome globe layer: built on sequencing (C2, flagged on neurons); per-cell packages 35.30.13 | It's one project: the atlas fit (AWS scale, about 35 times today's site count), then the globe layer built on it, then its synthetic test. I'm adding it to the  |
| 2 | 91 | status report, result or explanation; no action of its own | What exists already: |
| 2 | 92 | status report, result or explanation; no action of its own | * Epimutation burden (also called epigenetic mutation load): count the sites where a person sits far outside a healthy cohort's range, across the whole array. I |
| 2 | 93 | status report, result or explanation; no action of its own | * Epigenetic clocks and methylation risk scores: combine many sites into one number, trained on cohorts. |
| 2 | 94 | status report, result or explanation; no action of its own | * Genome-wide methylation entropy: the leukemia study's own measure, on deep sequencing against normal controls. |
| 2 | 95 | status report, result or explanation; no action of its own | So a whole-genome "how abnormal is this person" score exists. Its known weaknesses come from leaning on a population: |
| 2 | 96 | status report, result or explanation; no action of its own | 1. The cohort's normal mixes everything that varies between healthy people: age, sex, ancestry, inherited variants that shift marks at thousands of sites, and a |
| 2 | 97 | status report, result or explanation; no action of its own | 2. Laboratory and chip batches shift values. The cohort has to be run on the same platform, often in the same lab, to be comparable. |
| 2 | 98 | status report, result or explanation; no action of its own | 3. The unit is relative. "4 standard deviations from this cohort" means something different in every cohort. |
| 2 | 99 | status report, result or explanation; no action of its own | What ours does differently: |
| 2 | 100 | status report, result or explanation; no action of its own | * Composition is removed per person. Each site's expectation is the atlas's healthy cells in that person's own measured mix, not a cohort average. |
| 2 | 101 | status report, result or explanation; no action of its own | * The scale is fixed by physics. Floor and ceiling come from the law, in bits, so a reading means the same thing in any lab. |
| 2 | 102 | status report, result or explanation; no action of its own | * IAM-A references the sample to itself, from its own molecules, so laboratory batch largely drops out. |
| 2 | 103 | status report, result or explanation; no action of its own | Where we have to be careful: |
| 2 | 104 | status report, result or explanation; no action of its own | * The atlas is still a measured reference of healthy cells from many donors. Met-A's baseline is still measured: the derived version missed in step 3 and is bei |
| 2 | 105 | status report, result or explanation; no action of its own | * Inherited variation still applies to us. People differ at many sites for genetic reasons, and those differences sit inside the atlas's donor spread. The whole |
| 2 | 107 | status report, result or explanation; no action of its own | Megakaryocytes, three sources: |
| 2 | 108 | status report, result or explanation; no action of its own | Source	What it is	Use |
| 2 | 109 | megakaryocytes: GSE87197 column in 40.18; GSE206818 cannot build a counting column (no reads); BLUEPRINT S004AV/S004BT added in 60.21 | BLUEPRINT S004AV and S004BT	Bulk purified megakaryocytes from cord blood, 2 donors, whole-genome sequencing	Build the column |
| 2 | 110 | megakaryocytes: GSE87197 column in 40.18; GSE206818 cannot build a counting column (no reads); BLUEPRINT S004AV/S004BT added in 60.21 | GSE206818	Platelets from 3 healthy adults, whole-genome sequencing. Platelets carry the megakaryocyte genome	Check the cord-blood column against adult cells |
| 2 | 111 | megakaryocytes: GSE87197 column in 40.18; GSE206818 cannot build a counting column (no reads); BLUEPRINT S004AV/S004BT added in 60.21 | GSE87197 donor 3	Single megakaryocytes, still unread	Test: plant into a spent plasma |
| 2 | 112 | megakaryocytes: GSE87197 column in 40.18; GSE206818 cannot build a counting column (no reads); BLUEPRINT S004AV/S004BT added in 60.21 | The GSE206818 study says what's at stake: it estimates megakaryocytes supply about 26% of healthy plasma cell-free DNA. That's the size of the gap in our plasma |
| 2 | 113 | status report, result or explanation; no action of its own | * Two limits: |
| 2 | 114 | status report, result or explanation; no action of its own | * BLUEPRINT's megakaryocytes are neonatal, which is why the adult platelets are the check. |
| 2 | 115 | status report, result or explanation; no action of its own | * The platelet data is merged into one track with no per-donor files and no raw reads, because of donor privacy. So it gives a level at each site but no spread  |
| 2 | 116 | status report, result or explanation; no action of its own | BLUEPRINT more broadly: 205 bisulfite-sequenced samples, read from the labels. |
| 2 | 117 | BLUEPRINT cell types the atlas lacks: added as 60.21 (level-only) | * Cell types the atlas only has from arrays: naive B (3 donors), regulatory T (2), eosinophils (2), mature neutrophils (7). The neutrophils were spent developin |
| 2 | 118 | BLUEPRINT cell types the atlas lacks: added as 60.21 (level-only) | * Cell types the atlas lacks entirely: plasma cells (5), dendritic cells (2), megakaryocytes (2), red-cell precursors (2). |
| 2 | 119 | BLUEPRINT cell types the atlas lacks: added as 60.21 (level-only) | * A second lab for CD4 T, memory B, monocytes, NK and the CD8 subsets. That's the biggest help with the problem that most of the sequencing comes from one lab. |
| 2 | 120 | status report, result or explanation; no action of its own | * Excluded: cultured or stimulated samples. |
| 2 | 121 | B and T on arrays: in plan (60.4); GSE252045 and GSE288076 named in DEV_METAA_ATLAS_CURVE_01.md | B and T cells for Met-A on the atlas: six unread healthy sorted-cell EPIC series with raw files, none of them in the atlas. |
| 2 | 122 | B and T on arrays: in plan (60.4); GSE252045 and GSE288076 named in DEV_METAA_ATLAS_CURVE_01.md | * Development: GSE252045. |
| 2 | 123 | B and T on arrays: in plan (60.4); GSE252045 and GSE288076 named in DEV_METAA_ATLAS_CURVE_01.md | * Confirmation, each from a different lab: GSE174666, GSE192451, GSE288076, GSE146093 and GSE184500. |
| 2 | 124 | status report, result or explanation; no action of its own | * Next: I'll write the test record, with the quality rule set before anything is read. |
| 2 | 125 | status report, result or explanation; no action of its own | From your last question, added to the plan: two tests of IAM's law to run on v2.1 before any whole-genome build: |
| 2 | 126 | floor-bounded prior: was missing as a branch; added as 60.20, as a check, not a refit (v2.1 is commissioned) | 1. A floor-bounded prior: each healthy held level kept within the derived floor's limits. The test is whether that hurts the fit anywhere. |
| 2 | 127 | held-error check on every column: was missing; added as 60.19 | 2. The held-error check on every cell column: at firmly held sites, the atlas level should equal 1 minus that cell's measured held error. That checks each colum |
| 2 | 128 | prior-art check: added as 61.5 | I also added a prior-art search before any public claim. Whole-genome healthy atlases already exist; what we have to establish is whether any of them pools labs |
| 2 | 130 | status report, result or explanation; no action of its own | What exists: |
| 2 | 131 | status report, result or explanation; no action of its own | * The published sequencing atlas of human cell types. Deep whole-genome bisulfite sequencing of purified healthy cells, about 39 cell types and roughly 200 samp |
| 2 | 132 | status report, result or explanation; no action of its own | * BLUEPRINT, ENCODE, Roadmap and the international reference-epigenome program publish whole-genome methylation of healthy cells and tissues too. |
| 2 | 133 | status report, result or explanation; no action of its own | So "a whole-genome healthy reference per cell type" isn't new, and we shouldn't present it as new. |
| 2 | 134 | status report, result or explanation; no action of its own | What's different about ours: |
| 2 | 135 | prior-art check: added as 61.5 | 1. Many labs and both platforms in one model. Published atlases are usually one lab's average per cell type. Ours pools labs and platforms, removes each lab's t |
| 2 | 136 | status report, result or explanation; no action of its own | 2. What it's used for. Others use their atlases for cell-of-origin markers. We use ours to build each person's own expected healthy self, from their own cell mi |
| 2 | 137 | status report, result or explanation; no action of its own | Where IAM's law comes in, and where it doesn't: |
| 2 | 138 | status report, result or explanation; no action of its own | * Building the atlas: anyone could do it the same way. The sampling is statistics. Someone with the same data and model gets the same atlas, and that's how it s |
| 2 | 139 | floor-bounded prior: added as 60.20 | * The law can constrain it, and that part would be ours. The derived floor says no held mark can sit closer to fully marked than 1 − ε_min, and no held-absent m |
| 2 | 140 | held-error check: added as 60.19 | * The law gives a check no one else has. Layer 1 found that the atlas's level at firmly held sites equals 1 minus the cell's own held error, which is what the l |
| 2 | 141 | status report, result or explanation; no action of its own | * The law gives the readings their meaning. Floor, ceiling, entropy against the floor, and each site's sensitivity dH/dε at its held level are what turn a depar |
| 2 | 142 | status report, result or explanation; no action of its own | Bottom line: the atlas is a ruler anyone can rebuild. The law bounds it, checks it, and decides what its readings mean. |
| 2 | 145 | status report, result or explanation; no action of its own | INFORMATION SPHERE (and globe list) |
| 2 | 152 | status report, result or explanation; no action of its own | How each control behaves now: |
| 2 | 153 | status report, result or explanation; no action of its own | Control	What you see |
| 2 | 154 | status report, result or explanation; no action of its own | Drag	Turns the sphere |
| 2 | 155 | status report, result or explanation; no action of its own | Click a family in the legend	Only that family; the sphere turns to face its strongest territory and stops turning. The readout says how strongly its sites gathe |
| 2 | 156 | status report, result or explanation; no action of its own | Click it again, or "Show every family"	All families back |
| 2 | 157 | status report, result or explanation; no action of its own | Light one cell type	Turns to face that cell's sites and lights them bright. The menu shows each cell's site count; with none at the current setting, the page sa |
| 2 | 158 | status report, result or explanation; no action of its own | "Stands apart" slider	Left colors more sites (all 191,458 at 0); right colors only the strongest (1,581 in view at max) |
| 2 | 159 | status report, result or explanation; no action of its own | Both / Unmethylated / Methylated	Shows only that kind of site |
| 2 | 160 | status report, result or explanation; no action of its own | Territories, Labels, Every site, Far side	Each on or off. Far side and Every site are faint by design |
| 2 | 161 | status report, result or explanation; no action of its own | Layer	"This sample" (each site's change, red or blue) or "What groups the sites" (the atlas map) |
| 2 | 162 | status report, result or explanation; no action of its own | Fade slider	Quiets everything except the change. On a healthy sample it goes dark and says why: no territory changed |
| 2 | 163 | status report, result or explanation; no action of its own | Hover	Names the territory under the pointer |
| 2 | 164 | status report, result or explanation; no action of its own | Every button also has a tooltip, and there's a "How to use" line above the legend. |
| 2 | 165 | status report, result or explanation; no action of its own | Two honest limits: |
| 2 | 166 | status report, result or explanation; no action of its own | * Changed territories were checked with a marked stand-in, because neither example blood changes one. A real changed sample is the true test. |
| 2 | 167 | status report, result or explanation; no action of its own | * The test browser: the renders ran the page's own script in a JavaScript engine with a stand-in canvas, not a full browser. Text labels and mouse feel weren't  |
| 2 | 168 | status report, result or explanation; no action of its own | Also in this push: distance to the ceiling. Every atlas reading (Met-A for the whole sample and each cell, and the territories) now carries its saturation, its  |
| 2 | 180 | status report, result or explanation; no action of its own | Globes that exist now |
| 2 | 181 | status report, result or explanation; no action of its own | Globe	Built for	What it shows	Status |
| 2 | 182 | status report, result or explanation; no action of its own | 1. IAM-Atlas globes	Every one of the atlas's 79 healthy cell types (Globe Explorer)	Each cell's healthy methylation at every site, plus how sure the atlas is	Bu |
| 2 | 183 | status report, result or explanation; no action of its own | 2. Report globe: whole genome	Every array the one command reads (whole blood, isolated cells, tissue)	The sample against its own expected healthy self: the atla |
| 2 | 184 | status report, result or explanation; no action of its own | 3. Report globe: Met-A's 6,000 sites	Same samples	Met-A against its healthy reference. It uses the same sites and the same blocks of 10 the C-Score reads, so th |
| 2 | 185 | status report, result or explanation; no action of its own | 4. Report globe: cells found	Same samples	Each cell the Deconvolver found, with its share, on that cell's healthy map	Built |
| 2 | 186 | status report, result or explanation; no action of its own | 5. Report globe: each cell's change	Same samples	Each found cell's change at its own identity sites	Development (same fix) |
| 2 | 187 | status report, result or explanation; no action of its own | 6. Pair globe	Two samples from one person (tumor and normal tissue, or plasma and buffy coat)	One sample against the person's own other sample	Built |
| 2 | 188 | status report, result or explanation; no action of its own | 7. IAM-A whole-genome globes	Sequencing samples (single molecules)	One sample against its own predicted counting error, the same sample read twice, two people s |
| 2 | 189 | status report, result or explanation; no action of its own | 8. Information sphere	The atlas (the healthy map)	Sites placed by how they behave across cells	Built today, development, no sample layer yet |
| 2 | 190 | status report, result or explanation; no action of its own | The C-Scores for each cell and for the whole genome, built today, are numbers in the report. They aren't their own globe layers yet. On the globe, the C-Score i |
| 2 | 191 | status report, result or explanation; no action of its own | When the chain builds a globe: every time the one command reads a sample, the report gets globes 2–5 automatically. A same-person pair gets globe 6. The atlas g |
| 2 | 192 | status report, result or explanation; no action of its own | cfDNA (plasma). It fits as follows; the runner is built and waits for matched real data: |
| 2 | 193 | status report, result or explanation; no action of its own | 1. Best case, one blood draw: buffy coat and plasma from the same tube. The buffy coat is the person's own healthy reference for their plasma (pair globe 6). An |
| 2 | 194 | status report, result or explanation; no action of its own | 2. Plasma alone: the Deconvolver finds the sources (blood cells, liver, and so on), and the plasma is read against the atlas mixed at those shares (globe 2). In |
| 2 | 195 | status report, result or explanation; no action of its own | 3. Sequenced plasma: IAM-A reads single molecules (globe 7), with the read-count correction for Met-A. |
| 2 | 196 | status report, result or explanation; no action of its own | 4. Where the information sphere helps most: tissue of origin. Tissues hold separate territories, so DNA shed from an unexpected tissue should light that tissue' |
| 2 | 197 | status report, result or explanation; no action of its own | The information sphere with the address globe, in order: |
| 2 | 198 | status report, result or explanation; no action of its own | 1. Sample layer: the person's own expected normal, as on globe 2, painted on the sphere. A territory lights up when that family's sites depart together; the sli |
| 2 | 199 | five uses of the information sphere: in plan (35.4) | 2. The five simulated uses: distance from your own healthy state, drift over time, sites failing together, origin of cell-free DNA, quality control. Synthetic,  |
| 2 | 200 | status report, result or explanation; no action of its own | 3. The null on real healthy bloods: they should stay dark. This is the step that decides whether it can go into reports. |
| 2 | 201 | in plan (a step of the to-do list) | 4. The linked pair in every report: the address globe and the information sphere side by side. Selecting a patch on one shows where those sites land on the othe |
| 2 | 211 | status report, result or explanation; no action of its own | What each is for, put simply: |
| 2 | 212 | status report, result or explanation; no action of its own | Question	Globe |
| 2 | 213 | status report, result or explanation; no action of its own | What does healthy look like?	1 (atlas) |
| 2 | 214 | status report, result or explanation; no action of its own | Which cells are in this sample?	4 |
| 2 | 215 | status report, result or explanation; no action of its own | Has this sample changed from its own normal, and where on the DNA?	2, 3 |
| 2 | 216 | status report, result or explanation; no action of its own | Which cell changed?	5 |
| 2 | 217 | status report, result or explanation; no action of its own | Has this person changed from their own earlier or other sample?	6 |
| 2 | 218 | status report, result or explanation; no action of its own | Is the change real or just counting?	7 |
| 2 | 219 | status report, result or explanation; no action of its own | What kind of change is it, and which cell's program?	8 |
| 2 | 225 | status report, result or explanation; no action of its own | the blood lineages ended up next to each other on their own, as did the tissues. Bone-marrow progenitors, B and NK cells, T cells and myeloid cells sit side by  |
| 2 | 227 | in plan (a step of the to-do list) | The information sphere |
| 2 | 228 | status report, result or explanation; no action of its own | What it is. The address globes and the C-Score globes place each site by where it sits on the DNA: neighbors on the globe are neighbors on a chromosome. The inf |
| 2 | 229 | status report, result or explanation; no action of its own | The math, step by step: |
| 2 | 230 | status report, result or explanation; no action of its own | 1. Each site becomes a point on a sphere. A site's level in each cell type is a probability b (tagged or not). Take the square roots of b and 1−b in every cell  |
| 2 | 231 | status report, result or explanation; no action of its own | 2. The angle between two sites is the true distance. θ = arccos(u₁·u₂) is the Fisher–Rao (Bhattacharyya) angle of information geometry. For nearby states, the r |
| 2 | 232 | status report, result or explanation; no action of its own | 3. Each site's own level is taken out first. Without this step, the methylation level decides where a site lands (test 8's failure). With it, a site is placed o |
| 2 | 233 | status report, result or explanation; no action of its own | 4. The sphere is drawn in its three leading directions. They keep 39% of the variance, and distances agree with the full space at 0.82. The empty sky is left da |
| 2 | 234 | status report, result or explanation; no action of its own | What the colors mean. Your question was what groups the sites: it's cell identity. Each site is colored by the cell where it departs most from its own level, an |
| 2 | 235 | status report, result or explanation; no action of its own | * Lineages fall together. The blood lineages (bone marrow progenitors, B and NK cells, T cells, myeloid cells) sit side by side, and gut, pancreas, lung, kidney |
| 2 | 236 | status report, result or explanation; no action of its own | * Each cell has two poles. A cell's methylated and unmethylated sites gather on opposite sides of the sphere. For T cells each side is tight (0.88 and 0.86 on a |
| 2 | 237 | status report, result or explanation; no action of its own | The blind spot it covers. Everything we have reads change in genome order: |
| 2 | 238 | status report, result or explanation; no action of its own | * The address globe shows a change confined to one stretch of a chromosome as a patch. |
| 2 | 239 | status report, result or explanation; no action of its own | * The C-Score asks whether departures cluster along the DNA. |
| 2 | 240 | status report, result or explanation; no action of its own | A cell's identity program, though, is thousands of sites spread over every chromosome. When that program slips, the address globe shows scattered dots, and the  |
| 2 | 241 | status report, result or explanation; no action of its own | How it differs from the C-Score globes. Same sample, two different orders of neighbors: |
| 2 | 242 | status report, result or explanation; no action of its own | C-Score / address globes	Information sphere |
| 2 | 243 | status report, result or explanation; no action of its own | Neighbors are	next to each other on the DNA	doing the same job in the same cells |
| 2 | 244 | status report, result or explanation; no action of its own | Answers	where on the genome a change is	what kind of change it is, and in which cell's program |
| 2 | 245 | status report, result or explanation; no action of its own | Misses	a program spread over many chromosomes	where on a chromosome the change is |
| 2 | 246 | fourth C-Score on the information sphere: was missing; added as 61.7 | That suggests a fourth member of the C-Score family: the same clustering statistic, computed over neighbors on the information sphere instead of the DNA. I'll a |
| 2 | 247 | status report, result or explanation; no action of its own | What it gives the chain, once a sample can be drawn on it (next step, 35.2): |
| 2 | 248 | status report, result or explanation; no action of its own | * a sample layer showing which cell's program is moving; |
| 2 | 249 | status report, result or explanation; no action of its own | * a distance from the person's own healthy state, measured as an angle; |
| 2 | 250 | status report, result or explanation; no action of its own | * repeated draws from the same person tracing a path. |
| 2 | 251 | status report, result or explanation; no action of its own | Synthetic test first, as always. |
| 2 | 252 | status report, result or explanation; no action of its own | Honest limits: |
| 2 | 253 | status report, result or explanation; no action of its own | * It is still development. |
| 2 | 254 | status report, result or explanation; no action of its own | * Three directions keep 39% of the variance. |
| 2 | 255 | status report, result or explanation; no action of its own | * About half the sites carry no single cell's identity (drawn as faint dust). |
| 2 | 256 | status report, result or explanation; no action of its own | * It hasn't yet read a person. |
| 2 | 258 | status report, result or explanation; no action of its own | (The information sphere: the machinery is ready, the next construction isn't written yet. The chain builds the sphere itself (run_sample.py --build info-sphere) |
| 2 | 259 | status report, result or explanation; no action of its own | * Synthetic tests: passed (test 7). |
| 2 | 260 | status report, result or explanation; no action of its own | * First test on real cells (test 8, all 79 atlas cell types): two of three checks met. |
| 2 | 261 | status report, result or explanation; no action of its own | * Groups of sites that behave alike land together: 12 of 12. |
| 2 | 262 | status report, result or explanation; no action of its own | * Distances are kept: 0.85 agreement, against a bar of 0.7. |
| 2 | 263 | status report, result or explanation; no action of its own | * Failed: sites that share only a methylation level also land together, 88 of 100 such groups. On real cells, a site's place on the sphere is set mostly by how  |
| 2 | 264 | status report, result or explanation; no action of its own | * The fix: take the level out before the geometry is applied, by centering each site's profile across the cells on its own mean. Then rerun with the level-match |
| 2 | 265 | status report, result or explanation; no action of its own | That construction has to be written and committed before it runs, tested synthetic first, then rerun on the real cells. That's one machine per test. I'll write  |
| 2 | 269 | status report, result or explanation; no action of its own | What the information sphere could be used for |
| 2 | 270 | status report, result or explanation; no action of its own | At heart it gives a true distance between two methylation states: the Fisher–Rao angle, from the same mathematics as the entropy our instruments measure. The ev |
| 2 | 271 | uses of the information sphere: in plan (35.4) | * How far a person is from their own healthy state. We'd measure it as an angle on the sphere instead of a score we define, so a sample has a position, a distan |
| 2 | 272 | uses of the information sphere: in plan (35.4) | * Tracking one person over time. Repeated draws trace a path. A slow drift toward a cancer-like region could show up before any single reading leaves Normal. Th |
| 2 | 273 | uses of the information sphere: in plan (35.4) | * Finding which sites fail together, with no list given in advance. Test 7 showed it finds a group of sites that behave alike and doesn't invent one when there' |
| 2 | 274 | uses of the information sphere: in plan (35.4) | * Where a fragment of DNA came from. Healthy tissues sit in different regions of the sphere, so cell-free DNA could be placed by where it lands. |
| 2 | 275 | uses of the information sphere: in plan (35.4) | * Quality control. An array or lab sitting off the healthy map stands out on its own. |
| 2 | 276 | status report, result or explanation; no action of its own | Next comes the same test on the atlas's real healthy cells, written before it runs. If it passes, the first real use I'd build is the first one, distance from y |
| 2 | 278 | status report, result or explanation; no action of its own | Information sphere vs. the address sphere |
| 2 | 279 | status report, result or explanation; no action of its own | They don't replace each other. They answer different questions about the same sample, so I'd keep both. |
| 2 | 280 | status report, result or explanation; no action of its own | Address sphere (built)	Information sphere (in development) |
| 2 | 281 | status report, result or explanation; no action of its own | A site's place is set by	where it sits on the DNA	how it behaves across cell types |
| 2 | 282 | status report, result or explanation; no action of its own | Answers	Where on the genome a change is	What kind of change it is, and how far the state has moved |
| 2 | 283 | status report, result or explanation; no action of its own | Neighbors are	sites next to each other on the chromosome	sites that act alike, wherever they are |
| 2 | 284 | status report, result or explanation; no action of its own | Its strength	clustering along the DNA (the C-Score's question), the microwave-sky tools, covers the whole globe	a true distance between states (Fisher–Rao), gro |
| 2 | 285 | status report, result or explanation; no action of its own | Its limit	can't tell you two distant sites belong to the same program	can't tell you where on a chromosome a change is; covers about 7% of the globe |
| 2 | 286 | status report, result or explanation; no action of its own | A concrete example: take a loss along one stretch of chromosome 2. On the address sphere it lights as one patch. On the information sphere those same sites may  |
| 2 | 287 | address globe and information sphere side by side: in plan (35.8) | What I'd build for the report: the two globes side by side, sharing the same sites, so selecting a patch on one shows where those sites land on the other. That  |
| 2 | 290 | status report, result or explanation; no action of its own | CMB tools that fit the information sphere. These are ideas to test, synthetic first: |
| 2 | 291 | seven sky tools on the information sphere: in plan (35.5) | 1. Power spectrum (C_ℓ): shows at what scale the variation lives, broad programs or fine detail. Healthy and tumor would get a spectrum each. Since 93% of this  |
| 2 | 292 | seven sky tools on the information sphere: in plan (35.5) | 2. Cross-spectrum of two readings. Noise doesn't repeat between two independent readings, so it drops out and only real shared structure is left. That's exactly |
| 2 | 293 | seven sky tools on the information sphere: in plan (35.5) | 3. Component separation. CMB teams separate the galaxy from the background using several frequencies. Here each cell type plays the part of a frequency, which w |
| 2 | 294 | seven sky tools on the information sphere: in plan (35.5) | 4. Dipole. The largest pattern on the CMB is our own motion. On a person's sphere, the dipole would be the net direction their state has moved, such as a drift  |
| 2 | 295 | seven sky tools on the information sphere: in plan (35.5) | 5. Spherical wavelets (needlets) and matched filters: find compact spots at every size at once. |
| 2 | 296 | seven sky tools on the information sphere: in plan (35.5) | 6. Shape statistics (Minkowski functionals): describe the shape of the changed areas, not just how many there are. That would tell a structured change from rand |
| 2 | 297 | seven sky tools on the information sphere: in plan (35.5) | 7. Stacking: average the map around every site of one kind, such as all gene starts, to bring up a signal too weak to see at any single site. |
| 2 | 298 | cross-system uses (grid, batteries, qubits, fisheries) and the scale-free spectrum test: private ideas, correctly off the public plan | Uses outside our work. One honest point first: the square-root geometry is old mathematics. It is the Bhattacharyya distance (1943), and in population genetics  |
| 2 | 299 | cross-system uses (grid, batteries, qubits, fisheries) and the scale-free spectrum test: private ideas, correctly off the public plan | * Quantum computing: this distance is the classical version of quantum fidelity, the closeness of two qubit states, and qubit states already live on a sphere. T |
| 2 | 300 | cross-system uses (grid, batteries, qubits, fisheries) and the scale-free spectrum test: private ideas, correctly off the public plan | * Fisheries and population genetics: allele frequencies across rivers or hatcheries make a sphere. A stock drifting away from its wild origin would show as a me |
| 2 | 301 | cross-system uses (grid, batteries, qubits, fisheries) and the scale-free spectrum test: private ideas, correctly off the public plan | * Microbiome and ecology: species proportions across samples, where the related Hellinger distance is already standard. |
| 2 | 302 | cross-system uses (grid, batteries, qubits, fisheries) and the scale-free spectrum test: private ideas, correctly off the public plan | * Semiconductors: failure probabilities across process conditions; a drifting fab line would show as motion on the sphere. |
| 2 | 303 | cross-system uses (grid, batteries, qubits, fisheries) and the scale-free spectrum test: private ideas, correctly off the public plan | * Power grids: each substation's state probabilities across operating conditions make a sky. A grid moving toward an unstable region would show as a dipole befo |
| 2 | 304 | cross-system uses (grid, batteries, qubits, fisheries) and the scale-free spectrum test: private ideas, correctly off the public plan | The pattern: wherever there are many units, each with probabilities across conditions, and the question is whether one unit is drifting from its own normal, the |
| 2 | 307 | prior-art check: added as 61.5 | Yes, I think the contribution is that combination. The square-root geometry is old, and so are the sky-mapping tools. Putting a whole system's information on a  |
| 2 | 308 | cross-system uses (grid, batteries, qubits, fisheries) and the scale-free spectrum test: private ideas, correctly off the public plan | A spherical RTCA (real-time contingency analysis) |
| 2 | 309 | cross-system uses (grid, batteries, qubits, fisheries) and the scale-free spectrum test: private ideas, correctly off the public plan | This fits the grid naturally, because RTCA already has the structure the sphere needs: |
| 2 | 310 | cross-system uses (grid, batteries, qubits, fisheries) and the scale-free spectrum test: private ideas, correctly off the public plan | * The sites: every line, transformer and bus. |
| 2 | 311 | cross-system uses (grid, batteries, qubits, fisheries) and the scale-free spectrum test: private ideas, correctly off the public plan | * The "cell types": the contingency list. Each element's response across all N-1 outages, such as loading as a share of its limit, becomes its square-root vecto |
| 2 | 312 | cross-system uses (grid, batteries, qubits, fisheries) and the scale-free spectrum test: private ideas, correctly off the public plan | * What that layout shows: elements that respond alike to outages land together, wherever they sit geographically. A weak corridor shows up as one patch even if  |
| 2 | 313 | cross-system uses (grid, batteries, qubits, fisheries) and the scale-free spectrum test: private ideas, correctly off the public plan | * Self-reference: each element is compared with its own normal for that hour, season and topology, not only with a fixed limit. That's IAM-A's idea applied to t |
| 2 | 314 | cross-system uses (grid, batteries, qubits, fisheries) and the scale-free spectrum test: private ideas, correctly off the public plan | * The tools: |
| 2 | 315 | cross-system uses (grid, batteries, qubits, fisheries) and the scale-free spectrum test: private ideas, correctly off the public plan | * Change map: lights the elements drifting from their own normal before any of them alarm. |
| 2 | 316 | cross-system uses (grid, batteries, qubits, fisheries) and the scale-free spectrum test: private ideas, correctly off the public plan | * Dipole: the net direction the whole system is moving through the day. |
| 2 | 317 | cross-system uses (grid, batteries, qubits, fisheries) and the scale-free spectrum test: private ideas, correctly off the public plan | * Cross-spectrum: compares two consecutive state-estimator snapshots and cancels measurement noise, so only real movement remains. |
| 2 | 318 | cross-system uses (grid, batteries, qubits, fisheries) and the scale-free spectrum test: private ideas, correctly off the public plan | * Shape statistics: tell a localized problem (one patch) from system-wide stress (spread out). |
| 2 | 319 | cross-system uses (grid, batteries, qubits, fisheries) and the scale-free spectrum test: private ideas, correctly off the public plan | * What the operator sees: one turning globe that stays dark when the system is holding its normal. Patches appear where it's drifting, ranked by how far they've |
| 2 | 320 | cross-system uses (grid, batteries, qubits, fisheries) and the scale-free spectrum test: private ideas, correctly off the public plan | How I'd build it without touching anything from your job: |
| 2 | 321 | cross-system uses (grid, batteries, qubits, fisheries) and the scale-free spectrum test: private ideas, correctly off the public plan | 1. Public synthetic grids only. Public test systems (the IEEE 118-bus case and Texas A&M's synthetic grids) come with open power-flow tools, so every number wou |
| 2 | 322 | cross-system uses (grid, batteries, qubits, fisheries) and the scale-free spectrum test: private ideas, correctly off the public plan | 2. A synthetic test first, as always. Plant a slow overload drift in one corridor and check that the globe lights it before RTCA's own violation list does, and  |
| 2 | 323 | cross-system uses (grid, batteries, qubits, fisheries) and the scale-free spectrum test: private ideas, correctly off the public plan | 3. Keep it separate from your employer. Develop it on your own time and your own machines, with public data. Before showing anyone at work, check your employmen |
| 2 | 324 | cross-system uses (grid, batteries, qubits, fisheries) and the scale-free spectrum test: private ideas, correctly off the public plan | This goes on the whiteboard list behind the human work, the same way the salmon project runs in machine downtime. When there's a gap, I'll start with the prior- |
| 2 | 326 | cross-system uses (grid, batteries, qubits, fisheries) and the scale-free spectrum test: private ideas, correctly off the public plan | One thing first about the RTCA idea: if it might be worth money, I'm keeping it out of the public repo, webpage, book and day logs. Publishing it there could co |
| 2 | 328 | cross-system uses (grid, batteries, qubits, fisheries) and the scale-free spectrum test: private ideas, correctly off the public plan | IAM and the sphere aren't separate. They're two layers of the same thing, and they fit together naturally. |
| 2 | 329 | cross-system uses (grid, batteries, qubits, fisheries) and the scale-free spectrum test: private ideas, correctly off the public plan | * IAM is the measurement. IAM's Law tells you what to measure at each unit and what that unit's own normal is. For methylation, that's the held error against th |
| 2 | 330 | cross-system uses (grid, batteries, qubits, fisheries) and the scale-free spectrum test: private ideas, correctly off the public plan | * The sphere is the geometry. It decides where each unit sits and how far apart two states are. |
| 2 | 331 | cross-system uses (grid, batteries, qubits, fisheries) and the scale-free spectrum test: private ideas, correctly off the public plan | * The CMB tools read the map: clustering, the dipole, cross-spectra. |
| 2 | 332 | cross-system uses (grid, batteries, qubits, fisheries) and the scale-free spectrum test: private ideas, correctly off the public plan | They aren't independent. The sphere's distance (Fisher–Rao) is, by established mathematics, the small-scale form of relative entropy: entropy differences, measu |
| 2 | 333 | cross-system uses (grid, batteries, qubits, fisheries) and the scale-free spectrum test: private ideas, correctly off the public plan | What "building IAM in" would look like: |
| 2 | 334 | cross-system uses (grid, batteries, qubits, fisheries) and the scale-free spectrum test: private ideas, correctly off the public plan | 1. Place: each unit sits on the sphere by how it behaves across conditions. |
| 2 | 335 | cross-system uses (grid, batteries, qubits, fisheries) and the scale-free spectrum test: private ideas, correctly off the public plan | 2. Layer: each unit's IAM reading against its own prediction, as in IAM-A. Normal is dark; a unit losing its held order lights up. |
| 2 | 336 | cross-system uses (grid, batteries, qubits, fisheries) and the scale-free spectrum test: private ideas, correctly off the public plan | 3. Lever: each system's own physical levers set its prediction. For methylation, that's writing and erasing (f and g). For a battery, it could be degradation se |
| 2 | 337 | cross-system uses (grid, batteries, qubits, fisheries) and the scale-free spectrum test: private ideas, correctly off the public plan | 4. Read: the same CMB tools on every system: where change clusters, which way the whole system is drifting, and what repeats between two readings. |
| 2 | 338 | cross-system uses (grid, batteries, qubits, fisheries) and the scale-free spectrum test: private ideas, correctly off the public plan | Designs as well as monitoring: in principle, yes. A design is a predicted position on the sphere, and two designs can be compared by where their states land and |
| 2 | 339 | cross-system uses (grid, batteries, qubits, fisheries) and the scale-free spectrum test: private ideas, correctly off the public plan | Like the RTCA, the battery and grid versions stay out of everything public. I'm still waiting on the workflow test before the Operator Guide section goes up. |
| 2 | 342 | cross-system uses (grid, batteries, qubits, fisheries) and the scale-free spectrum test: private ideas, correctly off the public plan | What's real, not just analogy: |
| 2 | 343 | cross-system uses (grid, batteries, qubits, fisheries) and the scale-free spectrum test: private ideas, correctly off the public plan | * The CMB is itself a sphere. It's the surface of last scattering, a slice of our past light cone seen from one point. The tools we borrowed were built for exac |
| 2 | 344 | cross-system uses (grid, batteries, qubits, fisheries) and the scale-free spectrum test: private ideas, correctly off the public plan | * The information sphere is the natural shape of probability. Any set of probabilities, under the square-root step, lands on a sphere, whatever the system. That |
| 2 | 345 | cross-system uses (grid, batteries, qubits, fisheries) and the scale-free spectrum test: private ideas, correctly off the public plan | * That gives IAM one space to live in. Instead of a ladder of separate applications, from the hydrogen atom up to the horizon, every system's held order becomes |
| 2 | 346 | cross-system uses (grid, batteries, qubits, fisheries) and the scale-free spectrum test: private ideas, correctly off the public plan | What's analogy until we derive it: |
| 2 | 347 | cross-system uses (grid, batteries, qubits, fisheries) and the scale-free spectrum test: private ideas, correctly off the public plan | * They're spheres in different spaces. The expanding cosmic sphere is a horizon in spacetime; the information sphere is a space of states. World lines run throu |
| 2 | 348 | cross-system uses (grid, batteries, qubits, fisheries) and the scale-free spectrum test: private ideas, correctly off the public plan | * Saying they're one and the same is a physics claim. It would need a derivation from IAM's Law and a prediction someone could check. Right now it's a strong in |
| 2 | 349 | cross-system uses (grid, batteries, qubits, fisheries) and the scale-free spectrum test: private ideas, correctly off the public plan | A test that could turn the intuition into evidence: if IAM really is one law at every scale, the same reading should look alike across systems. For example, the |
| 2 | 350 | cross-system uses (grid, batteries, qubits, fisheries) and the scale-free spectrum test: private ideas, correctly off the public plan | I'm noting this privately, not in the public repo, alongside the other cross-system ideas. |

File 1 is the to-do lists (pasted-text-2026-11-05T21-47-58.txt); file 2 is the atlas and information-sphere notes (pasted-text-2026-11-05T21-49-15.txt).
