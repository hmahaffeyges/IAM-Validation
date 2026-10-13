# IAM to-do: every open item, by track (updated 2026-10-13)

Built from the author's lists of 10-08 to 10-12 and the record; nothing dropped from the 10-12 morning list.
Feasibility: **L** = laptop only, **B** = needs a rented machine, **S** = search first, **A** = author decision.
Rules for every item: synthetic tests first, seal before reading, one cancer at a time, public data only, American spelling, every push through the gate.

## 0. Running now / next hours

| # | Item | Status | Next step |
|---|---|---|---|
| N1 | **Stage B, EpiQC sequencing (A2)** | on an on-demand c6a.32xlarge after four spot reclaims; 55 of 71 at about 1 PM Pacific | when 71 of 71: **terminate the machine** (`spot_from_box.py release i-08550fe70a8e58348`), laptop reader finishes, score once under the seal, review, report |
| N2 | **Atlas v2 public deposit** | 4.67 GB pulled from S3, checked file by file against the manifest, packed into 7 archives | author approves deposit details (title, author, MIT license, files); publish; then **one push**: DOI into atlas README, book atlas chapter, status fact `atlas_v2`; test and commit `fetch_atlas.py` on the live record |
| N3 | Day summary 2026-10-12 in the development log | due before tomorrow's first push (gate rule 5) | write at end of day |
| N4b | **IAM-A laboratory failure: cause diagnosed 2026-10-13** | MIXED - the three disagreeing libraries read 11-14% low at the file head, controls flat; commissioning stays HELD | write the read-across-the-file rule, planted test, seal, then re-read the whole stage (the 5 diagnosis libraries are spent) |
| N5 | **Lung today (author: do not end the day on lung failure)** | LUSC v1, v2 failed synthetic tests (stroma; no lung stroma in atlas); 40 tumors unread | v3: real lung stroma (GSE146376 airway smooth muscle 54, GSE107226 lung fibroblasts 4) and lung interstitial macrophages in the panel and the direction test; run once; if pass seal, fetch, score tonight; if fail, TCGA-LUAD with v3 |
| N4 | Live book PDF carries the funding paragraph | website rebuild was pending | confirm in the live PDF |

## A. IAM-A commissioning (the instrument that tests the law)

| # | Item | Status | Next step |
|---|---|---|---|
| A1 | Narrow scope, bars first: sorted neutrophils, WGBS, pinned pipeline, same-run references | C1 decided (0.98, no correction); C2-C5 sealed | score with A2 |
| A2 | EpiQC identical DNA, 7 lines x 5 kits x 2 labs | Stage A done (C1 met); Stage B running (N1) | score once; TruSeq may fail C2, recorded not adjusted |
| A3 | Other-lab healthy neutrophils | lead from atlas search: GSE59163 (RRBS, 13 healthy) | verify from files; design |
| A4 | Second physics lever | none public found | fatty liver without SAM loss (C1); DNMT1-reduced mice; methyl-donor diets |
| A5 | Per-territory IAM-A (held vs gained) | reader built and passed | seal a prediction on MCF-7 territories before reading |
| A6 | Met-A to IAM-A link as a standing sealed prediction | dose series no result; donors off curve | seal a link window on EpiQC (same DNA, arrays and sequencing) |
| A7 | Oral cancer tumor/normal WGBS, 4 patients | per-site tables only in S3 | trace source, re-fetch read-level files, after A1 |
| A8 | **Neurons, second cell type** (DEV-IAMA-NEURON-01) | reference built by the rule (2 of 5 donors inside Normal); decision made: keep it, add a sorted-only reference labeled as a post-reading rule change | seal the sorted-only reference's test on the two unread labs (Feinberg GSE96612, Yi GSE107729 controls) before opening them |
| A9 | Neurons: copy error in non-dividing cells | balance section DERIVED in book; age test is a PREDICTION | derivation note, then synthetic test |
| A10 | Neurons: post-mortem time | not checked | check copy error against post-mortem time in the Yi controls before trusting post-mortem neurons |
| A11 | Neurons on Met-A (both instruments, same cells) | array sets: Pishva (Maastricht, 39 donors), Labrie (33 arrays); new: GSE306226 (20 donors, 4 brain cell types, 2025), GSE234520 (43 donors) | build a neuron Met-A reference by the SOP new-cell rule |

## M. Met-A commissioning extensions

| # | Item | Status | Next step |
|---|---|---|---|
| M1 | **Whole blood back into scope (bar 7)** | GSE167202 held unread (clinical counts too noisy); new lead **GSE127824** (cord blood, matched flow counts, independent group) | assess the cord-blood caveat (nucleated red cells, newborn mix); if usable, synthetic tests, seal, read |
| M2 | **Neutrophil reference from the atlas** (U4 first case) | plan written; current reference unchanged | derive sites and healthy reference keeping per-sample noise; pass the same bars on held-out arrays (GSE224807 granulocytes); re-commission |
| M3 | Reference balance: 6 arrays, one lab, 5 male / 1 female | open problem in book | solved by M2 if it passes |
| M4 | 450K neutrophil reference (GSE88824, 8 donors) | pending | build and freeze |
| M5 | C-score bars narrowly unmet | development | author decision on one-sided reading and absolute repeatability bar |
| M6 | Healthy Met-A across ages (donors 20-39) | S | search purified neutrophil arrays across ages |

## AT. Atlas v2 (showcase, deposit, commissioning, growth)

| # | Item | Status | Next step |
|---|---|---|---|
| AT1 | Showcase folder, README, book chapter, main README | **done** today (what is new; questions builders would ask; built for IAM, independent of its physics); class-era REBUILD removed | — |
| AT2 | Public deposit with DOI | N2 | — |
| AT3 | **U2, composition-blind sites for tissue tests: commission first** | first finding: lung blind sites failed (no lung stroma) | bars on independent purified cells |
| AT4 | U1, profiles on independent labs | leads: GSE224807 (141 donors, 6 blood types, 2023), GSE184269, GSE103541, GSE252045, GSE59065; GSE123995 hepatocytes; GSE140295 endothelium; GSE52025 fibroblasts | split each into build and held-out data before reading; verify IDATs and donor counts from files |
| AT5 | U3, blood composition | CD4/CD8 outside bar | find the source of the shortfall |
| AT6 | U4, per-cell identity sites | held-out 87.7 % (arrays) and 73.1 % (sequencing) Normal, bar 95 % | more purified samples per cell on one platform |
| AT7 | **Lung stroma into the atlas** | candidates: GSE146376 airway smooth muscle (54 healthy donors, EPIC), GSE107226 lung fibroblasts (4 normal, 450K) | culture check against uncultured lung; build/held-out split; v2.1 |
| AT8 | 19 cell types with no independent candidate (incl. adult colon epithelium, lung alveolar and breast epithelium on arrays) | "none" is provisional (GEO keywords only, one pass for some types) | deeper search: ArrayExpress, ENA, controlled-access (BLUEPRINT, DEEP, IHEC) |
| AT9 | New cell types | leads: HSPC sequencing GSE87196, cord-blood HSC GSE273074, adipocyte WGBS GSE111445 | after AT7 |
| AT12 | **Gold mine map**: classify every candidate from today's search and the 2025 compendium (build / held-out test / truth set / research / not usable), map by cell type, priority list | sub-agent running (metadata only) | review; commit map; feed AT4, AT7-AT9, M1, A3, A11 |
| AT13 | **Master Inventory** (author 2026-10-12): one generated document of every data set: in S3 (format, readable), public not downloaded (gold-mine map and candidate lists), status (unread / designed / sealed / scored / not suitable / prior reading), use and value, ranked next jobs by impact and cost | after AT12 | generator script from S3 check, data_spent.csv, notes, gold-mine map; older inventories frozen or made inputs; push rule: a test note naming a set not in the inventory is refused |
| AT10 | Six-month search rule | gate rule 19; last search 2026-10-12 | next due by 2027-04-12 |
| AT11 | Versioning: each addition = new version, own DOI, re-commissioned before the chain switches | rule written | first use with v2.1 |

## R. Replication and public record

| # | Item | Status | Next step |
|---|---|---|---|
| R1 | Tissue scorers call the chain's Met-A step | scorers carry their own copy of the math | one Met-A function in the chain; scorers call it |
| R2 | Tissue tests print a chain report | outcome note and scored file only | report per scored pair set |
| R3 | **Replication page per public result** (milestones only, author scope) | none yet | generate from the results register; push rule refusing a milestone without one |
| R4 | **Clean-room rerun** | Met-A neutrophil rebuild done earlier | liver pilot (after N2), then breast, colon; page says "not yet rerun" until passed |
| R5 | Class-era guides (COMPONENT_MAP, REVIEWER_MANIFEST, RUNBOOK) still describe the old chain and atlas | flagged in log | rewrite whole, with R3 |
| R6 | American spelling in the book and technical notes | donor-facing text done (rule 16) | separate pass with every book check rerun |
| R7 | Social media wording | true now: code, seals and records public; not yet: one-command rerun | update after R4 |

## B. Cancer tracks (one at a time)

| # | Item | Status | Next step |
|---|---|---|---|
| B1 | MCF-7, T47D vs breast normals | gain 4/4, loss 3/4 FINGERPRINT | done |
| B2 | TCGA-BRCA, 53 patients | FINGERPRINT (23 of 53) | second independent breast set for replication |
| B3 | Colorectal GSE131013, 88 patients | FINGERPRINT (28 of 88) | replication: GSE199057 not suitable; find another same-slide set |
| B11 | **Liver TCGA-LIHC, 50 patients** | **FINGERPRINT (22 of 50)** today | replication on fresh data: GEO liver arrays (check slide pairing), PRJNA762641 sequencing (box job about $30-50, ledger first) |
| B12 | **Lung** | lung 1 (GSE94785) NO FINGERPRINT; lung 2 (GSE83842) and lung 3 (TCGA-LUSC) not suitable, tumors unread | after AT7 (lung stroma in the atlas): redesign; LUSC and TCGA-LUAD tumors still unread |
| B10 | HepG2 / A549 read both ways | open (one arm only, batch-scored) | via B11/B12 replications |
| B13 | **Liver in blood; the systemic question** (DEV-LIVER-BLOOD-01) | HCC plasma vs healthy candidates; transplant recurrence data controlled-access only | sealed test of commissioned Met-A on neutrophils of cancer patients vs healthy donors; liver cfDNA after B8 |
| B4 | Colon adenomas GSE288652 (pre-cancer) | in hand | does Met-A rise before cancer? |
| B5 | Colorectal GSE166212 (EPIC); liver tissue GSE304513 | in hand | read the designs |
| B6 | Cell-free DNA sets (liver GSE298812, lung GSE277078, TNBC GSE207998, prostate GSE119260, others) | in hand | needs healthy cfDNA references first |
| B7 | Blood cancers (AML GSE159907, GSE175758; CLL GSE136724, GSE295456; GSE237299) | in hand | references for blasts and B cells (SOP new-cell rule) |
| B8 | Tumor dilution series | S | sets the detection limit, required before any early-detection claim |
| B9 | Replication package (kit, open predictions, preprint) | — | after R3-R4 |
| B14 | Dogs' cancers (fundraiser page names them) | not started | search after the human tracks |

## G. Gauge questions (measured, never set)

| # | Question | Next step |
|---|---|---|
| G1 | Where on the gauge does cancer begin? | sealed test across healthy mucosa, adenoma, adjacent normal, tumor (B4, B3) |
| G2 | A Warburg line? (conjecture) | author derivation first |
| G3 | What reads below Normal? | search a public senescence set with raw IDATs |
| G4 | Neurons: the low end | A8-A11 |

## C. Physics and supporting work

| # | Item | Next step |
|---|---|---|
| C1 | Fatty liver without SAM loss (GSE233768; GSE85772, GSE231727) | check liver SAM in that diet model |
| C2 | Temperature: hamster hibernation (designed) | download and read as designed (B) |
| C3 | Aging across species | redo the cross-species reading with the reference-free reader |
| C4 | Passage series GSE268145 (copy error with divisions) | design and simulate |
| C5 | Lavage set 2: newer EPIC revision in Stage 1 | add the manifest revision; isolate the third address pair |
| C6 | Genome-wide reading, same-run references | write bars |
| C7 | Reproducibility backlog | next tier |
| C8 | 450K disease readings blocked (slide batch) | after a 450K fix |
| C9 | Fish (Salmonid), other animals on the S3 data | after commissioning |
| C10 | Methionine (parked: too few divisions) | wait for a weeks-long restriction set |
| C11 | Washout memory (DNMT1 degron) | needs a lab-produced public set |

## O. Operations and fundraiser

| # | Item | Next step |
|---|---|---|
| O1 | Terminate Stage B on-demand machine at the end (it only stops) | N1 |
| O2 | Spot reclaims (four today) | launcher has subnet and `--ondemand`; consider multi-type launch next time |
| O3 | Per-run cost in the ledger | activate the `run` cost-allocation tag once AWS lists it |
| O4 | GoFundMe page | paste the 5 American-spelling lines if not done; repo copy and checks (rules 15-16) follow the page |
| O5 | Workstation to replace rented hours | pricing on the fundraiser page; decide when funded |

## L. Levers and candidate instruments (2026-10-13)

Registers: `CANON/levers.json` (board in `Biological_Physics/MethylPhys/doors/DEV_LEVERS_01.md`) and `CANON/instruments.json`
(board `Biological_Physics/MethylPhys/INSTRUMENTS.md`). Every lever and instrument carries the reading that would break it.

| id | item | stands | next |
|---|---|---|---|
| L1 | **Temperature lever** | designed; cross-species is weak | find hibernator tissue, warm and cold, same animals (S) |
| L2 | **Division-rate lever** (neurons) | reference rule sealed | read the two independent laboratories (L) |
| L3 | **Methyl-supply lever** | parked | write the design from DNMT1's SAM constant, predicted shift fixed first (L) |
| L4 | **Write and erase lever** (test 3) | met in simulation; copy and de novo separate only by size | the run-length spectrum instrument (after IAM-A) |
| L5 | **Clustering meter**, last check | blocked: no unspent same-person repeat arrays | keep searching (S) |
| L6 | **Territory contrast** v2 | 3 of 6 bars met | decide v3 or retire (L) |
| L7 | **Whole-blood Met-A** | per-person out; group comparisons work in one series | find cancer cohorts with same-series healthy controls (S) |
| L8 | **Distance to the ceiling** | done on 219 pairs; breach edges committed | DEV-BREACH-01 (H2) |
| L9 | **Sequencing trim rule** | planted test, 6 of 6 bars | one short box read, then rerun the held lab check (B) |
| L10 | Run-length spectrum; plasma fragment tail | waiting on IAM-A | after L9 |

## H. Hypotheses and new designs (2026-10-13)

| id | item | stands | next |
|---|---|---|---|
| H1 | **DEV-FIELD-01**: healthy donor, other breast, other quadrant, adjacent, tumor (GSE287331, EPIC v1, 446 arrays, one series) | SEALED 2026-10-13 (planted test met; slide and position checks done) | download and calibrate 446 arrays, score once (B) |
| H2 | **DEV-BREACH-01**: the breach line from lesions caught before cancer with known outcome | DESIGNED on GSE281307 (untreated: 40 progressed, 46 not; slides shared); SEALED 2026-10-13; GSE108123 (lung precancer) second | download, calibrate, score once (B) |
| H3 | **DEV-ERASE-01**: the 0.74 kT gap - erasure or missed restores (TET knockout; neurons; field vs tumor) | to design | search TET-knockout sets, declare spent (S) |
| H4 | **HYP-SATURATION-01** falsifiers 5 and 6 on the 219 tumors (vessel/metabolic signatures; recurrence vs shuffle) | written | run both (L) |
| H5 | HYP-SATURATION-01 falsifier 7 (program then disorder) | written | needs IAM-A discordance, then an ordered lesion series |
| H6 | **TCGA split**: which cancers explore, which stay sealed | to write | commit before opening any (L) |
| H7 | Loss side of phi outside the cell: TET rates, maintenance lag | to search | literature, metadata only (S) |
| H8 | **Spread as cascade** (falsifier 8): paired primary/metastasis, distant tissue before seeding, blood or cfDNA localized vs spread | search not started | search, declare spent, bars first (S) |

## Closed today (2026-10-12)
Liver FINGERPRINT (B11). Lung 1 NO FINGERPRINT; lung 2, lung 3 and colorectal replication not suitable (tumors unread). Neuron reference built.
Fingerprint results in one register written to four places; plain-language explanation for every result (rule 14); live fundraiser text held to
the records (rule 15); American spelling (rule 16); one lessons file (rule 17); one funding statement (rule 18); six-month atlas search (rule 19).
Atlas showcased; class-era atlas removed; atlas claims checked and locked (status fact `atlas_v2`). First atlas search and lung stroma
candidates recorded. Lessons T15, T16, C6 (third occurrence).
