cat > TODO_2026-10-01.md <<'EOF'
# IAM — master to-do list (updated 2026-10-01, 11:00 UTC)
Status: **RUNNING** (on a box now) · **NEXT** (ready to start) · **WAITING** (needs data or a decision) · **DONE** (recorded)

## A. Human tests — Met-A, IAM-A, C-score (one cell type: neutrophils)
| # | task | status |
|---|---|---|
| A1 | PROC-AML-SERIAL-01: Met-A + C-score, 10 AML patients read at diagnosis → remission 1 → remission 2 (GSE315367); healthy C-score line from Salas EPIC | **RUNNING** box 1 |
| A2 | PROC-TUMOUR-01: IAM-A, primary tumour vs same patient's adjacent normal (early-onset colorectal; oral SCC) | **RUNNING** box 1 |
| A3 | C-score healthy band: more held-out healthy EPIC arrays (purified + whole blood) so the band is not set from 91 arrays alone | NEXT after A1 |
| A4 | Met-A vs IAM-A on the SAME neutrophils: find public array + bisulfite sequencing from the same donors | NEXT (search) |
| A5 | Fix the per-molecule IAM-A floor so it uses the same statistic as the reading (PROC-MOLECULE-01 lesson), then re-test detection | NEXT |
| A6 | MDS/CML (GSE315451, 126 marrow): Met-A descriptive on the myeloid clone | WAITING on A1 result |
| A7 | Fixed reference DNA on every slide (single-patient tare) — design for a real lab | WAITING (lab) |
| A8 | Tier lines re-set on the new scale; report A + interval only until commissioning | WAITING on A1–A5 |
| A9 | VAL triage: which of the 200+ VALs survive the platform floor + slide tare | NEXT |
| A10 | Ledger of settled cell results (for the cell part of the book) | NEXT |
| — | Done today: reference floors per cell type/platform; near on/off sites; whole-blood shared sites; slide tare (EPIC-Italy 329: 2 of 3 pass); neutrophil sky map; ENCODE (architecture, cancer > normal) | DONE |

## B. Salmon, fish and animal tests (Chelan PUD route)
| # | task | status |
|---|---|---|
| B1 | PROC-CHARR-01: brook charr, 40 males, ambient vs +2 °C, selected vs control, one lab (pre-registered) — genome index + timing pilot | **RUNNING** box 2 |
| B2 | B1 full run: all 40 fish at equal depth (N set from the pilot) | NEXT after pilot |
| B3 | Rimouski Atlantic salmon: captive vs wild adults + offspring, WGBS (PRJNA892473, 1.2 TB) — hatchery effect and inheritance | NEXT after B2 |
| B4 | 580-species methylome atlas (PRJNA802599): error in kT vs body temperature and lifespan — read the vertebrate lifespan paper in full FIRST, then pre-register | NEXT (reading) |
| B5 | Coho milt WGBS, two rivers (PRJNA678281): per-fish reading on whole-genome data vs RRBS | WAITING on B2 |
| B6 | Le Luyer coho hatchery vs wild RRBS (PRJNA389610), on the trout index we already have | WAITING on B2 |
| B7 | Methow re-read with the library-quality control learned in B1/B2 | WAITING on B2 |
| B8 | Chelan PUD pilot design: archived tissue × PIT/PBT returns; questions for your supervisor (tissues, storage, PIT/PBT link, who holds the archive) | WAITING (you) |
| B9 | Salmon paper for a fisheries reader (management question first, physics in the appendix) | WAITING on B2–B3 |

## C. Cosmology book — *Law and Order: The Thermodynamics of Informational Actualization* (Parts 1, 2, 5)
| # | task | status |
|---|---|---|
| C1 | Read every Part 1/2/5 paper in full and audit | DONE (50 docs) |
| C2 | Level 1 Runs A and B to R−1 < 0.01 | **RUNNING** box 2 |
| C3 | One documented extraction of every chain number from the final chain files (all 18) | NEXT after C2 |
| C4 | Check which MGCAMB µ form Level 1 used (normalised vs not: µ(0) 0.865 vs 0.908) from the MGCAMB source | NEXT |
| C5 | Apply the 8 recurring corrections everywhere (n = 7/2; 15 + 2 chains; β fixed, not sampled; 1 − µ is not growth; drop 5.5σ / 444,000× / 8.9σ; "per e-fold peaks today"; drop η 0.815; photon-sector observables) | NEXT |
| C6 | Physics-language rewrites flagged by the readers (IAM's Law, Holographic, Variational, w(z) far future, DM/DE partners, BH paradox, measurement problem, GRF essay, BH working document); replace repo copies + give you copies | NEXT |
| C7 | Rewrite the CC and baryon papers on the "observed relation, derivation open" basis | NEXT |
| C8 | Update the outline (IAM_BOOK_OUTLINE_PARTS_1_2.md) with the audit: what stands, what moves to Part 5 open problems, what is dropped | NEXT |
| C9 | Predictions register: remove the derivation suite's unsourced "DESI" Test 9 (Δχ² 31.2); add the corrected fσ8 result | NEXT |
| C10 | Speculative items to discuss with you before anything is written: gravitational engineering / propulsion notes; the β_m dated record | WAITING (you) |
| C11 | Write Part 1 (law, virial theorem, 37 orders) and Part 2 chapters, re-running every number and figure | WAITING on C3–C8 |
| C12 | Part 5: interpretation chapters + open problems + the whole web | WAITING on C11 |
| C13 | Compile the book (LaTeX on a box) | WAITING on C11 |

## D. Housekeeping
- S3 upload/download links for the boxes expire about 7 Oct — renew before then.
- Every push goes through the canon-checked push, with copies of the changed files.
EOF
echo ok