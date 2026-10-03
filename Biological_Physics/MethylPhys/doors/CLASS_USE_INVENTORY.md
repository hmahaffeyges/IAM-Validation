# Class-use inventory and removal plan — 2026-09-28

**The rule (author, 2026-09-28):** a cell's class is the name of the floor it is divided by. Nothing else in the instrument reads it.
Anything that could lead a reader to treat a class as a group of cells — a class score, a class gauge, pooling by class — is removed.

Scan: every `.py/.md/.tex/.json` under `Biological_Physics/MethylPhys` outside RETIRED and generated outputs; 182 files touched.
Raw hits: H_min references 2,038 (allowed); "class A / class score" 337; per-class pooling or aggregation wording 631;
class-map reads 115; class-keyed lookups other than H_min 44. The regex is coarse — many "per-class" hits are the floors themselves,
which stay. The table below is what was read by hand in the live chain.

## Stays — the one allowed use
| where | what |
|---|---|
| `hmin_calibration/` (G-002) | fits one floor per class — that IS the floor |
| A-scoring H_min lookup | cell → class → H_min. The only reader of the class map once v2 lands |

## Removed in the live chain (the reading depends on it — goes at the v2 switch-over, when per-cell identity loci replace it)
| where | class use | replaced by |
|---|---|---|
| cpg_conductor.py (archived privately) Stage B (`stage_b_identity`) | the class gauge: A over a class marker union; internal "is this blood-like" gate; `composition_verified` | per-cell identity loci (v2 acceptance test); composition from the deconvolver alone |
| iamatlas_gauge_identity_loci_v1_0.json (archived privately) | identity loci keyed by class (`["immune"]`) — every blood cell is read on the immune set | per-cell identity loci |
| the class-era deconvolver (archived privately) twin merge | two cells merge only if they share a class ("Same class required") | the sample-level twin test (non-overlapping CpGs), no class condition |
| the class-era deconvolver presence gate | "per-class fraction CI", "class-level solve" wording and output | per-cell presence only |
| `cpg_conductor.py` l.280 | `groups = {"immune": ["immune"], "haematopoietic_progenitor": ["progenitor","stem_adult"]}` | removed |
| disease_matching.py (archived privately) | per-class identity CpGs; "per-class H_min floor" check | per-cell identity loci; floor check stays (it is the H_min lookup) |
| Stage 2d / trace panel | "threshold per class" | per-cell detector constant |

## Removed now (dead code or text; no reading changes)
| where | what |
|---|---|
| iamatlas_a_scoring.py (archived privately) `score_per_class()` | "8 class A-scores" — defined, called by nothing |
| build_methylphys.py (archived privately) | header "A per cell and per class"; the internal-gate paragraph on the Story tab; CMB twin row "posterior per CpG per class" |
| SOP (372 hits), OM (om_data.py (archived privately), build_operations_manual.py (archived privately), switching_order.py (archived privately), om_part3.py (archived privately)), RUNBOOK, README, HANDOFF | class A-score, class gauge, class-level reading passages. Floors per class stay, worded as floors |
| kit tests test_tiers.py (archived privately), test_gauge_switch.py (archived privately), test_percell_physics.py (archived privately), test_a_score_canonical.py (archived privately) | class-keyed assertions |

## Record, not live (marked, not rewritten)
manual/om_lib.py (archived privately), ../RETIRED_2026-10/MethylPhys/papers/build_mphys_issue002.py (archived privately), Issue 002/003, `IAM_Hubble2Methyl_Alpha_Omega_5.tex`,
`Mahaffey_2026_cell_thermodynamics.tex`, the atlas v0.1 vault READMEs: historical documents of how the classes were reasoned.
Each gets a one-line banner: *"Record. In the commissioned instrument a class is only the floor a cell is divided by."*

## The guard (added with the removals)
kit/class_guard.py (archived privately), run by the release check and by [`build_all.py`](../chain/build_all.py) before every push: fails if any chain module other than the
H_min lookup reads the class map, groups or averages by class, or prints a class A; fails if any live document or report tab uses
"class A", "class score", "class gauge" or "class reading". Record documents are exempt by banner, the way the vocabulary scan is.
