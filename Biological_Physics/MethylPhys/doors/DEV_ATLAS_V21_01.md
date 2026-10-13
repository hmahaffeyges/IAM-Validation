# DEV-ATLAS-V21-01 — IAM-Atlas v2.1: roster, build/held-out split and plan (design, 2026-10-12; no v2.1 data read)

**Author decision (2026-10-12):** build v2.1, run the joint fit and commission it, as one project; the rented machine is approved. Each version
gets its own DOI and re-passes commissioning before the chain switches; results scored on v2 stay on v2.

**Prep** (metadata, file listings and open-access papers only; nothing opened): `data/DEV_ATLAS_V21_01/atlas_v21_plan.md` (plan, split rule,
culture check, compute and download estimate, open questions) and `atlas_v21_roster.csv` (one row per series and cell type, with role, donor
lists, verified facts, entry-rule status, risks). Re-query scripts: `fetch_meta.py`, `fetch_papers.py`, `summarize_meta.py`. Source: the gold-mine
map (`data/DEV_ATLAS_COMMISSION_01/gold_mine_map.md`). Limits: papers of 7 series were not open access; donor counts where GEO has no donor field
count arrays; EGA, dbGaP, IHEC, ENA-only and CNCB were not searched; the compute estimate is a scaling argument, not a dry run.

**Split rule, fixed now (before any array is read):** whole series to one role by default; a series already read by any method builds and does
not test; a large unread series is split by donor (natural sort of GEO donor IDs; the first 12 odd ranks build, all others held out); all arrays of
one donor take the same role; at most 12 donors per source and cell type build; a set used to build cannot test, and a test set shares no GEO
contributor with any build source of the cell it tests. Donor lists are in the roster; any change is a new dated entry, never an edit.

**Decisions taken on the prep's recommendations** (author standing rule: the most defensible choice):
- granulocyte fractions enter as neutrophils only if marker-selected (CD15+ or CD16+) and passing the sample-level twin test against the atlas
  neutrophils;
- 12-donor cap adopted;
- 450K-only new cells allowed, with EPIC-only loci recorded as not measured (no filling);
- deferred to a later version: EPIC v2 thymocytes (until the Stage 1 EPIC v2 read check), sperm, thymus, cord-blood stem cells and placenta (no
  use needs them; each would add an assumed source term), adipocyte sequencing;
- cord-blood sets GSE68456 and GSE82084 build nothing, so GSE127824 stays a clean truth set;
- GSE306226 glia, GSE146552 ovarian and tubal epithelium, E-MTAB-4957 children's colon and ileum epithelium: conditional, each with the purity or
  identity check named in the plan, decided before the build;
- GSE132547 and GSE131989: used only after the independence checks named in the plan;
- sex inferred from X and Y probes at Stage 1, after the split is sealed;
- lung stroma: research-only in v2.1 (every public set is cultured); the culture check in the plan runs first;
- compute: NBLOCK 1,400 so 16 or more fits run at once; estimated 21-30 hours at 16 processes on the v2 machine type, entered in the ledger before
  launch.

**Firm build (prep count, provisional until Stage 1):** 428 new samples from 13 independent series; 6 new cell types; 80 cells in all.

**New cell types checked against the cancers and blood tests planned (author, 2026-10-12; gold-mine map and roster, metadata only).**
- Blood cancers: bone-marrow stem cells and promyelocytes (GSE49618) are in the firm build; cord-blood stem cells conditional.
- T-cell leukemia that starts in the thymus: thymocytes (GSE269608) are on EPIC v2, which Stage 1 does not read yet. An EPIC v2 read check
  would open them and other EPIC v2 sets; added to the job list.
- Myeloma (plasma cells) and B-cell subsets: GSE118255 is RRBS, which the build rule does not admit; held-out test only.
- Melanoma: melanocyte sets are cultured (GSE166069), too few donors (GSE103413) or of unverified access (GSE16368); the culture check decides
  whether the cultured set can be used, as for lung stroma.
- Placenta (GSE271697): matters for blood tests in pregnancy (placental DNA in the mother's blood), not for any test planned now; kept research
  until a pregnancy-relevant blood test is designed.
- Sperm, dura fibroblasts, nasal epithelium, myoblasts, tonsil NK intermediates: no planned test needs them.

**Next:** the culture check and the conditional checks (laptop); ledger entry; then the build run.

**Test data:** none (design)
