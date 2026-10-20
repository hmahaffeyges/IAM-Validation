# Group-A: commissioned for groups of whole bloods run in one series, 2026-10-20

**Status: COMMISSIONED 2026-10-20** for comparing two groups of whole bloods run in one series (EPIC v1), under the author's standing
instruction to commission every instrument that meets its pre-registered bars. Its sealed rule (DEV_GROUPA_COMMISSION_01.md, sealed
2026-10-20 before any array of the set was opened) set three bars; all three are met. Not a diagnostic test.

**The instrument's job:** two groups of healthy people read as no different after the tare, and a group shift planted into healthy arrays is found.

**What it reads.** Whether a whole group has moved: a trial arm, an exposure, a cohort. One person's whole-blood Met-A reading is too noisy
to call a small change (healthy people differ by about 2 % after the tare), but the average of a group is not. Each person is read by the
one command (`chain/MethylPhys_Interface/run_sample.py`, unchanged); each reading is tared against healthy arrays run the same way (same slide when it holds at least
3 healthy references, else the same series); the two groups' tared means are compared (difference, standard error, t). Nothing is fitted.
A difference of at least 2 standard errors is a group that moved. Chain: `chain/stage_groupa.py`.

**What it has passed** (GSE318173, the 278 healthy controls of a study run in one laboratory that no IAM instrument had read before; scored once):

| bar | sealed | result |
|---|---|---|
| 1. the arrays are read | Stage 0 admits at least half; at least 90 % of those read | Stage 0 admitted 278 of 278; **263 read** (15 withheld for coverage, each with its reason) - met |
| 2. two healthy groups read the same (women, men) | difference under 2 % and under 2 standard errors | women 0.99698 (n 75), men 1.00059 (n 188): **-0.36 %, -1.206 SE** - met |
| 3. a planted 1 % loss of the neutrophils' pattern is found | planted half above unplanted by more than 2 standard errors | planted 1.01883 (n 136), unplanted 0.99868 (n 127): **+2.00 %, 7.253 SE** - met |

**Reported with it** (not scored; `data/GROUPA_COMMISSIONING/scored_groupa_01.json` "reported", `tandem_groupa_summary.txt`):
- The healthy spread after the tare in this new laboratory: relative SD 0.02225, the same as the corrected whole-blood spread.
- Untared, women and men differed by -0.41 % (-1.425 SE); the spread of slide medians went from 0.016 untared to 0.01283 tared.
- The C-Score: 4 of 263 healthy arrays above its upper limit 1.152.
- Two instruments, two kinds of damage. The planted loss is scattered. On the same array, planting moved Met-A by up to 0.0357 (mean
  +0.0198) and the C-Score by up to 0.0060 (mean +0.0016): Met-A reads the scattered loss and the C-Score, which reads clustered loss, does
  not call it. On the unplanted half the two routes agree to 0.0000 in both readings.
- The cell shares (IAM-Deconvolver against IAM-Atlas v2.1), women against men: neutrophils 0.42849 against 0.43281 (t -0.198);
  monocytes 0.11719 against 0.1349 (t -3.141), CD4 T cells 0.20597 against 0.18194 (t 2.578), NK cells 0.07483 against 0.08369 (t -2.503).
  The mixes differ between the sexes and Met-A still reads them the same: the expected reading follows each person's own mix.

**Reference sets:** the tare's healthy references are healthy arrays of the same series (here GSE318173's own healthy controls, at least 3
on the slide, else the series); each person's expected reading comes from Met-A's commissioned references (blood composition from the
purified arrays of GSE110554 and GSE167998; the neutrophil reference from 6 purified neutrophil arrays).

**Synthetic test:** chain/MethylPhys_Interface/run_sample.py, GROUPA_COMMISSIONING/rehearsal_groupa_summary.txt

**Code and outputs:** every array was read by the chain, `chain/MethylPhys_Interface/run_sample.py`, unchanged, on GitHub's machines
(`.github/workflows/groupa.yml`, run 39503114606 at the sealed commit a9b94b07); the chain's `chain/stage_groupa.py` gives bars 2 and 3 to
every printed digit from the committed rows (`data/GROUPA_COMMISSIONING/scored_groupa_01_rows_idat.csv`, `scored_groupa_01_rows_values.csv`;
release check V10). Outputs: `scored_groupa_01.json`, `scored_groupa_01_summary.txt`, `tandem_groupa_summary.txt`, and `read_01/` (each
machine's rows and the scorer's log). The sealed test's own reading and scoring scripts are listed in DEV_GROUPA_COMMISSION_01.md.

**Scope.** EPIC v1 arrays; whole blood; two groups within one series (same laboratory, same run), each person read against same-run
healthy references. Groups in any other specimen or platform are read and labeled preliminary.

**What it would take to go further.** Patients and healthy controls run in the same series, one cancer at a time, under a rule sealed
before the arrays are read (GRP-004's next step); other specimens and platforms each need their own sealed test.

**Milestone:** listed in `Biological_Physics/README.md`, Advancements.

**In plain words** (Group-A commissioned): one blood sample cannot show a 1 % change, but a group can. On 278 healthy people from a
laboratory we had never read, women and men came out the same, and a 1 % loss planted into half of them was found at more than seven
times the noise. To go further it needs patients and healthy controls from the same laboratory run, one cancer at a time.

<!-- PLAIN:BEGIN (generated by CANON/results_to_tex.py from CANON/results_register.json; do not edit by hand) -->
**In plain words** (Group-A commissioned for groups of whole bloods): One blood sample cannot show a 1 % change, but a group can. Group-A reads each person with the same command as every report, centers each reading on healthy people run the same way, and compares the groups. On 278 healthy people from a laboratory we had never read, women and men came out the same, and a 1 % loss planted into half of them was found at more than seven times the noise. It ran on free public computers, from a rule written before any of the data was opened.
<!-- PLAIN:END -->
