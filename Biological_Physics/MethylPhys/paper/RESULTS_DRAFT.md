# Results (draft; held, not sent)

_Every number below is a named value read by `build_paper.py` from a committed output (`values.json`); the built copy is `build/RESULTS_DRAFT.md`.
A section marked **[text to come]** names its question and its record; its prose is written from that record before the draft goes to
review. The draft is held until the chain has been tested further on plasma and on the chosen tests._

The instruments are reported as a family. Each answers one question about a sample, and each sits beside the one that covers its blind spot.

## R1. A single sample is read on its own, against a ruler of healthy cells

A sample is read without a cohort or matched controls. Its reading's two ends are set by physics: the energy with which a cell holds a
mark, {{M_cell}} k_BT per ATP (R2). Its healthy middle comes from an atlas of purified healthy cells. Its mix comes from the
sample itself, through the Deconvolver.

**Arrays.** IAM-Atlas v2.1 holds 79 purified healthy cell types, with a 95 % range at every site **[text to come: from
`atlas/v2.1/README.md` and its commissioning note]**.

**Sequencing.** An array reads light, which never quite reaches 0 or 1; sequencing counts molecules. So a second ruler was built: the IAM
Sequencing Atlas, 57 healthy cell types counted from whole-genome bisulfite sequencing at every CpG, each cell's donors sealed into two
halves. Every healthy sequenced input was read by the one command against the donors of the other half. All {{seq_healthy_n}} read Normal
on every reading: sorted-cell Met-A {{seq_metaa_sorted_healthy}}, the islands {{seq_islands_healthy}}, the domains {{seq_domains_healthy}}.
The atlas is checked as a sky survey is checked: noise maps by donor and by read, a null test on the sphere, the genome-distance spectrum,
the bridge to IAM-Atlas v2.1, and isotropy by chromosome (Chapter on the IAM Sequencing Atlas; `CHECKLIST_STATUS.md`).

**Another laboratory.** Healthy blood cells sequenced by an outside laboratory and read against the sequencing atlas did not read Normal:
own-cell Met-A {{bp_own_metaa}}, whole-sample Met-A {{bp_whole_metaa}}, the islands {{bp_islands}}. The cause was measured, not assumed.
Each library carries its own floor, the rate of false "methylated" calls at sites no healthy cell marks: {{bp_floor}} in the outside
laboratory against {{atlas_floor}} in the atlas's own. The islands follow that floor. The floor does not reach every library's islands in
the same way: in plasma the islands carry {{plasma_islands_floor_share}} of the library's floor, a single library-wide rate, while inside
the atlas's laboratory they carry {{atlas_islands_floor_share}} of it. A laboratory's chemistry is therefore part of what a ruler must
carry before a sample from any laboratory reads Normal. This is reported as an open result, not a finished one.

## R2. A thermodynamic bound sets the floor and the ceiling of methylation fidelity

Holding one mark costs {{M_cell}} k_BT per ATP at {{T_cell}} K **[text to come: the derivation, the measured holding energy and the prediction made before the data, from the records
named in the outline]**.

## R3. A sample's own loosely held share sets its expected error (IAM-A)

**[text to come: from the IAM-A records and `doors/DEV_REPLICATE_01.md`]**

## R4. The published leukemia data read again

The 36 samples of a published whole-genome bisulfite study of pediatric pre-B acute lymphoblastic leukemia were read by the full working
chain. Pass 1, at fixed depth, before the sequencing atlas existed: held error {{all_held_err_leuk}} in the leukemias against
{{all_held_err_cord}} in cord-blood B cells (p {{all_held_err_p}}), with a loose share of {{all_loose_leuk}} in the leukemias. Read by the full
working chain against the 57-cell IAM Sequencing Atlas, Met-A in the B-cell column read {{hopkins3_leuk_metaa_b}} in the 31 leukemias, {{hopkins3_ctrl_metaa_b}} in the Reh line,
and {{hopkins3_ko_metaa_b}} in its UHRF1 knockout. Group-A gave no call: the study's three healthy references leave each only two others
for its tare. Each leukemia is also kept by its recorded cytogenetic subtype in the A-score database, for our own look at where each kind
sits; nothing is declared from it.

## R5. Loss of a maintenance factor read against the bound

Knocking out UHRF1 raised the held error from {{uhrf1_held_err_ctrl}} to {{uhrf1_held_err_ko}}, a factor of {{uhrf1_held_err_ratio}}
**[text to come: the entropy turnover in heterochromatic domains, from `doors/DEV_ERASE_01.md`]**.

## R6. How a change lies along the genome (the C-Scores and the territories)

In a replay of a published prostate series ({{r7_arrays}} arrays read), the islands read {{r7_islands_normal}} in normal tissue and
{{r7_islands_tumor}} in tumor (p {{r7_islands_p}}) **[text to come: the C-Score records and DEV-TERRITORIES-01, with the transfer
function at the islands]**.

## R7. Beyond deep sequencing: cell-free DNA

Thirty healthy plasmas, read at about one read per site against the sequencing atlas: Met-A at the identity sites of the cells found read
{{plasma_identity_metaa}}. The islands carry each library's floor (slope {{plasma_floor_slope}}), and read {{plasma_islands_floor}} once
every library is brought to a common floor. The domains read {{plasma_domains}} and follow the share given to erythrocyte progenitors
(rho {{plasma_ep_rho}}). That is an open question, stated as one. The matched white cells and plasma of 23 healthy donors stay sealed
for the test that counts, until the tuning plasmas read Normal.

## R8. What each reading cannot see, and how the chain checks itself

**[text to come: the blind-spot table from `CANON/catalogue.json`, and the results that went against the method, from the outline]**
