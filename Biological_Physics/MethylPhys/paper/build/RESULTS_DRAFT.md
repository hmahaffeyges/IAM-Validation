# Results (draft; held, not sent)

_Every number below is a named value read by `build_paper.py` from a committed output (`values.json`); the built copy is `build/RESULTS_DRAFT.md`.
A section marked **[text to come]** names its question and its record; its prose is written from that record before the draft goes to
review. The draft is held until the chain has been tested further on plasma and on the chosen tests._

The instruments are reported as a family. Each answers one question about a sample, and each sits beside the one that covers its blind spot.

## R1. A single sample is read on its own, against a ruler of healthy cells

A sample is read without a cohort or matched controls. Its reading's two ends are set by physics: the energy with which a cell holds a
mark, 20.94 k_BT per ATP (R2). Its healthy middle comes from an atlas of purified healthy cells. Its mix comes from the
sample itself, through the Deconvolver.

**Arrays.** IAM-Atlas v2.1 holds 79 purified healthy cell types, with a 95 % range at every site **[text to come: from
`atlas/v2.1/README.md` and its commissioning note]**.

**Sequencing.** An array reads light, which never quite reaches 0 or 1; sequencing counts molecules. So a second ruler was built: the IAM
Sequencing Atlas, 57 healthy cell types counted from whole-genome bisulfite sequencing at every CpG, each cell's donors sealed into two
halves. Every healthy sequenced input was read by the one command against the donors of the other half. All 140 read Normal
on every reading: sorted-cell Met-A 1.0075, the islands 1.0118, the domains 1.0014.
The atlas is checked as a sky survey is checked: noise maps by donor and by read, a null test on the sphere, the genome-distance spectrum,
the bridge to IAM-Atlas v2.1, and isotropy by chromosome (Chapter on the IAM Sequencing Atlas; `CHECKLIST_STATUS.md`).

**Another laboratory.** Healthy blood cells sequenced by an outside laboratory and read against the sequencing atlas did not read Normal:
own-cell Met-A 0.7134, whole-sample Met-A 0.9479, the islands 0.2924. The cause was measured, not assumed.
Each library carries its own floor, the rate of false "methylated" calls at sites no healthy cell marks: 0.00154 in the outside
laboratory against 0.00626 in the atlas's own. The islands follow that floor. The floor does not reach every library's islands in
the same way: in plasma the islands carry 1.005 of the library's floor, a single library-wide rate, while inside
the atlas's laboratory they carry 0.291 of it. A laboratory's chemistry is therefore part of what a ruler must
carry before a sample from any laboratory reads Normal. This is reported as an open result, not a finished one.

## R2. A thermodynamic bound sets the floor and the ceiling of methylation fidelity

Holding one mark costs 20.94 k_BT per ATP at 310.15 K **[text to come: the derivation, the measured holding energy and the prediction made before the data, from the records
named in the outline]**.

## R3. A sample's own loosely held share sets its expected error (IAM-A)

**[text to come: from the IAM-A records and `doors/DEV_REPLICATE_01.md`]**

## R4. The published leukemia data read again

The 36 samples of a published whole-genome bisulfite study of pediatric pre-B acute lymphoblastic leukemia were read by the full working
chain. Pass 1, at fixed depth, before the sequencing atlas existed: held error 0.04121 in the leukemias against
0.03668 in cord-blood B cells (p 0.022), with a loose share of 0.1208 in the leukemias. Read by the full
working chain against the 57-cell IAM Sequencing Atlas, Met-A in the B-cell column read 1.0410 in the 31 leukemias, 1.0837 in the Reh line,
and 2.2320 in its UHRF1 knockout. Group-A gave no call: the study's three healthy references leave each only two others
for its tare. Each leukemia is also kept by its recorded cytogenetic subtype in the A-score database, for our own look at where each kind
sits; nothing is declared from it.

## R5. Loss of a maintenance factor read against the bound

Knocking out UHRF1 raised the held error from 0.03216 to 0.09407, a factor of 2.925
**[text to come: the entropy turnover in heterochromatic domains, from `doors/DEV_ERASE_01.md`]**.

## R6. How a change lies along the genome (the C-Scores and the territories)

In a replay of a published prostate series (237 of 238 arrays read), the islands read 1.3587 in normal tissue and
1.4400 in tumor (p 1.2e-06) **[text to come: the C-Score records and DEV-TERRITORIES-01, with the transfer
function at the islands]**.

## R7. Beyond deep sequencing: cell-free DNA

Thirty healthy plasmas, read at about one read per site against the sequencing atlas: Met-A at the identity sites of the cells found read
0.9918. The islands carry each library's floor (slope 0.939), and read 0.9272 once
every library is brought to a common floor. The domains read 0.8666 and follow the share given to erythrocyte progenitors
(rho -0.928). That is an open question, stated as one. The matched white cells and plasma of 23 healthy donors stay sealed
for the test that counts, until the tuning plasmas read Normal.

## R8. What each reading cannot see, and how the chain checks itself

**[text to come: the blind-spot table from `CANON/catalogue.json`, and the results that went against the method, from the outline]**
