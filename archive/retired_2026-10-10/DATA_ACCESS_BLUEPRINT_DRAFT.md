# Data access request — BLUEPRINT mature neutrophil WGBS (DRAFT for the author to submit)

**Dataset:** EGAD00001001201 (BLUEPRINT, Bisulfite-Seq for mature neutrophil: 6 samples, one laboratory, CNAG; study EGAS00001000418).
**Data Access Committee:** EGAC00001000135 (BLUEPRINT DAC). **How:** create an EGA account (ega-archive.org), "Request access" on the
dataset page; the DAC usually asks for the BLUEPRINT Data Access Agreement signed by the applicant and an authorised signatory of the
applicant's organisation.

**Applicant:** Heath W. Mahaffey, IAMPerformance (independent research). [Organisation signatory: to be named by the author.]

**Project title:** Commissioning a single-molecule copy-fidelity reading (IAM-A) of healthy human neutrophils.

**Lay summary (≤ 200 words).** We are building an open, fully documented measurement of how faithfully a cell copies its DNA methylation
pattern, read molecule by molecule from bisulfite sequencing. The reading is calibrated on healthy purified neutrophils. Before it can be
used on any other data it must be shown to read healthy neutrophils from an independent laboratory as healthy, after correcting for
laboratory and library-kit effects with healthy references sequenced in the same laboratory. Public data contain at most two healthy
neutrophil donors per laboratory and kit; the BLUEPRINT set provides six from one laboratory, which is exactly what this check needs.
The data will be used only for this methodological check, in aggregate; no attempt will be made to identify any donor; no genotypes
will be called, stored or shared; only summary error rates per sample are published.

**Data use.** Download the aligned or raw reads of the 6 samples; re-align with a fixed public pipeline; compute per-sample copy-error
rates; publish summary statistics only. Data held encrypted on a single access-controlled cloud volume, deleted at project end.

**Publications.** Results in the open IAM-Validation repository and book, acknowledging BLUEPRINT per its publication policy.
