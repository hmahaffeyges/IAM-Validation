# DEV-GROUP-01 — whole blood on arrays as a GROUP instrument: does it stay null when nothing is there? (design, 2026-10-13; NOT sealed)

**The author's question.** Per-person detection in blood on arrays is out: healthy people differ by 6.36 % while a tissue-sized
signal is 2-7 % (DEV-BLOOD-SENSITIVITY-01). But a group comparison faces the spread divided by the square root of the group size, so
with 50 people a side a mean loss under 1 % separates. If the gauge can be shown to find a difference where cancer is present AND to
find nothing where it is absent, across many public cohorts, that is evidence for IAM that costs nothing and spends no new data type.

**The half that carries the weight is the second one.** Showing a difference between cancer patients and healthy donors is a modest
claim; many methods do that, and a laboratory batch effect can do it by accident. Showing that two HEALTHY groups from different
laboratories do NOT separate is the claim almost nobody tests, and it is the one that makes the first half mean anything.

## Step 1, the sham arm (this step): two healthy groups, two laboratories, no disease anywhere

Both sets are whole blood, both read by the chain, both already spent on other questions (declared below). Nothing here is a
detection claim: this is a method check on the instrument.

**Read two ways, because the chain tares within a series:**
- **untared** (`A`): the raw reading. A difference here is a laboratory or batch effect, pure and simple.
- **tared** (`A_rel`): each array against healthy references from its own series. If the tare works, a difference here should vanish.

**Bars, fixed before the readings are compared:**
1. **The sham must be null in the TARED reading.** The two healthy groups' means must differ by less than **2 %** - the bottom of the
   range tumors show - and the difference must be smaller than twice its own standard error. A larger difference means laboratory
   effects can counterfeit a cancer-sized group signal, and the group instrument is NOT usable across laboratories.
2. **The untared reading is reported, not barred.** It is expected to differ; the size of that difference is the measurement of how
   much work the tare is doing, which is a number we have never had.
3. **A group study design follows only if bar 1 is met**, and it must then be run one cancer at a time, with the healthy comparison
   group drawn from a laboratory that did not produce the cancer arrays - the harder configuration, deliberately.

**Prior reading (data spending).** GSE184269 is already read and spent (the C-score band's held-out test); GSE315366 is already read and spent (the first band test). Both are in `CANON/data_spent.csv`. This step is DEVELOPMENT use of readings already taken - no new array is opened and no bar for either
of those tests is affected. Neither set may later serve as a held-out test for a blood group reading.

**Known confound, stated before the run.** The two sets were read in different chain runs. If bar 1 fails, a chain-version difference
is a candidate cause and must be excluded before the laboratory is blamed.

**Synthetic test:** none - this is a comparison of readings the chain has already produced, not a new reading. Its honesty rests on
the bars above being fixed before the two groups are compared, which this commit timestamps.

**In plain words.** We can already tell, in a tube of blood, that one GROUP of people differs from another - we just cannot do it for
one person. Before claiming that is useful for cancer, we have to show the opposite case first: two groups of HEALTHY people, read in
two different laboratories, must come out the same. If they do not, then any difference we later see between patients and healthy
donors could just be the laboratory. **What it would take to finish:** this step is one comparison of readings we already have; if it
passes, the next step is one cancer at a time, with the healthy group deliberately taken from a different laboratory than the patients.

## Outcome of step 1, run once (2026-10-13): BAR MET, and the untared number is the headline

`sham_01.py`, outputs `sham_01_output.csv` and `sham_01_summary.txt`. 35 clonal-haematopoiesis-negative whole bloods from one
laboratory, 23 whole bloods from another, all healthy, all read by the chain.

| reading | lab 1 mean (SD) | lab 2 mean (SD) | difference | in standard errors |
|---|---|---|---|---|
| **untared** | 1.0088 (0.0229) | 0.9723 (0.0196) | **+3.69 %** | 6.49 |
| **tared** | 1.0008 (0.0233) | 0.9992 (0.0201) | **+0.16 %** | 0.28 |

**Bar 1 MET**: the tared sham is null at 0.16 % and 0.28 standard errors, against a bar of 2 % and two standard errors.

**The untared number is the one that matters, and it is a warning.** Two groups of HEALTHY people, read in two different
laboratories, differ by 3.69 % in the raw reading - squarely inside the 2-7 % range that tumor tissue shows, at six and a half
standard errors. **A laboratory difference alone can counterfeit a cancer-sized group signal.** Any study comparing patients in one
laboratory with healthy donors in another is measuring the laboratories.

**What the tare passing does and does not establish.** It is partly BY CONSTRUCTION: the tared reading divides each array by healthy
references from its own series, so a healthy group is expected to land near 1. The sham cannot therefore be read as "the tare removes
laboratory effects in general". What it does establish is narrower and still useful: after taring, the two laboratories agree to
0.16 % AND the spread within each is preserved (SD 0.023 and 0.020, against the 6.36 % between-person spread measured on the untared
reading) - so taring does not inflate the noise it removes the offset from, and the residual laboratory offset is an order of
magnitude below the smallest tissue-sized signal.

**The design rule this forces, written before any cancer cohort is opened.** A group comparison is only admissible when the patient
arrays and the healthy arrays were run in the SAME series, so both can be tared against the same healthy references - exactly the
configuration the tissue tests already use, where each tumor is read against the same patient's own normal tissue. Cohorts without
their own healthy controls in the same series are not usable for this, however large they are. That removes a large part of the
public archive from this design, and saying so now is cheaper than discovering it after a result.

**In plain words.** Two groups of healthy people, from two different laboratories, were compared. In the raw reading they differed by
as much as a tumor would - which means a careless study could "find cancer" that is really just a change of laboratory. After the
chain's correction, which compares each person against healthy people measured on the same machine run, the two groups agreed to
0.16 %, and that correction is doing the work of about 3.7 %. The lesson is a rule rather than a result: patients and healthy
comparisons must be measured in the same run, or the comparison is not worth making.

**Synthetic test:** none for this step, by the reasoning above; the bars were committed before the comparison (`ba45a71d`).

**Not a milestone:** a design and a method check; nothing sealed, no detection claim.
