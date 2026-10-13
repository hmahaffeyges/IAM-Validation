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

**Prior reading (data spending).** GSE184269 was read as the C-score band's held-out test and GSE315366 as the first band test; both
are in `CANON/data_spent.csv`. This step is DEVELOPMENT use of readings already taken - no new array is opened and no bar for either
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

**Not a milestone:** a design and a method check; nothing sealed, no detection claim.
