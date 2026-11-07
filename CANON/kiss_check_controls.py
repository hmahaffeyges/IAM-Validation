#!/usr/bin/env python3
"""Controls for the KISS check: each case must be refused or accepted as stated, or the check does not run (checked_push.sh: &&)."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import kiss_check as K
HANSEN = ("**Step 2: the rule (sealed 2026-10-15 by the author, \"Seal as drafted\"; before any count).**\n\n- **S1:** at least 2 of the 3 donors read within "
          "+/- 0.0067 of the chain's self-reference line.\n- **Reported, not scored:** the lab shift.\n")
JOB = "**The instrument's job:** a healthy sample from another laboratory reads Normal on IAM-A's gauge.\n"
SRC = "**Sources:** GSE144691 and GSE98837, public on GEO and SRA: raw reads, so chemistry and sequencer error are measured.\n"
REFL = "**Reference:** the IAM Sequencing Atlas (public, Zenodo 10.5281/zenodo.24029328), healthy cells only.\n"
GOOD = JOB + SRC + REFL + "**Sealed 2026-10-16.**\n- **S1:** at least 80 percent of healthy samples passing the shape check read inside band v1.\n"
cases = [("C1 tonight's Hansen rule (no job, no sources)", "", HANSEN, True),
         ("C2 four scored bars", "", GOOD + "- **S2:** x\n- **S3:** y\n- **S4:** z\n", True),
         ("C3 a controlled-access source", "", JOB + "**Sources:** Loyfer 2023, public .pat files; raw reads on EGA EGAS00001006791.\n**Sealed.**\n- **S1:** x\n", True),
         ("C4 job line too short", "", "**The instrument's job:** reads Normal.\n" + SRC + "**Sealed.**\n- **S1:** x\n", True),
         ("C5 a correct seal", "", GOOD, False),
         ("C6 an outcome added to an old record (no new seal)", HANSEN, HANSEN + "\n## Outcome: NOT MET\n", False),
         ("C7 three bars plus reported lines", "", GOOD + "- **S2:** x\n- **S3:** y\n- **Reported, not scored:** z\n", False),
         ("C8 a synthetic with no range of the data (tonight's copies synthetic)", "", "**The copies synthetic (written before it runs).** Bars: the constructed plasma reads 1.0 at 4 and 12 copies with the flag.\n", True),
         ("C9 a range line without numbers", "", "**The copies synthetic (written before it runs).** Bars: the constructed plasma reads 1.0 at 4 and 12 copies with the flag.\n**Range of the data:** real and synthetic alike.\n", True),
         ("C10 a synthetic with its range of the data", "", "**The copies synthetic (written before it runs).** Bars: the constructed plasma reads 1.0 at 4 and 12 copies with the flag.\n**Range of the data:** real plasmas 1 to 20 calls a site, copies 1.2 to 270; synthetic 1 to 20 molecules, copies 1 to 300.\n", False),
         ("C11 a range paragraph that wraps", "", "**The copies synthetic (written before it runs).** Bars: x.\n**Range of the data:** real plasmas 1 to 20 calls a site;\nsynthetic 1 to 20 molecules.\n\nNext paragraph.\n", False)]
T_JOB = "**The instrument's job:** a person's tumor reads abnormal against a healthy reference.\n" + SRC
cases += [
 ("C12 no reference line", "", JOB + SRC + "**Sealed.**\n- **S1:** each healthy sample reads within 0.95 to 1.05.\n", True),
 ("C13 tonight's tumor against own normal", "", T_JOB + "**Reference:** each person's own normal tissue (GSE86340 'normal').\n**Sealed.**\n- **S1:** at least 4 of the 5 tumors read outside 0.95 to 1.05.\n", True),
 ("C14 the same-laboratory tare", "", JOB + SRC + "**Reference:** the IAM Sequencing Atlas, tared on 3 of the laboratory's own healthy samples.\n**Sealed.**\n- **S1:** each healthy sample reads within 0.95 to 1.05.\n", True),
 ("C15 same-run healthy arrays", "", JOB + SRC + "**Reference:** IAM-Atlas v2.1 with the same-run tare on the run's healthy arrays.\n**Sealed.**\n- **S1:** each healthy array reads within 0.95 to 1.05.\n", True),
 ("C16 the CD4 test's median-only bars", "", JOB + SRC + REFL + "**Sealed.**\n- **S1:** the tared own-cell Met-A median within 0.95 to 1.05.\n- **S2:** the whole-sample median within 0.95 to 1.05.\n", True),
 ("C17 bars in a sentence, median only, no reference", "", "**The CD4 test (written and committed before any donor is read).**\n\n" + JOB + SRC + "Bars (scorer): 1. the tared own-cell median within 0.95 to 1.05; 2. the whole-sample median within 0.95 to 1.05.\n", True),
 ("C18 bars in a sentence, per person, atlas reference", "", "**The healthy test (written and committed before any sample is read).**\n\n" + JOB + SRC + REFL + "Bars (scorer): 1. at least 90 percent of the healthy samples read within 0.95 to 1.05; 2. each laboratory's median within 0.03 of 1.\n", False),
 ("C19 atlas reference that says no tare is used", "", JOB + SRC + "**Reference:** IAM-Atlas v2.1 only; no same-run tare, no controls.\n**Sealed.**\n- **S1:** each healthy array reads within 0.95 to 1.05.\n", False),
]
bad = []
for name, old, new, refuse in cases:
    F = K.findings("x.md", old, new); ok = bool(F) == refuse
    print(f"  {name}: {'REFUSED' if F else 'ACCEPTED'} (want {'REFUSE' if refuse else 'ACCEPT'})" + (f" - {F[0][:90]}" if F else "")); bad += [] if ok else [name]
print("kiss_check controls: " + (f"ok ({len(cases)} of {len(cases)})" if not bad else "FAILED: " + ", ".join(bad))); sys.exit(1 if bad else 0)
