#!/usr/bin/env python3
"""Controls for the KISS check: each case must be refused or accepted as stated, or the check does not run (checked_push.sh: &&)."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import kiss_check as K
HANSEN = ("**Step 2: the rule (sealed 2026-10-15 by the author, \"Seal as drafted\"; before any count).**\n\n- **S1:** at least 2 of the 3 donors read within "
          "+/- 0.0067 of the chain's self-reference line.\n- **Reported, not scored:** the lab shift.\n")
JOB = "**The instrument's job:** a healthy sample from another laboratory reads Normal on IAM-A's gauge.\n"
SRC = "**Sources:** GSE144691 and GSE98837, public on GEO and SRA: raw reads, so chemistry and sequencer error are measured.\n"
GOOD = JOB + SRC + "**Sealed 2026-10-16.**\n- **S1:** at least 80 percent of healthy samples passing the shape check read inside band v1.\n"
cases = [("C1 tonight's Hansen rule (no job, no sources)", "", HANSEN, True),
         ("C2 four scored bars", "", GOOD + "- **S2:** x\n- **S3:** y\n- **S4:** z\n", True),
         ("C3 a controlled-access source", "", JOB + "**Sources:** Loyfer 2023, public .pat files; raw reads on EGA EGAS00001006791.\n**Sealed.**\n- **S1:** x\n", True),
         ("C4 job line too short", "", "**The instrument's job:** reads Normal.\n" + SRC + "**Sealed.**\n- **S1:** x\n", True),
         ("C5 a correct seal", "", GOOD, False),
         ("C6 an outcome added to an old record (no new seal)", HANSEN, HANSEN + "\n## Outcome: NOT MET\n", False),
         ("C7 three bars plus reported lines", "", GOOD + "- **S2:** x\n- **S3:** y\n- **Reported, not scored:** z\n", False)]
bad = []
for name, old, new, refuse in cases:
    F = K.findings("x.md", old, new); ok = bool(F) == refuse
    print(f"  {name}: {'REFUSED' if F else 'ACCEPTED'} (want {'REFUSE' if refuse else 'ACCEPT'})" + (f" - {F[0][:90]}" if F else "")); bad += [] if ok else [name]
print("kiss_check controls: " + (f"ok ({len(cases)} of {len(cases)})" if not bad else "FAILED: " + ", ".join(bad))); sys.exit(1 if bad else 0)
