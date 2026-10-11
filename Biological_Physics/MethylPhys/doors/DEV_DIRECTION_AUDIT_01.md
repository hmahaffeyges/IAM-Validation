# DEV-DIRECTION-AUDIT-01 — which methylation direction each instrument cannot see (development; 2026-10-11)

Author rule (2026-10-11): every instrument is checked for the direction (loss: on to off; gain: off to on) it cannot see, by a planted test on
healthy data, and the result is written into its limits.

## IAM-A (Stage Q) and its C-score under planted gain
`data/DEV_DIRECTION_AUDIT_01/plant_gain_iama.py`, output `plant_gain_iama_liver_MBY.csv` (healthy liver RRBS, ENCODE ENCFF000MBY; Stage Q's own
pat_site_table and cscore). A share r of the normally unmethylated molecules is made fully methylated, with copy error eps_g per call.
1. **IAM-A sees gain only indirectly.** Gained molecules pass Stage Q's >= 80 % rule and enter the reading (opportunities 412,504 -> 862,270-898,533 at
   r = 0.05). IAM-A then reads a mixture of held and gained territory: down when the gained territory copies with less error, up when with more
   (r = 0.05: 0.7907 / 0.9297 / 1.1589 for eps_g = 0.5 / 1 / 2 x eps). IAM-A alone cannot separate gain from a change in holding.
   Next: read the two territories separately (held: healthy beta >= 0.8; gained: healthy beta <= 0.2), each with its own copy error.
2. **The IAM-A C-score does not hold its derived null on RRBS.** Healthy liver reads C = 15.3773 (null 1 +- 0.2722, 27 blocks of 1,000 sites):
   RRBS regions differ block to block in healthy cells. Planted changes move C without order. Not usable on RRBS until re-based on healthy
   same-pipeline files. (Developed on whole-genome Loyfer files.)
3. **Planting calibration.** eps_g = eps per call reads IAM-A 0.9544 / 0.9297, not 1: Stage Q counts isolated errors only. The next plant is set in
   Stage Q units.
Met-A reads both directions by construction (methylated and unmethylated identity sites); its planted check is pending. Fingerprint: Arm B = loss, Arm G = gain.

## Draft for the book, SOP and OM (author, 2026-10-11): goes in only when the conditions below are met
Why there are two instruments, by what each can and cannot see:

| | Met-A (arrays) | IAM-A (sequencing) |
|---|---|---|
| What it reads | fixed identity sites, both directions: sites that should be methylated and sites that should not | the cost of holding marks: copy error on the molecules a cell keeps methylated |
| Sees loss | yes | yes |
| Sees gain | yes, directly, at the unmethylated identity sites | only indirectly, as a mixture it cannot name |
| Blind spot | only the sites it was built on; nothing about how faithfully marks are copied | which direction moved, unless held and gained territory are read separately |

Conditions before this enters the book (Part VI instruments chapter), the SOP and the OM:
1. Met-A planted-gain and planted-loss test on healthy arrays (the "yes / yes" column is by construction until then).
2. Per-territory IAM-A reader (held vs gained copy error) built and recovering planted values in Stage Q units.
3. The IAM-A C-score re-based on healthy RRBS, or its RRBS limit written into its own row.
When all three are committed, this table (with the numbers from those outputs) goes into the book, SOP and OM in the same push.
