# DEV-DIRECTION-AUDIT-01 — which methylation direction each instrument cannot see (development; 2026-10-11)

Author rule (2026-10-11): every instrument is checked for the direction (loss: on to off; gain: off to on) it cannot see, by a planted test on
healthy data, and the result is written into its limits.

## IAM-A (Stage Q) and its C-score under planted gain
`data/DEV_DIRECTION_AUDIT_01/plant_gain_iama.py`, output `plant_gain_iama_liver_MBY.csv` (healthy liver RRBS, ENCODE ENCFF000MBY; Stage Q's own
pat_site_table and cscore). A share r of the normally unmethylated molecules is made fully methylated, with copy error eps_g per call.
1. **IAM-A sees gain only indirectly.** Gained molecules pass Stage Q's >= 80 % rule and enter the reading (opportunities 412,504 -> 881,990-898,533 at
   r = 0.05). IAM-A then reads a mixture of held and gained territory: down when the gained territory copies with less error, up when with more
   (r = 0.05: 0.7907 / 0.9297 / 1.1589 for eps_g = 0.5 / 1 / 2 x eps). IAM-A alone cannot separate gain from a change in holding.
   Next: read the two territories separately (held: healthy beta >= 0.8; gained: healthy beta <= 0.2), each with its own copy error.
2. **The IAM-A C-score does not hold its derived null on RRBS.** Healthy liver reads C = 15.3773 (null 1 +- 0.2722, 27 blocks of 1,000 sites):
   RRBS regions differ block to block in healthy cells. Planted changes move C without order. Not usable on RRBS until re-based on healthy
   same-pipeline files. (Developed on whole-genome Loyfer files.)
3. **Planting calibration.** eps_g = eps per call reads IAM-A 0.9544 / 0.9297, not 1: Stage Q counts isolated errors only. The next plant is set in
   Stage Q units.
Met-A reads both directions by construction (methylated and unmethylated identity sites); its planted check is pending. Fingerprint: Arm B = loss, Arm G = gain.
