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
**Met-A under planted gain and loss (condition 1, done).** `data/DEV_DIRECTION_AUDIT_01/plant_meta.py`, output `plant_meta_neutrophils.csv`: the six
physical Salas neutrophil EPIC arrays (12 GEO records; GSE167998 re-deposits GSE110554's arrays), commissioned path, Met-A before any tare. These
are the reference arrays, so this measures the direction and size of the response, not accuracy on new data. Baseline 0.9651-1.0235. A share d of
molecules gaining the mark at the unmethylated identity sites, or losing it at the methylated ones: d 0.02 gain 1.0738-1.1274, loss 1.0749-1.1292;
d 0.05 gain 1.2163-1.2644, loss 1.2188-1.2685; d 0.10 gain 1.4136-1.4551, loss 1.4178-1.4620. Met-A sees both directions with nearly equal
sensitivity; it does not say which direction moved (a direction reading would split the two site sets). Fingerprint: Arm B = loss, Arm G = gain.

## Draft for the book, SOP and OM (author, 2026-10-11): goes in only when the conditions below are met
Why there are two instruments, by what each can and cannot see:

| | Met-A (arrays) | IAM-A (sequencing) |
|---|---|---|
| What it reads | fixed identity sites, both directions: sites that should be methylated and sites that should not | the cost of holding marks: copy error on the molecules a cell keeps methylated |
| Sees loss | yes | yes |
| Sees gain | yes, directly, at the unmethylated identity sites | only indirectly, as a mixture it cannot name |
| Blind spot | only the sites it was built on; nothing about how faithfully marks are copied | which direction moved, unless held and gained territory are read separately |

Conditions before this enters the book (Part VI instruments chapter), the SOP and the OM:
1. Met-A planted-gain and planted-loss test on healthy arrays: DONE 2026-10-11 (above).
2. Per-territory IAM-A reader (held vs gained copy error) built and recovering planted values in Stage Q units: DONE 2026-10-11 (below).
3. The IAM-A C-score re-based on healthy RRBS, or its RRBS limit written into its own row: DONE 2026-10-11, limit stated (below).
When all three are committed, this table (with the numbers from those outputs) goes into the book, SOP and OM in the same push.

## Per-territory IAM-A (condition 2, done)
`data/DEV_DIRECTION_AUDIT_01/territory_eps.py`: Stage Q's rule, qualifying molecules split by healthy beta (held >= 0.8, gained <= 0.2).
Built-in check (`territory_eps_liver_MBY_check.txt`): over all qualifying molecules it equals Stage Q's eps exactly (0.037332971316641775). Healthy liver
(reference ENCFF000MCD) already differs by territory: held 0.026875, other 0.052666. Recovery test (`recover_territory.py`,
`recover_territory_liver_MBY.json`): gained molecules planted with the exact call patterns of held molecules; the reader returns the expected
gained eps to 1e-12 (0.078602 at r 0.01, 0.089216 at r 0.05) and held eps unchanged (0.026875). Note: donors were drawn per pattern line, not
per molecule, so the copied patterns carry more error (0.084874-0.09142) than held territory; a plant meant to equal held territory must weight
donors by count.

## IAM-A C-score on RRBS (condition 3, done: limit)
`data/DEV_DIRECTION_AUDIT_01/cscore_by_territory.py`, output `cscore_by_territory_healthy.csv` (18 healthy RRBS files, Stage Q's formula and
block size; all-territory liver ENCFF000MBY equals Stage Q's cscore, 15.3773). Healthy all-territory C 1.6498-41.8264 against a derived null of 1.
Held territory only: MCF 10A 1.2889-2.3710, liver ENCFF000MCD 14.2257; the other files have under 10 blocks of held sites. Mixing territories of
different copy error explains part of the inflation, not all of it, and RRBS gives too few held sites. **Limit: the IAM-A C-score is a
whole-genome reading; it is not read on RRBS.**
