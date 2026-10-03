# DEV-SKYSTAT-01 — stage 12 sky statistics (development; check written 2026-10-03 before any data were read)

**Commissioning step 6.** Check: the look-elsewhere correction by simulation holds its stated false-positive rate on healthy arrays.

**State before reading.** `chain/sky_statistics.py` has the masked spectrum, band powers and the spatially shuffled null; the look-elsewhere
correction by simulation is listed as **not built** (chain/TOOLKIT.md, stage 12). The check cannot run until it exists. Stage 12 also sits
downstream of stage 11 (DEV-SKY-01); nothing downstream is accepted while upstream is open. **Not run; not wired. Author decision needed**:
whether to build the by-simulation correction (and its false-positive target) after stage 11 is settled.

---
## Outcome
Not run (reason above, recorded before data).
