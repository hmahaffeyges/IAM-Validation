# PROC-AML-PROG-01 — pre-registration (written 2026-10-01, before any AML array is read on these floors)

**Question.** Read against the healthy floor of its OWN compartment, does an AML blast population sit outside Normal on Met-A?
(Fixes the AML-SERIAL-01 cause: blasts were read against the neutrophil floor on lineage-wide sites.)

**Data.** GSE63409 (450K, raw IDATs, our Stage 1, all 74 pass): healthy bone marrow, 5 donors × HSC, MPP, L-MPP (CD34+CD38−) and
CMP, GMP, MEP (CD34+CD38+); AML patients sorted CD34+CD38− (14), CD34+CD38+ (15), CD34− (15). Same platform, same lab, for floors and readings.

**Floors (fixed now).** Two compartment floors: *primitive* = HSC+MPP+L-MPP (15 arrays); *committed* = CMP+GMP+MEP (15 arrays).
Sites per compartment, chosen on healthy arrays only (and without the array being read): across-array SD ≤ 0.05 and mean β in 0.80–0.95
(methylated channel) or 0.05–0.20 (unmethylated channel), up to 3,000 per channel, most stable first (Met-A reference floors v1.1, moderate rule).
Met-A = mean per-site H(β) over the sites / the same on the healthy compartment arrays. Normal = 0.95–1.05 (one gauge).

**Predictions.**
- G1 (healthy precision): each healthy array read leave-one-out against its own compartment: ≥ 80 % in Normal (30 arrays).
- P1 (AML seen): AML CD34+CD38− read against *primitive*, AML CD34+CD38+ against *committed*: ≥ 80 % outside Normal (29 arrays).
- P2 (direction): of the AML readings outside Normal, ≥ 70 % read above 1.05 (pattern loss raises entropy).
Descriptive: AML CD34− against both floors; per-patient CD34+ vs CD34− of the same patient; per-channel A.
**Stated limits now.** 5 healthy donors per type; sorted cells, not blood; the CD34+ AML fractions contain residual normal progenitors;
a pass shows Met-A separates leukaemic from healthy progenitors on the same platform, not detection in blood.
