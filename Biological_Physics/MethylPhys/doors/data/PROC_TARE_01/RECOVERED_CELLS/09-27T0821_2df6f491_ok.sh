set -e; W="$(pwd)"; MP="$W/iamrepo/Biological_Physics/MethylPhys"
cp handoff/munich_controls.csv "$MP/kit/results/FINDING_GSE125105_controls.csv"
cat > "$MP/doors/FINDING_GSE125105_LOW_SIGNAL.md" <<'EOF'
# Finding 2026-09-27 — GSE125105 (Munich) arrays are low-signal, and the intake gate that should refuse them never fires

**Why it was looked at.** The author: one laboratory fails every procedure while three pass — "once is a result, twice is
interesting, every time is screaming that something may be off." GSE125105 was the outlier in PROC-TARE-01 (most compressed
T_scale 0.890, largest T_offset +0.043), carried the largest of the retired laboratory zeros (−0.035), and in PROC-SKY-01 was
the laboratory on which the on-array noise model failed (3 of 12 in range; 12/12, 11/12, 10/12 elsewhere).

**What was measured** (three arrays per laboratory through Stage 1 with the control probes kept; `kit/results/FINDING_GSE125105_controls.csv`):

| median | GSE87571 | GSE42861 | GSE111629 | **GSE125105** |
|---|---|---|---|---|
| non-polymorphic control G / R (raw signal, methylation-independent) | 8,582 / 14,427 | 5,959 / 10,773 | 4,823 / 7,252 | **1,291 / 2,268** |
| bisulfite-conversion II, R | 32,559 | 20,159 | 16,590 | **7,966** |
| hybridisation, G | 27,410 | 14,726 | 20,392 | **12,494** |
| negative controls G / R (background) | 311 / 370 | 165 / 270 | 215 / 280 | 175 / 275 |
| probes at background, poobah p > 0.05 | 0.9 % | 1.5 % | 5.4 % | **12.5 % (10.8–17.2)** |
| SNP homozygous-cluster SD (β = 0 / 1), PROC-SKY-01 | 0.017 / 0.016 | 0.025 / 0.025 | 0.027 / 0.048 | **0.069 / 0.072** |
| chip decoded → scanned (IDAT header) | 3 months | — | 1 month | **8 months** |

Same background, one-sixth the signal. Every anomaly this laboratory has shown follows from that: wide SNP clusters, compressed
dynamic range, the largest zero, an on-array noise term (built from its SNP probes) that over-predicts its cg scatter.

**Why the chain let it through.** `stage_0_intake.py` carries the SOP §15/§17 gates (detected fraction; call rate ≥ 0.98
PROCEED, 0.95–0.98 PENALTY, < 0.95 QUARANTINE). Munich at 0.875 is a QUARANTINE by the SOP as written. But Stage 1 never hands
Stage 0 the per-probe detection numbers, the check is recorded `DEFERRED_PENDING_STAGE1_DECODER`, and the code sets
`advance = True`. A deferred check was acting as a pass — the exact thing the standing rule forbids. Commissioning runs also
use `--no-intake`.

**What it is not.** Not the physics, not the atlas, not the pipeline map. An address whose signal sits at background carries no
information about β; averaging it in injects ≈ 0.5. This is an instrument-quality fact about specific arrays, measured on the
array itself.

**Decisions.**
1. PROC-INTAKE-01 (pre-registered separately): Stage 1 returns the per-probe detection mask and control summary; Stage 0.5/0.7
   run on real numbers; failed probes are masked before any mean; a deferred check never advances an IDAT input.
2. GSE125105 stays a commissioning laboratory for its own detection-panel line only until PROC-INTAKE-01 has run on its 80
   panel arrays; anything cross-laboratory that used it is re-read afterwards (the retired band and zero already are gone).
3. PROC-SKY-01 B2 is **not** re-scored. Its outcome stands as sealed; this finding is recorded beside it as the probable cause of
   the fourth laboratory's miss, to be tested by re-running the same script once the gate exists.
EOF
cat > "$MP/doors/PROC_INTAKE_01_PREREG.md" <<'EOF'
# PROC-INTAKE-01 — pre-registration: the intake gate runs on the array's own numbers, and a deferred check never advances

**Written 2026-09-27, before any array is scored under these bars.** Follows FINDING_GSE125105_LOW_SIGNAL.md.

## The change (fixed)
1. Stage 1 (`stage_1_idat_calibration.py`) runs the decoder with per-probe detection (poobah, against this array's own negative
   controls) and the control probes kept, and returns: the beta vector, the detection-pass mask (p ≤ 0.05), the per-class
   control medians, and the fraction of probes at background.
2. `run_full` hands these to Stage 0.4 / 0.5 / 0.7. The gates are the SOP's, unchanged: detected fraction > 0.99 PASS,
   0.95–0.99 BORDERLINE, < 0.95 FAIL; call rate ≥ 0.98 PROCEED, 0.95–0.98 PROCEED_WITH_PENALTY, < 0.95 QUARANTINE.
3. Probes failing detection are **removed from the beta vector before any stage reads it** (no measurement at background).
4. `DEFERRED_PENDING_STAGE1_DECODER` on an IDAT input → `advance = False` (QUARANTINE_INTAKE_DEFERRED). On a betas-only input
   (no IDAT) intake cannot run; the bundle carries `intake_verified: False` and the Reading tab prints one line saying so.
   `--no-intake` is removed from the commissioning path.

## Bars
- **B1** each of the 12 GSE125105 panel arrays from PROC-SKY-01: call rate reported; those below 0.95 are QUARANTINED and
  produce no report (exit 2), those in 0.95–0.98 carry the PENALTY flag on the Reading tab.
- **B2** the gate is not refusing good arrays: of the first 100 GSE87571 IDAT pairs (sorted accession order) at least 95
  PROCEED. (An instrument check that the SOP threshold sits where the array's own noise says it should — not a definition of
  anything about people. If it fails, the threshold is reported against the observed call-rate distribution and NOT moved here.)
- **B3** masking does not move A on good arrays: on 12 GSE87571 arrays, per present cell, |A_masked − A_unmasked| < 0.002.
- **B4** a betas-only input renders with `intake_verified: False` and the one printed line; nothing else on the report changes.
- **B5** the negative control: a Stage 1 return with the mask deliberately withheld must QUARANTINE (deferred never advances).

## Decision rule
B1–B5 met → adopted; the Troubleshooting tab's "three ways of not giving you an answer" gains the intake line. B2 failed →
adopted anyway for B1/B3/B4/B5 but the call-rate thresholds are flagged UNCALIBRATED on the report (printed, not refused on),
exactly as the bisulfite row is today, and the distribution is recorded for the author's decision.
EOF
cd "$MP" && python3 - <<'PY'
p="doors/PLAN.md"; s=open(p,encoding="utf-8").read()
if "PROC-INTAKE-01" not in s:
    i=s.find("6. **Held-out Stage 2d**"); assert i>0
    s=s[:i]+"5a. **PROC-INTAKE-01** — the Stage 0 detection/call-rate gate runs on the array's own numbers (pre-registered 2026-09-27); failed probes masked before any mean; deferred never advances. Then re-run PROC-SKY-01's script with Munich's sub-threshold arrays flagged — the outcome stays sealed, this is the follow-up.\n"+s[i:]
L=s.split("\n"); k=next(i for i,l in enumerate(L) if l.lstrip().startswith("5.") and "ky" in l)
L[k]=L[k].rstrip()+"  ← PROC-SKY-01 sealed 2026-09-27: B2 failed (36/48; GSE125105 3/12 — see FINDING_GSE125105_LOW_SIGNAL.md). Panel scales retired. **Author's decision pending: withhold the sky, or draw the residual in β units with no σ (plus the serial difference map).**"
open(p,"w",encoding="utf-8").write("\n".join(L)); print("PLAN updated")
q="doors/ENHANCEMENTS.md"; t=open(q,encoding="utf-8").read()
if "0r. **GSE125105" not in t:
    i=t.find("\n9. **Webpage drafts"); t=t[:i]+"\n0r. **GSE125105 is low-signal and the intake gate never fired (2026-09-27, author: 'every time is screaming').** Non-polymorphic controls at 1/6 of Uppsala's, 12.5 % of probes at background, chip 8 months on the shelf before scan. SOP §17 would QUARANTINE at call rate 0.875, but Stage 0 records DEFERRED and advances - a deferred check acting as a pass. PROC-INTAKE-01 pre-registered. PROC-SKY-01 sealed as measured (B2 failed, sky withheld pending the author's decision between withhold and a β-unit residual sky with no σ).\n"+t[i:]
    open(q,"w",encoding="utf-8").write(t); print("ENHANCEMENTS 0r")
p="doors/CHAIN_COMMISSIONING.md"; s=open(p,encoding="utf-8").read()
if "PROC-SKY-01" not in s:
    s=s.rstrip()+"\n| B-15 | the sky's zero and spread with no population in them (m = 0; atlas posterior + this array's SNP-probe noise) | **NOT COMMISSIONED 2026-09-27** (PROC-SKY-01). B1, B3, B4, B5 met; B2 failed - robust SD of z in [0.7, 1.4] on 36 of 48 (bar 40): GSE125105 3/12, whose arrays are low-signal (FINDING_GSE125105_LOW_SIGNAL.md). The four panel residual scales retired as population layers. Diagnostic: the m = 0 sky shows a −0.03 to −0.04 offset off the identity loci, ≈ 0 on them - the pipeline map's fitting range, not compression. |\n"
    open(p,"w",encoding="utf-8").write(s); print("register B-15")
PY
cd "$W/iamrepo" && git add -A Biological_Physics && git status --short | wc -l | awk '{print "staged (incl. parked sky edits): "$1}'