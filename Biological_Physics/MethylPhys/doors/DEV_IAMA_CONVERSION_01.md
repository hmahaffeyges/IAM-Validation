# DEV-IAMA-CONVERSION-01 — what incomplete bisulfite conversion does to IAM-A (written 2026-10-10, before the refused files are read)

**Why.** Intake refuses a run below conversion 0.98 (ENCODE standard). In Box Run 2 whole blood every Swift library sits at 0.979–0.980,
so that standard alone decides which kit is read. The limit needs a physical basis.

**Derivation (nothing fitted).** A copy error is a CpG left unmethylated on a methylated molecule. It reads as T only if bisulfite converts it
(probability c, measured on the lambda spike-in CHH calls); unconverted, it reads as C and the error is hidden. Methylated C reads C either way.
So the measured copy error is ε_meas = c · ε. Two consequences:
1. Correction: ε = ε_meas / c, with c from the run's own alignment record.
2. Bias without the correction: d ln H / d ln ε = log2((1−ε)/ε) · ε / H(ε) = 0.752 at ε 0.043, so IAM-A_rel moves by about 0.75 · (c − c_ref)/c_ref
   against references at conversion c_ref. A bias under 0.01 needs |c − c_ref| ≤ 0.013. An absolute limit of 0.98 does not follow; what matters is
   the conversion difference from the references, and it can be divided out.

**Prediction (TruSeq repeat libraries, same donor, same kit, read before this note: rep1 only).**
| donor | c rep1 | c rep2 | ε_rep2 / ε_rep1 predicted (c2/c1) | IAM-A rep2 / rep1 |
|---|---|---|---|---|
| Sample2 | 0.99016 | 0.97001 | 0.9796 | 0.9846 |
| Sample3 | 0.99108 | 0.97383 | 0.9826 | 0.9869 |
| Sample4 | 0.99079 | 0.97132 | 0.9803 | 0.9852 |
**Read:** after dividing by c, |IAM-A(rep2) − IAM-A(rep1)| is smaller than before in at least 2 of 3 donors, and ≤ 0.02 in 3 of 3 (the repeat bar).
The predicted effect (~1.5 %) is close to repeat noise (0.009 measured on neutrophils), so this is a weak first test. It does not decide the
correction; the decision needs another laboratory. The Swift libraries (0.979–0.980, no c difference within donor) are read for the record
(`score_conversion_01.py`, committed with this note).
