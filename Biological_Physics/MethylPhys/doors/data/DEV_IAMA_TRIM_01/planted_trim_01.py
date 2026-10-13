"""DEV-IAMA-TRIM-01 planted test, written and committed 2026-10-13 BEFORE it was run, with its bars in this docstring.

# NO CHAIN: synthetic reads and the trim rule's own arithmetic. The copy error is planted directly, so no reading stage is needed;
# what is under test is whether the trim finds the right part of a file to skip and whether it recovers a planted value.

Each case builds a synthetic library as a list of windows. A window carries (a) the raw-read diagnostic d - C among C+T outside CpG -
and (b) a planted copy error eps. A HEAD ANOMALY sets both differently for the first windows: the chemistry differs there (d shifted)
AND the reading is biased there (eps shifted). An UNBIASED library has neither.

BARS, written before running:
 1. RECOVERY: with a head anomaly over 10 % of the file and the head's eps biased by -12 % (the size measured in the cause run), the
    trimmed read must recover the library's true eps to within 1 %.
 2. NO HARM: on an unbiased library (no head anomaly), the rule must skip nothing and move the reading by less than the sealed
    tolerance of 0.03 relative.
 3. REFUSAL: a library whose chemistry is off throughout must be REFUSED, not trimmed to nothing.
 4. SIGN BLINDNESS: the rule must trim the same windows whether the head's eps bias is negative or POSITIVE - the diagnostic must not
    know which way the reading moved, since in the cause run the sign differed between libraries.
 6. THE REAL CONFIGURATION (added before running, after noticing bar 1 is too easy): in the actual failure the reading was taken
    from the HEAD ALONE - the sealed decision reports part 0 reproducing each library's Stage B value to 0.0 % - so the head's bias
    arrived undiluted. Bar 6 therefore reads only the head, as Stage B did, and requires the trimmed read to recover the true eps to
    within 1 % from a starting error of about -12 %, not -1 %.
 5. NOT A TUNING KNOB: with the head's chemistry normal but its eps biased, the rule must NOT trim (it cannot see a bias it has no
    independent evidence for), and the test records that as expected behaviour rather than a failure - the honest limit of the rule.

Writes planted_trim_01_output.csv and planted_trim_01_summary.txt. Usage: python3 planted_trim_01.py [OUTDIR]
"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = sys.argv[1] if len(sys.argv) > 1 else HERE
sys.path.insert(0, HERE)
from trim_rule_01 import decide_trim, K_SD, MAX_TRIM
rg = np.random.default_rng(20261013)
NW, D_OK, D_SD, EPS_TRUE = 40, 0.006, 0.0004, 0.0480      # windows per library; healthy diagnostic level and its window noise

def library(head_windows=0, d_shift=0.0, eps_bias=0.0, d_shift_all=0.0):
    d = rg.normal(D_OK + d_shift_all, D_SD, NW)
    eps = rg.normal(EPS_TRUE, EPS_TRUE * 0.01, NW)
    if head_windows:
        d[:head_windows] += d_shift
        eps[:head_windows] *= (1 + eps_bias)
    return d, eps

def trimmed_eps(d, eps):
    o = decide_trim(d)
    kept = eps if o["decision"] == "READ ALL" else eps[o["skip_windows"]:]
    return o, (float(np.mean(kept)) if len(kept) else np.nan)

rows, L = [], ["DEV-IAMA-TRIM-01 planted test", "",
               f"fixed before running: K_SD = {K_SD}, MAX_TRIM = {MAX_TRIM:.0%}, {NW} windows per library, "
               f"healthy diagnostic {D_OK} (window SD {D_SD}), true eps {EPS_TRUE}", ""]
# bar 1: recovery
d, eps = library(head_windows=4, d_shift=0.004, eps_bias=-0.12)
o, e = trimmed_eps(d, eps); untrimmed = float(np.mean(eps))
rec_err = abs(e / EPS_TRUE - 1); bar1 = rec_err <= 0.01
rows.append(dict(case="1_recovery", decision=o["decision"], skipped=o["skip_windows"], eps_read=e, eps_true=EPS_TRUE,
                 rel_error=rec_err, eps_untrimmed=untrimmed, passed=bar1))
L += ["BAR 1 - recovery: head anomaly over 10 % of the file, head eps biased -12 %",
      f"  rule: {o['decision']}, skipped {o['skip_windows']} of {NW} windows ({o['trimmed_fraction']:.0%})",
      f"  untrimmed read {untrimmed:.5f} ({untrimmed / EPS_TRUE - 1:+.2%} off truth) | trimmed read {e:.5f} "
      f"({e / EPS_TRUE - 1:+.2%} off truth) -> {'PASS' if bar1 else 'FAIL'} (bar: within 1 %)", ""]
# bar 2: no harm
d, eps = library()
o, e = trimmed_eps(d, eps); move = abs(e / float(np.mean(eps)) - 1); bar2 = (o["skip_windows"] == 0) and move < 0.03
rows.append(dict(case="2_no_harm", decision=o["decision"], skipped=o["skip_windows"], eps_read=e, rel_move=move, passed=bar2))
L += ["BAR 2 - no harm: an unbiased library",
      f"  rule: {o['decision']}, skipped {o['skip_windows']} | reading moved {move:.4%} -> {'PASS' if bar2 else 'FAIL'} "
      f"(bar: skip nothing, move under 3 %)", ""]
# bar 3: refusal
d, eps = library(head_windows=NW, d_shift=0.0, eps_bias=0.0, d_shift_all=0.01)
d[NW // 2:] -= 0.01                                      # the later half is normal, the first half is off throughout
o, e = trimmed_eps(d, eps); bar3 = o["decision"] == "REFUSE"
rows.append(dict(case="3_refusal", decision=o["decision"], skipped=o["skip_windows"], trimmed_fraction=o["trimmed_fraction"], passed=bar3))
L += ["BAR 3 - refusal: chemistry off across the whole first half",
      f"  rule: {o['decision']}, would skip {o['trimmed_fraction']:.0%} -> {'PASS' if bar3 else 'FAIL'} (bar: REFUSE above "
      f"{MAX_TRIM:.0%})", ""]
# bar 4: sign blindness
d1, e1 = library(head_windows=4, d_shift=0.004, eps_bias=-0.12)
d2, e2 = library(head_windows=4, d_shift=0.004, eps_bias=+0.12)
o1, _ = trimmed_eps(d1, e1); o2, _ = trimmed_eps(d2, e2); bar4 = o1["skip_windows"] == o2["skip_windows"]
rows.append(dict(case="4_sign_blind", decision=f"{o1['decision']}/{o2['decision']}",
                 skipped=f"{o1['skip_windows']}/{o2['skip_windows']}", passed=bar4))
L += ["BAR 4 - sign blindness: the same chemistry anomaly with the eps bias negative, then positive",
      f"  skipped {o1['skip_windows']} and {o2['skip_windows']} windows -> {'PASS' if bar4 else 'FAIL'} (bar: identical)", ""]
# bar 5: the honest limit
d, eps = library(head_windows=4, d_shift=0.0, eps_bias=-0.12)
o, e = trimmed_eps(d, eps); bar5 = o["skip_windows"] == 0
rows.append(dict(case="5_declared_limit", decision=o["decision"], skipped=o["skip_windows"], eps_read=e, passed=bar5))
L += ["BAR 5 - the declared limit: the head's chemistry is NORMAL but its reading is biased",
      f"  rule: {o['decision']}, skipped {o['skip_windows']} | reading {e:.5f} ({e / EPS_TRUE - 1:+.2%} off truth) -> "
      f"{'PASS' if bar5 else 'FAIL'} (bar: trim NOTHING - the rule may not act on a bias it has no independent evidence for)",
      "  This is the rule's honest limit, declared rather than hidden: it removes anomalies the chemistry can see, and a bias with no",
      "  independent signature survives it. Such a library is caught by the commissioning check, not by the trim.", ""]
# bar 6: the real configuration - the reading taken from the head alone, as Stage B took it
d, eps = library(head_windows=4, d_shift=0.004, eps_bias=-0.12)
head_only = float(np.mean(eps[:4]))                        # what Stage B effectively read
o, e = trimmed_eps(d, eps)
err_before, err_after = head_only / EPS_TRUE - 1, e / EPS_TRUE - 1
bar6 = abs(err_after) <= 0.01 and abs(err_before) > 0.03
rows.append(dict(case="6_real_configuration", decision=o["decision"], skipped=o["skip_windows"], eps_head_only=head_only,
                 eps_read=e, rel_error_before=err_before, rel_error_after=err_after, passed=bar6))
L += ["BAR 6 - the real configuration: the reading taken from the HEAD ALONE, as Stage B took it",
      f"  head-only read {head_only:.5f} ({err_before:+.2%} off truth - outside the sealed 3 % tolerance, which is the failure we saw)",
      f"  after the rule: {o['decision']}, skipped {o['skip_windows']} windows, read {e:.5f} ({err_after:+.2%} off truth) -> "
      f"{'PASS' if bar6 else 'FAIL'} (bar: within 1 %, from a starting error above 3 %)", ""]
allok = all(r["passed"] for r in rows)
L += [("ALL FIVE BARS MET: the rule recovers a planted head bias, leaves a clean library alone, refuses a library that is anomalous "
       "throughout, is blind to the direction of the reading's error, and does not act without independent evidence."
       if allok else "NOT ALL BARS MET: the failing bar is above, and the rule is not used on any real file until it is fixed.")]
pd.DataFrame(rows).to_csv(os.path.join(OUT, "planted_trim_01_output.csv"), index=False)
open(os.path.join(OUT, "planted_trim_01_summary.txt"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
