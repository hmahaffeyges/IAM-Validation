"""DEV-IAMA-TRIM-01: the trim rule, and nothing else. Written 2026-10-13 BEFORE it was exercised on anything.

The problem it solves (DEV-IAMA-READACROSS-01): the head of a sequencing run reads differently, the effect is library-specific in
sign, and the sealed commissioning values were taken from the head. Reading more of the file and pooling cannot work - pooling still
counts the biased block, so it divides a bias instead of removing it. The head must be EXCLUDED, and how much to exclude must be
decided by a diagnostic measured independently of IAM-A, or the trim becomes a tuning knob.

# NO CHAIN: this file computes no physics. It decides which part of a raw read file to skip, from the reads' own base composition.
# The reading itself stays the chain's Stage Q, applied to whatever this rule admits.

THE DIAGNOSTIC (raw reads only - no alignment, no reference, no IAM-A). In a bisulfite library almost every cytosine outside a CpG
is converted to thymine, so in the READ sequence the fraction of C among C+T at positions NOT followed by G is near zero and is a
property of the chemistry. It needs no alignment, so it can be computed in windows along the file for pennies, and anyone can check
it with a few lines of code.

    d(window) = count(C not followed by G) / count((C or T) not followed by G)

THE RULE, fixed here:
 1. read the file in windows of WINDOW reads;
 2. take the reference level as the MEDIAN d over the windows in the last half of the file, with its spread (median absolute
    deviation, scaled to a standard deviation);
 3. walk from the start and skip windows while |d - reference| > K_SD standard deviations of the reference level;
 4. stop skipping at the first window inside the band, and never skip more than MAX_TRIM of the file (if the rule would exceed
    that, the library is REFUSED rather than trimmed - a library that is anomalous throughout is not a trimming problem);
 5. the reading is then taken from the first admitted window to the end, by the chain's Stage Q, unchanged.

K_SD = 5 and MAX_TRIM = 0.25 are fixed now, before any real file is trimmed, and are not to be adjusted after seeing a reading.
"""
import numpy as np
WINDOW, K_SD, MAX_TRIM = 50_000, 5.0, 0.25

def window_diagnostic(seqs):
    """d for one window of read sequences: C among C+T at positions not followed by G. Raw sequence only."""
    c = t = 0
    for s in seqs:
        for i in range(len(s) - 1):
            if s[i + 1] == "G": continue
            if s[i] == "C": c += 1
            elif s[i] == "T": t += 1
    return c / (c + t) if (c + t) else np.nan

def decide_trim(d, window=WINDOW, k_sd=K_SD, max_trim=MAX_TRIM):
    """Given the per-window diagnostic along a file, return how many windows to skip and why.

    The reference level is the median of the LAST HALF of the file, with a robust standard deviation (1.4826 x MAD). Windows are
    skipped from the start while they sit further than k_sd robust standard deviations from that level.
    """
    d = np.asarray(d, dtype=float); n = len(d)
    tail = d[n // 2:]
    ref = float(np.nanmedian(tail)); sd = float(1.4826 * np.nanmedian(np.abs(tail - ref))) or float(np.nanstd(tail)) or 1e-9
    skip = 0
    while skip < n and abs(d[skip] - ref) > k_sd * sd: skip += 1
    frac = skip / n if n else 0.0
    if frac > max_trim:
        return dict(decision="REFUSE", skip_windows=skip, trimmed_fraction=round(frac, 4), reference=round(ref, 6),
                    robust_sd=round(sd, 6), reason=f"the rule would skip {frac:.0%} of the file, more than the {max_trim:.0%} limit: "
                                                   "the library is anomalous throughout, which is not a trimming problem")
    return dict(decision="TRIM" if skip else "READ ALL", skip_windows=skip, trimmed_fraction=round(frac, 4),
                reference=round(ref, 6), robust_sd=round(sd, 6), reads_skipped=skip * window,
                reason=(f"the first {skip} window(s) sit further than {k_sd} robust SD from the file's own later level"
                        if skip else "no window at the head departs from the file's own later level"))
