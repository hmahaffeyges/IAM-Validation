"""Adapter trimming for rrbs_iama (2026-10-10). Rule of Trim Galore (Illumina universal adapter AGATCGGAAGAGC): cut the read at the first
full match; otherwise at a 3' prefix of the adapter of >= 5 bases. Reads shorter than 20 bases after trimming are dropped by rrbs_iama itself.
The adapter is not bisulfite-converted, so its CG reads as methylated and can never show a copy error; untrimmed, it adds error-free calls."""
AD = "AGATCGGAAGAGC"
def trim(s):
    j = s.find(AD)
    if j >= 0: return s[:j]
    for k in range(len(AD) - 1, 4, -1):
        if s.endswith(AD[:k]): return s[:-k]
    return s
def trimmed(lines):
    for i, l in enumerate(lines):
        yield (trim(l.rstrip("\n")) + "\n") if i % 4 == 1 else l
