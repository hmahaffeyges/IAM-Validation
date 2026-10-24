"""DEV-ERASE-01 block plan (committed before any read): for one sample's runs (srr, spots, in run order), 20 equal blocks totalling 10,000,000
read pairs, centred at (k + 1/2)/20 of the sample's spots; a block that crosses into the next run continues there. Prints srr start end
(1-based spot ids, inclusive), one block piece per line."""
import sys, pandas as pd
N, K = 10_000_000, 20
def plan(runs, n=None):   # 2026-10-23: n, the pairs to take (default N, the recount's); plasma_seq_01 takes every pair
    tot = sum(s for _, s in runs); size = min(n or N, tot) // K; out = []
    for k in range(K):
        g0 = max(0, min(tot - size, int((k + 0.5) * tot / K) - size // 2)); left = size; pos = g0
        for r, s in runs:
            if left == 0: break
            if pos >= s: pos -= s; continue
            take = min(left, s - pos); out.append((r, pos + 1, pos + take)); left -= take; pos = 0
    return out
def chunk(runs, k, n):   # 2026-10-24 (split alignment, plan 49.8): the k-th of n contiguous ranges of the sample's spots, every pair once
    tot = sum(s for _, s in runs); g0, g1 = k * tot // n, (k + 1) * tot // n; out = []; pos = 0
    for r, s in runs:
        a, b = max(g0, pos), min(g1, pos + s)
        if a < b: out.append((r, a - pos + 1, b - pos))
        pos += s
    return out
if __name__ == "__main__":
    S = pd.read_csv(sys.argv[1]); x = S[S.gsm == sys.argv[2]]
    R = list(zip(x.srr, x.spots))
    if len(sys.argv) > 3 and "/" in sys.argv[3]: k, n = map(int, sys.argv[3].split("/")); B = chunk(R, k, n)   # CHUNK=k/n
    else: B = plan(R, int(sys.argv[3]) if len(sys.argv) > 3 else None)
    for r, a, b in B: print(r, a, b)
