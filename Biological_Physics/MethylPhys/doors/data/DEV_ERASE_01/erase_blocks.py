"""DEV-ERASE-01 block plan (committed before any read): for one sample's runs (srr, spots, in run order), 20 equal blocks totalling 10,000,000
read pairs, centred at (k + 1/2)/20 of the sample's spots; a block that crosses into the next run continues there. Prints srr start end
(1-based spot ids, inclusive), one block piece per line."""
import sys, pandas as pd
N, K = 10_000_000, 20
def plan(runs):
    tot = sum(s for _, s in runs); size = min(N, tot) // K; out = []
    for k in range(K):
        g0 = max(0, min(tot - size, int((k + 0.5) * tot / K) - size // 2)); left = size; pos = g0
        for r, s in runs:
            if left == 0: break
            if pos >= s: pos -= s; continue
            take = min(left, s - pos); out.append((r, pos + 1, pos + take)); left -= take; pos = 0
    return out
if __name__ == "__main__":
    S = pd.read_csv(sys.argv[1]); x = S[S.gsm == sys.argv[2]]
    for r, a, b in plan(list(zip(x.srr, x.spots))): print(r, a, b)
