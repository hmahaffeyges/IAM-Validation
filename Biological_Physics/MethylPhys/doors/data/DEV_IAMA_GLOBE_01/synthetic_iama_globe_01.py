"""DEV-IAMA-GLOBE-01, synthetic test 1 (written and committed 2026-10-22 before it runs; exploratory, not sealed; to-do step 29).
The instrument's job: on the IAM-A globe, a healthy sample shows no departures from its own predicted held error, and a stretch of genome
where held marks are lost more often lights up there.
A synthetic .pat (the format chain Stage Q2 reads) over chr1 and chr2 of hg19's CpG index (4,448,805 CpGs): molecules of 8 consecutive CpGs,
about 6 held-error opportunities per CpG, all methylated except isolated losses at the rate that makes the held error equal to the
self-reference line's prediction for this sample's loose share (all held, so the line's firm rate). The planted copy doubles the loss rate
over one stretch of 50,000 CpGs (chr1, from CpG 1,000,001). Both read by chain/stage_q2_globe.py (count_pat_sites, departures), unchanged.
Reported: (1) the per-site sums equal count_pat's own genome totals on the same lines; (2) changed blocks on the healthy copy; (3) on the
planted copy, the share of the stretch's blocks changed and the changed blocks outside it.
Usage: python3 synthetic_iama_globe_01.py OUTDIR   (writes synthetic_iama_globe_01_summary.txt)"""
import os, sys, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "..", "..", "..", "chain"))
import stage_q2_selfref as Q2, stage_q2_globe as S
out = sys.argv[1]; os.makedirs(out, exist_ok=True); rng = np.random.default_rng(20261022)
N = 2284470 + 2164335; L8 = 8; DEPTH = 6; nmol = int(N * DEPTH / (L8 - 2)) * 2      # odd and even lines alike
firm = Q2.predict(0.0)
r = firm
for _ in range(50): r = firm / (1 - r) ** 2                                        # isolated loss: r (1 - r)^2 = the predicted held error
PL = (1_000_001, 1_050_000)
def lines(planted):
    g = np.random.default_rng(7); st = g.integers(1, N - L8, nmol); st.sort()
    for i in range(0, nmol, 200_000):
        s = st[i:i + 200_000]; rate = np.where(planted & (s >= PL[0]) & (s <= PL[1]), 2 * r, r)
        T = g.random((len(s), L8)) < rate[:, None]; T[:, 0] = T[:, -1] = False
        for a, row in zip(s, T):
            c = "chr1" if a <= 2284470 else "chr2"
            yield f"{c}\t{a}\t{''.join('T' if x else 'C' for x in row)}\t1\n"
Lo = ["DEV-IAMA-GLOBE-01 synthetic test 1 (exploratory, not sealed): chr1 and chr2, about 6 held-error opportunities per CpG, "
      f"{nmol:,} molecules; predicted held error (line, loose share 0) {firm:.4f}; planted: loss rate doubled over CpG {PL[0]:,} to {PL[1]:,}"]
for planted in (False, True):
    R = S.count_pat_sites(lines(planted)); c = Q2.count_pat(lines(planted), Q2.hg19_tiles())
    same = (R["total_err"] == round(c["held_err"] * c["held_opp"])) and (R["total_opp"] == c["held_opp"]) and abs(R["err"].sum() - R["total_err"]) < 0.5 and abs(R["opp"].sum() - R["total_opp"]) < 0.5
    D = S.departures(R["err"], R["opp"], firm); tag = "planted" if planted else "healthy"
    Lo.append(f"{tag}: per-site sums equal count_pat's totals: {same} (errors {R['total_err']:,}, opportunities {R['total_opp']:,}; held error {R['total_err'] / R['total_opp']:.4f})")
    Lo.append(f"{tag}: {D['blocks']:,} blocks, median {D['opp_per_block']:.0f} opportunities per block, line {D['line']:.2f}; changed {int(D['changed'].sum())}")
    if planted:
        idx, rows, n = S.site_owner(); pos = np.full(n, -1); pos[rows] = idx; p_ord = pos[D["order"]][D["measured"][:D["blocks"] * S.BLOCK]].reshape(D["blocks"], S.BLOCK)
        inside = ((p_ord >= PL[0]) & (p_ord <= PL[1])).all(1); edge = ((p_ord >= PL[0]) & (p_ord <= PL[1])).any(1) & ~inside
        Lo.append(f"planted: stretch blocks (all 10 sites inside) {int(inside.sum())}, changed {int((D['changed'] & inside).sum())} "
                  f"({(D['changed'] & inside).sum() / max(1, inside.sum()):.0%}); edge blocks changed {int((D['changed'] & edge).sum())} of {int(edge.sum())}; "
                  f"changed outside the stretch {int((D['changed'] & ~inside & ~edge).sum())}")
    print(Lo[-1], flush=True)
open(os.path.join(out, "synthetic_iama_globe_01_summary.txt"), "w").write("\n".join(Lo) + "\n"); print("\n".join(Lo))
