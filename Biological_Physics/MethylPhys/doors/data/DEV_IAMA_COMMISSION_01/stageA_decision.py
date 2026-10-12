"""DEV-IAMA-COMMISSION-01 Stage A stop decision (rules sealed 2026-10-11, DEV_IAMA_COMMISSION_01.md 'How the libraries are read', items 6-7).
Reads stageA_libraries.csv (read_libraries_01.py on the 7 TruSeq replicate-1 libraries). Decides only what the seal says Stage A decides:
  C1 (from score_iama_commission_01: every c1_iama >= C1_MIN) -> Stage B runs only if MET;
  Stage B depth: 10,000,000 pairs, or 10,000,000 x 2,400,000 / median opportunities if that median is below 2,400,000 (cap 25,000,000).
C2-C5 are not scored here. Writes stageA_decision_output.txt. Usage: python3 stageA_decision.py"""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); import score_iama_commission_01 as S
L = pd.read_csv(os.path.join(HERE, "stageA_libraries.csv")); c1 = L.c1_iama.dropna()
met = len(c1) == 7 and bool((c1 >= S.C1_MIN).all()); med = float(L.opp.median())
pairs = 10_000_000 if med >= 2_400_000 else min(25_000_000, int(round(10_000_000 * 2_400_000 / med, -5)))
out = [f"libraries read: {len(L)}; refused by Stage Q0 (conversion < 0.98): {int((L.q0 != 'PROCEED').sum())} ({', '.join(L[L.q0 != 'PROCEED'].line)})",
       f"conversion (human chr1 CHH): {L.conversion.min():.5f} - {L.conversion.max():.5f}",
       f"C1: c1_iama {c1.min():.4f} - {c1.max():.4f} (bar >= {S.C1_MIN}) -> {'MET' if met else 'NOT MET'}",
       f"median opportunities {med:,.0f} -> Stage B depth {pairs:,} read pairs",
       "DECISION: " + ("Stage B runs (71 libraries)" if met else "STOP: Stage B does not run")]
open(os.path.join(HERE, "stageA_decision_output.txt"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
