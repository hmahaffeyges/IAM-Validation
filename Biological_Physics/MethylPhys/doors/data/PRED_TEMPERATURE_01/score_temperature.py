"""PRED-TEMPERATURE-01 stage T3 scorer, written and sealed 2026-10-17 before any plant read was counted. Run once on the counted samples.
usage: score_temperature.py <folder of out/<gsm>.json>
Rule (PRED_TEMPERATURE_01.md, stage T3): gates per sample (chain stage TL: >= 20,000 qualifying molecules, chloroplast conversion failure <= 1 %,
corrected held error above zero); a sample that fails a gate is left out with its reason. Replicates at one temperature: the mean of their corrected
held errors. An accession enters when both temperatures have a usable sample. Pairs are (10 C, 16 C). Bars B1, B2 as chain stage TL.score.
Information only (not bars): the cold/warm ratio of the sequencer error and of the conversion failure, and the reading repeated on the pairs whose two
sequencer errors differ by less than 20 %."""
import os, sys, glob, json, statistics
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "..", "chain")))
sys.path.insert(0, os.path.join(HERE, "chain"))
import stage_temperature as TL
S = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.json")))]
left = [dict(gsm=s["gsm"], accession=s["accession"], temp=s["temp"], reason=s["reason"]) for s in S if not s["usable"]]
by = {}
for s in S:
    if s["usable"]: by.setdefault(s["accession"], {}).setdefault(s["temp"], []).append(s)
pairs, P = [], []
for a, d in sorted(by.items()):
    if "10C" in d and "16C" in d:
        c, w = (statistics.mean(x["eps_corr"] for x in d[t]) for t in ("10C", "16C"))
        sc, sw = (statistics.mean(x["sub_err"] for x in d[t]) for t in ("10C", "16C"))
        fc, fw = (statistics.mean(x["conv_fail"] for x in d[t]) for t in ("10C", "16C"))
        pairs.append((c, w)); P.append(dict(accession=a, eps_10C=c, eps_16C=w, ratio=TL.logodds(c) / TL.logodds(w), sub_10C=sc, sub_16C=sw, conv_10C=fc, conv_16C=fw))
R = TL.score(pairs, 10, 16)
close = [(p["eps_10C"], p["eps_16C"]) for p in P if abs(p["sub_10C"] / p["sub_16C"] - 1) < 0.20]
info = dict(sub_err_ratio_cold_over_warm=statistics.median(p["sub_10C"] / p["sub_16C"] for p in P) if P else None,
            conv_fail_ratio_cold_over_warm=statistics.median(p["conv_10C"] / p["conv_16C"] for p in P if p["conv_16C"]) if P else None,
            reading_on_pairs_with_close_sequencer_error=TL.score(close, 10, 16) if len(close) >= 10 else f"{len(close)} pairs (fewer than 10)")
out = dict(test="PRED-TEMPERATURE-01 stage T3", samples=len(S), left_out=left, pairs=P, reading=R, information_only=info, verdict="MET" if R["met"] else "NOT MET")
TL.Q2.write_json(out, sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "t3_scored.json"), manifest=["PHY-057"])
print(json.dumps(dict(samples=len(S), left_out=len(left), pairs=R["pairs"], median_ratio=R["median_ratio"], ci95=R["ci95"], predicted=R["predicted"], window=R["window"], verdict=out["verdict"])))
