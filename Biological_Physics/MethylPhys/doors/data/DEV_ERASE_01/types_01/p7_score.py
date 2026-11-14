"""PHY-090 P7 score (written and tested on planted counts before any library is read; 2026-11-13 night). LSK progenitors, GSE72630, the
laboratory of P4 (mature B) and P5 (iNKT), TET2 and TET3 removed. Prediction: knockout over wild type 0.86, sigma 0.035 (the erasers'
share in LSK at least TET2's alone, PHY-085 0.900, and about all three in stem cells, 0.864; sigma from replicate spread about 2 % and a
laboratory term about 3 %). Read in the agreed classes (WHY_IAM.md rule 4): pull within 2 passes, 2 to 3 watch, beyond 3 a real miss.
Null test: the three wild types agree, max |ln(WTi/WTj)| <= 0.05, else not assessed. Each arm is the mean of its libraries' lone-loss
held error (held_err_length; the one command, --dev-count-only). Rehearsal (spent PHY-085 LSK, TET2 alone, 0.900): pull +1.14.
P8 (exploration grade, stated now): with P4, P5 and P7 from one laboratory, if the held error is regulated toward a common level, the
three wild types spread less than the three knockouts (sd of ln level). Usage: python3 p7_score.py COUNTS_DIR. Writes p7_summary.txt."""
import os, sys, json, glob, math, itertools
H = os.path.dirname(os.path.abspath(__file__)); WT = ("GSM1866966", "GSM1866967", "GSM1866968"); KO = ("GSM1866969", "GSM1866970", "GSM1866971")
CENTER, SIGMA = 0.86, 0.035
def classify(p):
    p = round(p, 6)
    return "passes" if abs(p) <= 2 else ("watch" if abs(p) <= 3 else "a real miss")
def main(d):
    C = {json.load(open(f))["gsm"]: json.load(open(f)) for f in glob.glob(os.path.join(d, "*_count.json"))}
    miss = [g for g in WT + KO if g not in C]
    if miss: raise SystemExit(f"missing counts: {miss}")
    w = [C[g]["held_err_length"] for g in WT]; k = [C[g]["held_err_length"] for g in KO]
    null = max(abs(math.log(a / b)) for a, b in itertools.combinations(w, 2))
    r = (sum(k) / 3) / (sum(w) / 3); p = (r - CENTER) / SIGMA
    v = "NOT ASSESSED (wild types differ by more than 0.05)" if null > 0.05 else classify(p)
    L = [f"PHY-090 P7 (LSK, GSE72630): knockout over wild type {r:.3f}; prediction {CENTER} (sigma {SIGMA}); pull {p:+.2f}: {v}",
         f"null test: max |ln(WT ratio)| {null:.3f} (limit 0.05)"] + [f"{g} {'WT' if g in WT else 'KO'} held_err_length {C[g]['held_err_length']:.5f} loose {C[g].get('loose_share', float('nan')):.3f}" for g in WT + KO]
    P4, P5 = os.path.join(H, "counts"), os.path.join(H, "counts_p5")
    if os.path.isdir(P4) and os.path.isdir(P5):
        def arms(dd, wt):
            X = [json.load(open(f)) for f in glob.glob(os.path.join(dd, "*_count.json"))]
            return (sum(x["held_err_length"] for x in X if x["gsm"] in wt) / len(wt), sum(x["held_err_length"] for x in X if x["gsm"] not in wt) / (len(X) - len(wt)))
        b, i = arms(P4, ("GSM4907469", "GSM4907470")), arms(P5, ("GSM1855585", "GSM1855586"))
        lw = [math.log(x) for x in (b[0], i[0], sum(w) / 3)]; lk = [math.log(x) for x in (b[1], i[1], sum(k) / 3)]
        sd = lambda v: (sum((x - sum(v) / 3) ** 2 for x in v) / 2) ** 0.5
        L.append(f"P8 (exploration): one laboratory, B / iNKT / LSK: sd ln(level) wild type {sd(lw):.3f}, knockout {sd(lk):.3f}: "
                 f"{'wild types closer (the governor direction)' if sd(lw) < sd(lk) else 'wild types not closer'}")
    open(os.path.join(H, "p7_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
if __name__ == "__main__":
    main(sys.argv[1])
