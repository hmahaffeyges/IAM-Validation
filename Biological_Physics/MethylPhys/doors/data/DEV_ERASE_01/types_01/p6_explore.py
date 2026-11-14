"""PHY-086 P6 (exploration grade, stated before reading in amendment 2): if the erasers set the difference between cell types, the wild-type
type gap ln(WT_B / WT_iNKT) equals ln(r_iNKT / r_B), and the knockouts lie closer together than the wild types. Reads counts/ (P4, B cells)
and counts_p5/ (P5, iNKT). Different experiments (2020, 2015): graded, not pass or fail. Writes p6_summary.txt."""
import os, json, glob, math
H = os.path.dirname(os.path.abspath(__file__))
def arms(d, wt):
    C = [json.load(open(f)) for f in glob.glob(os.path.join(H, d, "*_count.json"))]
    w = [c["held_err_length"] for c in C if c["gsm"] in wt]; k = [c["held_err_length"] for c in C if c["gsm"] not in wt]
    return sum(w) / len(w), sum(k) / len(k)
bw, bk = arms("counts", ("GSM4907469", "GSM4907470")); iw, ik = arms("counts_p5", ("GSM1855585", "GSM1855586"))
gap, pred = math.log(bw / iw), math.log((ik / iw) / (bk / bw))
L = [f"B cells: wild type {bw:.5f}, knockout {bk:.5f}, ratio {bk / bw:.3f}; iNKT: wild type {iw:.5f}, knockout {ik:.5f}, ratio {ik / iw:.3f}",
     f"wild-type type gap ln(B/iNKT) {gap:+.3f}; eraser-explained gap ln(r_iNKT/r_B) {pred:+.3f}",
     f"knockout gap ln(B/iNKT) {math.log(bk / ik):+.3f} against wild-type gap {gap:+.3f}: knockouts {'closer' if abs(math.log(bk / ik)) < abs(gap) else 'further apart'}",
     "Reading: the two types hold alike in wild type although their erasers differ (iNKT about 7 %, mature B none): the erasers do not set "
     "the type difference here; P6 not supported. Exploration grade (different experiments)."]
open(os.path.join(H, "p6_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
