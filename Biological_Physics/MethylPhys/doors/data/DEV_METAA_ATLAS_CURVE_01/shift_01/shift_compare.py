"""DEV-METAA-ATLAS-CURVE-01, the sequencing-array shift at the B and T identity sites (diagnostic on spent atlas inputs; written before any
of its readings exists; nothing scored). Each sequencing input of IAM-Atlas v2.1's B and T columns (seq_inputs.csv) read by the one command
(run_sample.py --pat, its own cell as the specimen) against its own column; set beside the same columns' array readings from the
development read (bt_02/bt_curve_dev_summary.txt). Per column: the sequencing inputs' Met-A, the arrays' median Met-A, and the difference.
Usage: python3 shift_compare.py BUNDLE_DIR OUT.txt"""
import sys, os, re, json, glob, statistics as st, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); S = pd.read_csv(os.path.join(H, "seq_inputs.csv"))
arr = {}
for l in open(os.path.join(H, "..", "bt_02", "bt_curve_dev_summary.txt")):
    m = re.match(r"atlas column (.+?): (\d+) of \d+ read, median ([0-9.]+)", l)
    if m: arr[m.group(1)] = float(m.group(3))
def met_a(B, cell):
    for c in (B.get("met_a_each_cell") or {}).get("cells") or []:
        if c.get("cell") == cell and c.get("A") is not None: return c["A"], "each cell"
    m = B.get("met_a") or {}
    return (m.get("A"), "sorted cell") if m.get("A") is not None else (None, (m.get("refusal") or m.get("reason") or "not read")[:80])
L = ["column | sample | sequencing Met-A | how read"]; per = {}
for r in S.itertuples():
    f = glob.glob(os.path.join(sys.argv[1], "**", f"{r.gsm}_bundle.json"), recursive=True)
    if not f: L.append(f"{r.cell} | {r.gsm} | missing |"); continue
    a, how = met_a(json.load(open(f[0])), r.cell); L.append(f"{r.cell} | {r.gsm} | {a} | {how}")
    if a is not None: per.setdefault(r.cell, []).append(a)
L += ["", "column | sequencing inputs read | sequencing median | array median (development read) | sequencing minus array"]
for c in sorted(set(S.cell)):
    v = per.get(c, []); s = st.median(v) if v else None; a = arr.get(c)
    L.append(f"{c} | {len(v)} | {'' if s is None else f'{s:.4f}'} | {'' if a is None else f'{a:.4f}'} | {'' if s is None or a is None else f'{s - a:+.4f}'}")
open(sys.argv[2], "w").write("\n".join(L) + "\n"); print("\n".join(L))
