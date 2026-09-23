#!/usr/bin/env python3
"""Read the two measurements and state what they say: the healthy null, and the limit of detection."""
import json
import os
import statistics as st

CL = ["immune", "progenitor", "stem_adult", "stem_pluri", "secretory", "cycling", "terminal", "stromal"]
FLOOR = {"terminal": 0.0298, "stem_pluri": 0.02, "stem_adult": 0.02, "progenitor": 0.02,
         "cycling": 0.02, "immune": 0.02, "secretory": 0.02, "stromal": 0.02}


def q(v):
    v = sorted(v)
    if not v:
        return None
    return {"n": len(v), "min": round(v[0], 4), "median": round(st.median(v), 4),
            "max": round(v[-1], 4), "mean": round(sum(v) / len(v), 4)}


def null_scan():
    p = "handoff/floor_scan.json"
    if not os.path.exists(p):
        print("no null scan yet"); return
    d = {k: v for k, v in json.load(open(p)).items() if "error" not in v}
    print("=" * 104)
    print(f"THE HEALTHY NULL - {len(d)} healthy Uppsala donors, class-level solve (floor-masked values included)")
    print("=" * 104)
    print(f"  {'class':<12} {'arrays>0':>9} {'median%':>9} {'max%':>8} {'mean%':>8} {'>=floor':>8}  "
          f"needlet: arrays>0, median%")
    for c in CL:
        vals = [d[k]["class_level_pct"].get(c, 0.0) for k in d]
        nz = [v for v in vals if v]
        above = sum(1 for v in vals if v / 100.0 >= FLOOR[c])
        nv = [(d[k]["needlet_frac"].get(c) or 0) * 100 for k in d]
        nnz = [v for v in nv if v]
        s = q(nz)
        print(f"  {c:<12} {len(nz):>4}/{len(d):<4} "
              f"{(s['median'] if s else 0):>9} {(s['max'] if s else 0):>8} {(s['mean'] if s else 0):>8} "
              f"{above:>8}  {len(nnz)}/{len(d)}, {round(st.median(nnz),3) if nnz else 0}")
    agree = [d[k].get("agreement") for k in d]
    l1 = [d[k].get("L1_class") for k in d if isinstance(d[k].get("L1_class"), (int, float))]
    print(f"\n  solver agreement: {agree.count('AGREE')} AGREE, {agree.count('DISAGREE')} DISAGREE "
          f"| L1 between solvers: {q(l1)}")
    dis = [k for k in d if d[k].get("agreement") == "DISAGREE"]
    for k in dis[:6]:
        print(f"     DISAGREE {k}: class-level {d[k]['class_level_pct']} | needlet "
              f"{ {c: round(v, 4) for c, v in (d[k]['needlet_frac'] or {}).items() if v} }")
    return d


def lod():
    p = "handoff/dilution.json"
    if not os.path.exists(p):
        print("\nno dilution series yet"); return
    raw = json.load(open(p))
    ref = raw.get("_reference_A", {})
    rows = [v for k, v in raw.items() if not k.startswith("_") and "error" not in v]
    if not rows:
        print("\nno completed mixtures"); return
    print()
    print("=" * 104)
    print(f"LIMIT OF DETECTION - {len(rows)} mixtures: real healthy blood with a reference class spiked in")
    print("=" * 104)
    for cls in sorted({r["spike_class"] for r in rows}):
        sub = [r for r in rows if r["spike_class"] == cls]
        hosts = sorted({r["host"] for r in sub})
        print(f"\n  {cls.upper()}  (a pure specimen of this class would read A = "
              f"{ref.get(cls, {}).get('A')})")
        print(f"    {'spiked':>7} | " + " | ".join(f"{h[-4:]:>22}" for h in hosts) + " |  A(spike class) mean")
        print(f"    {'':>7} | " + " | ".join(f"{'class-lvl  rollup  ndlt':>22}" for h in hosts) + " |")
        fracs = sorted({r["spike_fraction"] for r in sub})
        zero = {}
        for f in fracs:
            cells, As = [], []
            for h in hosts:
                m = [r for r in sub if r["host"] == h and r["spike_fraction"] == f]
                if not m:
                    cells.append(f"{'-':>22}"); continue
                r = m[0]
                if f == 0:
                    zero[h] = max(r["recovered_class_level_pct"], r["recovered_cell_rollup_pct"],
                                  (r["recovered_needlet_frac"] or 0) * 100)
                cells.append(f"{r['recovered_class_level_pct']:>9}{r['recovered_cell_rollup_pct']:>8}"
                             f"{(r['recovered_needlet_frac'] or 0)*100:>6.2f}")
                if r["spike_class_A_on_identity_loci"] is not None:
                    As.append(r["spike_class_A_on_identity_loci"])
            am = round(sum(As) / len(As), 4) if As else None
            print(f"    {f*100:>6.2f}% | " + " | ".join(cells) + f" |  {am}")
        # the smallest spike whose class-level estimate clears every host's zero-spike reading
        ceiling = max(zero.values()) if zero else 0.0
        det = None
        for f in fracs:
            if f == 0:
                continue
            got = [r["recovered_class_level_pct"] for r in sub if r["spike_fraction"] == f]
            if got and min(got) > ceiling and min(got) > 0:
                det = f; break
        print(f"    zero-spike reading, worst host: {ceiling}% (this is the noise a floor must clear)")
        print(f"    smallest spike recovered above it on EVERY host: "
              f"{'none of the fractions tested' if det is None else f'{det*100:.2f}%'}")
        imm = [r["immune_A_abs"] for r in sub if r["immune_A_abs"] is not None]
        print(f"    immune A'' across these mixtures: {q(imm)}  (the commissioned gauge must not wander)")


if __name__ == "__main__":
    null_scan()
    lod()
