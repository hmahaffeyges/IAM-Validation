"""stage_groupa.py - Group-A: has a whole group moved? (commissioned 2026-10-20, doors/COMMISSIONING_NOTE_GROUPA.md)

The instrument's job: two groups of healthy people read as no different after the tare, and a group shift planted into healthy arrays is found.

What it does. One person's whole-blood Met-A is too noisy to call a small change (healthy people differ by about 2 % after the tare), but the
average of a group is not. Group-A takes each person's Met-A reading from the one command (run_sample.py, unchanged), tares each reading
against healthy arrays run the same way (same slide when it holds at least 3 healthy references, else the same series; the C-Score band
test's tare, word for word), and compares the two groups' tared means (Welch: difference, standard error, t). Nothing is fitted.

  groups differ        |t| >= 2
  no difference found  |t| <  2, with the smallest difference this design could have found printed (2 x the standard error)

Groups must be run in the SAME series (same laboratory, same run): across laboratories the untared readings differ by several percent
(DEV-GROUP-01). Commissioned for whole blood, EPIC v1, two groups within one series; anything else is read and labeled preliminary.

Usage:
  python3 stage_groupa.py collect BUNDLE_DIR SAMPLES.csv OUT.csv     one row per person from run_sample.py bundles (<id>_bundle.json);
                                                                     SAMPLES.csv needs id, slide, and your group column (optional: healthy_reference)
  python3 stage_groupa.py compare READINGS.csv GROUP_COLUMN A B      tare, compare group A with group B, print the call
"""
import os, sys, json, numpy as np, pandas as pd
GROUPA_COMMISSIONING = {"status": "COMMISSIONED", "date": "2026-10-20", "reading": "Group-A", "platform": "EPIC v1", "specimens": ["whole blood"],
    "scope": "two groups within one series, each person read by run_sample.py and tared against same-run healthy references",
    "rule": "groups differ when |t| >= 2", "record": "doors/COMMISSIONING_NOTE_GROUPA.md",
    "evidence": "DEV-GROUPA-COMMISSION-01, 278 healthy controls of GSE318173: women vs men 0.36 % apart (1.2 SE); a planted 1 % loss found, +2.00 % at 7.3 SE"}
PRELIMINARY = "This score is preliminary until Group-A is officially commissioned on {what}."

def tare(r, healthy):
    import conductor_v3 as C
    hh = r[healthy & r.A.notna()]; out = []
    for _, x in r.iterrows():
        if pd.isna(x.A): out.append((None, None, 0)); continue
        res = None
        for scope in ("slide", "series"):
            ref = hh[(hh.gsm != x.gsm) & ((hh.slide == x.slide) if scope == "slide" else True)]
            if len(ref) >= 3: res = (float(x.A / ref.A.median()), scope, len(ref)); break
        out.append(res or (None, None, 0))
    r["A_rel"], r["ref_scope"], r["n_refs"] = zip(*out)
    cr = []
    for _, x in r.iterrows():
        v = None
        if pd.notna(x.C):
            for scope in ("slide", "series"):
                ref = hh[(hh.gsm != x.gsm) & ((hh.slide == x.slide) if scope == "slide" else True)].C.dropna()
                if len(ref) >= 3: v = float(x.C / ref.median()); break
        cr.append(v)
    r["C_rel"] = cr
    r["state"] = r.A_rel.map(lambda a: None if pd.isna(a) else ("Normal" if 0.95 <= a <= 1.05 else ("above Normal" if a > 1.05 else "below Normal")))
    return r

def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); se = float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
    d = float(a.mean() - b.mean()); return dict(n1=len(a), n2=len(b), mean1=round(float(a.mean()), 5), mean2=round(float(b.mean()), 5), difference=round(d, 5),
                                                 rel_difference=round(d / float(np.concatenate([a, b]).mean()), 5), se=round(se, 5), t=round(d / se, 3))

def compare(rows, group_col, a, b, healthy=None, specimen="whole blood", platform="EPIC v1"):
    """rows: one row per person with gsm, A (untared Met-A), C (optional), slide, group_col. healthy: boolean Series of the arrays the tare may
    use as references (default: every array). Returns the comparison, the call and, when needed, the preliminary label."""
    r = rows.copy()
    if "C" not in r: r["C"] = np.nan
    r = tare(r, pd.Series(True, index=r.index) if healthy is None else healthy.reindex(r.index).fillna(False).astype(bool))
    x, y = r[r[group_col] == a].A_rel.dropna(), r[r[group_col] == b].A_rel.dropna()
    why = []
    if len(x) < 3 or len(y) < 3: why.append(f"group {a!r} has {len(x)} tared readings and group {b!r} has {len(y)} (at least 3 each required)")
    untared = int(r.A.notna().sum() - r.A_rel.notna().sum())
    if why: return dict(call=None, reason="; ".join(why), untared=untared, rows=r)
    s = welch(x, y); s["call"] = "groups differ" if abs(s["t"]) >= 2 else "no difference found"
    s["smallest_findable_difference"] = round(2 * s["se"], 5); s["untared"] = untared
    s["ref_scope"] = r.ref_scope.value_counts().to_dict()
    if specimen != "whole blood" or platform != "EPIC v1":
        s["commissioning"] = PRELIMINARY.format(what=f"{specimen}, {platform}")
    s["rows"] = r; return s

def collect(bundle_dir, samples_csv, out_csv):
    S = pd.read_csv(samples_csv); L = []
    for _, s in S.iterrows():
        p = os.path.join(bundle_dir, f"{s['id']}_bundle.json")
        b = json.load(open(p)) if os.path.exists(p) else {}
        m = b.get("met_a") or {}; c = b.get("met_a_cscore") or {}
        L.append(dict(s, gsm=s["id"], A=m.get("A"), C=c.get("C"), withheld=None if m.get("A") is not None else (m.get("reason") or ("no bundle" if not b else "no reason recorded"))))
    pd.DataFrame(L).to_csv(out_csv, index=False); return len(L)

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    if len(sys.argv) >= 5 and sys.argv[1] == "collect":
        print(f"collected {collect(sys.argv[2], sys.argv[3], sys.argv[4])} people into {sys.argv[4]}")
    elif len(sys.argv) >= 5 and sys.argv[1] == "compare":
        R = pd.read_csv(sys.argv[2]); g, a, b = sys.argv[3], sys.argv[4], sys.argv[5]
        for col in (g,): R[col] = R[col].astype(str)
        h = R["healthy_reference"].astype(bool) if "healthy_reference" in R else None
        o = compare(R, g, a, b, h)
        if o.get("call") is None: print("Group-A: no call -", o["reason"]); sys.exit(2)
        print(f"Group-A ({GROUPA_COMMISSIONING['status']} {GROUPA_COMMISSIONING['date']}, {GROUPA_COMMISSIONING['scope']})\n"
              f"  {a}: tared mean {o['mean1']} (n {o['n1']})   {b}: tared mean {o['mean2']} (n {o['n2']})\n"
              f"  difference {100*o['rel_difference']:+.2f} %, {o['t']} standard errors -> {o['call']}\n"
              f"  smallest difference this design could have found: {100*o['smallest_findable_difference']:.2f} % (2 x the standard error)\n"
              f"  tare references: {o['ref_scope']}; readings with no tare: {o['untared']}" + (f"\n  {o['commissioning']}" if o.get("commissioning") else ""))
    else:
        print(__doc__); sys.exit(1)
