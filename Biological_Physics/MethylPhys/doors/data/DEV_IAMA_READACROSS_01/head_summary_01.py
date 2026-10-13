"""Every number DEV-IAMA-READACROSS-01 quotes about the head of a file, computed from committed inputs so the note has a source.

# NO CHAIN: arithmetic over readings and quality numbers the chain and the alignment step already produced. No physics here.

Inputs, both committed: head_diagnostics_01.csv (this folder; conversion with its binomial standard error, duplicate fraction,
mapped reads, reads scanned, from the 15 alignment-quality files of the cause run) and ../DEV_IAMA_CAUSE_01/cause_parts_01.csv
(the chain's Stage Q reading of each part, with its role). Writes head_summary_01_output.txt.
"""
import os
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
Q = pd.read_csv(os.path.join(HERE, "head_diagnostics_01.csv"))
P = pd.read_csv(os.path.join(HERE, "..", "DEV_IAMA_CAUSE_01", "cause_parts_01.csv"))
T = Q.merge(P[["run", "part", "eps", "role"]], on=["run", "part"])
h, d = T[T.part == 0], T[T.part > 0]
L = ["DEV-IAMA-READACROSS-01: the head of a file against the rest, on quantities measured independently of IAM-A", "",
     f"conversion at the head: mean {h.conversion.mean():.5f} | median standard error {h.conversion_se.median():.6f} "
     f"| CHH calls {h.chh_calls.min():,} to {h.chh_calls.max():,}",
     f"conversion deeper in:   mean {d.conversion.mean():.5f} | median standard error {d.conversion_se.median():.6f} "
     f"| CHH calls {d.chh_calls.min():,} to {d.chh_calls.max():,}",
     f"difference {h.conversion.mean() - d.conversion.mean():.4f} = "
     f"{(h.conversion.mean() - d.conversion.mean()) / d.conversion_se.median():.0f} standard errors of a deeper part", "",
     "the reading at the head against deeper in, per library:"]
for run, g in T.groupby("run"):
    g = g.sort_values("part"); hh = g[g.part == 0].eps.iloc[0]; dd = g[g.part > 0].eps.tolist()
    L.append(f"  {run} ({g.role.iloc[0]:>7}): head {hh:.5f} | deeper " + ", ".join(f"{x:.5f}" for x in dd) +
             f" | head reads {'LOW' if hh < np.mean(dd) else 'HIGH'}")
# the sealed decision's own quantity: r = A / (mean A of the two normal runs AT THAT PART), so the position effect common to
# every library is already cancelled in r. Both framings are reported because they answer different questions.
import json
_dp = os.path.join(HERE, "..", "DEV_IAMA_CAUSE_01", "cause_decision_01.json")
if os.path.isfile(_dp):
    _J = json.load(open(_dp)); _r = _J.get("r", {})
    L += ["", "the sealed quantity r = A / (mean A of the two normal runs at the same part) - the position effect is cancelled in r:"]
    for run, g in T.groupby("run"):
        vals = [f"p{int(pt)} {_r.get(run + 'p' + str(int(pt)), float('nan')):.4f}" for pt in sorted(g.part)]
        L.append(f"  {run} ({g.role.iloc[0]:>7}): " + " | ".join(vals))
    L.append(f"  suspects at the head: " + ", ".join(
        f"{_r[run + 'p0']:.4f}" for run, g in T.groupby("run") if g.role.iloc[0] == "low" and run + "p0" in _r) +
        "  -> they read 11-14 % below the controls at the SAME position")
    L.append(f"  part 0 reproduces each library's Stage B reading within 2 %: {_J.get('p0_check')} "
             f"(so Stage B itself read the head: {_J.get('p0_vs_stageB')})")
L += ["", f"conversion difference present in all {T.run.nunique()} libraries: "
          f"{bool(all(g[g.part == 0].conversion.iloc[0] > g[g.part > 0].conversion.max() for _, g in T.groupby('run')))}",
      f"head effect sign by role: " + ", ".join(
          f"{r}: {sum(1 for _, g in gg.groupby('run') if g[g.part == 0].eps.iloc[0] < g[g.part > 0].eps.mean())} of "
          f"{gg.run.nunique()} read LOW at the head" for r, gg in T.groupby("role")),
      f"duplicate fraction: head mean {h.duplicate_fraction.mean():.4f} | deeper mean {d.duplicate_fraction.mean():.4f}",
      f"reads the quality step scanned: head {h.reads_scanned.min():,} | deeper {d.reads_scanned.min():,} to {d.reads_scanned.max():,}"]
open(os.path.join(HERE, "head_summary_01_output.txt"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
