#!/usr/bin/env python3
"""Regenerate example_runs/RUN_INDEX.csv from the evidence ledgers - never typed.

Three series, three different things, and the distinction is the point (author's question, 2026-09-25):

  VAL-###          the pre-atlas validation record. Historical and frozen; nothing new joins it.
  PROC-XXX-##      a TEST of the instrument: a question, bars fixed in a pre-registration before any data is
                   read, and a sealed outcome. This is how a test is marked, and it lives in doors/.
  RUN-YYYYMMDD-NN  one execution of the chain on one specimen. A run makes no claim and passes no bar, so it
                   is neither a VAL nor a PROC - but it is evidence and has to be findable later.

A run's record is its ledger row, written wherever the operator keeps it. This index collects the rows from
every ledger under the tree so there is one table of every run: what was measured, on which specimen, with
which panel and which chain commit.
"""
import csv
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "example_runs", "RUN_INDEX.csv")
COLS = ["run_id", "specimen", "cohort", "substrate", "platform", "run_timestamp_utc", "chain_commit",
        "immune_A_abs", "immune_placement", "trace_secretory", "trace_cycling", "stage0", "report", "ledger"]


def rows(paths):
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                yield p, d


def main(search=None):
    paths = sorted(set(glob.glob(search or os.path.join(HERE, "**", "evidence_ledger*.jsonl"),
                                 recursive=True)))
    out = []
    for p, d in rows(paths):
        out.append({
            "run_id": d.get("run_id") or "",
            "specimen": d.get("sample_id") or d.get("patient_id") or "",
            "cohort": d.get("cov.cohort") or d.get("lab") or "",
            "substrate": d.get("substrate") or "",
            "platform": d.get("array_type") or d.get("pipeline") or "",
            "run_timestamp_utc": d.get("run_timestamp_utc") or "",
            "chain_commit": (d.get("chain_commit") or "")[:12],
            "immune_A_abs": d.get("A_abs.immune", ""),
            "immune_placement": d.get("placement.immune", ""),
            "trace_secretory": d.get("trace.secretory", ""),
            "trace_cycling": d.get("trace.cycling", ""),
            "stage0": d.get("stage0_verdict") or "",
            "report": d.get("report") or "",
            "ledger": os.path.relpath(p, HERE)})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in sorted(out, key=lambda x: (x["run_id"] or "zz", x["specimen"])):
            w.writerow(r)
    print("RUN_INDEX.csv: %d runs from %d ledger(s)" % (len(out), len(paths)))
    for r in out[:6]:
        print("   %-18s %-12s %-14s %s" % (r["run_id"], r["specimen"], r["substrate"], r["report"][:46]))
    return len(out)


if __name__ == "__main__":
    main()
