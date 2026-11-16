#!/usr/bin/env python3
"""Register A8 (exploration on spent atlas data; 2026-11-15). Counts two atlas-lab libraries never counted in PHY-050 (it kept 1-3 per type):
donor 30 F's naive CD8 (GSM5652297) and effector-memory CD8 (GSM5652294), so the same-donor naive vs memory reading has a second donor. Same
command as every other atlas count: percall_count.py's one() = run_sample.py --geo-pat --dev-count-only (chain Stage Q2). Writes xout/<GSM>.json.
Then copy the two counts into percall_out/ by a later edit and re-run a8_same_donor_gh.py. Usage (explore.yml): a8_count_two.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import percall_count as P
for g, cell in (("GSM5652297", "Naive T cells CD8"), ("GSM5652294", "T effector memory CD8")):
    r = P.one((g, cell)); print(f"::notice::{g} {cell}: {r[1] if isinstance(r, tuple) else r}")
