#!/usr/bin/env python3
"""Planted cases for CANON/design_check.py: a complete sheet passes, and each of the 2026-11-13 mistakes is refused."""
import os, sys, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import design_check as D
OK = {"physics": {"where_iam_law": "the candidate's hold against thermal kicks at 310 K"},
      "data": {"set": "GSE161463", "protocol_read": "sheared, end repaired, methylated adapters, standard directional WGBS from 1 ug",
               "spent_first": "no: spent sets hold no eraser-removed mature B cells", "alternatives": "GSE210026 (PBAT, refused)"},
      "kiss": {"smallest": "two wild type, two knockout: the fewest that measure replicate scatter", "samples_why": "each arm needs two"},
      "machines": {"platform": "github", "github_free_at_design": 17, "s_per_pair": "0.0005 (PHY-085 timing.json)", "pairs": 10000000,
                   "peak_mem_gb": 6, "machine_mem_gb": 16, "parallel_per_machine": 1, "expected_hours": 2.2, "time_limit_hours": 6, "cost": "free"},
      "save": {"resume": "ckpt"}, "watchers": {"count": 1, "stop_condition": "all four runs completed"},
      "canary": {"instrument_check": {"qc_failed_max": 0.05, "mapped_min": 0.95, "s_per_pair_max": 0.0015}},
      "cosmologist": "instrument check on a 2-million-pair canary before any full library"}
def case(label, edit, expect_refused):
    S = copy.deepcopy(OK); edit(S); F = D.sheet_findings(S, label)
    ok = bool(F) == expect_refused
    print(("ok   " if ok else "FAIL ") + label + ("" if ok else f" (findings: {F})")); return ok
R = [case("complete sheet passes", lambda S: None, False),
     case("missing question refused", lambda S: S["kiss"].pop("smallest"), True),
     case("google without a reason refused", lambda S: S["machines"].update(platform="google"), True),
     case("three 6 GB counts on a 16 GB machine refused (PHY-087, 11-13)", lambda S: S["machines"].update(parallel_per_machine=3), True),
     case("unmeasured memory with 3 per machine refused", lambda S: S["machines"].update(peak_mem_gb="not measured", parallel_per_machine=3), True),
     case("PBAT library without a method answer refused (GSE210026, 11-13)", lambda S: S["data"].update(protocol_read="post-bisulfite adapter tagging, about 500 cells"), True),
     case("1.75 h promised for 3.4 h of aligning refused (Google, 11-13)", lambda S: S["machines"].update(s_per_pair="0.00123", expected_hours=1.75), True),
     case("time limit under 1.3 x expected refused", lambda S: S["machines"].update(time_limit_hours=2.5), True),
     case("two watchers refused", lambda S: S["watchers"].update(count=2), True),
     case("canary without numbers refused", lambda S: S["canary"].update(instrument_check="looks fine"), True),
     case("count-only canary with its own three numbers passes", lambda S: S["canary"].update(instrument_check={"rows_min": 14000, "halves_corr_min": 0.7, "hours_per_sample_max": 3}), False)]
print("design_check_controls: " + ("ok" if all(R) else "FAILED")); sys.exit(0 if all(R) else 1)
