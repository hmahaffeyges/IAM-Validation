"""Negative controls for ledger_guard.check (2026-10-12). Runs on the current RUNS.md; writes ledger_guard_test_output.txt. Usage: python3 ledger_guard_test.py"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); import ledger_guard as G
md = open(os.path.join(HERE, "..", "RUNS.md")).read()
c = [("listed WORKBOX01 (must pass)", G.check("WORKBOX01", md)[0] is True),
     ("listed BOXRUN10_stageB (must pass)", G.check("BOXRUN10_stageB", md)[0] is True),
     ("unlisted TEST_UNLISTED (must refuse)", G.check("TEST_UNLISTED", md)[0] is False),
     ("row without dollar estimate (must refuse)", G.check("WORKBOX01", md.replace("about 5 hours, $1-2", "about 5 hours"))[0] is False),
     ("prefix of a tag, WORKBOX0 (must refuse)", G.check("WORKBOX0", md)[0] is False),
     ("no Upcoming section (must refuse)", G.check("WORKBOX01", md.split("## Upcoming")[0])[0] is False)]
out = [("PASS " if ok else "FAIL ") + n for n, ok in c] + ["LEDGER GUARD CONTROLS " + ("PASSED" if all(ok for _, ok in c) else "FAILED")]
open(os.path.join(HERE, "ledger_guard_test_output.txt"), "w").write("\n".join(out) + "\n"); print("\n".join(out)); sys.exit(0 if all(ok for _, ok in c) else 1)
