#!/usr/bin/env python3
"""Controls for the manifest freshness gate: each case must be refused or accepted as stated, or the gate does not run (checked_push.sh: &&)."""
import os, sys, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import manifest_fresh as MF
row = lambda i, n="": {"id": i, "next_step": n, "record": "Biological_Physics/MethylPhys/doors/DEV_X.md"}
OLD = {"PHY-047": row("PHY-047", "old"), "PHY-055": row("PHY-055")}
UPD = dict(OLD, **{"PHY-047": row("PHY-047", "new")})
LOG0 = "### 2026-10-15 · old\n"; LOG1 = LOG0 + "### 2026-10-16 · Hansen run 4 aligning (PHY-047)\n"; LOGX = LOG0 + "### 2026-10-16 · Cell throughput corrected\n"
J0 = {"i-1": {"job_id": "i-1", "state": "running", "manifest": ["PHY-047"]}}; J1 = {"i-1": {"job_id": "i-1", "state": "failed", "manifest": ["PHY-047"]}}
S_OK = [("x.json", {"utc": "2026-10-16T01:00:00Z", "manifest": ["PHY-047"]})]; S_NO = [("x.json", {"utc": "2026-10-16T01:00:00Z", "manifest": []})]
S_OLD = [("x.json", {"utc": "2026-10-15T01:00:00Z"})]
cases = [("C1 tonight's miss: log and ledger updated, manifest not", (OLD, OLD, LOG0, LOG1, J0, J1, [], []), True),
         ("C2 log heading names no row", (OLD, UPD, LOG0, LOGX, J0, J0, [], []), True),
         ("C3 job changed state, its row unchanged", (OLD, OLD, LOG0, LOG0, J0, J1, [], []), True),
         ("C4 new job with no manifest row", (OLD, OLD, LOG0, LOG0, {}, {"i-2": {"job_id": "i-2", "state": "running"}}, [], []), True),
         ("C5 record changed, its row unchanged", (OLD, OLD, LOG0, LOG0, J0, J0, ["Biological_Physics/MethylPhys/doors/DEV_X.md"], []), True),
         ("C6 chain result with no manifest row", (OLD, OLD, LOG0, LOG0, J0, J0, [], S_NO), True),
         ("C7 correct push: log, job and record all with the row updated", (OLD, UPD, LOG0, LOG1, J0, J1, ["Biological_Physics/MethylPhys/doors/DEV_X.md"], S_OK), False),
         ("C8 chain result from before the cutover", (OLD, OLD, LOG0, LOG0, J0, J0, [], S_OLD), False),
         ("C9 nothing changed", (OLD, OLD, LOG0, LOG0, J0, J0, [], []), False)]
bad = []
for name, args, refuse in cases:
    F = MF.findings(*args); ok = bool(F) == refuse
    print(f"  {name}: {'REFUSED' if F else 'ACCEPTED'} (want {'REFUSE' if refuse else 'ACCEPT'})" + (f" - {F[0][:80]}" if F else "")); bad += [] if ok else [name]
print("manifest_fresh controls: " + (f"ok ({len(cases)} of {len(cases)})" if not bad else "FAILED: " + ", ".join(bad))); sys.exit(1 if bad else 0)
