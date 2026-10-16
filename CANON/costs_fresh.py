#!/usr/bin/env python3
"""Cost freshness (author 2026-10-16: the page showed 2026-10-12 costs on 2026-10-16; "that is how stale data leaks in"). The page reads
CANON/costs_snapshot.json, and the website build cannot refresh it (no AWS access there). A push is refused when the snapshot is older than
today (UTC); the checkpoint refreshes it (CANON/costs_snapshot.py). Usage: python3 CANON/costs_fresh.py"""
import os, json, datetime, sys
J = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "costs_snapshot.json"), encoding="utf-8"))
today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
ok = J["as_of"] >= today
print("costs_fresh: " + ("ok (" + J["as_of"] + ")" if ok else f"REFUSED: costs_snapshot.json is from {J['as_of']}, today is {today}; run python3 CANON/costs_snapshot.py (needs AWS billing access)")); sys.exit(0 if ok else 1)
