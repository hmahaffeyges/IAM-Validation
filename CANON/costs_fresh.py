#!/usr/bin/env python3
"""Cost freshness (author 2026-10-16: the page showed 2026-10-12 costs on 2026-10-16; "that is how stale data leaks in"). The page reads
CANON/costs_snapshot.json, and the website build cannot refresh it (no AWS access there). A push is refused when the snapshot is older than
today (UTC); the checkpoint refreshes it (CANON/costs_snapshot.py). Usage: python3 CANON/costs_fresh.py"""
import os, json, datetime, sys
J = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "costs_snapshot.json"), encoding="utf-8"))
today = datetime.date.today().isoformat()   # the same clock costs_snapshot.py writes with
ok = J["as_of"] >= today
_hold = os.path.join(os.path.dirname(os.path.abspath(__file__)), "billing_unreadable.json")   # 2026-10-20: AWS unreadable, declared and dated
if not ok and os.path.exists(_hold) and J["as_of"] >= (datetime.date.fromisoformat(json.load(open(_hold))["since"]) - datetime.timedelta(days=1)).isoformat():
    print(f"costs_fresh: ok (as of {J['as_of']}; billing unreadable since {json.load(open(_hold))['since']}, declared in CANON/billing_unreadable.json)"); sys.exit(0)
print("costs_fresh: " + ("ok (" + J["as_of"] + ")" if ok else f"REFUSED: costs_snapshot.json is from {J['as_of']}, today is {today}; run python3 CANON/costs_snapshot.py (needs AWS billing access)")); sys.exit(0 if ok else 1)
