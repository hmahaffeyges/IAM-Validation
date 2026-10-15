"""Atlas v2.1 fit progress for the README and the page (author 2026-10-15: the atlas run summary must update itself). Reads the box's
watcher file (s3://.../results/ATLAS_V21/fit/watch.json, written every 10 minutes on the box) and fit_started_utc from
CANON/v21_readiness.json, and writes CANON/fit_progress.json: blocks done, rate, expected finish in Pacific time. Run by checkpoint.sh;
without AWS credentials it leaves the file unchanged. Usage: python3 CANON/fit_progress.py"""
import os, json, datetime
from zoneinfo import ZoneInfo
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); OUT = os.path.join(ROOT, "CANON", "fit_progress.json")
try:
    import boto3
    W = json.loads(boto3.client("s3").get_object(Bucket="methylphys-data-945451304272-us-west-2-an", Key="results/ATLAS_V21/fit/watch.json")["Body"].read())
except Exception as e:
    print("fit_progress: unchanged (no watcher reading:", type(e).__name__ + ")"); raise SystemExit(0)
V = json.load(open(os.path.join(ROOT, "CANON", "v21_readiness.json"), encoding="utf-8"))
st = next(v["fit_started_utc"] for v in V.values() if isinstance(v, dict) and v.get("fit_started_utc"))
t0 = datetime.datetime.fromisoformat(st.replace("Z", "+00:00")); now = datetime.datetime.fromisoformat(W["utc"].replace("Z", "+00:00"))
done, of = int(W["blocks_done"]), int(W["of"]); h = (now - t0).total_seconds() / 3600; rate = done / h if h > 0 and done else 0
if W.get("blocks_last_3h"): rate = W["blocks_last_3h"] / 3.0   # the current rate, past the slow first wave
eta = now + datetime.timedelta(hours=(of - done) / rate) if rate else None; PT = ZoneInfo("America/Los_Angeles")
O = {"_rule": "written by CANON/fit_progress.py from the box watcher; do not edit by hand", "as_of_utc": W["utc"], "status": W["status"],
     "blocks_done": done, "blocks": of, "percent": round(100 * done / of, 1), "blocks_per_hour": round(rate, 1), "rate_basis": "blocks finished in the last 3 hours" if W.get("blocks_last_3h") else "average since the start",
     "expected_finish_pacific": eta.astimezone(PT).strftime("%A %B %-d, about %-I %p Pacific") if eta and W["status"] != "DONE" else None}
json.dump(O, open(OUT, "w", encoding="utf-8"), indent=1); open(OUT, "a").write("\n"); print("fit_progress:", O["blocks_done"], "of", of, "|", O["expected_finish_pacific"])
