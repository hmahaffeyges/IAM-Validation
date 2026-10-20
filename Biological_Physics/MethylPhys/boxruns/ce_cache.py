"""One shared, cached door to AWS Cost Explorer (2026-10-20). Cost Explorer charges $0.01 per request, and the ledger and snapshot scripts
asked it on nearly every commit: 953 requests, $10.34 with tax, by 2026-10-20 (the author's bill). Billing updates about once a day, so each
distinct request is sent at most once every 12 hours; repeats are answered from ~/.cache/iam_ce/ (outside the repository).
Usage: from ce_cache import cost_and_usage; r = cost_and_usage(TimePeriod=..., Granularity=..., Metrics=[...], ...)"""
import os, json, time, hashlib
MAX_AGE_S = 12 * 3600
D = os.path.join(os.path.expanduser("~"), ".cache", "iam_ce")
def cost_and_usage(**kw):
    os.makedirs(D, exist_ok=True)
    f = os.path.join(D, hashlib.sha256(json.dumps(kw, sort_keys=True, default=str).encode()).hexdigest()[:24] + ".json")
    if os.path.exists(f) and time.time() - os.path.getmtime(f) < MAX_AGE_S: return json.load(open(f))
    import boto3
    r = boto3.client("ce", region_name="us-east-1").get_cost_and_usage(**kw); r.pop("ResponseMetadata", None)
    json.dump(r, open(f + ".tmp", "w"), default=str); os.replace(f + ".tmp", f); return r
