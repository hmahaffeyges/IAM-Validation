"""Every box job publishes its own code to S3 before it does any work (author, 2026-10-13: every piece of code, every document, everything
for reproduction goes into S3 and the repo, and eventually Zenodo). Usage, first lines of any *_box.py job:
    from publish_code import publish_code
    publish_code("results/<RUN>/code/", ["this_script.py", "inputs.csv", ...], job="<short name>")
Uploads each file and code_index.json (sha256 per file, repo commit if the job knows it, time) to that prefix; raises on any failure, so a
job that cannot publish its code does not run. CANON/box_code_check.py compares the index with the repo at the job's commit."""
import os, json, time, hashlib
BUCKET = "methylphys-data-945451304272-us-west-2-an"
def publish_code(prefix, files, job, commit=None):
    import boto3
    s3 = boto3.client("s3", region_name="us-west-2"); prefix = prefix.rstrip("/") + "/"
    idx = {"job": job, "commit": commit or os.environ.get("REPO_COMMIT", ""), "published_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": {}}
    for f in files:
        idx["files"][os.path.basename(f)] = hashlib.sha256(open(f, "rb").read()).hexdigest()
        s3.upload_file(f, BUCKET, prefix + os.path.basename(f))
    open("code_index.json", "w").write(json.dumps(idx, indent=1)); s3.upload_file("code_index.json", BUCKET, prefix + "code_index.json")
    for f in files: s3.head_object(Bucket=BUCKET, Key=prefix + os.path.basename(f))   # confirm every upload landed
    return idx
