"""Upload one finished atlas v2.1 block (its parquet, draws, prior, warnings and log) to S3; a missing block is reported, not created."""
import os, sys, boto3
B = "methylphys-data-945451304272-us-west-2-an"; P = os.environ.get("S3P", "results/ATLAS_V21/fit/blocks/")
OUTD = os.environ.get("OUTD", "/home/ubuntu/data/atlas_v21/blocks_v21"); i = int(sys.argv[1]); b = f"{OUTD}/block_{i:05d}"
s3 = boto3.client("s3", region_name="us-west-2"); fs = [b + x for x in (".parquet", "_draws.npz", "_prior.parquet", "_warnings.json")] + [f"logs/block_{i}.log"]
miss = [f for f in fs if not os.path.exists(f)]
for f in fs:
    if os.path.exists(f): s3.upload_file(f, B, P + os.path.basename(f))
print(i, "uploaded", len(fs) - len(miss), "missing", [os.path.basename(m) for m in miss])
