"""Start the on-demand box for a run (2026-10-12). Refused unless the run is in the published ledger with its estimate (ledger_guard).
Tags the box with the run name for per-run billing. Usage: python3 start_box.py TAG"""
import sys, boto3, ledger_guard
tag = sys.argv[1]; ledger_guard.require(tag); BOX = "i-0a040dddcaebdd65d"; ec2 = boto3.client("ec2", region_name="us-west-2")
ec2.create_tags(Resources=[BOX], Tags=[{"Key": "run", "Value": tag}]); ec2.start_instances(InstanceIds=[BOX]); print("started", BOX, "for", tag)
