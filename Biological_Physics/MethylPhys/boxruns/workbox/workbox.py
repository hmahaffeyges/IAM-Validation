"""Work machine (2026-10-12; author: use AWS for justified work instead of tying up the laptop, frugally). A small spot instance started from
the box image runs ONE committed job list (jobs_*.sh) unattended: no login, results saved to S3 item by item, then it terminates itself. A hard
time cap (shutdown at MAXMIN minutes) means it cannot run on forever. Uses the box's S3 role (instance profile methylphys-box-s3).
Usage: python3 workbox.py launch NAME AMI TYPE JOBS.sh MAXMIN   |   python3 workbox.py dryrun NAME AMI TYPE JOBS.sh MAXMIN"""
import os, sys, boto3
mode, name, ami, typ, jobs, maxmin = sys.argv[1:7]
ec2 = boto3.client("ec2", region_name="us-west-2")
ud = f"""#!/bin/bash
shutdown -h +{int(maxmin)}
sudo -u ubuntu bash -lc 'cd ~/IAM-Validation && git pull -q origin main && bash Biological_Physics/MethylPhys/boxruns/workbox/{jobs} {name}' > /var/log/workbox.out 2>&1
shutdown -h now
"""
kw = dict(ImageId=ami, InstanceType=typ, MinCount=1, MaxCount=1, SubnetId="subnet-0243614badb21eb0d", SecurityGroupIds=["sg-06f40a9c356f019de"],
          KeyName="methylphys-data", IamInstanceProfile={"Name": "methylphys-box-s3"}, UserData=ud, InstanceInitiatedShutdownBehavior="terminate",
          InstanceMarketOptions={"MarketType": "spot", "SpotOptions": {"SpotInstanceType": "one-time", "InstanceInterruptionBehavior": "terminate"}},
          TagSpecifications=[{"ResourceType": "instance", "Tags": [{"Key": "Name", "Value": "methylphys-workbox"}, {"Key": "run", "Value": name}]}])
if mode == "dryrun":
    try: ec2.run_instances(DryRun=True, **kw)
    except Exception as e: print(str(e)[:200])
else:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); import ledger_guard; ledger_guard.require(name)
    i = ec2.run_instances(**kw)["Instances"][0]["InstanceId"]; print(name, i, typ)
