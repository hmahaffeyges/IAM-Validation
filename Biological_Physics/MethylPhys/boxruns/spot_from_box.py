"""Start the box's disk on a spot instance (2026-10-11; author: cheapest method until funded). Steps:
  image  : AMI from the STOPPED on-demand box (i-0a040dddcaebdd65d), no reboot, tagged with the run name.
  launch : spot instance of TYPE from that AMI in the box's subnet and security group, with cloud-init told NOT to regenerate SSH host
           keys, and the box's Elastic IP moved to it, so the same ssh alias and host fingerprint keep working. Tagged Name=methylphys-cpu-01
           and run=<RUN> (cost per run in AWS billing).
  release: terminate the spot instance and move the Elastic IP back to the on-demand box (which stays stopped).
Usage: python3 spot_from_box.py image RUN | launch RUN AMI [TYPE] | release INSTANCE"""
import sys, time, boto3
R = "us-west-2"; BOX = "i-0a040dddcaebdd65d"; ec2 = boto3.client("ec2", region_name=R)
def box(): return ec2.describe_instances(InstanceIds=[BOX])["Reservations"][0]["Instances"][0]
def eip(): return ec2.describe_addresses()["Addresses"][0]
mode = sys.argv[1]
if mode == "image":
    b = box(); assert b["State"]["Name"] == "stopped", "stop the box first"
    a = ec2.create_image(InstanceId=BOX, Name=f"methylphys-box-{sys.argv[2]}-{int(time.time())}", NoReboot=True,
                         TagSpecifications=[{"ResourceType": "image", "Tags": [{"Key": "run", "Value": sys.argv[2]}]}])["ImageId"]
    ec2.get_waiter("image_available").wait(ImageIds=[a], WaiterConfig={"Delay": 20, "MaxAttempts": 90}); print(a)
elif mode == "launch":
    sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__))); import ledger_guard; ledger_guard.require(sys.argv[2])
    run, ami = sys.argv[2], sys.argv[3]; typ = sys.argv[4] if len(sys.argv) > 4 else "m7a.32xlarge"; b = box()
    ud = "#cloud-config\nssh_deletekeys: false\nssh_genkeytypes: []\n"
    i = ec2.run_instances(ImageId=ami, InstanceType=typ, MinCount=1, MaxCount=1, KeyName=b["KeyName"], SubnetId=b["SubnetId"],
                          SecurityGroupIds=[g["GroupId"] for g in b["SecurityGroups"]], UserData=ud, InstanceInitiatedShutdownBehavior="terminate",
                          InstanceMarketOptions={"MarketType": "spot", "SpotOptions": {"SpotInstanceType": "one-time", "InstanceInterruptionBehavior": "terminate"}},
                          TagSpecifications=[{"ResourceType": "instance", "Tags": [{"Key": "Name", "Value": "methylphys-cpu-01"}, {"Key": "run", "Value": run}]},
                                             {"ResourceType": "volume", "Tags": [{"Key": "run", "Value": run}]}])["Instances"][0]["InstanceId"]
    ec2.get_waiter("instance_running").wait(InstanceIds=[i]); e = eip()
    ec2.associate_address(AllocationId=e["AllocationId"], InstanceId=i, AllowReassociation=True); print(i, e["PublicIp"])
elif mode == "release":
    i = sys.argv[2]; assert i != BOX; ec2.terminate_instances(InstanceIds=[i]); ec2.get_waiter("instance_terminated").wait(InstanceIds=[i])
    ec2.associate_address(AllocationId=eip()["AllocationId"], InstanceId=BOX, AllowReassociation=True); print("released", i)
