"""DEV-COMPOSITION-TRUTH-04 Stage 1 read check (2026-10-11; supersedes the name-only check of platform_check.py as the deciding test).
The chain reads an array by BEAD ADDRESS through the standard EPIC v1 manifest (methylprep MethylationEPIC_v-1-0_B4); a custom chip is read
correctly only if every probe it shares with that manifest has the same address. Checks:
 1. Address identity: the submitters' custom-manifest addresses (columns M, U of GSE167202_ProcessedBetaValues.txt.gz, S3) against the B4
    manifest's AddressB_ID, AddressA_ID for every shared probe.
 2. Beads on the chip: every chain site's address(es) present in GSM5098953's IDAT (an array with no neutrophil count, so never in the bar-7
    scoring set; it is excluded in any case). Only probe names and addresses are used: no beta value or composition is looked at.
 3. Stage 1 runs: probes chain Stage 1 calibrated on GSM5098953 (stage1_probe_names_GSM5098953.txt.gz); chain sites missing there are reported.
Writes stage1_read_check_output.txt. Usage: python3 stage1_read_check.py CUSTOM_ADDR.csv B4_MANIFEST.csv.gz GRN.idat.gz  (needs methylprep)"""
import os, sys, gzip, json, pandas as pd, methylprep
from methylprep.files import IdatDataset
HERE = os.path.dirname(os.path.abspath(__file__)); RM = os.path.join(HERE, "../../../chain/Runtime Matrices/Met_A_Floors")
C = pd.read_csv(sys.argv[1]).set_index("IlmnID"); M = pd.read_csv(sys.argv[2], usecols=["IlmnID", "AddressA_ID", "AddressB_ID"], low_memory=False).set_index("IlmnID")
nrm = lambda x: "" if pd.isna(x) or str(x) in ("NA", "nan", "") else str(int(float(x)))
both = C.index.intersection(M.index)
same = (C.loc[both, "U"].map(nrm) == M.loc[both, "AddressA_ID"].map(nrm)) & (C.loc[both, "M"].map(nrm) == M.loc[both, "AddressB_ID"].map(nrm))
addr = set(int(float(x)) for x in IdatDataset(sys.argv[3], channel=methylprep.models.Channel.GREEN).probe_means.index)
names = set(gzip.open(os.path.join(HERE, "stage1_probe_names_GSM5098953.txt.gz"), "rt").read().split())
def sites(o):
    out = set()
    if isinstance(o, dict): [out.update(sites(v)) for v in o.values()]
    elif isinstance(o, list):
        for x in o:
            if isinstance(x, str) and x[:2] in ("cg", "ch"): out.add(x)
            else: out.update(sites(x))
    return out
B_ = json.load(open(os.path.join(RM, "blood_composition_EPIC_v1.json"))); F = json.load(open(os.path.join(RM, "metA_floors_v1_3.json")))
L = {"Met-A floor sites": sites(F), "composition markers": set(B_["markers"]) if isinstance(B_["markers"][0], str) else sites(B_["markers"]),
     "composition neutrophil sites": set(B_["neutrophil_sites"]), "noise sites": sites(json.load(open(os.path.join(RM, "noise_sites_EPIC_v1.json"))))}
ina = lambda v: pd.isna(v) or int(float(v)) in addr
out = [f"1. address identity: {len(both):,} probes shared with the EPIC v1 manifest; identical {int(same.sum()):,}; different {int((~same).sum())}"]
ok = bool(same.all())
for k, s in L.items():
    off = [x for x in s if not (ina(M.loc[x, "AddressA_ID"]) and ina(M.loc[x, "AddressB_ID"]))]; miss = s - names; ok &= not off
    out.append(f"2-3. {k}: {len(s):,} sites; beads absent from the chip {len(off)}; not calibrated on GSM5098953 (its own quality check) {len(miss)}")
out.append(f"Stage 1 on GSM5098953: {len(names):,} probes calibrated")
out.append("STAGE 1 READ CHECK " + ("PASSED: every chain site is on the chip at the standard EPIC v1 address; per-array losses as on any EPIC array" if ok else "FAILED"))
open(os.path.join(HERE, "stage1_read_check_output.txt"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
