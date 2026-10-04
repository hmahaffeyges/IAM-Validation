#!/usr/bin/env python3
"""run1_driver.py - Box Run 1 driver (development mode). Runs the jobs of boxruns/run1/JOBS.md in order, resumes after a stop,
logs to S3, and stops the instance at the end and on a crash.

What JOBS.md defines (summary; JOBS.md is the source)
-----------------------------------------------------
Inputs are in s3://methylphys-data-945451304272-us-west-2-an/ (prefixes in INPUTS below, about 151 GB). Outputs go to
results/BOXRUN1/<job>/ in the same bucket, with a progress log at results/BOXRUN1/log.txt. Every job works from the calibrated
betas of DEV-BASE-CHAIN-01 (results/DEV_BASE_CHAIN_01/betas) where they exist and calibrates only the arrays that lack them.

  A  Commissioning check: self-tare II, then the median tare, on purified neutrophils. Bars from doors/CHAIN_COMMISSIONING.md
     (stage 5 and stage 8, DEV-SELFTARE-02): same-person replicates (GSE250556) within-person SD <= 0.020 and >= 95 % Normal;
     purified neutrophils from other laboratories Normal on tared A; floor arrays 6/6 Normal. Set: the 68-array set plus the 2
     GSE128733 arrays. Output: one table per bar and one row per array.
  B  Every chain test set read again with the adopted tare: Met-A, noise index, C-score, direction and composition, one row per
     array, per set (and the difference map on the longitudinal set).
  C  Met-A C-score on every held-out healthy array: spread (median, 2.5-97.5 %) per laboratory and per set. No band is set.
  D  Sky statistics with the hard mask and the apodised mask side by side, healthy replicates against the block-shuffle null.
     Bars (DEV-SKY-02): median band power ratio 0.9-1.1 in every band; look-elsewhere rate at the stated 8.4 %.
     Needs the apodised-mask code (GitHub session task 1, PR #18: chain/sky_statistics.py masked_spectrum(..., mask="hard"|"apodised",
     apod_deg, taper)). The driver detects it: if the installed sky_statistics has no `mask` parameter, D is BLOCKED ("apodised mask
     (PR #18) not merged yet"); if it has it, D runs both masks on the same block shuffles. Width: --apod-deg (default 2.0).
  E  Atlas composition (atlas_e) on bloods with known composition. Runs only if GSE112618 or GSE182379 has been downloaded first;
     JOBS.md gives no S3 location for them, so their location is passed with --job-e-prefix / --job-e-dir. Without one, or when the
     given location holds no files, E is SKIPPED with the reason in the log and in state.json (not a failure; exit code unaffected).

Default locations (the author's instruction; --options override them)
  GSE128733 (job A, 2 EPIC arrays)  s3://<bucket>/downloads/G_chain_tests/neutrophil_ref_GSE128733/   (--gse128733-prefix / --gse128733-dir).
                                    A missing or empty location is a WARNING (log and state.json); job A continues without the 2 arrays.
  atlas v2 (jobs D and E)           s3://methylphys-data-945451304272-us-west-2-an/atlas_v2/IAMAtlas_v2.parquet, synced to
                                    <work>/data/atlas_v2/  (--atlas-v2 / CPG_ATLAS_V2_PARQUET override it).
  S3 access                         boto3 with the default credential chain (credentials configured on the box; the instance has no
                                    IAM role). --aws-region / --aws-profile are passed to the boto3 session only when given.

How it runs
-----------
    # plan only (no S3, no jobs, no shutdown)
    python3 run1_driver.py --work /home/ubuntu/data/boxrun1 --dry-run
    # the run (boto3, default credential chain; GSE128733 and the atlas from their default locations; stops the instance at the end
    # and on a crash)
    python3 run1_driver.py --work /home/ubuntu/data/boxrun1 --python /home/ubuntu/env/bin/python
    # job E as well, once GSE112618 / GSE182379 are in the bucket
    python3 run1_driver.py --work /home/ubuntu/data/boxrun1 --python /home/ubuntu/env/bin/python --job-e-prefix <key prefix>
    # local test (JOBS.md "still to do" 4): 3 arrays per set, shutdown switched off, a local directory standing in for the bucket
    python3 run1_driver.py --work /tmp/r1 --s3-local-root /tmp/fake_bucket --limit-arrays 3 --no-shutdown

Each job runs as a subprocess (by default `run1_driver.py worker <JOB>`, which reads arrays through the chain's own entry point
chain/MethylPhys_Interface/run_sample.py). State lives in <work>/results/BOXRUN1/state.json (per job: status, attempts, output files
with sizes, bars). A rerun skips jobs whose status is done and whose recorded outputs are still there; a job left running by a stop
is wiped and started again (per-array readings are cached outside the job's output folder, so a restarted job does not re-read them).
The log and the state are copied to S3 at every job boundary and at the end, inside try/finally, before the shutdown.

Exit codes: 0 every job done and every bar met (E skipped counts as done); 1 crash (a job command failed, or the driver itself);
3 a job could not run (a requirement named in the log is missing); 4 a commissioning bar was not met. 2 is argparse's usage error.
Worst wins: 1 > 3 > 4 > 0.

Development mode: these are development readings; nothing here sets a bar, a band or a frozen value. No credentials are stored here.
"""
import argparse
import csv
import datetime as _dt
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import traceback

# ------------------------------------------------------------------------------------------------ constants from JOBS.md
BUCKET = "methylphys-data-945451304272-us-west-2-an"
RESULTS_PREFIX = "results/BOXRUN1"            # outputs: results/BOXRUN1/<job>/, log: results/BOXRUN1/log.txt
LOG_NAME = "log.txt"
STATE_NAME = "state.json"                     # kept beside the log under results/BOXRUN1/
BETAS_PREFIX = "results/DEV_BASE_CHAIN_01/betas"

# key -> (S3 prefix, files, GB, used by) - the JOBS.md input table
INPUTS = {
    "betas":            (BETAS_PREFIX, 56, 26.6, "ABC"),
    "GSE250556":        ("downloads/G_chain_tests/GSE250556", 131, 1.1, "ACD"),
    "neutrophil_ref":   ("downloads/G_chain_tests/neutrophil_ref", 28, 9.1, "A"),
    "healthy_repeat":   ("downloads/G_chain_tests/healthy_repeat", 66, 20.6, "BCD"),
    "infection":        ("downloads/G_chain_tests/infection", 43, 39.4, "B"),
    "myeloid":          ("downloads/G_chain_tests/myeloid", 41, 21.2, "B"),
    "autoimmune":       ("downloads/G_chain_tests/autoimmune", 22, 14.8, "B"),
    "prediagnosis":     ("downloads/G_chain_tests/prediagnosis", 10, 7.7, "B"),
    "longitudinal":     ("downloads/G_chain_tests/longitudinal", 18, 7.6, "B"),
    "neutrophil_state": ("downloads/G_chain_tests/neutrophil_state", 8, 2.5, "B"),
}
# the 2 GSE128733 purified-neutrophil arrays of job A (2 EPIC arrays; location given by the author, JOBS.md input table).
# Default for job A; --gse128733-prefix / --gse128733-dir override it. Size from the JOBS.md table (4 files, 0.03 GB).
# A missing or empty location is a warning: job A continues without the 2 arrays.
GSE128733_PREFIX_DEFAULT = "downloads/G_chain_tests/neutrophil_ref_GSE128733"
GSE128733_GB = 0.03
# atlas v2 parquet (jobs D and E): s3://methylphys-data-945451304272-us-west-2-an/atlas_v2/IAMAtlas_v2.parquet (location confirmed
# by the author). Synced to <work>/data/atlas_v2/IAMAtlas_v2.parquet and used when neither --atlas-v2 nor CPG_ATLAS_V2_PARQUET is given.
ATLAS_V2_KEY_DEFAULT = "atlas_v2/IAMAtlas_v2.parquet"
ATLAS_JOBS = "DE"
B_SETS = ["healthy_repeat", "infection", "myeloid", "autoimmune", "prediagnosis", "longitudinal", "neutrophil_state"]

JOB_ORDER = ["A", "B", "C", "D", "E"]
JOB_TITLES = {
    "A": "Commissioning check: self-tare II, then the median tare, on purified neutrophils",
    "B": "Every chain test set read again with the adopted tare",
    "C": "Met-A C-score on every held-out healthy array (spread per laboratory and per set)",
    "D": "Sky statistics with the apodised mask (hard and apodised mask side by side)",
    "E": "Atlas composition (atlas_e) on bloods with known composition",
}
# files a job command must leave in its output folder for the driver to call it done
REQUIRED_OUTPUTS = {"A": ["A_arrays.csv"], "B": ["B_all.csv"], "C": ["C_arrays.csv", "C_spread_by_lab.csv", "C_spread_by_set.csv"],
                    "D": ["D_sky.csv"], "E": ["E_arrays.csv"]}

# bars (carried from doors/CHAIN_COMMISSIONING.md / DEV-SELFTARE-02 / DEV-SKY-02; none is set here)
NORMAL = (0.95, 1.05)                         # Normal range of tared Met-A (DEV-SELFTARE-02 readings)
BAR_REPL_SD = 0.020
BAR_REPL_NORMAL_FRAC = 0.95
BAR_FLOOR_N = 6
BAR_SKY_RATIO = (0.9, 1.1)
BAR_SKY_LEE_RATE = 0.084
SKY_BAR_SERIES = "GSE250556"                  # DEV-SKY-02: the band-ratio bar is the median over the GSE250556 arrays
APOD_DEG_DEFAULT = 2.0                        # PR #18 default width of the apodised mask (degrees)
D_BLOCKED_PR18 = "apodised mask (PR #18) not merged yet"

# sets of job A (from the repository notes): other laboratories = the DEV-BASE-CHAIN-01 (b) set enlarged by DEV-INTAKE-02 check 6
OTHER_LAB_SERIES = ["GSE247193", "GSE247195", "GSE122244", "GSE118144", "GSE167998"]
FLOOR_SERIES = "GSE110554"
# the two GSE128733 purified-neutrophil arrays (development log, DEV-PAIRED-01): not in the DEV-BASE-CHAIN-01 manifest
GSE128733_ARRAYS = [{"series": "GSE128733", "gsm": "GSM3684010", "slide": "200357150019", "specimen": "isolated neutrophils",
                     "healthy": "True", "person": "", "plat": "EPIC_v1", "title": "Sample6"},
                    {"series": "GSE128733", "gsm": "GSM3684011", "slide": "200357150019", "specimen": "isolated neutrophils",
                     "healthy": "True", "person": "", "plat": "EPIC_v1", "title": "Sample7"}]
# job E series named in JOBS.md; specimen passed to run_sample.py (GSE182379 = constructed mixtures, read as whole blood by the chain)
E_SERIES_SPECIMEN = {"GSE112618": "whole blood", "GSE182379": "constructed DNA mixture"}

EXIT_OK, EXIT_CRASH, EXIT_BLOCKED, EXIT_BARS = 0, 1, 3, 4
EXIT_HELP = ("exit codes: 0 every selected job done (E skipped counts as done) and every bar met; 1 crash (a job command or the driver); "
             "3 a job blocked by a missing requirement; 4 a commissioning bar not met; 2 usage error. Worst wins: 1 > 3 > 4 > 0.")
_EXIT_RANK = {EXIT_CRASH: 3, EXIT_BLOCKED: 2, EXIT_BARS: 1, EXIT_OK: 0}

HERE = os.path.dirname(os.path.abspath(__file__))
MP_DEFAULT = os.path.normpath(os.path.join(HERE, "..", ".."))          # Biological_Physics/MethylPhys
CHAIN_DEFAULT = os.path.join(MP_DEFAULT, "chain")
MANIFEST_DEFAULT = os.path.join(MP_DEFAULT, "doors", "data", "DEV_BASE_CHAIN_01", "manifest.csv")
DEFAULT_SHUTDOWN = "sudo shutdown -h now"     # EC2: an instance-initiated shutdown stops an EBS-backed instance (default behaviour)


def worst(*codes):
    return max(codes, key=lambda c: _EXIT_RANK.get(c, 3))


def utcnow():
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


class Stopped(BaseException):
    """Raised in the main thread by SIGTERM / SIGHUP so the finally block (log upload, shutdown) runs."""


# ------------------------------------------------------------------------------------------------ log
class Log:
    def __init__(self, path, echo=True):
        self.path = path; self.echo = echo
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def __call__(self, *parts):
        line = f"{utcnow()} " + " ".join(str(p) for p in parts)
        with open(self.path, "a") as f:
            f.write(line + "\n")
        if self.echo:
            print(line, flush=True)


# ------------------------------------------------------------------------------------------------ S3 (injectable)
def _s3_missing(e):
    """True if a boto3/botocore error says the key is not there (ClientError 404 / NoSuchKey / NotFound)."""
    code = str(((getattr(e, "response", None) or {}).get("Error") or {}).get("Code", ""))
    return code in ("404", "NoSuchKey", "NotFound")


class Boto3Store:
    """The bucket through boto3 with the DEFAULT credential chain (e.g. environment variables, or ~/.aws/credentials and ~/.aws/config
    on the box): no keys, no role assumption and no profile are set here; --aws-profile / --aws-region are passed on only when given.
    Sync semantics: a download skips a local file that already exists with the same size; an upload skips a key
    that already exists with the same size and is not older than the local file."""

    def __init__(self, bucket=BUCKET, region=None, profile=None, client=None):
        self.bucket = bucket; self.region = region; self.profile = profile
        if client is None:
            try:
                import boto3
            except ImportError:
                raise SystemExit("boto3 is not installed in the python running run1_driver.py; it is needed for S3 "
                                 "(pip install boto3), or use --s3-local-root for a local test")
            kw = {"region_name": region}
            if profile:
                kw["profile_name"] = profile
            client = boto3.session.Session(**kw).client("s3")
        self.client = client

    def uri(self, key):
        return f"s3://{self.bucket}/{key.lstrip('/')}"

    def _list(self, prefix):
        """(key, size, last_modified) of every object under prefix/ (paginated)."""
        pfx = prefix.strip("/") + "/"
        for page in self.client.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=pfx):
            for o in page.get("Contents", []) or []:
                if not o["Key"].endswith("/"):
                    yield o["Key"], o["Size"], o.get("LastModified")

    def has_files(self, prefix):
        return next(iter(self._list(prefix)), None) is not None

    def download_prefix(self, prefix, dest):
        os.makedirs(dest, exist_ok=True); pfx = prefix.strip("/") + "/"
        for key, size, _lm in self._list(prefix):
            local = os.path.join(dest, *key[len(pfx):].split("/"))
            if os.path.isfile(local) and os.path.getsize(local) == size:
                continue
            os.makedirs(os.path.dirname(local), exist_ok=True)
            self.client.download_file(self.bucket, key, local)

    def _head_size(self, key):
        try:
            return self.client.head_object(Bucket=self.bucket, Key=key)["ContentLength"]
        except Exception as e:
            if _s3_missing(e):
                return None
            raise

    def download_file(self, key, dest):
        """True if the key existed and was copied, False if it is not there."""
        if self._head_size(key) is None:
            return False
        os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
        self.client.download_file(self.bucket, key, dest)
        return os.path.exists(dest)

    def sync_file(self, key, dest):
        """Copy one key to dest unless dest already has the same size. True if dest is there afterwards, False if the key is not."""
        size = self._head_size(key)
        if size is None:
            return False
        if not (os.path.isfile(dest) and os.path.getsize(dest) == size):
            os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
            self.client.download_file(self.bucket, key, dest)
        return os.path.exists(dest)

    def upload_file(self, path, key):
        self.client.upload_file(path, self.bucket, key.lstrip("/"))

    def upload_dir(self, path, prefix):
        pfx = prefix.strip("/") + "/"
        remote = {k: (sz, lm) for k, sz, lm in self._list(prefix)}
        for root, _d, files in os.walk(path):
            for fn in files:
                p = os.path.join(root, fn); key = pfx + os.path.relpath(p, path).replace(os.sep, "/")
                r = remote.get(key)
                if r and r[0] == os.path.getsize(p) and r[1] is not None and r[1].timestamp() >= os.path.getmtime(p):
                    continue
                self.client.upload_file(p, self.bucket, key)


class LocalStore:
    """A local directory standing in for the bucket (tests and the local driver test): <root>/<key>."""

    def __init__(self, root):
        self.root = os.path.abspath(root); self.bucket = f"local:{self.root}"

    def uri(self, key):
        return os.path.join(self.root, key)

    def download_prefix(self, prefix, dest):
        src = self.uri(prefix)
        if not os.path.isdir(src):
            raise RuntimeError(f"input prefix not found in the store: {prefix}")
        shutil.copytree(src, dest, dirs_exist_ok=True)

    def has_files(self, prefix):
        return any(files for _r, _d, files in os.walk(self.uri(prefix)))

    def download_file(self, key, dest):
        src = self.uri(key)
        if not os.path.isfile(src):
            return False
        os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True); shutil.copy2(src, dest); return True

    def sync_file(self, key, dest):
        src = self.uri(key)
        if not os.path.isfile(src):
            return False
        if not (os.path.isfile(dest) and os.path.getsize(dest) == os.path.getsize(src)):
            os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True); shutil.copy2(src, dest)
        return True

    def upload_file(self, path, key):
        dst = self.uri(key); os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy2(path, dst)

    def upload_dir(self, path, prefix):
        shutil.copytree(path, self.uri(prefix), dirs_exist_ok=True)


# ------------------------------------------------------------------------------------------------ shutdown (injectable)
def make_shutdown(cmd):
    def _shutdown(log):
        log("SHUTDOWN:", cmd)
        r = subprocess.run(shlex.split(cmd), capture_output=True, text=True)
        if r.returncode != 0:
            log("SHUTDOWN command failed", r.returncode, (r.stderr or r.stdout)[-300:])
        return r.returncode
    return _shutdown


# ------------------------------------------------------------------------------------------------ small statistics (stdlib)
def fnum(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v else None


def median(xs):
    xs = sorted(xs); n = len(xs)
    if not n:
        return None
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def percentile(xs, q):
    """Linear interpolation between order statistics (numpy's default)."""
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q / 100.0; f = int(k); c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def within_person_sd(rows, col, by="person"):
    """Pooled within-person SD: sqrt(sum of squared deviations from each person's mean / (n - number of persons))."""
    g = {}
    for r in rows:
        v = fnum(r.get(col))
        if v is not None and r.get(by):
            g.setdefault(r[by], []).append(v)
    ss = sum(sum((v - sum(vs) / len(vs)) ** 2 for v in vs) for vs in g.values()); dof = sum(len(vs) for vs in g.values()) - len(g)
    return (ss / dof) ** 0.5 if dof > 0 else None


def in_normal(v):
    return v is not None and NORMAL[0] <= v <= NORMAL[1]


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, cols=None):
    cols = cols or sorted({k for r in rows for k in r})
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})


# ------------------------------------------------------------------------------------------------ commissioning bars
def bars_A(out_dir):
    """Job A bars (CHAIN_COMMISSIONING.md stages 5 and 8 / DEV-SELFTARE-02) from A_arrays.csv; writes one table per bar and bars.json.
    Columns used: role (replicate | other_lab | floor), person, A_st (self-tare II), A_st_tared (self-tare II then median tare)."""
    rows = read_csv(os.path.join(out_dir, "A_arrays.csv"))
    rep = [r for r in rows if r.get("role") == "replicate"]
    oth = [r for r in rows if r.get("role") == "other_lab"]
    flo = [r for r in rows if r.get("role") == "floor"]
    bars = []
    # replicates
    vals = [fnum(r.get("A_st_tared")) for r in rep]; vals = [v for v in vals if v is not None]
    sd = within_person_sd(rep, "A_st_tared"); nn = sum(in_normal(v) for v in vals); frac = nn / len(vals) if vals else None
    write_csv(os.path.join(out_dir, "A_bar_replicates.csv"),
              [dict(r, in_normal=in_normal(fnum(r.get("A_st_tared")))) for r in rep],
              ["gsm", "series", "slide", "person", "A", "A_st", "A_st_tared", "ref_scope", "n_refs", "in_normal", "status"])
    bars.append({"name": "replicates within-person SD (GSE250556)", "measured": sd, "bar": f"<= {BAR_REPL_SD}",
                 "met": sd is not None and sd <= BAR_REPL_SD, "n": len(vals)})
    bars.append({"name": "replicates in Normal (GSE250556)", "measured": f"{nn}/{len(vals)}", "fraction": frac,
                 "bar": f">= {BAR_REPL_NORMAL_FRAC:.0%}", "met": frac is not None and frac >= BAR_REPL_NORMAL_FRAC, "n": len(vals)})
    # other laboratories: every array that has a tared A in Normal
    tared = [(r, fnum(r.get("A_st_tared"))) for r in oth]
    have = [(r, v) for r, v in tared if v is not None]; untared = [r["gsm"] for r, v in tared if v is None]
    on = sum(in_normal(v) for _, v in have)
    write_csv(os.path.join(out_dir, "A_bar_otherlab.csv"), [dict(r, in_normal=in_normal(v)) for r, v in tared],
              ["gsm", "series", "slide", "A", "A_st", "A_st_tared", "ref_scope", "n_refs", "in_normal", "status"])
    bars.append({"name": "other-laboratory purified neutrophils in Normal on tared A", "measured": f"{on}/{len(have)}", "bar": "all",
                 "met": bool(have) and on == len(have), "n": len(have), "not_tared": untared})
    # floor arrays: 6/6 Normal (self-tare II, as DEV-SELFTARE-02 read them)
    fv = [(r, fnum(r.get("A_st"))) for r in flo]; fn = sum(in_normal(v) for _, v in fv)
    write_csv(os.path.join(out_dir, "A_bar_floor.csv"), [dict(r, in_normal=in_normal(v)) for r, v in fv],
              ["gsm", "series", "slide", "A", "A_st", "A_st_tared", "in_normal", "status"])
    bars.append({"name": "floor arrays in Normal", "measured": f"{fn}/{len(fv)}", "bar": f"{BAR_FLOOR_N}/{BAR_FLOOR_N}",
                 "met": len(fv) == BAR_FLOOR_N and fn == BAR_FLOOR_N, "n": len(fv)})
    res = {"job": "A", "source": "doors/CHAIN_COMMISSIONING.md stage 5 and stage 8 (DEV-SELFTARE-02)", "normal": list(NORMAL),
           "bars": bars, "all_met": all(b["met"] for b in bars)}
    atomic_write_json(os.path.join(out_dir, "bars.json"), res)
    return res


def bars_D(out_dir):
    """Job D bars (DEV-SKY-02) per mask from D_sky.csv (one row per array and mask; columns mask, ratio_1..ratio_6, structure_beyond_null)."""
    rows = [r for r in read_csv(os.path.join(out_dir, "D_sky.csv")) if r.get("mask")]; bars = []   # rows without a sky carry no mask
    for mask in sorted({r.get("mask") for r in rows}):
        rs = [r for r in rows if r.get("mask") == mask]
        rb = [r for r in rs if r.get("series") == SKY_BAR_SERIES] or rs      # band bar: GSE250556 arrays (all rows if none is tagged)
        meds = [median([v for v in (fnum(r.get(f"ratio_{b}")) for r in rb) if v is not None]) for b in range(1, 7)]
        ok = all(m is not None and BAR_SKY_RATIO[0] <= m <= BAR_SKY_RATIO[1] for m in meds)
        flags = [str(r.get("structure_beyond_null")).strip().lower() in ("true", "1") for r in rs]
        rate = sum(flags) / len(flags) if flags else None
        bars.append({"name": f"{mask} mask: median band power / block-shuffle null, bands 1-6", "measured": meds,
                     "bar": f"{BAR_SKY_RATIO[0]}-{BAR_SKY_RATIO[1]} in every band", "met": ok, "n": len(rb), "band1": meds[0]})
        bars.append({"name": f"{mask} mask: look-elsewhere rate", "measured": rate, "bar": f"<= {BAR_SKY_LEE_RATE}",
                     "met": rate is not None and rate <= BAR_SKY_LEE_RATE, "n": len(rs)})
    res = {"job": "D", "source": "doors/CHAIN_COMMISSIONING.md (DEV-SKY-02)", "bars": bars, "all_met": bool(bars) and all(b["met"] for b in bars)}
    atomic_write_json(os.path.join(out_dir, "bars.json"), res)
    return res


BARS = {"A": bars_A, "D": bars_D}


# ------------------------------------------------------------------------------------------------ requirements per job
SKY_FUNCS = ("masked_spectrum", "shuffled_null", "sky_spectrum_record")


def sky_mask_support(chain_dir):
    """(True, '') if chain/sky_statistics.py has the apodised mask of PR #18 (a `mask` parameter on masked_spectrum, shuffled_null and
    sky_spectrum_record), else (False, why). Read from the source with ast, so the driver needs neither numpy nor healpy."""
    import ast
    p = os.path.join(chain_dir, "sky_statistics.py")
    try:
        tree = ast.parse(open(p, encoding="utf-8").read())
    except (OSError, SyntaxError) as e:
        return False, f"{p} cannot be read ({type(e).__name__})"
    args = {n.name: [a.arg for a in n.args.args + n.args.kwonlyargs] for n in tree.body if isinstance(n, ast.FunctionDef)}
    missing = [f for f in SKY_FUNCS if "mask" not in args.get(f, [])]
    if missing:
        return False, f"{p}: {', '.join(missing)} take no `mask` parameter"
    return True, ""


E_SKIP_NOT_GIVEN = ("job E skipped: inputs absent (GSE112618 / GSE182379 location not given: --job-e-prefix / --job-e-dir; "
                    "JOBS.md: job E runs only if one of them has been downloaded first)")


def atlas_desc(cfg):
    if cfg.atlas_v2_key:
        return f"{cfg.atlas_v2} (synced from s3://{BUCKET}/{cfg.atlas_v2_key}; --atlas-v2 / CPG_ATLAS_V2_PARQUET override it)"
    return f"{cfg.atlas_v2} (--atlas-v2 / CPG_ATLAS_V2_PARQUET)"


def job_e_absent(store, cfg):
    """None if job E has input files, else the reason it is skipped: no location given, or no file under any given location.
    A prefix is looked up in the store (in <work>/data/<prefix> with --skip-input-sync); a folder on the local disk."""
    if not (cfg.job_e_prefix or cfg.job_e_dir):
        return E_SKIP_NOT_GIVEN
    looked = []
    for p in cfg.job_e_prefix:
        if cfg.skip_input_sync:
            d = os.path.join(cfg.data_dir, p)
            if any(f for _r, _d, f in os.walk(d)):
                return None
            looked.append(d)
        else:
            if store.has_files(p):
                return None
            looked.append(store.uri(p.strip("/") + "/"))
    for d in cfg.job_e_dir:
        if any(f for _r, _d, f in os.walk(d)):
            return None
        looked.append(d)
    return f"job E skipped: inputs absent (no files under {', '.join(looked)})"


def requirements(job, cfg):
    """(status, reasons): status 'ok', 'blocked' (a requirement is missing) or 'skipped' (JOBS.md's own run condition is not met).
    Only for the default worker commands; a --job-cmd override is the operator's own command and is not checked."""
    reasons = []
    rs = os.path.join(cfg.chain_dir, "MethylPhys_Interface", "run_sample.py")
    if job in ("A", "B", "C", "E"):
        if not os.path.isfile(rs):
            reasons.append(f"chain entry point not found: {rs}")
        if not os.path.isfile(cfg.manifest):
            reasons.append(f"array manifest not found: {cfg.manifest}")
    if job == "A":
        if not (cfg.gse128733_prefix or cfg.gse128733_dir):
            reasons.append("the 2 GSE128733 arrays have no location: pass --gse128733-prefix <key prefix in the bucket> or "
                           f"--gse128733-dir <local folder> (default prefix {GSE128733_PREFIX_DEFAULT})")
        st = os.path.join(cfg.chain_dir, "Runtime Matrices", "Development", "dev_selftare_typeII_EPIC_v1.json")
        if not os.path.isfile(st):
            reasons.append(f"self-tare II runtime file not found: {st}")
    if job == "D":
        ok, why = sky_mask_support(cfg.chain_dir)
        if not ok:
            reasons.append(f"{D_BLOCKED_PR18} (JOBS.md: GitHub session task 1): {why}; job D runs the hard and apodised masks side by side")
        for f in ("dev_stages.py", "conductor_v3.py"):
            if not os.path.isfile(os.path.join(cfg.chain_dir, f)):
                reasons.append(f"chain module not found: {os.path.join(cfg.chain_dir, f)}")
        if not os.path.isfile(cfg.manifest):
            reasons.append(f"array manifest not found: {cfg.manifest}")
        if not (cfg.atlas_v2 and os.path.isfile(cfg.atlas_v2)):
            reasons.append(f"atlas v2 parquet not found: {atlas_desc(cfg)} (the sky needs the atlas v2 parent means)")
    if job == "E":
        if not (cfg.job_e_prefix or cfg.job_e_dir):
            return "skipped", [E_SKIP_NOT_GIVEN]
        if not (cfg.atlas_v2 and os.path.isfile(cfg.atlas_v2)):
            reasons.append(f"atlas v2 parquet not found: {atlas_desc(cfg)} (atlas_e needs it)")
    return ("blocked" if reasons else "ok"), reasons


def job_inputs(job, cfg):
    """[(S3 key prefix, local dir)] the job reads."""
    out = [(p, os.path.join(cfg.data_dir, p)) for k, (p, _n, _gb, used) in INPUTS.items() if job in used]
    if job == "D":      # JOBS.md: every job works from the calibrated betas where they exist (synced once; A-C use them too)
        out.insert(0, (BETAS_PREFIX, os.path.join(cfg.data_dir, BETAS_PREFIX)))
    if job == "A" and cfg.gse128733_prefix:
        out.append((cfg.gse128733_prefix, os.path.join(cfg.data_dir, cfg.gse128733_prefix)))
    if job == "E":
        out += [(p, os.path.join(cfg.data_dir, p)) for p in cfg.job_e_prefix]
    return out


def default_command(job, cfg, out_dir):
    c = [cfg.python, os.path.abspath(__file__), "worker", job, "--work", cfg.work, "--out", out_dir, "--chain-dir", cfg.chain_dir,
         "--manifest", cfg.manifest, "--workers", str(cfg.workers)]
    if cfg.limit_arrays:
        c += ["--limit-arrays", str(cfg.limit_arrays)]
    if cfg.atlas_v2:
        c += ["--atlas-v2", cfg.atlas_v2]
    if job == "D":
        c += ["--apod-deg", str(cfg.apod_deg)]
    if job == "A":
        for d in ([os.path.join(cfg.data_dir, cfg.gse128733_prefix)] if cfg.gse128733_prefix else []) + list(cfg.gse128733_dir or []):
            c += ["--extra-dir", d]
    if job == "E":
        for d in [os.path.join(cfg.data_dir, p) for p in cfg.job_e_prefix] + list(cfg.job_e_dir):
            c += ["--extra-dir", d]
    return c


def job_command(job, cfg, out_dir):
    if job in cfg.job_cmd:
        fmt = {"python": cfg.python, "driver": os.path.abspath(__file__), "work": cfg.work, "out": out_dir, "data": cfg.data_dir, "job": job}
        return [tok.format(**fmt) for tok in shlex.split(cfg.job_cmd[job])], True
    return default_command(job, cfg, out_dir), False


# ------------------------------------------------------------------------------------------------ state
class State:
    def __init__(self, path):
        self.path = path
        self.data = {"run": "BOXRUN1", "bucket": None, "created": utcnow(), "jobs": {}}

    def load(self):
        if os.path.exists(self.path):
            with open(self.path) as f:
                self.data = json.load(f)
            return True
        return False

    def job(self, j):
        return self.data["jobs"].setdefault(j, {"status": "pending", "attempts": 0})

    def save(self):
        self.data["updated"] = utcnow(); atomic_write_json(self.path, self.data)


def snapshot_outputs(out_dir):
    snap = {}
    for root, _d, files in os.walk(out_dir):
        for fn in files:
            p = os.path.join(root, fn); snap[os.path.relpath(p, out_dir)] = os.path.getsize(p)
    return snap


def outputs_intact(js, out_dir):
    rec = js.get("outputs") or {}
    if not rec:
        return False
    for rel, size in rec.items():
        p = os.path.join(out_dir, rel)
        if not os.path.isfile(p) or os.path.getsize(p) != size:
            return False
    return True


# ------------------------------------------------------------------------------------------------ the driver
class Config:
    pass


def sync_atlas(store, cfg, job, log):
    """Jobs D and E with the default atlas: copy s3://<bucket>/atlas_v2/IAMAtlas_v2.parquet to <work>/data/ unless it is already
    there with the same size. A failure is logged; requirements() then blocks the job with the reason."""
    if not cfg.atlas_v2_key or cfg.skip_input_sync:
        return
    t0 = time.time()
    try:
        if store.sync_file(cfg.atlas_v2_key, cfg.atlas_v2):
            log(f"[{job}] atlas v2 ready", cfg.atlas_v2_key, f"{time.time() - t0:.0f}s")
        else:
            log(f"[{job}] atlas v2 not found in the store: {cfg.atlas_v2_key}")
    except Exception as e:
        log(f"[{job}] atlas v2 sync failed: {type(e).__name__}: {str(e)[:300]}")


def make_config(a):
    cfg = Config()
    cfg.work = os.path.abspath(a.work)
    cfg.data_dir = os.path.join(cfg.work, "data")
    cfg.results_dir = os.path.join(cfg.work, *RESULTS_PREFIX.split("/"))
    cfg.cache_dir = os.path.join(cfg.work, "cache")
    cfg.chain_dir = os.path.abspath(a.chain_dir)
    cfg.manifest = os.path.abspath(a.manifest)
    cfg.python = a.python
    cfg.workers = a.workers
    cfg.limit_arrays = a.limit_arrays
    cfg.atlas_v2 = a.atlas_v2; cfg.atlas_v2_key = None
    if not cfg.atlas_v2:                       # default: the atlas in the bucket, synced to the work dir (jobs D and E)
        cfg.atlas_v2_key = ATLAS_V2_KEY_DEFAULT
        cfg.atlas_v2 = os.path.join(cfg.data_dir, *ATLAS_V2_KEY_DEFAULT.split("/"))
    cfg.apod_deg = a.apod_deg
    # job A: the default GSE128733 prefix unless --gse128733-prefix or --gse128733-dir is given (either one overrides it)
    cfg.gse128733_prefix = a.gse128733_prefix or (None if a.gse128733_dir else GSE128733_PREFIX_DEFAULT)
    cfg.gse128733_dir = a.gse128733_dir or []
    cfg.job_e_prefix = a.job_e_prefix or []
    cfg.job_e_dir = a.job_e_dir or []
    cfg.job_cmd = {}
    for s in a.job_cmd or []:
        k, _, v = s.partition("=")
        if k not in JOB_ORDER or not v:
            raise SystemExit(f"--job-cmd {s!r}: expected JOB=COMMAND with JOB in {JOB_ORDER}")
        cfg.job_cmd[k] = v
    cfg.only = [j for j in (a.only.split(",") if a.only else JOB_ORDER)]
    bad = [j for j in cfg.only if j not in JOB_ORDER]
    if bad:
        raise SystemExit(f"--only: unknown job(s) {bad}; jobs are {JOB_ORDER}")
    cfg.force = set(a.force or [])
    cfg.stop_on_failure = a.stop_on_failure
    cfg.skip_input_sync = a.skip_input_sync
    cfg.job_timeout = a.job_timeout
    return cfg


def print_plan(cfg, store_desc, state, shutdown_desc):
    p = print
    p("Box Run 1 plan (dry run: nothing is downloaded, run, uploaded or stopped)")
    p(f"  work dir      {cfg.work}")
    p(f"  store         {store_desc}")
    p(f"  log           {cfg.results_dir}/{LOG_NAME} -> {RESULTS_PREFIX}/{LOG_NAME}")
    p(f"  state         {cfg.results_dir}/{STATE_NAME} -> {RESULTS_PREFIX}/{STATE_NAME}")
    p(f"  shutdown      {shutdown_desc}")
    p(f"  atlas v2      {atlas_desc(cfg)}")
    seen = {}
    for j in JOB_ORDER:
        js = state.data["jobs"].get(j, {})
        sel = "selected" if j in cfg.only else "not selected (--only)"
        out_dir = os.path.join(cfg.results_dir, j)
        cmd, override = job_command(j, cfg, out_dir)
        stat, why = ("ok", []) if override else requirements(j, cfg)
        if cfg.atlas_v2_key and stat == "blocked":     # the default atlas is synced at run time, before the check
            why = [w for w in why if not w.startswith("atlas v2 parquet not found")]; stat = "blocked" if why else "ok"
        p(f"\n[{j}] {JOB_TITLES[j]}  - {sel}; state: {js.get('status', 'pending')}")
        for pre, loc in job_inputs(j, cfg):
            gb = next((v[2] for v in INPUTS.values() if v[0] == pre), GSE128733_GB if pre == GSE128733_PREFIX_DEFAULT else None)
            if j in cfg.only:
                seen[pre] = gb
            p(f"    input   s3://{BUCKET}/{pre}/" + (f"  ({gb} GB)" if gb is not None else ""))
        p(f"    command {' '.join(shlex.quote(c) for c in cmd)}" + ("  (override)" if override else ""))
        p(f"    outputs {out_dir}/ -> {RESULTS_PREFIX}/{j}/ ; required {REQUIRED_OUTPUTS[j]}")
        if j in BARS:
            p(f"    bars    checked by the driver ({BARS[j].__doc__.splitlines()[0]})")
        if j in ATLAS_JOBS and cfg.atlas_v2_key:
            p(f"    atlas   s3://{BUCKET}/{cfg.atlas_v2_key} -> {cfg.atlas_v2} (synced before the job)")
        if j == "E" and stat == "ok" and not override:
            why = ["inputs are checked for files at run time (an empty location skips E)"]
        p(f"    ready   {stat}" + ("".join(f"\n            - {w}" for w in why)))
    known = [g for g in seen.values() if g is not None]
    p(f"\nInputs of the selected jobs: {len(seen)} prefixes, about {sum(known):.1f} GB from the JOBS.md table"
      + (f" plus {len(seen) - len(known)} prefix(es) of unknown size" if len(known) < len(seen) else "") + "; downloaded once each.")


def sync_up(store, cfg, log, job=None):
    """Copy a job's outputs (if given), the log and the state to S3. Errors are logged, never raised."""
    ok = True
    try:
        if job and os.path.isdir(os.path.join(cfg.results_dir, job)):
            store.upload_dir(os.path.join(cfg.results_dir, job), f"{RESULTS_PREFIX}/{job}")
    except Exception as e:
        ok = False; log("S3 upload of job outputs failed", job, type(e).__name__, str(e)[:300])
    for name in (STATE_NAME, LOG_NAME):
        p = os.path.join(cfg.results_dir, name)
        try:
            if os.path.exists(p):
                store.upload_file(p, f"{RESULTS_PREFIX}/{name}")
        except Exception as e:
            ok = False; log("S3 upload failed", name, type(e).__name__, str(e)[:300])
    return ok


def gse128733_check(store, cfg):
    """Job A: (warnings, prefixes not to sync) for GSE128733 locations that are missing or hold no files. A prefix is looked up in the
    store (in <work>/data/<prefix> with --skip-input-sync); a --gse128733-dir folder on the local disk."""
    warn, skip = [], set()
    tail = "; job A continues without the 2 GSE128733 arrays (GSM3684010, GSM3684011)"
    if cfg.gse128733_prefix:
        p = cfg.gse128733_prefix
        if cfg.skip_input_sync:
            d = os.path.join(cfg.data_dir, p)
            if not any(f for _r, _d, f in os.walk(d)):
                warn.append(f"GSE128733 location missing or empty: no files under {d}{tail}")
        elif not store.has_files(p):
            warn.append(f"GSE128733 location missing or empty: no files under {store.uri(p.strip('/') + '/')}{tail}"); skip.add(p)
    for d in cfg.gse128733_dir:
        if not any(f for _r, _d, f in os.walk(d)):
            warn.append(f"GSE128733 location missing or empty: no files under {d}{tail}")
    return warn, skip


def sync_inputs(store, cfg, job, log, skip=()):
    for prefix, local in job_inputs(job, cfg):
        if prefix in skip:
            continue
        mark = os.path.join(cfg.data_dir, ".synced", prefix.strip("/").replace("/", "__"))
        if os.path.exists(mark):
            continue
        t0 = time.time(); log(f"[{job}] input sync", prefix)
        store.download_prefix(prefix, local)
        os.makedirs(os.path.dirname(mark), exist_ok=True); open(mark, "w").write(utcnow())
        log(f"[{job}] input ready", prefix, f"{time.time() - t0:.0f}s")


def run_command(cmd, log_path, cwd, timeout):
    """Run one job command; stdout/stderr to log_path. Kills the child if the driver is interrupted."""
    with open(log_path, "a") as fh:
        fh.write(f"# {utcnow()} {' '.join(shlex.quote(c) for c in cmd)}\n"); fh.flush()
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=cwd)
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait(); return -9
        except BaseException:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
            raise


def tail(path, n=2000):
    try:
        with open(path, errors="replace") as f:
            return f.read()[-n:]
    except OSError:
        return ""


def run_job(j, cfg, state, store, log):
    """Run one job; returns its exit-code contribution. Updates and saves the state; syncs to S3 at the boundary."""
    js = state.job(j); out_dir = os.path.join(cfg.results_dir, j)
    if js.get("status") == "done" and j not in cfg.force:
        if outputs_intact(js, out_dir):
            log(f"[{j}] done in an earlier run ({js.get('finished')}): skipped")
            return EXIT_BARS if (js.get("bars") and not js["bars"].get("all_met")) else EXIT_OK
        log(f"[{j}] marked done but its outputs are missing or changed: running it again")
    if js.get("status") in ("running", "interrupted"):
        log(f"[{j}] was interrupted in an earlier run (status {js['status']}, started {js.get('started')}): restarting it cleanly")
    cmd, override = job_command(j, cfg, out_dir)
    e_skip = job_e_absent(store, cfg) if (j == "E" and not override) else None
    if e_skip:
        stat, why = "skipped", [e_skip]
    else:
        if j in ATLAS_JOBS:
            sync_atlas(store, cfg, j, log)          # before the requirement check, which looks for the local file
        stat, why = ("ok", []) if override else requirements(j, cfg)
    if stat != "ok":
        js.update(status=stat, reason=why, finished=utcnow(), outputs={}); state.save()
        for w in why:
            log(f"[{j}] {stat.upper()}: {w}")
        sync_up(store, cfg, log, None)
        return EXIT_BLOCKED if stat == "blocked" else EXIT_OK
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)                         # clean start: nothing of an interrupted or failed attempt is kept
    os.makedirs(out_dir)
    js.update(status="running", attempts=js.get("attempts", 0) + 1, started=utcnow(), finished=None, reason=None, outputs={},
              bars=None, returncode=None, command=cmd, warnings=None)
    state.save()
    log(f"[{j}] START {JOB_TITLES[j]} (attempt {js['attempts']})")
    sync_up(store, cfg, log, None)
    t0 = time.time(); jlog = os.path.join(out_dir, f"job_{j}.log")
    try:
        skip = ()
        if j == "A":                                   # a missing or empty GSE128733 location: warning only, A goes on without it
            warn, skip = gse128733_check(store, cfg)
            if warn:
                js["warnings"] = warn; state.save()
                for w in warn:
                    log(f"[{j}] WARNING: {w}")
        if not cfg.skip_input_sync:
            sync_inputs(store, cfg, j, log, skip)
        rc = run_command(cmd, jlog, cfg.work, cfg.job_timeout)
    except BaseException:
        js.update(status="interrupted", finished=utcnow()); state.save(); raise
    js["returncode"] = rc; js["seconds"] = round(time.time() - t0, 1)
    missing = [f for f in REQUIRED_OUTPUTS[j] if not os.path.isfile(os.path.join(out_dir, f))]
    contrib = EXIT_OK
    if rc != 0 or missing:
        js.update(status="failed", finished=utcnow(),
                  reason=(f"exit {rc}" if rc != 0 else "") + (f"; missing outputs {missing}" if missing else ""))
        log(f"[{j}] FAILED ({js['reason']}) after {js['seconds']}s; last lines of {jlog}:\n{tail(jlog)}")
        contrib = EXIT_CRASH
    else:
        if j in BARS:
            try:
                res = BARS[j](out_dir); js["bars"] = res
                for b in res["bars"]:
                    log(f"[{j}] bar {'MET' if b['met'] else 'NOT MET'}: {b['name']}: {b['measured']} (bar {b['bar']})")
                if not res["all_met"]:
                    contrib = EXIT_BARS; log(f"[{j}] commissioning bars NOT all met (JOBS.md / CHAIN_COMMISSIONING.md)")
            except Exception as e:
                js.update(status="failed", finished=utcnow(), reason=f"bar check failed: {type(e).__name__}: {e}")
                log(f"[{j}] FAILED: bar check could not be read: {e}"); contrib = EXIT_CRASH
        if contrib != EXIT_CRASH:
            js.update(status="done", finished=utcnow(), outputs=snapshot_outputs(out_dir))
            log(f"[{j}] DONE in {js['seconds']}s ({len(js['outputs'])} files)")
    state.save()
    sync_up(store, cfg, log, j)
    return contrib


def restore_from_s3(store, cfg, state, log):
    """No local state (a new disk): take the state from S3 and the outputs of the jobs it records as done."""
    if store.download_file(f"{RESULTS_PREFIX}/{STATE_NAME}", state.path):
        state.load(); log("state restored from S3")
        for j, js in state.data.get("jobs", {}).items():
            if js.get("status") == "done":
                try:
                    store.download_prefix(f"{RESULTS_PREFIX}/{j}", os.path.join(cfg.results_dir, j))
                except Exception as e:
                    log(f"[{j}] outputs could not be restored from S3 ({e}); the job will run again")
        lp = os.path.join(cfg.results_dir, LOG_NAME + ".s3")
        if store.download_file(f"{RESULTS_PREFIX}/{LOG_NAME}", lp):
            with open(lp) as src, open(os.path.join(cfg.results_dir, LOG_NAME), "a") as dst:
                dst.write(src.read())
            os.remove(lp)


def build_parser():
    ap = argparse.ArgumentParser(prog="run1_driver.py", description="Box Run 1 driver (boxruns/run1/JOBS.md): jobs A-E, resumable, "
                                 "S3 log, instance stop at the end and on a crash. `run1_driver.py worker JOB ...` is the per-job worker.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=EXIT_HELP)
    ap.add_argument("--work", required=True, help="working directory on the box (inputs, cache, results/BOXRUN1, state)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit: no S3, no jobs, no shutdown")
    ap.add_argument("--no-shutdown", action="store_true", help="do not stop the instance at the end or on a crash (tests, local runs)")
    ap.add_argument("--shutdown-cmd", default=DEFAULT_SHUTDOWN, help=f"command that stops the instance (default: {DEFAULT_SHUTDOWN!r})")
    ap.add_argument("--only", help="comma-separated jobs to run (default A,B,C,D,E, in that order)")
    ap.add_argument("--force", action="append", choices=JOB_ORDER, help="run this job again even if it is done (repeatable)")
    ap.add_argument("--stop-on-failure", action="store_true", help="stop after the first failed job (default: record it and go on)")
    ap.add_argument("--job-timeout", type=float, default=None, help="seconds before a job command is killed (default: none)")
    g = ap.add_argument_group("S3")
    g.add_argument("--bucket", default=BUCKET, help=f"bucket (JOBS.md: {BUCKET})")
    g.add_argument("--aws-region", default=None, help="region of the boto3 session (default: boto3's own configuration)")
    g.add_argument("--aws-profile", default=None, help="named profile for the boto3 session (default: none, the default credential chain)")
    g.add_argument("--s3-local-root", default=None, help="use this local directory as the bucket instead of S3 (tests, local test)")
    g.add_argument("--skip-input-sync", action="store_true", help="do not download inputs (they are already under <work>/data/<prefix>)")
    g = ap.add_argument_group("chain and inputs")
    g.add_argument("--python", default=sys.executable, help="python of the chain environment (the box used /home/ubuntu/env/bin/python)")
    g.add_argument("--chain-dir", default=CHAIN_DEFAULT, help="Biological_Physics/MethylPhys/chain")
    g.add_argument("--manifest", default=MANIFEST_DEFAULT, help="array manifest (DEV-BASE-CHAIN-01)")
    g.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="arrays read in parallel inside a job")
    g.add_argument("--limit-arrays", type=int, default=None, help="read at most N arrays per set (JOBS.md local test: 3)")
    g.add_argument("--atlas-v2", default=os.environ.get("CPG_ATLAS_V2_PARQUET"),
                   help=f"atlas v2 parquet on the local disk (jobs D and E; default: s3://{BUCKET}/{ATLAS_V2_KEY_DEFAULT} synced to "
                        f"<work>/data/{ATLAS_V2_KEY_DEFAULT}; CPG_ATLAS_V2_PARQUET also overrides it)")
    g.add_argument("--apod-deg", type=float, default=APOD_DEG_DEFAULT,
                   help="job D: width of the apodised mask in degrees (PR #18 default 2.0; taper C2). PR #18 warns that the real sky's "
                        "scattered holes may need a small width (e.g. 0.5 deg)")
    g.add_argument("--gse128733-prefix", default=None,
                   help=f"key prefix in the bucket holding the 2 GSE128733 arrays (job A; default {GSE128733_PREFIX_DEFAULT})")
    g.add_argument("--gse128733-dir", action="append",
                   help="local folder holding the 2 GSE128733 IDAT pairs (job A; replaces the default prefix unless --gse128733-prefix is also given)")
    g.add_argument("--job-e-prefix", action="append",
                   help="key prefix in the bucket holding GSE112618 / GSE182379 (job E; repeatable). Without one (or with no files there) E is skipped")
    g.add_argument("--job-e-dir", action="append", help="local folder holding GSE112618 / GSE182379 IDATs (job E; repeatable)")
    g.add_argument("--job-cmd", action="append", metavar="JOB=COMMAND",
                   help="replace a job's command (tests, local runs); placeholders {python} {driver} {work} {out} {data} {job}")
    return ap


def main(argv=None, store=None, shutdown=None):
    """Returns the exit code. store / shutdown are injectable (tests): store has uri, has_files, download_prefix, download_file,
    sync_file, upload_file, upload_dir; shutdown(log) stops the instance."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["worker"]:
        return worker_main(argv[1:])
    a = build_parser().parse_args(argv)
    cfg = make_config(a)
    if shutdown is None:
        shutdown = make_shutdown(a.shutdown_cmd)
    state = State(os.path.join(cfg.results_dir, STATE_NAME))
    if a.dry_run:
        state.load()
        if store is not None:
            desc = getattr(store, "bucket", str(store))
        elif a.s3_local_root:
            desc = f"local:{os.path.abspath(a.s3_local_root)}"
        else:
            desc = (f"s3://{a.bucket} (boto3, default credential chain" + (f", profile {a.aws_profile}" if a.aws_profile else "")
                    + (f", region {a.aws_region}" if a.aws_region else "") + ")")
        print_plan(cfg, desc, state, "off (--no-shutdown)" if a.no_shutdown else a.shutdown_cmd)
        return EXIT_OK
    if store is None:
        if a.s3_local_root:
            store = LocalStore(a.s3_local_root)
        else:
            try:
                store = Boto3Store(a.bucket, a.aws_region, a.aws_profile)      # fails here, before any job, if boto3 is missing
            except SystemExit as e:                                             # no S3: log locally, still stop the instance
                print(f"run1_driver.py: {e}", file=sys.stderr)
                os.makedirs(cfg.results_dir, exist_ok=True)
                log = Log(os.path.join(cfg.results_dir, LOG_NAME))
                log(f"DRIVER CRASH before any job: {e}")
                log("shutdown skipped (--no-shutdown)" if a.no_shutdown else f"stopping the instance now (exit code {EXIT_CRASH})")
                if not a.no_shutdown:
                    try:
                        shutdown(log)
                    except Exception as e2:
                        log("SHUTDOWN failed:", type(e2).__name__, e2)
                return EXIT_CRASH

    os.makedirs(cfg.results_dir, exist_ok=True)
    log = Log(os.path.join(cfg.results_dir, LOG_NAME))
    lock = open(os.path.join(cfg.work, "driver.lock"), "w")
    try:
        import fcntl
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"another driver holds {cfg.work}/driver.lock; not starting", file=sys.stderr); return EXIT_CRASH
    except ImportError:
        pass

    def _on_signal(signum, _frame):
        raise Stopped(f"signal {signum}")
    old = {}
    for s in (signal.SIGTERM, signal.SIGHUP):
        try:
            old[s] = signal.signal(s, _on_signal)
        except (ValueError, OSError):       # not in the main thread
            pass

    rc = EXIT_CRASH
    try:
        if not state.load():
            restore_from_s3(store, cfg, state, log)
        state.data["bucket"] = getattr(store, "bucket", None); state.data.setdefault("runs", []).append({"started": utcnow(), "argv": argv})
        state.save()
        log(f"BOXRUN1 driver start: jobs {cfg.only}; work {cfg.work}; store {getattr(store, 'bucket', store)}; "
            f"shutdown {'off' if a.no_shutdown else a.shutdown_cmd}")
        rc = EXIT_OK
        for j in JOB_ORDER:
            if j not in cfg.only:
                continue
            c = run_job(j, cfg, state, store, log)
            rc = worst(rc, c)
            if c == EXIT_CRASH and cfg.stop_on_failure:
                log(f"[{j}] failed and --stop-on-failure is set: the remaining jobs are not run"); break
        summary = {j: state.data["jobs"].get(j, {}).get("status") for j in cfg.only}
        log(f"BOXRUN1 driver end: {summary}; exit code {rc}")
    except BaseException as e:
        rc = EXIT_CRASH
        try:
            log(f"DRIVER CRASH: {type(e).__name__}: {e}\n{traceback.format_exc()[-3000:]}")
            state.data.setdefault("runs", [{}])[-1]["crash"] = f"{type(e).__name__}: {e}"; state.save()
        except Exception:
            pass
    finally:
        try:
            state.data.setdefault("runs", [{}])[-1].update(finished=utcnow(), exit_code=rc); state.save()
        except Exception:
            pass
        for s, h in old.items():
            try:
                signal.signal(s, h)
            except (ValueError, OSError):
                pass
        log("shutdown skipped (--no-shutdown)" if a.no_shutdown else f"stopping the instance now (exit code {rc})")
        sync_up(store, cfg, log, None)              # last copy of the log and the state, before the instance stops
        if not a.no_shutdown:
            try:
                shutdown(log)
            except Exception as e:
                log("SHUTDOWN failed:", type(e).__name__, e)
        lock.close()
    return rc


# ================================================================================================ worker (runs on the box, chain env)
# One process per job: reads arrays through chain/MethylPhys_Interface/run_sample.py (the chain's entry point), from the
# DEV-BASE-CHAIN-01 betas where they exist (--betas) and from the IDAT pair otherwise (calibration, --save-betas), then tabulates.

IDAT_RX = re.compile(r"(GSM\d+)_.*?_?Grn\.idat(\.gz)?$")
FLAGS_ABC = ["--dev-selftare-ii", "--dev-direction"]      # DEV-FLAGS-01: development flags leave the reading unchanged
GROUPS8 = ["NEU", "EOS", "BASO", "MONO", "B", "NK", "CD4T", "CD8T"]


class WCtx:
    pass


def _extract_tars(d, log):
    """Extract tar archives under d once (marker per archive); the layout of the G_chain_tests prefixes is not recorded in the repository."""
    for root, _dirs, files in os.walk(d):
        if "/_x/" in root + "/":
            continue
        for fn in files:
            if re.search(r"\.(tar|tar\.gz|tgz)$", fn) and not fn.startswith("betas_"):
                p = os.path.join(root, fn); dest = os.path.join(d, "_x", fn); mark = dest + ".ok"
                if os.path.exists(mark):
                    continue
                os.makedirs(dest, exist_ok=True)
                try:
                    with tarfile.open(p) as t:
                        try:
                            t.extractall(dest, filter="data")
                        except TypeError:          # python without extraction filters
                            t.extractall(dest)
                    open(mark, "w").write("ok")
                except (tarfile.TarError, OSError) as e:
                    log(f"cannot extract {p}: {e}")


def idat_index(dirs, log):
    """gsm -> (grn, red) for every IDAT pair under dirs; also the set of GSE ids that appear in file or folder names."""
    pairs, series = {}, set()
    for d in dirs:
        if not os.path.isdir(d):
            continue
        _extract_tars(d, log)
        for root, _dirs, files in os.walk(d):
            series.update(re.findall(r"GSE\d+", root))
            for fn in files:
                series.update(re.findall(r"GSE\d+", fn))
                m = IDAT_RX.search(fn)
                if m:
                    grn = os.path.join(root, fn); red = grn.replace("_Grn.idat", "_Red.idat")
                    if os.path.exists(red):
                        pairs.setdefault(m.group(1), (grn, red))
    return pairs, series


def worker_parser():
    ap = argparse.ArgumentParser(prog="run1_driver.py worker", description="one Box Run 1 job (called by the driver)")
    ap.add_argument("job", choices=JOB_ORDER)
    ap.add_argument("--work", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--chain-dir", default=CHAIN_DEFAULT); ap.add_argument("--manifest", default=MANIFEST_DEFAULT)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4); ap.add_argument("--limit-arrays", type=int, default=None)
    ap.add_argument("--atlas-v2", default=None); ap.add_argument("--extra-dir", action="append", default=[])
    ap.add_argument("--apod-deg", type=float, default=APOD_DEG_DEFAULT)
    return ap


def worker_main(argv):
    a = worker_parser().parse_args(argv)
    c = WCtx()
    c.work = os.path.abspath(a.work); c.out = os.path.abspath(a.out); c.chain = os.path.abspath(a.chain_dir)
    c.data = os.path.join(c.work, "data"); c.cache = os.path.join(c.work, "cache"); c.workers = a.workers; c.limit = a.limit_arrays
    c.atlas = a.atlas_v2; c.apod_deg = a.apod_deg; c.extra = [os.path.abspath(d) for d in a.extra_dir]
    c.run_sample = os.path.join(c.chain, "MethylPhys_Interface", "run_sample.py")
    c.log = lambda *p: print(utcnow(), f"[worker {a.job}]", *p, flush=True)
    c.manifest = read_csv(a.manifest)
    c.by_gsm = {r["gsm"]: r for r in c.manifest}
    os.makedirs(c.out, exist_ok=True)
    fn = {"A": worker_A, "B": worker_B, "C": worker_C, "D": worker_D, "E": worker_E}[a.job]
    fn(c)
    return 0


def set_dir(c, key):
    return os.path.join(c.data, INPUTS[key][0])


def _limit(c, rows):
    return rows[:c.limit] if c.limit else rows


def betas_parquet(c, row):
    """The DEV-BASE-CHAIN-01 beta vector of this array (from results/DEV_BASE_CHAIN_01/betas/betas_<series>.tar), or None."""
    p = os.path.join(c.cache, "betas", f"{row['gsm']}.parquet")
    if os.path.exists(p):
        return p
    tp = os.path.join(c.data, BETAS_PREFIX, f"betas_{row['series']}.tar")
    if not os.path.exists(tp):
        return None
    try:
        with tarfile.open(tp) as t:
            m = t.getmember(f"{row['gsm']}.parquet")
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with t.extractfile(m) as src, open(p + ".part", "wb") as dst:
                shutil.copyfileobj(src, dst)
            os.replace(p + ".part", p)
            return p
    except (KeyError, tarfile.TarError, OSError):
        return None


def betas_csv(c, row):
    """run_sample.py --betas reads a two-column CSV (cpg_id, beta); the betas on S3 are parquet."""
    p = os.path.join(c.cache, "betas_csv", f"{row['gsm']}.csv")
    if os.path.exists(p):
        return p
    pq = betas_parquet(c, row)
    if pq is None:
        return None
    import pandas as pd
    b = pd.read_parquet(pq).iloc[:, 0]; b.index = b.index.astype(str); b.index.name = "cpg_id"
    os.makedirs(os.path.dirname(p), exist_ok=True); b.rename("beta").to_frame().to_csv(p + ".part"); os.replace(p + ".part", p)
    return p


def summarise_bundle(b):
    m = b.get("met_a") or {}; dev = b.get("development") or {}; st = dev.get("selftare_ii") or {}; di = dev.get("direction") or {}
    fr = (b.get("composition") or {}).get("fractions") or {}; ae = dev.get("atlas_e") or {}; dm = b.get("difference_map") or {}
    o = {"A": m.get("A"), "state": m.get("state", m.get("reason")), "N": m.get("noise_index"), "noise_gate": m.get("noise_gate"),
         "f_neu": m.get("fraction"), "n_sites": m.get("n_sites"), "C": (b.get("met_a_cscore") or {}).get("C"),
         "A_st": st.get("A_selftared"), "selftare_status": st.get("status"), "D": di.get("D"), "direction_status": di.get("status"),
         "refusal": b.get("refusal"), "atlas_e_status": ae.get("status"), "difference_map_status": dm.get("status")}
    for g in GROUPS8:
        o[f"comp_{g}"] = fr.get(g)
        o[f"atlas_e_{g}"] = (ae.get("fractions") or {}).get(g)
    return o


def read_array(c, row, flags, tag, idats, extra_args=()):
    """One array through run_sample.py; cached per tag under <work>/cache/readings/<tag>/<gsm>.json (kept across job restarts)."""
    gsm = row["gsm"]; cdir = os.path.join(c.cache, "readings", tag); cp = os.path.join(cdir, f"{gsm}.json")
    if os.path.exists(cp):
        with open(cp) as f:
            return json.load(f)
    os.makedirs(cdir, exist_ok=True)
    base = {k: row.get(k, "") for k in ("series", "gsm", "slide", "specimen", "healthy", "person", "plat", "title")}
    out_html = os.path.join(cdir, f"{gsm}.html")
    cmd = [sys.executable, c.run_sample, "--engine", "v3", "--specimen", row.get("specimen") or "whole blood", "--id", gsm,
           "--out", out_html, "--ledger", os.path.join(cdir, f"{gsm}_ledger.jsonl")] + list(flags) + list(extra_args)
    if c.atlas and any(f in flags for f in ("--dev-atlas-e", "--dev-sky", "--dev-nilc", "--dev-percell-b")):
        cmd += ["--atlas-v2", c.atlas]
    bc = betas_csv(c, row)
    if bc:
        cmd += ["--betas", bc]; base["input"] = "DEV_BASE_CHAIN_01 betas"
    elif gsm in idats:
        grn, red = idats[gsm]
        cmd += ["--grn", grn, "--red", red, "--save-betas", os.path.join(c.cache, "betas", f"{gsm}.parquet")]
        if row.get("plat"):
            cmd += ["--array-type", row["plat"]]
        base["input"] = "IDAT (calibrated here)"
    else:
        rec = dict(base, status="no_input", reason="no calibrated betas and no IDAT pair found in the job's inputs")
        return rec          # not cached: a later run with the input present reads it
    t0 = time.time()
    env = dict(os.environ, PYTHONPATH=c.chain, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(c.run_sample), env=env, timeout=3600)
        code, txt = r.returncode, r.stdout + "\n" + r.stderr
    except subprocess.TimeoutExpired as e:
        code, txt = -9, f"TIMEOUT {e}"
    with open(os.path.join(cdir, f"{gsm}.log"), "w") as f:
        f.write(" ".join(shlex.quote(x) for x in cmd) + "\n" + txt)
    bp = os.path.splitext(out_html)[0] + "_bundle.json"
    rec = dict(base, exit=code, seconds=round(time.time() - t0, 1))
    if os.path.exists(bp):
        with open(bp) as f:
            rec.update(summarise_bundle(json.load(f)))
        rec["status"] = "ok" if code == 0 else f"exit {code}"; rec["bundle"] = bp
    else:
        rec["status"] = ("stage0_stop" if code == 2 else "environment" if code == 3 else "crash")
        rec["reason"] = txt[-600:]
    if rec["status"] in ("ok", "stage0_stop"):
        atomic_write_json(cp, rec)
    return rec


def read_many(c, rows, flags, tag, idats, extra=None):
    from concurrent.futures import ThreadPoolExecutor
    extra = extra or {}
    with ThreadPoolExecutor(max(1, c.workers)) as ex:
        res = list(ex.map(lambda r: read_array(c, r, flags, tag, idats, extra.get(r["gsm"], ())), rows))
    n = {}
    for r in res:
        n[r["status"]] = n.get(r["status"], 0) + 1
    c.log(f"{tag}: {len(res)} arrays {n}")
    return res


def merge_rows(rows, results):
    """Manifest row + its reading; the reading fills fields the manifest row leaves empty or does not have."""
    return [dict(g, **{k: v for k, v in x.items() if k not in g or g[k] in ("", None)}) for g, x in zip(rows, results)]


def chain_tare(rows, col, out_col, healthy_only=True):
    """Median tare as chain Stage T (DEV-BASE-CHAIN-01 reference rule): references are the other healthy-labelled arrays of the same
    series and specimen with a value; same slide when >= 3, else the series (batch); >= 3 needed. Nothing is fitted."""
    for r in rows:
        v = fnum(r.get(col)); r[out_col] = None; r.setdefault("ref_scope", None); r.setdefault("n_refs", None)
        if v is None:
            continue
        pool = [x for x in rows if x is not r and x.get("series") == r.get("series") and x.get("specimen") == r.get("specimen")
                and (not healthy_only or str(x.get("healthy")) == "True") and fnum(x.get(col)) is not None]
        same = [x for x in pool if x.get("slide") and x.get("slide") == r.get("slide")]
        ref, scope = (same, "same slide") if len(same) >= 3 else (pool, "same series")
        if len(ref) >= 3:
            r[out_col] = v / median([fnum(x[col]) for x in ref]); r["ref_scope"] = scope; r["n_refs"] = len(ref)
        else:
            r["ref_scope"] = f"not tared: {len(ref)} references (>= 3 needed)"


def chain_tare_diff(rows, col, out_col):
    for r in rows:
        v = fnum(r.get(col)); r[out_col] = None
        if v is None:
            continue
        pool = [fnum(x[col]) for x in rows if x is not r and x.get("series") == r.get("series") and x.get("specimen") == r.get("specimen")
                and str(x.get("healthy")) == "True" and fnum(x.get(col)) is not None]
        if len(pool) >= 3:
            r[out_col] = v - median(pool)


ROW_COLS = ["set", "role", "series", "gsm", "slide", "specimen", "healthy", "person", "input", "status", "A", "A_st", "A_st_tared",
            "ref_scope", "n_refs", "state", "N", "noise_gate", "f_neu", "n_sites", "C", "D", "D_rel", "refusal"] + [f"comp_{g}" for g in GROUPS8]


def floor_gsms(c):
    f = json.load(open(os.path.join(c.chain, "Runtime Matrices", "Met_A_Floors", "metA_floors_v1_3.json")))
    e = f["platforms"]["EPIC"]["neutrophils"]
    refs = [x.split("_")[0] for x in e["refs"]]
    sentrix = {"_".join(x.split("_")[1:]) for x in e["refs"]}
    dups = {g for v in (e.get("duplicates_removed") or {}).values() for g in v}
    return refs, sentrix, dups


def worker_A(c):
    refs, _sx, _d = floor_gsms(c)
    idats, _ = idat_index([set_dir(c, "GSE250556"), set_dir(c, "neutrophil_ref")] + c.extra, c.log)
    rep = _limit(c, [dict(r, role="replicate") for r in c.manifest if r["series"] == "GSE250556"])
    oth = [dict(r, role="other_lab") for r in c.manifest if r["series"] in OTHER_LAB_SERIES and r["specimen"] == "isolated neutrophils"
           and r["healthy"] == "True"]
    g128 = [dict(r, role="other_lab") for r in GSE128733_ARRAYS if r["gsm"] in idats or betas_parquet(c, r)]
    if len(g128) < len(GSE128733_ARRAYS):
        c.log(f"WARNING: GSE128733 arrays without an input (no IDAT pair under {c.extra or 'no --extra-dir'}): "
              f"{[r['gsm'] for r in GSE128733_ARRAYS if r['gsm'] not in {x['gsm'] for x in g128}]}; job A continues without them")
    oth = _limit(c, oth) + g128
    flo = _limit(c, [dict(c.by_gsm[g], role="floor") for g in refs if g in c.by_gsm])
    c.log(f"set: replicates {len(rep)}, other laboratories {len(oth)} (incl. {len(g128)} GSE128733), floor {len(flo)}")
    rows = []
    for grp in (rep, oth, flo):
        res = read_many(c, grp, FLAGS_ABC, "abc", idats)
        rr = merge_rows(grp, res)
        chain_tare(rr, "A_st", "A_st_tared")       # self-tare II, then the median tare
        rows += rr
    write_csv(os.path.join(c.out, "A_arrays.csv"), rows, ROW_COLS)


def _set_rows(c, key, idats_all):
    """Manifest rows of a chain test set: arrays with an IDAT pair under the set's prefix, or whose series id appears there."""
    pairs, series = idat_index([set_dir(c, key)], c.log)
    idats_all.update(pairs)
    return [dict(r, set=key) for r in c.manifest if r["gsm"] in pairs or r["series"] in series]


def worker_B(c):
    idats = {}; allrows = []
    for key in B_SETS:
        rows = _limit(c, _set_rows(c, key, idats))
        if not rows:
            c.log(f"set {key}: no arrays found under {INPUTS[key][0]}"); continue
        res = read_many(c, rows, FLAGS_ABC, "abc", idats)
        rr = merge_rows(rows, res)
        chain_tare(rr, "A_st", "A_st_tared"); chain_tare_diff(rr, "D", "D_rel")
        write_csv(os.path.join(c.out, f"B_{key}.csv"), rr, ROW_COLS); allrows += rr
        if key == "longitudinal":
            difference_maps(c, rr, idats)
    write_csv(os.path.join(c.out, "B_all.csv"), allrows, ROW_COLS)


def difference_maps(c, rows, idats):
    """Stage 12b on the longitudinal set: each later array of a person against that person's first array (manifest order)."""
    by = {}
    for r in rows:
        if r.get("person") and r.get("status") == "ok":
            by.setdefault((r["series"], r["person"]), []).append(r)
    out = []
    for (ser, per), rs in by.items():
        if len(rs) < 2:
            continue
        first = rs[0]; pb = first.get("bundle"); pq = betas_parquet(c, first)
        for r in rs[1:]:
            if not (pb and pq):
                out.append({"series": ser, "person": per, "gsm": r["gsm"], "prior_gsm": first["gsm"], "status": "prior draw not available"}); continue
            pid = f"{ser}_{per}"
            # both draws need the same person identifier: re-read the first draw with it, then the later one against it
            p0 = read_array(c, first, [], "diffmap_prior", idats, ["--patient-id", pid])
            x = read_array(c, r, [], "diffmap", idats, ["--patient-id", pid, "--prior-betas", pq, "--prior-bundle", p0.get("bundle", "")])
            dm = {}
            if x.get("bundle") and os.path.exists(x["bundle"]):
                dm = json.load(open(x["bundle"])).get("difference_map") or {}
            out.append({"series": ser, "person": per, "gsm": r["gsm"], "prior_gsm": first["gsm"], "status": dm.get("status", x.get("status")),
                        **{k: v for k, v in dm.items() if isinstance(v, (int, float, str)) and k not in ("status", "stage", "note")}})
    write_csv(os.path.join(c.out, "B_longitudinal_difference_map.csv"), out)
    c.log(f"difference map: {len(out)} later draws")


def worker_C(c):
    _refs, sentrix, dups = floor_gsms(c)
    idats = {}
    hr_pairs, hr_series = idat_index([set_dir(c, "healthy_repeat")], c.log); idats.update(hr_pairs)
    g_pairs, _ = idat_index([set_dir(c, "GSE250556")], c.log); idats.update(g_pairs)
    rows = []
    for r in c.manifest:
        if r["healthy"] != "True" or r["plat"] != "EPIC_v1" or r["specimen"] not in ("whole blood", "isolated neutrophils"):
            continue
        if r["sentrix"] in sentrix or r["gsm"] in dups:
            continue                                   # not held out: the floor / reference arrays and their re-deposits
        setname = ("GSE250556" if r["series"] == "GSE250556" else "healthy_repeat" if (r["gsm"] in hr_pairs or r["series"] in hr_series)
                   else "DEV_BASE_CHAIN_01 betas")
        rows.append(dict(r, set=setname))
    by_set = {}
    for r in rows:
        by_set.setdefault(r["set"], []).append(r)
    rows = [x for s in sorted(by_set) for x in _limit(c, by_set[s])]
    res = read_many(c, rows, FLAGS_ABC, "abc", idats)
    rr = merge_rows(rows, res)
    write_csv(os.path.join(c.out, "C_arrays.csv"), rr, ROW_COLS)

    def spread(key):
        g = {}
        for r in rr:
            v = fnum(r.get("C"))
            if v is not None:
                g.setdefault(r[key], []).append(v)
        return [{key: k, "n": len(v), "median": median(v), "p2_5": percentile(v, 2.5), "p97_5": percentile(v, 97.5)} for k, v in sorted(g.items())]
    write_csv(os.path.join(c.out, "C_spread_by_lab.csv"), spread("series"), ["series", "n", "median", "p2_5", "p97_5"])
    write_csv(os.path.join(c.out, "C_spread_by_set.csv"), spread("set"), ["set", "n", "median", "p2_5", "p97_5"])


def sky_two_masks(z, DV, SS, apod_deg, n_null=None, n_lee=None, seed=20261004):
    """DEV-SKY-02 sky of one array (dev_stages.sky_from_z), read with the hard and the apodised mask side by side on the SAME
    block shuffles: per mask the six band powers over the mean of the first n_null shuffles, and the look-elsewhere statistic
    T = max band ratio with p over n_lee shuffles. The hard-mask numbers follow dev_stages.sky_from_z (same seed, same draw order)."""
    import numpy as np
    n_null = DV.SKY_NULL_N if n_null is None else n_null
    n_lee = DV.SKY_LEE_N if n_lee is None else n_lee
    m = DV._mapping(); zz = z.reindex(m.index).dropna(); mm = m.loc[zz.index]; npix = 12 * SS.NSIDE * SS.NSIDE
    pix = mm.pix.values.astype(np.int64); vals = zz.values; rng = np.random.default_rng(seed)
    sky = DV._pixmap(vals, pix, npix)
    orders = [DV.block_shuffle_order(mm.chr.values, rng=rng) for _ in range(max(n_null, n_lee))]
    out = []
    for mask in ("hard", "apodised"):
        kw = {"mask": mask} if mask == "hard" else {"mask": mask, "apod_deg": apod_deg}
        cl, f_sky, good = SS.masked_spectrum(sky, **kw); bp = SS.bandpowers(cl)
        nb = np.array([SS.bandpowers(SS.masked_spectrum(DV._pixmap(vals[o], pix, npix), **kw)[0]) for o in orders])
        mu = nb[:n_null].mean(0); r = bp / mu; T = float(np.max(r)); Tn = np.max(nb[:n_lee] / mu, axis=1)
        p = float((np.sum(Tn >= T) + 1) / (len(Tn) + 1))
        rec = {"mask": mask, "apod_deg": apod_deg if mask == "apodised" else None, "f_sky": round(f_sky, 4), "n_pix": int(good.sum()),
               "n_sites": int(len(zz)), "lee_T": round(T, 4), "lee_p": round(p, 4), "structure_beyond_null": bool(p < 0.05),
               "null": f"within-chromosome block shuffle, {n_null} shuffles for the band null, {n_lee} for look-elsewhere (same shuffles, both masks)"}
        if mask == "apodised" and hasattr(SS, "apodised_mask"):
            rec["w2"] = round(float(np.mean(SS.apodised_mask(good, apod_deg) ** 2)), 4)
        rec.update({f"ratio_{i + 1}": round(float(x), 4) for i, x in enumerate(r)})
        rec.update({f"bandpower_{i + 1}": float(x) for i, x in enumerate(bp)})
        out.append(rec)
    return out


def load_beta(c, row, idats):
    """Stage 1 beta vector (pandas Series): the DEV-BASE-CHAIN-01 betas, else calibrated here from the IDAT pair (run_sample.py --save-betas)."""
    import pandas as pd
    p = betas_parquet(c, row)
    if p is None and row["gsm"] in idats:
        read_array(c, row, [], "d_calibrate", idats); p = betas_parquet(c, row)
    if p is None:
        return None
    b = pd.read_parquet(p).iloc[:, 0].astype("float64"); b.index = b.index.astype(str)
    return b


def sky_module_has_mask(SS):
    import inspect
    try:
        return all("mask" in inspect.signature(getattr(SS, f)).parameters for f in SKY_FUNCS)
    except (AttributeError, TypeError, ValueError):
        return False


def worker_D(c):
    """Job D: healthy whole-blood arrays of GSE250556 and healthy_repeat, residual sky against the block-shuffle null (DEV-SKY-02),
    hard and apodised mask side by side. One row per array and mask in D_sky.csv; the driver checks the DEV-SKY-02 bars per mask."""
    sys.path.insert(0, c.chain)
    import sky_statistics as SS
    if not sky_module_has_mask(SS):
        raise SystemExit(f"job D BLOCKED: {D_BLOCKED_PR18} ({SS.__file__} has no `mask` parameter); nothing was run")
    import importlib.util
    if importlib.util.find_spec("healpy") is None:
        raise SystemExit("job D: healpy is not installed in the chain environment (the sky needs it); nothing was run")
    if not (c.atlas and os.path.exists(c.atlas)):
        raise SystemExit("job D: --atlas-v2 parquet not given or not found; nothing was run")
    import dev_stages as DV
    import conductor_v3 as C3
    g_pairs, _ = idat_index([set_dir(c, "GSE250556")], c.log)
    hr_pairs, hr_series = idat_index([set_dir(c, "healthy_repeat")], c.log)
    idats = dict(g_pairs, **hr_pairs)
    rows = []
    for key, sel in (("GSE250556", lambda r: r["series"] == "GSE250556"),
                     ("healthy_repeat", lambda r: r["series"] != "GSE250556" and (r["gsm"] in hr_pairs or r["series"] in hr_series))):
        rows += _limit(c, [dict(r, set=key) for r in c.manifest if sel(r) and r["healthy"] == "True" and r["specimen"] == "whole blood"
                           and r["plat"] == "EPIC_v1"])
    c.log(f"job D: {len(rows)} healthy whole-blood arrays; apodised mask width {c.apod_deg} deg")
    AP = DV.load_atlas_parents(c.atlas); out = []
    for r in rows:
        base = {k: r.get(k, "") for k in ("set", "series", "gsm", "slide", "person")}
        try:
            b = load_beta(c, r, idats)
            if b is None:
                out.append(dict(base, status="no_input")); continue
            fr = (C3.stage_a_composition(b) or {}).get("fractions")
            if not fr:
                out.append(dict(base, status="no composition")); continue
            for rec in sky_two_masks(DV.residual_z(b, fr, AP), DV, SS, c.apod_deg):
                out.append(dict(base, status="ok", **rec))
        except Exception as e:
            out.append(dict(base, status=f"error: {type(e).__name__}: {str(e)[:200]}"))
    ok = [x for x in out if x.get("status") == "ok"]
    c.log(f"job D: {len(ok) // 2} arrays read with both masks; {len(out) - len(ok)} arrays without a sky")
    cols = (["set", "series", "gsm", "slide", "person", "status", "mask", "apod_deg", "w2", "f_sky", "n_pix", "n_sites"]
            + [f"ratio_{i}" for i in range(1, 7)] + ["lee_T", "lee_p", "structure_beyond_null"] + [f"bandpower_{i}" for i in range(1, 7)] + ["null"])
    write_csv(os.path.join(c.out, "D_sky.csv"), ok + [x for x in out if x.get("status") != "ok"], cols)


def worker_E(c):
    idats, _ = idat_index(c.extra, c.log)
    rows = []
    for gsm, (grn, _red) in sorted(idats.items()):
        ser = next((s for s in E_SERIES_SPECIMEN if s in grn), None)
        if ser is None:
            continue
        rows.append({"series": ser, "gsm": gsm, "slide": "", "specimen": E_SERIES_SPECIMEN[ser], "healthy": "", "person": "", "plat": "EPIC_v1", "title": ""})
    by = {}
    for r in rows:
        by.setdefault(r["series"], []).append(r)
    rows = [x for s in sorted(by) for x in _limit(c, by[s])]
    c.log(f"job E arrays: { {s: len(v) for s, v in by.items()} }")
    res = read_many(c, rows, ["--dev-atlas-e"], "e", idats)
    rr = merge_rows(rows, res)
    write_csv(os.path.join(c.out, "E_arrays.csv"), rr, ["series", "gsm", "specimen", "input", "status", "atlas_e_status"] +
              [f"atlas_e_{g}" for g in GROUPS8] + [f"comp_{g}" for g in GROUPS8] + ["A", "state", "refusal"])


if __name__ == "__main__":
    sys.exit(main())
