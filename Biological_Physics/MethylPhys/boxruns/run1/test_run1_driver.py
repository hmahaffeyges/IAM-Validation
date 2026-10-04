"""Tests for run1_driver.py: fake inputs, fake job commands, a local directory as the bucket, a fake shutdown hook.
Nothing here touches S3 or AWS.  Run:  python3 -m pytest Biological_Physics/MethylPhys/boxruns/run1/test_run1_driver.py -q"""
import csv
import json
import os
import subprocess
import sys
import textwrap

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run1_driver as D  # noqa: E402

PY = sys.executable

FAKE_JOB = textwrap.dedent(r'''
    import csv, json, os, signal, sys
    job, out, plan_path, order_path = sys.argv[1:5]
    plan = json.load(open(plan_path)); mode = plan.get(job, "ok")
    with open(order_path, "a") as f: f.write(job + "\n")
    open(os.path.join(out, "partial.txt"), "w").write("started")
    if mode == "fail":
        print("fake job failing on purpose", flush=True); sys.exit(1)
    if mode in ("kill_parent", "term_parent"):
        plan[job] = "ok"; json.dump(plan, open(plan_path, "w"))       # the next run of this job completes
        os.kill(os.getppid(), signal.SIGKILL if mode == "kill_parent" else signal.SIGTERM)
        import time; time.sleep(30); sys.exit(0)
    def w(name, rows):
        cols = sorted({k for r in rows for k in r}) or ["x"]
        with open(os.path.join(out, name), "w", newline="") as f:
            d = csv.DictWriter(f, fieldnames=cols); d.writeheader(); [d.writerow(r) for r in rows]
    if job == "A":
        spread = 0.08 if mode == "barfail" else 0.005
        rows = []
        for i, (p, k) in enumerate([("P1", -1), ("P1", 0), ("P1", 1), ("P2", -1), ("P2", 0), ("P2", 1)]):
            rows.append(dict(role="replicate", gsm=f"GSMR{i}", series="GSE250556", person=p, A_st=1.0, A_st_tared=1.0 + k * spread))
        for i in range(4):
            rows.append(dict(role="other_lab", gsm=f"GSMO{i}", series="GSE247193", A_st=1.0, A_st_tared=1.0 + 0.01 * i))
        rows.append(dict(role="other_lab", gsm="GSM3684010", series="GSE128733", A_st=1.04, A_st_tared=""))
        for i in range(6):
            rows.append(dict(role="floor", gsm=f"GSMF{i}", series="GSE110554", A_st=1.0 + 0.005 * i, A_st_tared=1.0))
        w("A_arrays.csv", rows)
    elif job == "B":
        w("B_all.csv", [dict(gsm="GSM1", A=1.0)])
    elif job == "C":
        for n in ("C_arrays.csv", "C_spread_by_lab.csv", "C_spread_by_set.csv"): w(n, [dict(gsm="GSM1", C=1.0)])
    elif job == "D":
        rows = [dict(mask=m, gsm=f"G{i}", structure_beyond_null=False, **{f"ratio_{b}": 1.0 for b in range(1, 7)})
                for m in ("hard", "apodised") for i in range(5)]
        w("D_sky.csv", rows)
    elif job == "E":
        w("E_arrays.csv", [dict(gsm="GSM1", atlas_e_NEU=0.6)])
    os.remove(os.path.join(out, "partial.txt"))
''')


class Recorder:
    def __init__(self):
        self.calls = 0

    def __call__(self, log):
        self.calls += 1; log("fake shutdown"); return 0


def make_bucket(root):
    for _k, (pre, *_r) in D.INPUTS.items():
        os.makedirs(os.path.join(root, pre), exist_ok=True)
        open(os.path.join(root, pre, "input.txt"), "w").write(pre)
    return root


@pytest.fixture
def env(tmp_path):
    bucket = make_bucket(str(tmp_path / "bucket"))
    script = tmp_path / "fake_job.py"; script.write_text(FAKE_JOB)
    plan = tmp_path / "plan.json"; plan.write_text("{}")
    order = tmp_path / "order.txt"
    work = str(tmp_path / "work")
    args = ["--work", work] + [f"--job-cmd={j}={PY} {script} {{job}} {{out}} {plan} {order}" for j in D.JOB_ORDER]
    return dict(tmp=tmp_path, bucket=bucket, plan=plan, order=order, work=work, args=args)


def order(env):
    return open(env["order"]).read().split() if os.path.exists(env["order"]) else []


def set_plan(env, **modes):
    env["plan"].write_text(json.dumps(modes))


def state(env):
    return json.load(open(os.path.join(env["work"], "results", "BOXRUN1", "state.json")))


def s3(env, *parts):
    return os.path.join(env["bucket"], "results", "BOXRUN1", *parts)


# ------------------------------------------------------------------------------------------------ order, sync, shutdown
def test_all_jobs_run_in_order_and_sync(env):
    sh = Recorder()
    rc = D.main(env["args"], store=D.LocalStore(env["bucket"]), shutdown=sh)
    assert rc == D.EXIT_OK
    assert order(env) == ["A", "B", "C", "D", "E"]
    st = state(env)
    assert [st["jobs"][j]["status"] for j in D.JOB_ORDER] == ["done"] * 5
    assert st["jobs"]["A"]["bars"]["all_met"] and st["jobs"]["D"]["bars"]["all_met"]
    for j in D.JOB_ORDER:                                           # outputs, log and state on the (fake) bucket
        assert os.path.isfile(s3(env, j, D.REQUIRED_OUTPUTS[j][0]))
    assert os.path.isfile(s3(env, "log.txt")) and os.path.isfile(s3(env, "state.json"))
    assert "BOXRUN1 driver end" in open(s3(env, "log.txt")).read()
    assert os.path.isfile(os.path.join(env["work"], "data", "downloads", "G_chain_tests", "infection", "input.txt"))  # inputs synced
    assert sh.calls == 1


def test_rerun_skips_everything_done(env):
    D.main(env["args"], store=D.LocalStore(env["bucket"]), shutdown=Recorder())
    sh = Recorder()
    rc = D.main(env["args"], store=D.LocalStore(env["bucket"]), shutdown=sh)
    assert rc == D.EXIT_OK and order(env) == ["A", "B", "C", "D", "E"]       # nothing ran a second time
    assert sh.calls == 1


def test_done_job_with_damaged_output_runs_again(env):
    D.main(env["args"], store=D.LocalStore(env["bucket"]), shutdown=Recorder())
    with open(os.path.join(env["work"], "results", "BOXRUN1", "B", "B_all.csv"), "a") as f:
        f.write("extra\n")
    D.main(env["args"], store=D.LocalStore(env["bucket"]), shutdown=Recorder())
    assert order(env) == ["A", "B", "C", "D", "E", "B"]


# ------------------------------------------------------------------------------------------------ resume after a stop
def _cli(env, extra=()):
    marker = env["tmp"] / "shutdown_calls.txt"
    cmd = [PY, os.path.join(HERE, "run1_driver.py"), *env["args"], "--s3-local-root", env["bucket"],
           "--shutdown-cmd", f"{PY} -c \"open('{marker}', 'a').write('x')\"", *extra]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return r, (open(marker).read().count("x") if os.path.exists(marker) else 0)


def test_resume_after_hard_stop_mid_run(env):
    set_plan(env, C="kill_parent")                                  # the instance dies while job C runs (SIGKILL: no cleanup at all)
    r, n_shut = _cli(env)
    assert r.returncode == -9
    st = state(env)
    assert [st["jobs"][j]["status"] for j in "ABC"] == ["done", "done", "running"]
    assert os.path.exists(os.path.join(env["work"], "results", "BOXRUN1", "C", "partial.txt"))
    r, n_shut = _cli(env)                                           # rerun: A, B skipped, C restarted cleanly, then D, E
    assert r.returncode == 0, r.stdout + r.stderr
    assert order(env) == ["A", "B", "C", "C", "D", "E"]
    st = state(env)
    assert all(st["jobs"][j]["status"] == "done" for j in D.JOB_ORDER) and st["jobs"]["C"]["attempts"] == 2
    assert not os.path.exists(os.path.join(env["work"], "results", "BOXRUN1", "C", "partial.txt"))
    log = open(s3(env, "log.txt")).read()
    assert "[A] done in an earlier run" in log and "[C] was interrupted in an earlier run" in log
    assert n_shut == 1


def test_sigterm_mid_job_uploads_log_and_shuts_down(env):
    set_plan(env, B="term_parent")
    r, n_shut = _cli(env)
    assert r.returncode == D.EXIT_CRASH
    assert n_shut == 1
    st = json.load(open(s3(env, "state.json")))                     # the state on S3 records the interrupted job
    assert st["jobs"]["B"]["status"] == "interrupted" and "C" not in st["jobs"]
    assert "DRIVER CRASH: Stopped" in open(s3(env, "log.txt")).read()
    r, n_shut = _cli(env)
    assert r.returncode == 0 and order(env) == ["A", "B", "B", "C", "D", "E"] and n_shut == 2


def test_restore_state_from_s3_on_a_new_disk(env):
    D.main(env["args"], store=D.LocalStore(env["bucket"]), shutdown=Recorder())
    import shutil
    shutil.rmtree(env["work"])                                      # new disk: only S3 is left
    rc = D.main(env["args"], store=D.LocalStore(env["bucket"]), shutdown=Recorder())
    assert rc == D.EXIT_OK and order(env) == ["A", "B", "C", "D", "E"]


# ------------------------------------------------------------------------------------------------ crash
def test_crash_in_job_uploads_log_and_shuts_down(env):
    set_plan(env, B="fail")
    sh = Recorder()
    rc = D.main(env["args"], store=D.LocalStore(env["bucket"]), shutdown=sh)
    assert rc == D.EXIT_CRASH
    assert order(env) == ["A", "B", "C", "D", "E"]                   # the other jobs still ran
    assert sh.calls == 1
    log = open(s3(env, "log.txt")).read()
    assert "[B] FAILED (exit 1" in log and "fake job failing on purpose" in log
    assert json.load(open(s3(env, "state.json")))["jobs"]["B"]["status"] == "failed"
    assert os.path.isfile(s3(env, "B", "job_B.log"))
    set_plan(env)                                                   # rerun: only B runs again
    rc = D.main(env["args"], store=D.LocalStore(env["bucket"]), shutdown=Recorder())
    assert rc == D.EXIT_OK and order(env)[5:] == ["B"]


def test_stop_on_failure(env):
    set_plan(env, B="fail")
    rc = D.main(env["args"] + ["--stop-on-failure"], store=D.LocalStore(env["bucket"]), shutdown=Recorder())
    assert rc == D.EXIT_CRASH and order(env) == ["A", "B"]


def test_driver_crash_still_uploads_and_shuts_down(env):
    import shutil
    shutil.rmtree(os.path.join(env["bucket"], "downloads", "G_chain_tests", "myeloid"))   # an input prefix is missing (B)
    sh = Recorder()
    rc = D.main(env["args"] + ["--stop-on-failure"], store=D.LocalStore(env["bucket"]), shutdown=sh)
    assert rc == D.EXIT_CRASH and sh.calls == 1 and order(env) == ["A"]
    log = open(s3(env, "log.txt")).read()
    assert "DRIVER CRASH: RuntimeError: input prefix not found in the store: downloads/G_chain_tests/myeloid" in log
    assert json.load(open(s3(env, "state.json")))["jobs"]["B"]["status"] == "interrupted"


def test_no_shutdown_flag_suppresses_shutdown(env):
    set_plan(env, B="fail")
    sh = Recorder()
    rc = D.main(env["args"] + ["--no-shutdown"], store=D.LocalStore(env["bucket"]), shutdown=sh)
    assert rc == D.EXIT_CRASH and sh.calls == 0
    assert "shutdown skipped (--no-shutdown)" in open(s3(env, "log.txt")).read()


def test_no_shutdown_cli_and_sigterm(env):
    set_plan(env, A="term_parent")
    r, n_shut = _cli(env, ["--no-shutdown"])
    assert r.returncode == D.EXIT_CRASH and n_shut == 0


# ------------------------------------------------------------------------------------------------ commissioning bars
def test_bar_failure_is_reported(env):
    set_plan(env, A="barfail")
    sh = Recorder()
    rc = D.main(env["args"], store=D.LocalStore(env["bucket"]), shutdown=sh)
    assert rc == D.EXIT_BARS and sh.calls == 1
    bars = json.load(open(s3(env, "A", "bars.json")))
    assert not bars["all_met"]
    by = {b["name"]: b for b in bars["bars"]}
    assert not by["replicates within-person SD (GSE250556)"]["met"] and by["replicates within-person SD (GSE250556)"]["measured"] > 0.02
    assert by["floor arrays in Normal"]["met"] and by["other-laboratory purified neutrophils in Normal on tared A"]["met"]
    assert by["other-laboratory purified neutrophils in Normal on tared A"]["not_tared"] == ["GSM3684010"]
    for t in ("A_bar_replicates.csv", "A_bar_otherlab.csv", "A_bar_floor.csv"):   # one table per bar
        assert os.path.isfile(s3(env, "A", t))
    log = open(s3(env, "log.txt")).read()
    assert "[A] bar NOT MET: replicates within-person SD (GSE250556)" in log
    assert "[A] commissioning bars NOT all met" in log
    assert state(env)["jobs"]["A"]["status"] == "done"             # a bar not met is a reading, not a crash
    rc = D.main(env["args"], store=D.LocalStore(env["bucket"]), shutdown=Recorder())
    assert rc == D.EXIT_BARS                                        # still reported when A is skipped on a rerun


def test_bars_D_band_and_lee(tmp_path):
    rows = []
    for i in range(10):
        rows.append(dict(mask="hard", ratio_1=1.84, **{f"ratio_{b}": 1.0 for b in range(2, 7)}, structure_beyond_null=i < 9))
        rows.append(dict(mask="apodised", **{f"ratio_{b}": 1.0 for b in range(1, 7)}, structure_beyond_null=i < 0))
    D.write_csv(str(tmp_path / "D_sky.csv"), rows)
    res = D.bars_D(str(tmp_path))
    by = {b["name"]: b for b in res["bars"]}
    assert not by["hard mask: median band power / block-shuffle null, bands 1-6"]["met"]
    assert by["hard mask: median band power / block-shuffle null, bands 1-6"]["band1"] == pytest.approx(1.84)
    assert not by["hard mask: look-elsewhere rate"]["met"]
    assert by["apodised mask: median band power / block-shuffle null, bands 1-6"]["met"] and by["apodised mask: look-elsewhere rate"]["met"]
    assert not res["all_met"]


def test_floor_bar_needs_six_arrays(tmp_path):
    rows = [dict(role="floor", gsm=f"F{i}", A_st=1.0) for i in range(5)]
    D.write_csv(str(tmp_path / "A_arrays.csv"), rows)
    by = {b["name"]: b for b in D.bars_A(str(tmp_path))["bars"]}
    assert by["floor arrays in Normal"]["measured"] == "5/5" and not by["floor arrays in Normal"]["met"]


# ------------------------------------------------------------------------------------------------ default commands: D blocked, E skipped
def test_default_D_blocked_and_E_skipped(tmp_path):
    bucket = make_bucket(str(tmp_path / "bucket")); sh = Recorder()
    chain = pre_pr18_chain(tmp_path)                                  # main's sky_statistics before PR #18: no `mask` parameter
    rc = D.main(["--work", str(tmp_path / "w"), "--only", "D,E", "--chain-dir", chain], store=D.LocalStore(bucket), shutdown=sh)
    assert rc == D.EXIT_BLOCKED and sh.calls == 1
    st = json.load(open(os.path.join(bucket, "results", "BOXRUN1", "state.json")))
    assert st["jobs"]["D"]["status"] == "blocked" and "apodised mask (PR #18) not merged yet" in " ".join(st["jobs"]["D"]["reason"])
    assert st["jobs"]["E"]["status"] == "skipped" and "GSE112618 or GSE182379" in " ".join(st["jobs"]["E"]["reason"])
    log = open(os.path.join(bucket, "results", "BOXRUN1", "log.txt")).read()
    assert "[D] BLOCKED: apodised mask (PR #18) not merged yet" in log


def test_default_A_blocked_without_gse128733(tmp_path):
    bucket = make_bucket(str(tmp_path / "bucket"))
    rc = D.main(["--work", str(tmp_path / "w"), "--only", "A"], store=D.LocalStore(bucket), shutdown=Recorder())
    assert rc == D.EXIT_BLOCKED
    st = json.load(open(os.path.join(bucket, "results", "BOXRUN1", "state.json")))
    assert "GSE128733" in " ".join(st["jobs"]["A"]["reason"])


def test_dry_run_touches_nothing(env, capsys):
    class NoStore:
        bucket = "none"

        def __getattr__(self, k):
            raise AssertionError(f"store used in a dry run: {k}")
    sh = Recorder()
    rc = D.main(env["args"] + ["--dry-run"], store=NoStore(), shutdown=sh)
    out = capsys.readouterr().out
    assert rc == 0 and sh.calls == 0 and order(env) == [] and not os.path.exists(env["work"])
    assert "[A] Commissioning check" in out and "[E] Atlas composition" in out and "about 150.6 GB" in out


def test_usage_errors():
    with pytest.raises(SystemExit):
        D.main(["--work", "/tmp/x", "--only", "Z"])
    with pytest.raises(SystemExit):
        D.main(["--work", "/tmp/x", "--job-cmd", "Q=true"])


# ------------------------------------------------------------------------------------------------ the real worker A on a fake chain
FAKE_RUN_SAMPLE = textwrap.dedent(r'''
    import argparse, json, os, sys
    ap = argparse.ArgumentParser(); ap.add_argument("--id"); ap.add_argument("--out"); ap.add_argument("--specimen")
    ap.add_argument("--betas"); ap.add_argument("--grn"); ap.add_argument("--red")
    a, _ = ap.parse_known_args()
    assert a.betas or (a.grn and a.red)
    v = json.load(open(os.environ["FAKE_VALUES"]))[a.id]
    b = {"met_a": {"A": v[0], "noise_index": 0.1, "fraction": 0.6}, "met_a_cscore": {"C": v[2]},
         "composition": {"fractions": {"NEU": 0.6}}, "development": {"selftare_ii": {"status": "OK", "A_selftared": v[1]},
         "direction": {"status": "OK", "D": 0.0}}}
    json.dump(b, open(os.path.splitext(a.out)[0] + "_bundle.json", "w")); print(a.id, "ok")
''')


def test_workers_A_B_C_end_to_end_on_fake_chain(tmp_path, monkeypatch):
    chain = tmp_path / "chain"
    (chain / "MethylPhys_Interface").mkdir(parents=True)
    (chain / "MethylPhys_Interface" / "run_sample.py").write_text(FAKE_RUN_SAMPLE)
    (chain / "Runtime Matrices" / "Development").mkdir(parents=True)
    (chain / "Runtime Matrices" / "Development" / "dev_selftare_typeII_EPIC_v1.json").write_text("{}")
    (chain / "Runtime Matrices" / "Met_A_Floors").mkdir(parents=True)
    floor = [f"GSM90001{i}" for i in range(6)]
    (chain / "Runtime Matrices" / "Met_A_Floors" / "metA_floors_v1_3.json").write_text(json.dumps(
        {"platforms": {"EPIC": {"neutrophils": {"refs": [f"{g}_200000000001_R0{i + 1}C01" for i, g in enumerate(floor)]}}}}))
    bucket = make_bucket(str(tmp_path / "bucket"))
    man, values = [], {}

    def add(series, gsm, slide, spec, person, A, Ast, idat_prefix):
        man.append(dict(series=series, gsm=gsm, grn="", sentrix=f"{slide}_R01C01", slide=slide, plat="EPIC_v1", specimen=spec, sex="", age="",
                        healthy="True", person=person, no_intake="False", intake_ready="True", title=gsm))
        values[gsm] = [A, Ast, 1.1]
        d = os.path.join(idat_prefix, "idat"); os.makedirs(d, exist_ok=True)
        for ch in ("Grn", "Red"):
            open(os.path.join(d, f"{gsm}_{slide}_R01C01_{ch}.idat.gz"), "w").write("x")
    g250 = os.path.join(bucket, "downloads", "G_chain_tests", "GSE250556")
    nref = os.path.join(bucket, "downloads", "G_chain_tests", "neutrophil_ref")
    for i, (p, ast) in enumerate([("subjectA", 0.99), ("subjectA", 1.00), ("subjectA", 1.01), ("subjectB", 1.00), ("subjectB", 1.005), ("subjectB", 0.995)]):
        add("GSE250556", f"GSM90002{i}", "205832330169", "whole blood", p, 1.2, ast, g250)
    for i, ast in enumerate([0.98, 1.0, 1.02, 1.0]):
        add("GSE247193", f"GSM90003{i}", "300000000001", "isolated neutrophils", "", 1.15, ast, nref)
    for i, g in enumerate(floor):
        add("GSE110554", g, "200000000001", "isolated neutrophils", "", 1.0, 1.0 + 0.004 * i, nref)
    add("GSE247193", "GSM900040", "300000000001", "PBMC", "", 1.0, 1.0, nref)            # not an isolated neutrophil: not in the set
    gdir = tmp_path / "gse128733"
    for g in ("GSM3684010", "GSM3684011"):
        values[g] = [1.166, 1.043, 1.4]
        for ch in ("Grn", "Red"):
            (gdir / "idat").mkdir(parents=True, exist_ok=True); (gdir / "idat" / f"{g}_200357150019_R01C01_{ch}.idat").write_text("x")
    inf = os.path.join(bucket, "downloads", "G_chain_tests", "infection", "GSE999001")          # a chain test set for job B
    for i in range(4):
        add("GSE999001", f"GSM90005{i}", "400000000001", "whole blood", "", 1.1, 1.05, inf)
    manifest = tmp_path / "manifest.csv"
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(man[0])); w.writeheader(); [w.writerow(r) for r in man]
    vals = tmp_path / "values.json"; vals.write_text(json.dumps(values)); monkeypatch.setenv("FAKE_VALUES", str(vals))
    work = tmp_path / "work"; sh = Recorder()
    rc = D.main(["--work", str(work), "--only", "A", "--chain-dir", str(chain), "--manifest", str(manifest), "--python", PY,
                 "--workers", "4", "--gse128733-dir", str(gdir)], store=D.LocalStore(bucket), shutdown=sh)
    log = open(os.path.join(bucket, "results", "BOXRUN1", "log.txt")).read()
    assert rc == D.EXIT_OK, log
    rows = list(csv.DictReader(open(os.path.join(work, "results", "BOXRUN1", "A", "A_arrays.csv"))))
    by = {r["gsm"]: r for r in rows}
    assert {r["role"] for r in rows} == {"replicate", "other_lab", "floor"} and "GSM900040" not in by
    assert len(rows) == 6 + 4 + 2 + 6
    # GSM900020: self-tare II 0.99 over the median of the other five on its slide (1.0)
    assert float(by["GSM900020"]["A_st_tared"]) == pytest.approx(0.99 / 1.0) and by["GSM900020"]["ref_scope"] == "same slide"
    assert float(by["GSM900030"]["A_st_tared"]) == pytest.approx(0.98 / 1.0)
    assert by["GSM3684010"]["A_st_tared"] == "" and by["GSM3684010"]["ref_scope"].startswith("not tared: 1 references")
    bars = json.load(open(os.path.join(work, "results", "BOXRUN1", "A", "bars.json")))
    assert bars["all_met"], bars
    assert [b["met"] for b in bars["bars"]] == [True, True, True, True]
    assert sorted(next(b for b in bars["bars"] if "other-laboratory" in b["name"])["not_tared"]) == ["GSM3684010", "GSM3684011"]
    # a restart of A re-uses the per-array readings in <work>/cache (they are outside the job's output folder)
    cached = os.listdir(os.path.join(work, "cache", "readings", "abc"))
    assert "GSM900020.json" in cached
    # jobs B and C through the same reader
    rc = D.main(["--work", str(work), "--only", "B,C", "--chain-dir", str(chain), "--manifest", str(manifest), "--python", PY,
                 "--workers", "4"], store=D.LocalStore(bucket), shutdown=sh)
    assert rc == D.EXIT_OK, open(os.path.join(bucket, "results", "BOXRUN1", "log.txt")).read()
    b = list(csv.DictReader(open(os.path.join(work, "results", "BOXRUN1", "B", "B_all.csv"))))
    assert [r["gsm"] for r in b] == ["GSM900050", "GSM900051", "GSM900052", "GSM900053"] and {r["set"] for r in b} == {"infection"}
    assert all(r["status"] == "ok" and float(r["A_st_tared"]) == pytest.approx(1.0) for r in b)
    assert os.path.isfile(os.path.join(work, "results", "BOXRUN1", "B", "B_infection.csv"))
    c = {r["gsm"]: r for r in csv.DictReader(open(os.path.join(work, "results", "BOXRUN1", "C", "C_arrays.csv")))}
    assert not set(floor) & set(c)                                    # the floor arrays are not held out
    assert sum(1 for r in c.values() if r["series"] == "GSE250556" and r["status"] == "ok") == 6
    lab = {r["series"]: r for r in csv.DictReader(open(os.path.join(work, "results", "BOXRUN1", "C", "C_spread_by_lab.csv")))}
    assert lab["GSE250556"]["n"] == "6" and float(lab["GSE250556"]["median"]) == pytest.approx(1.1)


# ------------------------------------------------------------------------------------------------ helpers
def test_statistics_helpers():
    assert D.median([3, 1, 2]) == 2 and D.median([1, 2, 3, 4]) == 2.5 and D.median([]) is None
    assert D.percentile([1, 2, 3, 4, 5], 50) == 3 and D.percentile([0, 10], 2.5) == pytest.approx(0.25)
    rows = [dict(person="a", v=1.0), dict(person="a", v=3.0), dict(person="b", v=5.0), dict(person="b", v=5.0)]
    assert D.within_person_sd(rows, "v") == pytest.approx((2.0 / 2) ** 0.5)
    assert D.worst(0, 4, 3) == 3 and D.worst(4, 1, 3) == 1 and D.worst(0, 0) == 0


def test_chain_tare_rule():
    rows = [dict(gsm=f"g{i}", series="S", specimen="whole blood", healthy="True", slide="s1" if i < 4 else "s2", v=1.0 + i / 100) for i in range(6)]
    rows.append(dict(gsm="sick", series="S", specimen="whole blood", healthy="False", slide="s1", v=2.0))
    D.chain_tare(rows, "v", "t")
    r0 = rows[0]
    assert r0["ref_scope"] == "same slide" and r0["n_refs"] == 3 and r0["t"] == pytest.approx(1.0 / 1.02)   # the unhealthy array is no reference
    r4 = rows[4]
    assert r4["ref_scope"] == "same series" and r4["n_refs"] == 5
    assert rows[6]["t"] == pytest.approx(2.0 / D.median([1.0, 1.01, 1.02, 1.03]))   # same slide (4 healthy refs)


def test_aws_cli_store_commands(tmp_path):
    calls = []

    class R:
        returncode = 0; stdout = ""; stderr = ""

    def fake_run(cmd, **kw):
        calls.append(cmd); return R()
    st = D.AwsCliStore(run=fake_run, region="us-west-2")
    st.upload_file("/x/log.txt", "results/BOXRUN1/log.txt")
    st.download_prefix("downloads/G_chain_tests/GSE250556", str(tmp_path / "in"))
    assert calls[0][:5] == ["aws", "s3", "cp", "/x/log.txt", f"s3://{D.BUCKET}/results/BOXRUN1/log.txt"]
    assert calls[1][:4] == ["aws", "s3", "sync", f"s3://{D.BUCKET}/downloads/G_chain_tests/GSE250556/"]
    assert "--region" in calls[0]


# ------------------------------------------------------------------------------------------------ job D: apodised mask (PR #18)
SKY_PATH = "Biological_Physics/MethylPhys/chain/sky_statistics.py"
PRE_PR18_REV = "b26d4d1"                                    # main when this branch started: sky_statistics without `mask`
PR18_REVS = ("origin/apodised-mask", "77c6fe6")             # PR #18 branch (read into a temp dir only; never merged here)


def _git_show(revs, path):
    for rev in revs:
        r = subprocess.run(["git", "-C", HERE, "show", f"{rev}:{path}"], capture_output=True, text=True)
        if r.returncode == 0:
            return r.stdout
    pytest.skip(f"{path} not available at {revs} in this clone")


def pre_pr18_chain(tmp_path):
    d = tmp_path / "chain_pre_pr18"; d.mkdir(exist_ok=True)
    (d / "sky_statistics.py").write_text(_git_show([PRE_PR18_REV], SKY_PATH))
    return str(d)


def pr18_sky_source():
    return _git_show(PR18_REVS, SKY_PATH)


def test_D_feature_detection(tmp_path):
    ok, why = D.sky_mask_support(pre_pr18_chain(tmp_path))
    assert not ok and "take no `mask` parameter" in why
    d = tmp_path / "chain_pr18"; d.mkdir(); (d / "sky_statistics.py").write_text(pr18_sky_source())
    assert D.sky_mask_support(str(d)) == (True, "")
    assert D.sky_mask_support(str(tmp_path / "nowhere"))[0] is False


def test_worker_D_refuses_pre_pr18_sky_statistics(tmp_path):
    pytest.importorskip("numpy")
    chain = pre_pr18_chain(tmp_path)
    man = tmp_path / "m.csv"; man.write_text("series,gsm,slide,plat,specimen,healthy,person\n")
    r = subprocess.run([PY, os.path.join(HERE, "run1_driver.py"), "worker", "D", "--work", str(tmp_path / "w"), "--out", str(tmp_path / "o"),
                        "--chain-dir", chain, "--manifest", str(man)], capture_output=True, text=True, timeout=60)
    assert r.returncode != 0 and "job D BLOCKED: apodised mask (PR #18) not merged yet" in r.stderr


FAKE_DV_TAIL = textwrap.dedent(r'''

    # ---- test overrides (small sky, small null)
    SKY_NULL_N, SKY_LEE_N = 4, 6
    def _mapping():
        m = pd.read_csv(os.path.join(HERE, "fake_map.csv"), index_col=0); m.index = m.index.astype(str)
        return m.sort_values(["chr", "pix"], kind="mergesort")
    def load_atlas_parents(p):
        return None
    def residual_z(beta, fractions, AP):
        return (beta - 0.5) * 10.0
''')


def _load_module(name, source, tmp_path):
    import importlib.util
    f = tmp_path / f"{name}.py"; f.write_text(source)
    spec = importlib.util.spec_from_file_location(name, f); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_sky_two_masks_hard_and_apodised(tmp_path):
    np = pytest.importorskip("numpy"); pd = pytest.importorskip("pandas")
    import types
    SS = _load_module("sky_pr18_copy", pr18_sky_source(), tmp_path)
    SS.NSIDE = 64                                                   # a small HEALPix map (49,152 pixels)
    real_dv = _load_module("dev_stages_copy", open(os.path.join(D.CHAIN_DEFAULT, "dev_stages.py")).read(), tmp_path)
    n = 3000; rng = np.random.default_rng(1)
    ids = [f"cg{i:08d}" for i in range(n)]
    m = pd.DataFrame({"pix": np.arange(n) % 2500, "chr": np.repeat([1, 2, 3], n // 3)}, index=ids)
    calls = []
    try:
        import healpy; assert healpy.__name__
    except ImportError:                                             # no healpy: stub the spectrum, keep the bookkeeping real
        def fake_ms(pix, lmax=255, mask="hard", apod_deg=2.0, taper="C2"):
            calls.append((mask, apod_deg)); good = np.isfinite(pix)
            v = np.where(good, pix, 0.0); return np.full(256, float(np.var(v)) + 1e-3), float(good.mean()), good
        SS.masked_spectrum = fake_ms
        SS.apodised_mask = lambda good, apod_deg=2.0, taper="C2": good.astype(float)
    DV = types.SimpleNamespace(SKY_NULL_N=4, SKY_LEE_N=6, _mapping=lambda: m.sort_values(["chr", "pix"], kind="mergesort"),
                               _pixmap=real_dv._pixmap, block_shuffle_order=real_dv.block_shuffle_order)
    z = pd.Series(rng.normal(size=n), index=ids)
    recs = D.sky_two_masks(z, DV, SS, apod_deg=2.0)
    assert [r["mask"] for r in recs] == ["hard", "apodised"]
    hard, apod = recs
    assert hard["apod_deg"] is None and apod["apod_deg"] == 2.0 and hard["n_pix"] == apod["n_pix"] == 2500
    assert 0 < apod["w2"] <= 1 and hard["f_sky"] == apod["f_sky"]
    for r in recs:
        assert all(np.isfinite(r[f"ratio_{b}"]) and r[f"ratio_{b}"] > 0 for b in range(1, 7))
        assert 0 < r["lee_p"] <= 1 and r["n_sites"] == n
    if calls:
        assert {c[0] for c in calls} == {"hard", "apodised"} and ("apodised", 2.0) in calls


def test_worker_D_end_to_end_with_pr18(tmp_path):
    np = pytest.importorskip("numpy"); pd = pytest.importorskip("pandas"); pytest.importorskip("pyarrow")
    pytest.importorskip("healpy"); pytest.importorskip("scipy")
    import tarfile
    chain = tmp_path / "chain"; chain.mkdir()
    (chain / "sky_statistics.py").write_text(pr18_sky_source().replace("NSIDE, LMAX, NPERM = 128, 255, 20", "NSIDE, LMAX, NPERM = 64, 255, 20"))
    (chain / "dev_stages.py").write_text(open(os.path.join(D.CHAIN_DEFAULT, "dev_stages.py")).read() + FAKE_DV_TAIL)
    (chain / "conductor_v3.py").write_text("def stage_a_composition(b):\n    return {'fractions': {'NEU': 0.6, 'CD4T': 0.4}}\n")
    n = 3000; ids = [f"cg{i:08d}" for i in range(n)]
    pd.DataFrame({"pix": np.arange(n) % 2500, "chr": np.repeat([1, 2, 3], n // 3)}, index=ids).to_csv(chain / "fake_map.csv")
    bucket = make_bucket(str(tmp_path / "bucket")); rng = np.random.default_rng(7)
    man = [dict(series="GSE250556", gsm=f"GSM95000{i}", slide="1", plat="EPIC_v1", specimen="whole blood", healthy="True",
                person=f"p{i % 2}") for i in range(4)]
    man += [dict(series="GSE888001", gsm=f"GSM95010{i}", slide="2", plat="EPIC_v1", specimen="whole blood", healthy="True", person="")
            for i in range(2)]
    man += [dict(series="GSE888001", gsm="GSM950199", slide="2", plat="EPIC_v1", specimen="whole blood", healthy="False", person="")]
    os.makedirs(os.path.join(bucket, "downloads", "G_chain_tests", "healthy_repeat", "GSE888001"), exist_ok=True)
    for ser in ("GSE250556", "GSE888001"):                          # the DEV-BASE-CHAIN-01 betas, packed as base_driver.py packed them
        tp = os.path.join(bucket, D.BETAS_PREFIX, f"betas_{ser}.tar")
        with tarfile.open(tp, "w") as t:
            for r in man:
                if r["series"] == ser:
                    f = tmp_path / f"{r['gsm']}.parquet"
                    pd.DataFrame({"beta": rng.uniform(0.05, 0.95, n)}, index=pd.Index(ids, name="cpg_id")).to_parquet(f)
                    t.add(f, arcname=f"{r['gsm']}.parquet")
    mp = tmp_path / "manifest.csv"
    with open(mp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(man[0])); w.writeheader(); [w.writerow(r) for r in man]
    atlas = tmp_path / "atlas.parquet"; atlas.write_text("not read by the fake dev_stages")
    sh = Recorder(); work = tmp_path / "work"
    rc = D.main(["--work", str(work), "--only", "D", "--chain-dir", str(chain), "--manifest", str(mp), "--python", PY,
                 "--atlas-v2", str(atlas), "--apod-deg", "3.0"], store=D.LocalStore(bucket), shutdown=sh)
    log = open(os.path.join(bucket, "results", "BOXRUN1", "log.txt")).read()
    jlog = os.path.join(work, "results", "BOXRUN1", "D", "job_D.log")
    assert rc in (D.EXIT_OK, D.EXIT_BARS), log + (open(jlog).read() if os.path.exists(jlog) else "")
    st = state({"work": str(work)})
    assert st["jobs"]["D"]["status"] == "done" and sh.calls == 1
    rows = list(csv.DictReader(open(os.path.join(bucket, "results", "BOXRUN1", "D", "D_sky.csv"))))
    assert len(rows) == 2 * 6 and {r["mask"] for r in rows} == {"hard", "apodised"}          # 4 replicates + 2 healthy repeats, both masks
    assert "GSM950199" not in {r["gsm"] for r in rows}                                       # not healthy: not in the set
    assert {r["apod_deg"] for r in rows if r["mask"] == "apodised"} == {"3.0"} and all(r["w2"] for r in rows if r["mask"] == "apodised")
    bars = json.load(open(os.path.join(bucket, "results", "BOXRUN1", "D", "bars.json")))
    assert [b["name"] for b in bars["bars"]] == [
        "apodised mask: median band power / block-shuffle null, bands 1-6", "apodised mask: look-elsewhere rate",
        "hard mask: median band power / block-shuffle null, bands 1-6", "hard mask: look-elsewhere rate"]
    assert bars["bars"][0]["n"] == 4 and bars["bars"][1]["n"] == 6       # band bar over GSE250556, look-elsewhere over all healthy arrays
    assert "--apod-deg 3.0" in " ".join(st["jobs"]["D"]["command"])
    assert ("bar MET" in log or "bar NOT MET" in log) and "[D] DONE" in log
