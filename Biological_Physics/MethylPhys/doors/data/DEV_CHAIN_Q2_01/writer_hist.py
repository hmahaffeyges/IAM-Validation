#!/usr/bin/env python3
"""WRITER test 1 (register Part P '### WRITER plan'; exploration). Which record history do the book's own converged chains prefer: the book's E(a) = exp(1 - 1/a)
or the forward histories computed by the missing-writer spider (Weyl energy of collapsed halos above 1e13 Msun/h, Clifton-Ellis-Tavakol 2013, Tinker 2008 and
Sheth-Tormen)? Runs ON A GITHUB MACHINE. Builds CAMB 1.6.5 with the repo's own equations_iam_level2.f90 (the Level-2 dual-sector code the chains ran on), with ONE
change: the line iam_Ea = exp(1 - 1/a) reads a history chosen at run time (IAM_EHIST: 0 book, 1 Weyl Tinker, 2 Weyl Sheth-Tormen, 3 off = LCDM growth). beta_m stays
0.15765 (fixed in advance, as in the chains). Importance test on the converged chain iam_level2_runD (Planck + RSD): for the chain's best-fit sample and 240
posterior draws, chi2 of the book's 7 RSD points (likelihood_rsd.py, same numbers) under each history. VALIDATION FIRST: in mode 0 the best-fit sample must
reproduce the chain's own chi2__iam_rsd and sigma8 (else the build is not the chain's and nothing else is reported as a result).
Limits stated in advance: only the RSD part is recomputed (the histories change the matter-sector growth; the photon sector is LCDM in every mode, so the CMB
spectra are the same up to lensing, which is not recomputed); importance reweighting, not a new chain."""
import os, sys, json, glob, subprocess, tarfile, shutil, re
R = os.getcwd()
def sh(c, **k): print("::notice::$ " + c[:160]); return subprocess.run(c, shell=True, check=True, **k)
if shutil.which("gfortran") is None: sh("sudo apt-get update -q && sudo apt-get install -y -q gfortran")
sh(f"{sys.executable} -m pip install -q numpy scipy")
W = "/tmp/cambsrc"; os.makedirs(W, exist_ok=True)
try:
    sh(f"{sys.executable} -m pip download -q camb==1.6.5 --no-binary :all: --no-deps -d {W}")
    tgz = glob.glob(f"{W}/camb-1.6.5*.tar.gz")[0]; tarfile.open(tgz).extractall(W); src = glob.glob(f"{W}/camb-1.6.5*/")[0]
except Exception as e:
    print("::notice::sdist unavailable (" + str(e)[:120] + "); cloning tag 1.6.5")
    sh(f"git clone -q --recursive --branch 1.6.5 https://github.com/cmbant/CAMB.git {W}/CAMB"); src = f"{W}/CAMB/"
f90 = open(f"{R}/Cosmological_Physics/camb_validation/equations_iam_level2.f90").read()
EH = {1: {"0.5": 0.8177661720934197, "1.0": 0.563404384985999, "2.0": 0.15077506567767232, "3.0": 0.018620092472552478},
      2: {"0.5": 0.8539482216511929, "1.0": 0.6295575089418138, "2.0": 0.2100721197324887, "3.0": 0.03646809437670566}}
import math
def lnrow(m): d = EH[m]; return [0.0] + [math.log(d[z]) for z in ("0.5", "1.0", "2.0", "3.0")]
L1, L2 = lnrow(1), lnrow(2)
fa = lambda v: ", ".join(f"{x:.10f}_dl" for x in v)
decl_old = "    real(dl) adotoa_matter, z_matter, iam_Ea, iam_grho_extra, iam_grho0\n"
decl_new = (decl_old + "    integer, save :: iam_mode = -1\n    character(len=16) :: iam_s\n    real(dl) :: iam_zz, iam_lnE\n"
            "    real(dl), parameter :: iam_zt(5) = [0._dl, 0.5_dl, 1._dl, 2._dl, 3._dl]\n"
            f"    real(dl), parameter :: iam_l1(5) = [{fa(L1)}]\n    real(dl), parameter :: iam_l2(5) = [{fa(L2)}]\n    integer :: iam_j\n")
line_old = "        iam_Ea = exp(1.0_dl - 1.0_dl / a)\n"
line_new = ("        if (iam_mode < 0) then\n            call get_environment_variable('IAM_EHIST', iam_s)\n            iam_mode = 0\n"
            "            if (len_trim(iam_s) > 0) read(iam_s, *) iam_mode\n        end if\n"
            "        if (iam_mode == 0) then\n            iam_Ea = exp(1.0_dl - 1.0_dl / a)\n        else if (iam_mode == 3) then\n            iam_Ea = 0.0_dl\n        else\n"
            "            iam_zz = 1.0_dl / a - 1.0_dl\n            iam_j = 4\n            do while (iam_j > 1 .and. iam_zz < iam_zt(iam_j))\n                iam_j = iam_j - 1\n            end do\n"
            "            if (iam_mode == 1) then\n                iam_lnE = iam_l1(iam_j) + (iam_l1(iam_j+1) - iam_l1(iam_j)) * (iam_zz - iam_zt(iam_j)) / (iam_zt(iam_j+1) - iam_zt(iam_j))\n"
            "            else\n                iam_lnE = iam_l2(iam_j) + (iam_l2(iam_j+1) - iam_l2(iam_j)) * (iam_zz - iam_zt(iam_j)) / (iam_zt(iam_j+1) - iam_zt(iam_j))\n            end if\n"
            "            iam_Ea = exp(min(0.0_dl, iam_lnE))\n        end if\n")
assert f90.count(decl_old) == 1 and f90.count(line_old) == 1, "f90 anchors not found"
f90 = f90.replace(decl_old, decl_new).replace(line_old, line_new)
open(f"{src}fortran/equations.f90", "w").write(f90)
sh(f"cd {src} && {sys.executable} -m pip install -q .", timeout=3000)
sh(f"cd /tmp && {sys.executable} -c 'import camb; print(camb.__version__, camb.__file__)'")

# chain samples (the converged chain the book reports; the file in the repo)
import numpy as np
cf = f"{R}/Cosmological_Physics/camb_validation/chains/iam_level2_runD.1.txt"
hdr = open(cf).readline().lstrip("#").split(); X = np.loadtxt(cf); col = {n: i for i, n in enumerate(hdr)}
best = int(np.argmin(X[:, col["minuslogpost"]]))
rng = np.random.default_rng(7); w = X[:, col["weight"]]; draws = rng.choice(len(X), size=240, replace=True, p=w / w.sum())
idx = [best] + [int(i) for i in draws]
P = [{k: float(X[i, col[k]]) for k in ("ombh2", "omch2", "cosmomc_theta", "tau", "As", "ns", "H0", "sigma8", "chi2__iam_rsd")} for i in idx]
json.dump(P, open("/tmp/samples.json", "w"))
print(f"::notice::WRITER chain rows {len(X)}, best-fit row {best}, minuslogpost {X[best, col['minuslogpost']]:.3f}, chain chi2_rsd {P[0]['chi2__iam_rsd']:.4f}, sigma8 {P[0]['sigma8']:.5f}")

worker = r'''
import os, sys, json, numpy as np, camb
D = np.array([[0.067, 0.423, 0.055], [0.150, 0.530, 0.160], [0.380, 0.497, 0.045], [0.510, 0.459, 0.038], [0.700, 0.473, 0.041], [0.850, 0.315, 0.095], [1.480, 0.462, 0.045]])
zs = sorted(set(D[:, 0].tolist()) | {0.0}, reverse=True)
out = []
for p in json.load(open("/tmp/samples.json")):
    try:
        pars = camb.set_params(ombh2=p["ombh2"], omch2=p["omch2"], cosmomc_theta=p["cosmomc_theta"], tau=p["tau"], As=p["As"], ns=p["ns"], mnu=0.06,
                               num_massive_neutrinos=1, nnu=3.046, halofit_version="mead2020", bbn_predictor="PArthENoPE_880.2_standard.dat",
                               theta_H0_range=[20, 100], WantCls=False)
        pars.set_matter_power(redshifts=zs, kmax=2.0)
        r = camb.get_results(pars); f = r.get_fsigma8()[::-1]; s8 = r.get_sigma8()[::-1]
        zz = sorted(zs); m = {z: f[i] for i, z in enumerate(zz)}
        th = np.array([m[z] for z in D[:, 0]]); chi2 = float(np.sum(((th - D[:, 1]) / D[:, 2]) ** 2))
        out.append({"chi2": chi2, "sigma8": float(s8[0]), "fs8": th.tolist(), "H0": float(r.Params.H0)})
    except Exception as e:
        out.append({"error": str(e)[:200]})
json.dump(out, open(f"/tmp/mode{os.environ['IAM_EHIST']}.json", "w"))
'''
open("/tmp/worker.py", "w").write(worker)
procs = []
for m in (0, 1, 2, 3):
    env = dict(os.environ, IAM_EHIST=str(m), OMP_NUM_THREADS="1")
    procs.append(subprocess.Popen([sys.executable, "/tmp/worker.py"], env=env, cwd="/tmp"))
for p in procs: p.wait()
M = {m: json.load(open(f"/tmp/mode{m}.json")) for m in (0, 1, 2, 3)}
names = {0: "book E(a)", 1: "Weyl Tinker >1e13", 2: "Weyl Sheth-Tormen >1e13", 3: "off (LCDM growth)"}
b0 = M[0][0]
val = (abs(b0.get("chi2", 1e9) - P[0]["chi2__iam_rsd"]) < 0.05 and abs(b0.get("sigma8", 0) - P[0]["sigma8"]) < 0.002)
res = {"validation": {"chain_chi2_rsd": P[0]["chi2__iam_rsd"], "rebuilt_chi2_rsd": b0.get("chi2"), "chain_sigma8": P[0]["sigma8"], "rebuilt_sigma8": b0.get("sigma8"),
                      "chain_H0": P[0]["H0"], "rebuilt_H0": b0.get("H0"), "passed": bool(val)}, "modes": {}}
for m in (0, 1, 2, 3):
    ok = [i for i in range(len(idx)) if "chi2" in M[m][i] and "chi2" in M[0][i]]
    d = np.array([M[m][i]["chi2"] - M[0][i]["chi2"] for i in ok]); c = np.array([M[m][i]["chi2"] for i in ok])
    wts = np.exp(-0.5 * (d - d.min())) if len(d) else np.array([])
    res["modes"][names[m]] = {"best_fit_chi2": M[m][0].get("chi2"), "best_fit_sigma8": M[m][0].get("sigma8"), "best_fit_fs8": M[m][0].get("fs8"),
                              "draws_ok": len(ok), "chi2_median": float(np.median(c)) if len(c) else None,
                              "delta_vs_book_median": float(np.median(d)) if len(d) else None, "delta_vs_book_16_84": [float(np.percentile(d, 16)), float(np.percentile(d, 84))] if len(d) else None,
                              "mean_importance_weight_vs_book": float(np.mean(np.exp(-0.5 * d))) if len(d) else None, "errors": sum(1 for x in M[m] if "error" in x)}
os.makedirs("xout", exist_ok=True); json.dump(res, open("xout/writer_hist.json", "w"), indent=1)
print("::notice::WRITER VALIDATION: " + json.dumps(res["validation"]))
for k, v in res["modes"].items():
    print(f"::notice::WRITER {k}: best-fit chi2_rsd {v['best_fit_chi2']}, sigma8 {v['best_fit_sigma8']}, median dchi2 vs book {v['delta_vs_book_median']} (16-84 {v['delta_vs_book_16_84']}), mean weight {v['mean_importance_weight_vs_book']}, draws {v['draws_ok']}, errors {v['errors']}")
