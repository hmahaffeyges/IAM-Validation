#!/usr/bin/env python3
"""
preflight.py — CPG_CMB self-bootstrapping environment check.

Run this once on any machine before running the chain. It will, for everything the chain needs:
  1. confirm the Python version is adequate (>= 3.9), and refuse clearly if not;
  2. check each required package, and INSTALL the correct version itself if it is missing
     (it never downgrades a working newer version the machine already has);
  3. decompress the atlas if it is still in .xz form;
  4. regenerate the cpg->HEALPix mapping if it is absent and the generator is present;
  5. report exactly what it found, installed, and fixed, and whether the chain is READY.

Usage:
    python3 preflight.py                 # core chain (works from a calibrated beta)
    python3 preflight.py --idat          # also install methylprep for raw IDAT decode
    python3 preflight.py --root /path    # set the engine root explicitly

Design notes:
  - Idempotent: safe to run repeatedly. It checks before it installs.
  - It only installs what is MISSING, so it will not fight a doctor's existing working stack.
  - Pins are minimum-versions known to run the chain (validated set: numpy 2.4, pandas 3.0,
    scipy 1.17, healpy 1.19, Pillow 12.1, reportlab 4.4 on Python 3.12). A missing package is
    installed at >= its minimum; a present package at or above the minimum is accepted as-is.
"""
import sys, subprocess, importlib, os, argparse
from pathlib import Path

MIN_PY = (3, 9)
# (import_name, pip_name, minimum_version)  -- core chain
CORE = [
    ("numpy",     "numpy",     "1.24"),
    ("pandas",    "pandas",    "1.5"),
    ("scipy",     "scipy",     "1.10"),
    ("healpy",    "healpy",    "1.16"),   # Cosmic Methylome Background render
    ("PIL",       "Pillow",    "9.0"),    # plate thumbnails
    ("reportlab", "reportlab", "4.0"),    # PDF output
]
IDAT = [("methylprep", "methylprep", "1.5")]  # only needed to decode raw .idat (Stage 1)


def _vtuple(s):
    out = []
    for p in str(s).split(".")[:3]:
        try: out.append(int(p))
        except ValueError: out.append(0)
    return tuple(out)


def check_python():
    ok = sys.version_info[:2] >= MIN_PY
    print(f"[python] {sys.version.split()[0]}  ({'OK' if ok else 'TOO OLD'}; need >= %d.%d)" % MIN_PY)
    if not ok:
        print("  ! This interpreter is too old. Install Python >= 3.9 and re-run with it, e.g.:")
        print("      macOS:  brew install python@3.12   then  python3.12 preflight.py")
        print("      Debian: sudo apt-get install -y python3.12 python3-pip   then  python3.12 preflight.py")
        sys.exit(2)
    return ok


def _pip_install(pip_name, minimum):
    spec = f"{pip_name}>={minimum}"
    for args in ([sys.executable, "-m", "pip", "install", "--break-system-packages", "-q", spec],
                 [sys.executable, "-m", "pip", "install", "-q", spec]):
        try:
            subprocess.check_call(args)
            return True
        except subprocess.CalledProcessError:
            continue
    return False


def ensure(import_name, pip_name, minimum, installed, failed):
    try:
        mod = importlib.import_module(import_name)
        have = getattr(mod, "__version__", "0")
        if _vtuple(have) >= _vtuple(minimum):
            print(f"[ok]   {pip_name}=={have}")
            return True
        print(f"[old]  {pip_name}=={have} < {minimum} -> upgrading")
    except Exception:
        print(f"[miss] {pip_name} not found -> installing >= {minimum}")
    if _pip_install(pip_name, minimum):
        importlib.invalidate_caches()
        try:
            mod = importlib.import_module(import_name)
            print(f"[got]  {pip_name}=={getattr(mod,'__version__','?')}")
            installed.append(pip_name); return True
        except Exception as e:
            print(f"[FAIL] {pip_name} installed but will not import: {e}")
    else:
        print(f"[FAIL] could not install {pip_name}")
    failed.append(pip_name); return False


def decompress_atlas(root):
    xz = root / "IAM_Atlas" / "IAMAtlasREBUILD.csv.xz"
    csv = root / "IAM_Atlas" / "IAMAtlasREBUILD.csv"
    if csv.exists():
        print(f"[ok]   atlas decompressed ({csv.stat().st_size//(1024*1024)} MB)")
        return True
    if not xz.exists():
        print("[warn] atlas not found (neither .csv nor .csv.xz) — chain cannot score without it")
        return False
    print("[..]   decompressing atlas (.xz -> .csv) ...")
    import lzma, shutil
    with lzma.open(xz) as f_in, open(csv, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    print(f"[got]  atlas decompressed ({csv.stat().st_size//(1024*1024)} MB)")
    return True


def ensure_mapping(root):
    npy = root / "Runtime Matrices" / "cpg healpix mapping" / "iamatlas_cpg_to_healpix_nside128.npy"
    if npy.exists():
        print("[ok]   cpg->HEALPix mapping present")
        return True
    gen = root / "Runtime Matrices" / "cpg healpix mapping" / "generate_cpg_healpix_mapping.py"
    if gen.exists():
        print("[..]   HEALPix mapping absent -> regenerating from generator ...")
        try:
            subprocess.check_call([sys.executable, str(gen)], cwd=str(root))
            print("[got]  HEALPix mapping regenerated")
            return True
        except Exception as e:
            print(f"[warn] could not regenerate mapping ({e}) — Cosmic Methylome panel will be skipped")
    else:
        print("[warn] HEALPix mapping absent and no generator — Cosmic Methylome panel will be skipped")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idat", action="store_true", help="also install methylprep for raw IDAT decode")
    ap.add_argument("--root", default=None, help="engine root (defaults to this script's folder)")
    args = ap.parse_args()

    root = Path(args.root) if args.root else Path(__file__).resolve().parent
    os.environ.setdefault("CPG_ENGINE_ROOT", str(root))
    os.environ.setdefault("CPG_ROOT", str(root))
    print("=" * 60)
    print("CPG_CMB preflight")
    print(f"engine root: {root}")
    print("=" * 60)

    check_python()
    installed, failed = [], []
    for imp, pip_name, mn in CORE:
        ensure(imp, pip_name, mn, installed, failed)
    if args.idat:
        for imp, pip_name, mn in IDAT:
            ensure(imp, pip_name, mn, installed, failed)
    else:
        print("[skip] methylprep (raw IDAT decode) — re-run with --idat if scoring from .idat files")

    decompress_atlas(root)
    ensure_mapping(root)

    print("=" * 60)
    if installed:
        print("installed: " + ", ".join(installed))
    if failed:
        print("STILL MISSING: " + ", ".join(failed))
        print("RESULT: NOT READY — resolve the failures above, then re-run preflight.")
        sys.exit(1)
    print("RESULT: READY. Set CPG_ENGINE_ROOT to this folder and run the chain.")
    print(f'  export CPG_ENGINE_ROOT="{root}"')


if __name__ == "__main__":
    main()
