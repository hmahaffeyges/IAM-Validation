"""The 3D sphere of IAM-Atlas v2.1 for the public page: drawn by the one command since 2026-10-22 (rule 23b-f). This file only runs
  chain/MethylPhys_Interface/run_sample.py --build atlas-sphere --build-out OUT [--build-input ATLAS_DIR]
(the drawing, unchanged, is chain/MethylPhys_Interface/reference_globes.py: atlas_sphere). Usage: python3 website/art/make_sphere_3d.py OUT [ATLAS_DIR]"""
import os, sys, subprocess
H = os.path.dirname(os.path.abspath(__file__)); RS = os.path.join(H, "..", "..", "Biological_Physics", "MethylPhys", "chain", "MethylPhys_Interface", "run_sample.py")
if __name__ == "__main__":
    sys.exit(subprocess.run([sys.executable, RS, "--build", "atlas-sphere", "--build-out", sys.argv[1]] + (["--build-input", sys.argv[2]] if len(sys.argv) > 2 else [])).returncode)
