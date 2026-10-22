"""Value pictures for the atlas globe's sliders: drawn by the one command since 2026-10-22 (rule 23b-f). This file only runs
  chain/MethylPhys_Interface/run_sample.py --build atlas-masks --build-out OUT
(the drawing, unchanged, is chain/MethylPhys_Interface/reference_globes.py: atlas_masks). Usage: python3 website/art/make_sphere_masks.py [OUT=website/art/out/sphere]"""
import os, sys, subprocess
H = os.path.dirname(os.path.abspath(__file__)); RS = os.path.join(H, "..", "..", "Biological_Physics", "MethylPhys", "chain", "MethylPhys_Interface", "run_sample.py")
if __name__ == "__main__":
    sys.exit(subprocess.run([sys.executable, RS, "--build", "atlas-masks", "--build-out", sys.argv[1] if len(sys.argv) > 1 else os.path.join(H, "out", "sphere")]).returncode)
