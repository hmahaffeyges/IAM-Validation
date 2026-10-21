"""Runs website/art/sphere_gh.sh from the exploration workflow (which runs Python scripts). Usage: python3 website/art/sphere_gh.py OUT"""
import subprocess, sys, os
sys.exit(subprocess.call(["bash", os.path.join(os.path.dirname(os.path.abspath(__file__)), "sphere_gh.sh"), sys.argv[1] if len(sys.argv) > 1 else "xout"]))
