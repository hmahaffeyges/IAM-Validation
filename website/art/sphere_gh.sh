#!/bin/bash
# The 3D sphere textures on GitHub's machines (to-do step 13, 2026-10-21): build, then pack for the workflow's artifact.
set -euo pipefail
OUT=${1:-xout}; mkdir -p "$OUT/sphere"
python3 website/art/make_sphere_3d.py "$OUT/sphere" || { echo "make_sphere_3d failed"; exit 1; }
tar -czf "$OUT/sphere.tar.gz" -C "$OUT" sphere && cp "$OUT/sphere/sphere.json" "$OUT/sphere.json" && echo "packed $(ls "$OUT/sphere" | wc -l) files"
