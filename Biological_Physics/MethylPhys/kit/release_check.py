#!/usr/bin/env python3
"""kit/release_check.py - entry point kept at its old path: runs chain/release_check_v3.py (chain v3 only) and returns its exit status.
The class-floor release check this file used to hold was retired with chain v2 on 2026-10-03 and is archived privately."""
import os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.exit(subprocess.call([sys.executable, os.path.join(HERE, "..", "chain", "release_check_v3.py")] + sys.argv[1:]))
