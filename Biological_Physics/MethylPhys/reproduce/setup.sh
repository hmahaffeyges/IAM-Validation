#!/bin/bash
# One-time setup for reproduce.py: a Python 3.11 environment with exactly the chain's pinned packages (the same command the chain's
# fresh-clone release check runs). Usage, from the repository root: bash Biological_Physics/MethylPhys/reproduce/setup.sh   (makes Biological_Physics/MethylPhys/reproduce/.venv)
set -eu
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH=$HOME/.local/bin:$PATH; }
uv venv -q -p 3.11 Biological_Physics/MethylPhys/reproduce/.venv
uv pip install -q -p Biological_Physics/MethylPhys/reproduce/.venv/bin/python -r Biological_Physics/MethylPhys/chain/requirements.txt requests
Biological_Physics/MethylPhys/reproduce/.venv/bin/python -c "import methylprep, pandas; print('ready: methylprep', methylprep.__version__, 'pandas', pandas.__version__)"
