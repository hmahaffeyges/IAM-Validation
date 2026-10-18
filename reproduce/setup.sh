#!/bin/bash
# One-time setup for reproduce.py: a Python 3.11 environment with exactly the chain's pinned packages (the same command the chain's
# fresh-clone release check runs). Usage, from the repository root: bash reproduce/setup.sh   (makes reproduce/.venv)
set -eu
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH=$HOME/.local/bin:$PATH; }
uv venv -q -p 3.11 reproduce/.venv
uv pip install -q -p reproduce/.venv/bin/python -r Biological_Physics/MethylPhys/chain/requirements.txt requests
reproduce/.venv/bin/python -c "import methylprep, pandas; print('ready: methylprep', methylprep.__version__, 'pandas', pandas.__version__)"
