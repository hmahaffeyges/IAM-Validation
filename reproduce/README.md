# Reproduce IAM's results from public data

Each test below starts from public accessions only (GEO, ArrayExpress, SRA), runs the committed code of the chain, and compares what it
makes with the result committed in this repository. Nothing is read from our own storage.

```bash
git clone https://github.com/hmahaffeyges/IAM-Validation.git && cd IAM-Validation
bash reproduce/setup.sh                       # once: Python 3.11 with the chain's pinned packages (reproduce/.venv)
python3 reproduce/reproduce.py --list         # the tests
python3 reproduce/reproduce.py cscore_band    # one test: prints each compared number and PASS or DIFF
python3 reproduce/reproduce.py --all          # every test
```

Each run writes `reproduce/out/<test>/verdict.json` (every comparison, the commit, the time) and a log per step. The tests and what each
compares are in [tests.json](tests.json).
