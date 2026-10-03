# chain/MethylPhys_Interface

How a person drives chain v3.

| file | |
|---|---|
| [`run_sample.py`](run_sample.py) | one command: IDAT pair, beta table or single-molecule input -> Stage 0 intake -> Stage 1 calibration -> conductor_v3 -> report_v3, plus Stage Q when sequencing input is given; writes the report, the bundle and a ledger row |
| [`report_v3.py`](report_v3.py) | renders the v3 report (one HTML page) from the bundle |

The class-floor engine's report builder and inventory generators that lived here were retired with that engine on 2026-10-03 and are archived privately.
