"""PRED-TEMPERATURE-01 stage T3 pair list (2026-10-17): the 111 accessions grown at both temperatures (t1_gse54292_design.json) joined to ENA's run
list for PRJNA236110 (t3_ena_runs_PRJNA236110.tsv, fetched 2026-10-17: run, sample, read count, FASTQ links and sizes; no read content).
Writes t3_pairs_runs.csv through the chain writer."""
import os, sys, json, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "..", "chain")))
import stage_q2_selfref as Q2
D = json.load(open(os.path.join(HERE, "t1_gse54292_design.json"))); S = pd.DataFrame(dict(gsm=D["gsm"], acc=D["acc"], temp=D["temp"])); S = S[S.acc.isin(D["both"])]
E = pd.read_csv(os.path.join(HERE, "t3_ena_runs_PRJNA236110.tsv"), sep="\t"); E = E[E.sample_alias.isin(S.gsm)]
m = S.merge(E, left_on="gsm", right_on="sample_alias")[["acc", "temp", "gsm", "run_accession", "read_count", "fastq_ftp", "fastq_bytes"]]
assert m.gsm.nunique() == len(S) == 233 and len(m) == 394, (m.gsm.nunique(), len(S), len(m))
Q2.write_csv(m, os.path.join(HERE, "t3_pairs_runs.csv"), manifest=["PHY-057"]); print(len(m), m.gsm.nunique(), m.acc.nunique())
