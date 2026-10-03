# PROC-REPL-V3-01 — exact commands (2026-10-03)

Repository commit: 87cfa658498fdd8d9554a38db3a50e60c6c2de9a. Box: ssh methylphys-cpu-01 (this run: m7a.32xlarge, 128 vCPU), python `/home/ubuntu/env/bin/python`
(methylprep 1.7.1, numpy 1.26.4, pandas 1.5.3).

## 1. Chain, unchanged, from the commit
```
git clone https://github.com/hmahaffeyges/IAM-Validation.git && cd IAM-Validation && git checkout 87cfa65
git archive --format=tar --prefix=x/ 87cfa65 Biological_Physics/MethylPhys/chain | tar -x -C /tmp
mkdir /tmp/x2 && mv /tmp/x/Biological_Physics/MethylPhys/chain /tmp/x2/chain && (cd /tmp/x2 && tar czf chain_v3_87cfa65.tgz chain)
# chain_v3_87cfa65.tgz as staged to the box in this run: sha256 1c59ba91d16088567658c060bf24e20dfe0c5089887eb85103f0ff3a8d66bbe8 (a rebuilt archive differs only by gzip timestamps; the file hashes below do not)
# frozen inputs, sha256:
#  conductor_v3.py                     dfdef09d8f9d6b0ea7875b3d5f8433e466e35541888112a79762f48f46821ad9
#  MethylPhys_Interface/run_sample.py  7002a79ce9cbadaff231e3d3c97b40f5ce4f8ce54c347c10b49cc80f144e0caa
#  stage_m_met_a.py                    0dd752575c668e6c9c5d17c4c290770a77f4444f10c76e36c1bdece34f838002
#  noise_gate_EPIC_v1.json             825f64f7b6c7077a6ff17ce44c28b5470ad55d4347d64edb57055b793f808480
#  noise_sites_EPIC_v1.json            3531561783261c577e436a0ac6b83799adbb791fe4ed40be7d0a48ec83f7aeb0
#  metA_floors_v1_3.json               2f64f8846f8a4b0d5d66964347807dde03067130f17d9ce97a8e4dc7efe7c9ae
#  neutrophil_reference_v1_1.json      a03531f07369d90bc96ce32ebc723bc59abb060ff9bc7b93042a28df47d9c023
#  blood_composition_EPIC_v1.json      5616b95e2095e1a5a771436a616b218489798f5fe8b5089cec7c6e4f46ff8693
```

## 2. Box job (workdir holds chain_v3_87cfa65.tgz, run_proc_repl_v3_01.py, s3post.json)
```
NP=32 /home/ubuntu/env/bin/python run_proc_repl_v3_01.py 2>&1 | tee run.log
```
The script (in this folder) does, in order:
```
curl -s -f -L -C - -o /home/ubuntu/data/G_chain_tests/GSE250556/GSE250556_RAW.tar \
  https://ftp.ncbi.nlm.nih.gov/geo/series/GSE250nnn/GSE250556/suppl/GSE250556_RAW.tar      # then tar -x into .../idat
curl -s -f -L -o .../GSE250556_series_matrix.txt.gz \
  https://ftp.ncbi.nlm.nih.gov/geo/series/GSE250nnn/GSE250556/matrix/GSE250556_series_matrix.txt.gz
# per array, pass 1 (first array run alone so methylprep's manifest download is not raced; then 32 processes):
/home/ubuntu/env/bin/python chain/MethylPhys_Interface/run_sample.py --grn <GSM>_<sentrix>_Grn.idat.gz --red <GSM>_<sentrix>_Red.idat.gz \
  --engine v3 --specimen "whole blood" --array-type EPIC_v1 --out reports/<GSM>.html --id <GSM> --ledger reports/ledger.jsonl --sex M --age <age>
#   (cwd chain/MethylPhys_Interface, PYTHONPATH=chain, OMP_NUM_THREADS=1; a failed array is retried once serially)
# per array with a pass-1 A, pass 2: <GSM>_refs.csv = columns id,A,f_neu,N of the other arrays on the same slide with an A
#   (>= 3 on every slide here; else all other GSE250556 arrays)
/home/ubuntu/env/bin/python chain/MethylPhys_Interface/run_sample.py ...same... --out reports/<GSM>_tared.html --ledger reports/ledger_tared.jsonl \
  --slide-ref-table reports/<GSM>_refs.csv
```
Every per-array command line and its full output: `reports.tgz` -> `logs/<GSM>.log`, `logs/<GSM>_tared.log`. Bundles: `reports/<GSM>[_tared]_bundle.json`.

## 3. S3 copies (no credentials on the box; time-limited presigned POST policy per prefix)
`s3sync.py <main job workdir>` ran beside the chain job and copied every new file every 45 s:
IDATs + series matrix -> `s3://methylphys-data-945451304272-us-west-2-an/downloads/G_chain_tests/GSE250556/` (128 IDATs + the platform csv that ships in the RAW tar),
outputs -> `s3://methylphys-data-945451304272-us-west-2-an/results/PROC_REPL_V3_01/`. 585 uploads, all HTTP 204 (`s3_sync.log`).
The chain script's own upload attempts used the global S3 endpoint and got HTTP 307 (`run.log`: "S3 failures 579"); no output was lost — the sync job copied them.

## 4. Tabulation (laptop, pandas)
doors/data/proc_repl_v3_01_readings.csv (archived privately) = `proc_repl_v3_01_box_readings.csv` + person/prep/replicate parsed from the title
(`^subject(\w)`, `_(pooled|unpooled)_`, `replicate(\d+)`) + RUN3 `A_rel_tared` from `doors/data/chain_v3_dev3_readings.csv` joined on gsm.
Statistics per `ANALYSIS_RULES_set_before_reading.md`.
