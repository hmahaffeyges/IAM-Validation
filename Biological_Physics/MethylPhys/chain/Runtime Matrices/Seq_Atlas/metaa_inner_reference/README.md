The IAM Sequencing Atlas's level at every Met-A identity site without the atlas's own molecule-end calls (stage_seq_atlas.metaa_inner_reference;
run_sample.py --build seq-metaa-inner), metaa_inner_reference_v1.parquet: per cell and site, the healthy donors' methylated and total calls after
the one-call trim at each molecule end (meth_inner, reads_inner) and the end calls themselves (meth_end, reads_end: the published per-donor counts,
metaa_donor_counts_v1, minus the trimmed). The trimmed donor counts are the donors' own molecule files read by run_sample.py --build seq-pat-lbeta
--dev-seq-trim-ends 1, then --build seq-metaa-donors. 787 of 905,266 donor-site rows had fewer published than trimmed calls and are left out
(counted in the build's record). Development (sequencing step 1: each library calibrated on its own calls).
