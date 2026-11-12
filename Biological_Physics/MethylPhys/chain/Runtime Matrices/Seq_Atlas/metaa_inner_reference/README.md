The IAM Sequencing Atlas's level at every Met-A identity site without the atlas's own molecule-end calls (stage_seq_atlas.metaa_inner_reference;
run_sample.py --build seq-metaa-inner), metaa_inner_reference_v1.parquet: per cell and site, the healthy donors' methylated and total calls after
the one-call trim at each molecule end (meth_inner, reads_inner) and the end calls themselves (meth_end, reads_end: the published per-donor counts,
metaa_donor_counts_v1, minus the trimmed). The trimmed donor counts are the donors' own molecule files read by run_sample.py --build seq-pat-lbeta
--dev-seq-trim-ends 1, then --build seq-metaa-donors. 787 of 905,266 donor-site rows had fewer published than trimmed calls and are left out
(counted in the build's record). Development (sequencing step 1: each library calibrated on its own calls).

metaa_inner_reference_t3.parquet (2026-11-12; the seventh self-calibration form): the same, with three calls left out at each molecule end
(the 156 donors counted by atlas_trim_donors.sh TRIMS=3, built by inner_reach_build.sh: run_sample.py --build seq-metaa-donors then
--build seq-metaa-inner; 285 of 905,266 donor-site rows left out). Used only at sites the read cell holds unmarked, whose end fault reaches
three calls; site by site its level matches the one-call reference (CD4 unmarked 0.0058 one-call, 0.0055 three-call, at the same sites;
marked 0.944 and 0.948), while molecules of six or fewer calls drop out entirely, so most marked sites lose their coverage (they are read at
one call). Development.
