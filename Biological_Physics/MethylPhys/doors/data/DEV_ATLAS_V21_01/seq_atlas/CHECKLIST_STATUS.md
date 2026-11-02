# IAM Sequencing Atlas: checklist status

Each item of `SEQ_ATLAS_CHECKLIST.md`, built or not yet, with where its record is. Development (plan 35.30). The records are in
`doors/DEV_ATLAS_V21_01.md` (the atlas), `doors/DEV_METAA_ATLAS_CURVE_01.md` (Met-A) and `doors/DEV_TERRITORIES_01.md` (territories).

| # | Status | Where |
|---|---|---|
| A1 | Built at the identity sites (each input's own reads) and at the island sites (per donor half); not yet at every CpG | `metaa_donor_counts`, `terr_half_counts` |
| A2 | Built: each cell's measured disorder at every CpG | the atlas pieces (`disorder`) |
| A3 | Built: level, spread, donors, reads per cell | the atlas pieces; `array_sites` |
| A4 | Built: blank where unmeasured, counts beside every value | every column (65535 blank) |
| A5 | Not yet: six cells have donors from two recorded sources (`donor_traits.csv`) | |
| A6 | Built for held error: the healthy held-error reference from the molecules | `held_reference_v1` |
| A7 | Built: sex, age, race, tissue and source per input, from GEO | `donor_traits.csv` |
| A8 | Built: 16-bit counts | every column |
| B1 | Built: IAM-A in the sample's mix; 0.942 of healthy inside its own band | `DEV_TERRITORIES_01.md`, pass 5 |
| B2 | Built: coupling-A, 0.993 inside its band | pass 5 |
| B3 | Built: Met-A in each cell, donor by donor, 1.0077 on healthy | `DEV_METAA_ATLAS_CURVE_01.md` |
| B4 | Built: islands 1.0122 and domains 1.0011 on healthy | `DEV_TERRITORIES_01.md` |
| B5 | Built: C-Score, 0.964 inside its limit | pass 5 |
| B6 | Built: the Deconvolver on the sequencing atlas; the megakaryocyte column provisional | `DEV_ATLAS_V21_01.md` |
| B7 | Built: every cell in a mix (in-mix readings) | `DEV_ATLAS_V21_01.md` |
| B8 | Built on sequencing: the counted distance to the entropy ceiling against the sample's own cell; synthetic met | `ceiling_seq_synthetic.txt` |
| B9 | Not yet on sequencing | |
| C1 | Built: the sky sphere, 56 cells, level and 95 % range | `sphere_cols`; `sphere_widths.txt` |
| C2 | Built: the information globe, and each sequenced sample's distance from its own healthy self against the sequencing atlas; the globe's change count flagged (outside healthy neurons above v2.1) | `globe_outside_3.txt` |
| C3 | Built: both noise maps on the sky sphere | `noise_sphere.json` |
| C4 | Built against v2.1 (the bridge); v2 is not in the repository | `bridge_summary.txt` |
| D1 | Built: two sealed donor halves | `seq_atlas_inputs.csv` |
| D2 | Built: read halves | the atlas pieces |
| D3 | Built: noise maps per cell | `noise_sphere.json` |
| D4 | Built: 95 % range and standard error per value, with a counting floor where donors agree exactly; synthetic version 3 met | `sphere_range_synthetic_v3.txt` |
| D5 | Built: sex chromosomes and poorly mapping probes masked, apodized edges | `stage_seq_atlas.sky_mask_sites` |
| D6 | Built on the sphere and along the genome; synthetics met | `noise_null_summary.txt`, `noise_spectrum_summary.txt` |
| D7 | Built: copies of each cell per site behind each in-mix reading on sequencing, flagged below one; synthetic met | `copies_seq_synthetic.txt` |
| D8 | Not yet as a covariance; the genome-distance spectrum gives its diagonal bands | `noise_spectrum_summary.txt` |
| D9 | Built at the islands: slope 1.0001 | `DEV_TERRITORIES_01.md` |
| D10 | Built: shuffled nulls on the sphere; counting map met in 56 of 56 cells | `noise_null_summary.txt` |
| D11 | Built against v2.1 (platform offset per cell); bisulfite against enzymatic not yet | `bridge_v21.csv` |
| D12 | Not available: no oxidative data in the inputs | |
| D13 | Not yet | |
| D14 | Built: counting map the same on every autosome (0 of 1232 beyond 4 standard errors) | `isotropy.txt` |
| D15 | Built: donor halves sealed; outside healthy samples kept unread by the build | `selfree_samples.csv` |
| D16 | Built through the one command from committed code; the columns rebuilt identical | `DEV_ATLAS_V21_01.md` |
| D17 | Built: each tool, its origin in cosmology, its use, the chain's part and the result | `GUIDEBOOK.md` |
