# The IAM Sequencing Atlas: a guidebook to its tools

The IAM Sequencing Atlas reads the methylome the way a cosmic microwave background survey reads the sky: one map per cell type, built
from independent observations, with its noise measured from the data, its foregrounds masked, and every statistic checked against a null
before it is trusted. This guidebook lists each tool, where it comes from in cosmology, what it means for a methylation sample, the part of
the chain that does it, and where its result is recorded. Every number lives in the outputs named here; none is typed into this page.

The chain's own command does every step (`chain/MethylPhys_Interface/run_sample.py`), from code in the repository and inputs listed in
`seq_atlas_inputs.csv`. The records are `doors/DEV_ATLAS_V21_01.md` (the atlas), `doors/DEV_METAA_ATLAS_CURVE_01.md` (Met-A) and
`doors/DEV_TERRITORIES_01.md` (the territories). The status of each item is in `CHECKLIST_STATUS.md`.

| Tool | In cosmology | In a methylation sample | In the chain | Result |
|---|---|---|---|---|
| Half-mission split (D1) | A survey's data split in two halves observed separately; their difference holds only noise | Every cell's donors split into two sealed halves before any build; half A minus half B is a map of how healthy people differ, plus counting | the donor halves in `seq_atlas_inputs.csv`; `stage_seq_atlas.noise_maps` | `noise_sphere.json` |
| Second split (D2) | A second, independent split (detector sets, years) to tell noise from sky | Each donor's reads split at random into two halves: a map of counting alone, available even for a cell with one donor | `stage_seq_atlas.noise_maps` (`--build seq-noise-cols`) | `noise_sphere.json` |
| Noise maps (D3) | Published noise maps beside the sky map | Both noise maps drawn on the sphere for every cell | `--build seq-noise-sphere` | `seq_noise_maps.jpg`, `noise_sphere.json` |
| 95 % range per value (D4) | Error bars on every pixel from the noise model | Each cell's level at each site given a range from its donors' spread, or from counting where the donors agree exactly | `stage_seq_atlas.range_of_average` (`--build seq-sphere-cols`) | `sphere_range_synthetic_v3.txt`, `sphere_widths_v3.txt`, `zero_widths_after.txt` |
| Masks and apodization (D5) | The galactic plane and point sources masked, with smooth mask edges so the mask itself adds no structure | The sex chromosomes (a real healthy difference between people) and probes that cannot be placed uniquely in the genome masked, with smooth edges | `stage_seq_atlas.sky_mask_sites` | `DEV_ATLAS_V21_01.md` |
| Power spectrum (D6) | The sky's power at each angular scale | The correlation of a map between two sites at each genome distance, and the map's power at each scale on the sphere | `stage_seq_atlas.genome_spectrum`, `sky_null` (`--build seq-noise-spectrum`, `--build seq-noise-null`) | `noise_spectrum_summary.txt`, `noise_null_summary.txt` |
| Cosmic variance (D7) | The floor set by the finite number of independent modes on the sky | The finite number of copies of a cell behind a reading: a cell's share times the reads at each site; below one copy per site a cell's reading is flagged | `stage_metaa_atlas.read_in_mix` (`copies_per_site`) | `copies_seq_synthetic.txt` |
| Covariance (D8) | The covariance of the sky between pixels | How healthy donors' levels move together between sites | the genome-distance spectrum gives its diagonal bands | `noise_spectrum_summary.txt` |
| Transfer function (D9) | What the pipeline does to a known simulated sky | A change planted into a real healthy sample's reads, read back by the chain, to measure how much of it the chain returns | `--dev-plant-territory`; `doors/data/DEV_TERRITORIES_01/seq_band_01/transfer_synthetic.py` | `DEV_TERRITORIES_01.md` |
| Null tests (D10) | Each statistic checked on maps that hold no signal | Each map checked against shuffles of itself; the counting map must stay inside its null | `stage_seq_atlas.sky_null` | `noise_null_summary.txt` |
| Frequency coverage (D11) | Several frequency bands, kept apart and cross-calibrated | Bisulfite and enzymatic sequencing kept apart; the sequencing atlas set beside the array atlas on the sites both measure | `stage_seq_atlas.bridge` (`--build seq-bridge`) | `bridge_v21.csv`, `bridge_summary.txt` |
| E and B modes (D12) | The two kinds of polarization separated | 5-methylcytosine separated from 5-hydroxymethylcytosine where oxidative data exist | not available in the inputs | |
| Higher-order statistics (D13) | Three- and four-point shapes, Minkowski functionals | The same shapes on healthy maps | not built | |
| Isotropy (D14) | The sky the same in every direction | Each chromosome's healthy mean against the others | `isotropy.py` | `isotropy.txt` |
| Blinding (D15) | Held-out skies and blind analysis | Held-out donors and laboratories reserved before the build and never read by it; every bar sealed before it is scored | `selfree_samples.csv`, `outside_data/` | `CHECKLIST_STATUS.md` |
| Reproducibility (D16) | Public pipelines that rebuild the published maps | The atlas rebuilt by the one command from committed code and listed inputs, byte for byte | the `--build` routes above | `DEV_ATLAS_V21_01.md` |

**The readings it serves.** Against this atlas the chain reads a sequenced sample's disorder at its own cell's identity sites (Met-A),
across the whole sample in its own cell mix, at the territories where cancer's early change shows, the cell shares (the
IAM-Deconvolver), the distance to the entropy ceiling, the report globe and information sphere, and a group of people against another
(Group-A). Each one's healthy test is in the records named above.
