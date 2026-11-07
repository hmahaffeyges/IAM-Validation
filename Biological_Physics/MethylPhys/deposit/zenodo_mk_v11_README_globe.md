# The Sequencing Atlas Globe, new version: the 58th cell, megakaryocytes

This version adds megakaryocytes from the IAM Sequencing Atlas version 1.1 (Zenodo 10.5281/zenodo.24029328). Every file of the version
before is carried here unchanged: the 57 cells' globe is exactly as before.

- `globe_megakaryocytes_slices.tar`: the megakaryocyte column at every CpG at NSIDE 2048 (one CpG to a pixel; level, low and high of its 95 %
  range, uint16 x 10,000, 65535 where no CpG has reads), built by `seq_globe_wg.py slice` from the column's 29 published pieces, each checked
  against the atlas deposit's SHA256 manifest first.
- `globe_wg_textures_megakaryocytes_n256.npz`: its Globe Viewer texture at NSIDE 256 (`seq_globe_wg.py assemble`).
- `globe_wg_noise_megakaryocytes.json`, `globe_wg_noise_maps_megakaryocytes_n256.npz`: both noise maps (donor halves: donor 1 against donor 2;
  read halves: every donor's reads split in two), the X and Y chromosomes masked, with the genome-distance spectrum and the sphere null tests
  (shuffles and sign flips), by `seq_globe_wg.py noise-part` and `noise-join`.
- `slice_report_megakaryocytes.json`, `CHECKSUMS_megakaryocytes.txt`, and the builder `seq_globe_wg.py`.

The column is provisional (two donors, about 4 reads a CpG): its range is wide and its donor-halves noise map holds the difference between two
people only. Rebuild: the steps in this record's main README, run on `pieces_megakaryocytes_29.tar` from the atlas version 1.1.
