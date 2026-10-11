# Clean-room rebuild of the commissioned Met-A record (2026-10-11)

Fresh `git clone --depth 1` of IAM-Validation at 52c7b7f into an empty folder, an empty home folder (no caches) except the three Illumina
manifests, copied in because this sandbox blocks methylprep's download host; Stage 1 now checks their sha256 (`_check_pins`), so a copied file and a
downloaded one are the same input. methylprep 1.7.1. Inputs: the 91 Salas purified arrays fetched from GEO by the scripts themselves.
Commands, in order, exactly as `COMMISSIONING_NOTE_METAA_NEUTROPHILS.md` lists them (steps 1-4). Wall time: calibration 117 min on a laptop.

Result: every commissioned matrix rebuilt identical (noise gate 0.149; 6,000 identity sites, healthy mean and shrunk SD difference 0; C-score
baseline median 1.0103; composition markers and profiles difference 0). `git status` after the rebuild: no file of the repo changed. The one
documented difference (re-chosen-site diagnostic, README line 8) reproduces digit for digit as committed in `reproduce_floor_v13_output.txt`.
