set -e; J=hpc/7f845615-656d-44ed-8c3d-e118473f0980; D=iamrepo/Biological_Physics/MethylPhys/doors; cp $J/blocks_lowerGI_all.csv $J/blocks_lowerGI_stool.csv $J/tumour_raised_tiles.csv $D/data/ && cp $J/tumour_tiles.csv.gz $D/data/ && cp remote_jobs/stool/colon_blocks.py remote_jobs/stool/tumour_scan.py $D/data/ 2>/dev/null || true
cat > $D/DEV_STOOL_01_OUTCOME.md <<'EOF'
# DEV-STOOL-01 — can stool carry enough colon-lining molecules for IAM-A and Met-A? (development, 2026-10-02)

**Run.** Box 1. (a) Marker regions for lower-GI epithelium (colon + small intestine treated as one cell type, since stool selects both) against the
Loyfer 2023 atlas, two backgrounds: every cell type ("all") and the cells that shed into stool ("stool": blood, immune, gut stroma, liver, squamous).
Both polarities searched. (b) 2-kb tile scan of the early-onset CRC tumour/normal pairs (6 usable pairs) for where the tumour's copy-error rise sits.

**(a) Markers.**
| background | strict regions | loose regions | CpGs |
|---|---|---|---|
| all cell types | 0 | 4 | 30 |
| stool-relevant cells | 1 (13 CpGs) | 28 | 214 + 13 |
All found regions are methylated in lower-GI epithelium (median β 0.79–0.89) and low in the background (0.10–0.19); the opposite polarity returned none.
Merging colon with small intestine and narrowing the background to stool-relevant cells raises the usable set from 1 region (6 CpGs, earlier run)
to 29 regions, 227 CpGs. That is enough sites to pull molecules from a deep stool library; whether enough tumour molecules survive in stool is not
assessed here (needs a stool methylome).

**(b) Tumour tiles: not assessed at this depth.** Only 80 tiles reached ≥ 20 qualifying molecules in both tissues in ≥ 5 pairs; 64 of them are on
chr1/chr16/chr10 pericentromeric repeats. Pooled over those tiles the tumour copy error is 1.16× normal (0.0539 vs 0.0466), the same direction as
PROC-TUMOUR-01, but where the rise sits genome-wide cannot be read from these libraries. Needs deeper WGBS or pooling at gene/region level.

**Next.** Score the 29 stool-background regions in the tumour pairs (region-level pooling instead of 2-kb tiles); look for a public stool or
colonic-effluent methylome to count lower-GI molecules per gram.
EOF
cd iamrepo && U="https://x-access-token:${GITHUB_TOKEN}@github.com/hmahaffeyges/IAM-Validation.git"; git add -A Biological_Physics/MethylPhys/doors && git -c user.name="Heath W. Mahaffey" -c user.email="hmahaffeyges@users.noreply.github.com" commit -q -m "DEV-STOOL-01: 29 lower-GI marker regions (227 CpGs) vs stool-relevant cells; tumour 2-kb tile scan not assessable at this depth (80 tiles, mostly repeats)" && bash CANON/checked_push.sh -q "$U" HEAD:main 2>&1 | grep -iE "error|rejected|FAILED|blocked"; git log --oneline -1