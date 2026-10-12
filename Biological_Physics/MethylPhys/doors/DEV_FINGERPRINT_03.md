# DEV-FINGERPRINT-03 — liver and lung, read both ways (candidates; metadata only, 2026-10-12; NOT designed, NOT sealed)

**Why.** HepG2 (liver) and A549 (lung) were scored by the loss arm alone, in one batch, before the meter read both directions (DEV-FINGERPRINT-02):
loss was not detected at that meter's sensitivity, and their gain excess was measured only after the fact. Whether liver and lung cancer cells carry
the fingerprint is open. It is decided only on liver and lung data no version of the instrument has read, by the two-way reading, synthetic tests
first, the seal before any cancer file is opened, one pair at a time.

**Search (author request 2026-10-12).** GEO, SRA, ENA, ENCODE and GDC metadata; no read, .pat, beta or IDAT opened. Full table with contributors,
pairing and risks: `data/DEV_FINGERPRINT_03/candidates_search_2026-10-12.json`. Runs, sizes and sample counts re-queried by
`data/DEV_FINGERPRINT_03/verify_candidates.py` (output `verify_candidates_output.txt`); where the search's own counts differ, the re-query holds
(PRJNA762641 has 132 runs, 4.97 TB). Searches were keyword-based, so sets indexed under other terms may be missing.

**Arrays (Met-A, tumour against the same patient's normal; the breast and colon design; laptop).**
- Lung: GSE94785 (450K, every pair on one slide), GSE235414 (EPIC v1), GSE158422 (EPIC v1, squamous), GSE83842 (450K, every pair on one slide).
- Liver: GSE136319, GSE89852, GSE136380, GSE77269. Their IDAT names carry no slide ID, so same-slide pairing is not yet known.
- TCGA-LIHC, -LUAD, -LUSC: analysed earlier by the retired class-gauge method only; allowed with the one-line Prior reading disclosure (rule 8).

**Sequencing (both instruments, both arms; needs the box).** Liver: PRJNA762641 (HCC with paired adjacent liver; the largest), PRJNA984754 (HCC
with paired adjacent tissue), GSE70090 (Feinberg; liver and lung, tumour and normal, BS-seq runs within PRJNA287622). Lung: GSE70090 and GSE79799 are
small; ENCODE HAIB WGBS of A549 is the same laboratory as the RRBS already read, a different assay, and is the weakest choice.

**Open before any design.** Tumour tissue against adjacent normal is a mixture design the fingerprint arms have not been tested on; their synthetic
tests must plant tissue-mix changes, gain and loss, before any seal. One cancer at a time: liver or lung first, chosen by the author.
