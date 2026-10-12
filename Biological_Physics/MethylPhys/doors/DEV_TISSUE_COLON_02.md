# DEV-TISSUE-COLON-02 — colorectal replication on GSE199057 (development; layout only, 2026-10-11; NOT sealed, no array opened)

**Data.** GSE199057 (Ghosh, Schultz, Chan et al.; EPIC v1, raw IDATs), a laboratory with no shared GEO contributors with GSE131013.
Layout from GEO metadata and file names only (`data/DEV_TISSUE_COLON_02/layout_02.py`, output `layout_02_output.txt`).

**Finding before any design.** The tumours sit on slides of their own: no tumour shares a slide with its own normal. DEV-TISSUE-COLON-01
and DEV-TISSUE-BREAST-01 compared each tumour with its own normal on the same slide or plate, because the processing plate dominated the
spread on real breast normals (lesson T10). Here that comparison is not available, so the COLON-01 design cannot be repeated as it is.

**Before any seal, still to design and test on synthetic data:** a reading that does not depend on the slide, for example each slide tared on
its own non-cancer arrays where they exist, and its power on the slides that carry only tumours; or a tumour-against-healthy-donor reading
with slide spread measured from the normals. If no slide-independent reading passes its synthetic tests, this set is recorded as not suitable.

**Inputs (2026-10-11).** `data/DEV_TISSUE_COLON_02/fetch_colon_02.py` fetches IDATs per sample from GEO. A tumour array is refused before any
request until `sealed_colon_02.json` is committed unchanged; the lock was tested three ways before any fetch (tumour refused with no seal and with
an uncommitted seal; allowed with a committed dummy seal, which was then removed). Only the 152 non-cancer arrays are fetched now.

**Design, written 2026-10-12 before any array of this set is read** (`data/DEV_TISSUE_COLON_02/colon02_synthetic.py`). The reading is COLON-01's
(blind-site Met-A, tumour against the same patient's normal); only the pairing changes. Because tumour and own normal are never on one slide,
the per-pair bar is built from pairs of two different patients' normals on DIFFERENT slides, so the slide-to-slide spread is inside the bar.
Tests 1-5 as COLON-01 (composition blend with a non-CRC normal from another slide), plus test 6: the NULL group call with the slide effect
doubled, because the tumours sit on slides of their own. If tests 2-6 do not all pass, this set is recorded as not suitable and is not sealed.

**Synthetic tests 1-6 run, 2026-10-12** (work machine 01, job 3; `data/DEV_TISSUE_COLON_02/colon02_synthetic.py`, outputs `colon02_synthetic_output.csv`,
`colon02_synthetic_summary.txt`; 150 non-cancer arrays, no tumour opened). Tests 2 (composition), 4 (planted 5 %) and 5 (power) passed.
**Test 3 (independence) failed:** the normals' Met-A follows their fitted epithelial share (Spearman -0.274, p 0.02). **Test 6 (doubled slide
offset) failed:** the group call reaches 0.072 under the null, above 0.05. The normals' Met-A differs by slide (Kruskal-Wallis p 2.42e-03), and
with no tumour on its own normal's slide, slide and cancer cannot be separated in this set.

**Decision, by the rule written before any array was read: GSE199057 is not suitable for this design and is not sealed.** Its tumour arrays stay
unread. **Not a milestone:** a design result (development).

