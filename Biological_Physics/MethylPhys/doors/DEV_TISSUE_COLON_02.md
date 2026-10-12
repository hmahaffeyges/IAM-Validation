# DEV-TISSUE-COLON-02 — colorectal replication on GSE199057 (development; layout only, 2026-10-11; NOT sealed, no array opened)

**Data.** GSE199057 (Ghosh, Schultz, Chan et al.; EPIC v1, raw IDATs), a laboratory with no shared GEO contributors with GSE131013.
Layout from GEO metadata and file names only (`data/DEV_TISSUE_COLON_02/layout_02.py`, output `layout_02_output.txt`).

**Finding before any design.** The tumours sit on slides of their own: no tumour shares a slide with its own normal. DEV-TISSUE-COLON-01
and DEV-TISSUE-BREAST-01 compared each tumour with its own normal on the same slide or plate, because the processing plate dominated the
spread on real breast normals (lesson T10). Here that comparison is not available, so the COLON-01 design cannot be repeated as it is.

**Before any seal, still to design and test on synthetic data:** a reading that does not depend on the slide, for example each slide tared on
its own non-cancer arrays where they exist, and its power on the slides that carry only tumours; or a tumour-against-healthy-donor reading
with slide spread measured from the normals. If no slide-independent reading passes its synthetic tests, this set is recorded as not suitable.
