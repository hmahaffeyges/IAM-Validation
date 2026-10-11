# DEV-TISSUE-COLON-01 — Met-A on colorectal tumour against the same patient's normal colon (development; design, 2026-10-11; NOT sealed)

**Data.** GSE131013 (colon, 450K, IDATs in S3 `downloads/C_tissue_tests/GSE131013/`): 48 healthy donors' colon mucosa, 96 adjacent normal, 96
tumours, all stage II (88 IIA, 8 IIB). Samples, plates and slides: `data/DEV_TISSUE_COLON_01/colon_samples.csv` (GEO annotations and the series'
Sentrix sheet). 91 patients have tumour and adjacent normal, each pair on the same slide (3 plates, 22 slides). Three arrays are flagged degraded
by the submitters (one normal, two tumours).

**Read before the seal:** healthy mucosa and adjacent normal arrays only (`fetch_colon.py normal`, `calib_colon.py normal`). Tumour arrays are
fetched and calibrated only after `sealed_colon_tissue.json` is committed; both scripts refuse otherwise. Lock tested both ways before use (no seal
and an uncommitted seal refuse; a committed seal opens), per LESSONS T11.

**Question (one).** Do patients' colorectal tumours read Met-A above their own adjacent normal, as a group, by a rule fixed in advance?
The second question this set can answer, whether adjacent normal already differs from healthy donors' mucosa (field effect), is a separate test,
designed after this one is scored.

**Reading (from DEV-TISSUE-BREAST-01, unchanged in form).** Met-A at composition-blind sites: sites whose atlas v2 value is within 0.05 of the
colon epithelium value in every cell of the colon panel (colon epithelium, colon fibroblasts, colon macrophages, smooth muscle, vascular
endothelium, CD4 and CD8 T cells, B cells, neutrophils, adipocytes). Each tumour read against its own patient's normal on the same slide.

**Synthetic tests required before any seal** (all on the real non-tumour arrays):
1. Spread of the normals and its sources (plate, slide, composition).
2. Composition shift alone must not cross the bar (<= 0.10): a normal blended with another donor's healthy mucosa (two real non-cancer arrays,
   the tissue mix changed, no methylation change); and with the atlas colon epithelium template.
3. Blind reading independent of each normal's fitted composition (|Spearman| < 0.2, p > 0.05).
4. Planted change found: 5 % change in the epithelial part found in >= 0.80 of stand-in pairs.
5. Power and false group calls on stand-in pairs (two different normals on one slide).
