# DOCS-AUDIT-01: every document checked for accuracy, generated from the canon where it can be, and consolidated

**Purpose (author):** search everything hand-written over the past month, confirm it is still accurate, weigh whether each fact it states can
be generated from the canon, and consolidate the documents so fewer copies can drift. The book review's method applies: each doubtful passage
is a flagged candidate with its evidence (canon, chain, git history); passages confirmed accurate are logged so they are not raised again;
nothing is changed until the author has seen it. Dated records (day logs, development log, door records, sealed preregistrations and outcomes)
stay hand-written and are never regenerated.

**Why now:** Met-A's 450K monocyte scope (COMMISSION-MONO450-01) reached the chain and the status fact but not the instruments, status and
README pages, because one generator (CANON/catalogue.py) held the scope as fixed text and the status fact pinned the old wording.

**Step 1 (reads only):** `data/DOCS_AUDIT_01/docs_inventory.py` on GitHub's machines against the full repository: which files the canon's
generators write (everything else is hand-written), every text document with its size, and the canon's own cross-document checks on the whole tree.

**Step 1, the inventory (GitHub explore run 42819987557, full repository):** the canon's 32 generators write 102 files (`data/DOCS_AUDIT_01/docs_generated.txt`); 4,873 text files
in all. Of the files changed in the past month (2,512 commits read), 101 are generated, 3,593 are records, data or code, and 159 are
hand-written documents (`handwritten_changed_30d.csv`): 77 book chapters (already reviewed and checked by verify_book), 10 top-level MethylPhys
documents, 7 manuals and SOP, 29 working notes and outputs, 2 website pages, 34 others. The canon's checks on the whole tree: number_trace
traces every number; doc_audit finds 2 missing paths (a run note naming TODO.md and PRIORITY_2026-10-13.md); status_check's findings are
box-run code records; canon_check finds warn-level retired terms in 121 files. Next: each of the 82 non-book documents read against the canon
(book method), with a generate, generate-the-facts, keep-as-record or merge recommendation per file.

**First fixes (the author agreed):** the whole-blood C-score's first commissioning (a healthy range from other studies' arrays) is kept as history,
as Met-A's first is (commissioning.json, STATUS.md); the Introduction's C-Score section states its atlas-only scope (sorted monocytes,
450K); IAM-A's catalogue entry says the baseline is read from the sample's own loose share on a line whose form follows from the cell's
design and whose two constants were measured once on 56 healthy cell types (foundation.json). The fundraiser's status paragraph is drafted
for the author. **Agreed structure:** one SOP (the operating manual; the OMs retired into it), one Decoder (the flowchart folded in when
finished), the book, the A-score database, Four Domains; the Q&A, the Introduction and the researcher papers hand-written; records kept;
working notes archived.
