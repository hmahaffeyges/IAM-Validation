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
