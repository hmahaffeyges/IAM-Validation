# The deposit manifest

Every file a test recalls, so the whole record can be re-run from a Zenodo deposit without any other source. Built by
`deposit_manifest.py` (`DEPOSIT_MANIFEST.csv`, one row per file; `DEPOSIT_MANIFEST.txt`, the totals). Files in the repository carry
their size and sha256. Inputs that live only in a public archive are staged in the project's Google bucket before the deposit, and their
checksums recorded as they are staged; the upload goes from the bucket to Zenodo, as a draft the author publishes. Nothing is uploaded until
the author says the package is ready. The manifest grows with every test: a new test's sample list is added to `LISTS`.
