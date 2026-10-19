"""Titles, descriptions and links for the researcher package and the book (2026-10-19, OTH-077). Run after zenodo_v4_box.sh, with the numbers it
reported, so every count on the record comes from the run. Usage: python zenodo_v4_metadata.py <commit> "<release check line>" <book checks passed>
<reproductions passed> <reproductions total>. Reads the Zenodo key from the environment (ZENODO*). American English; references by their work."""
import json, os, sys, requests
C, RC, BOOKN, RP, RT = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]); S = C[:8]
D = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "zenodo_v4_drafts.json")))
TOK = next(os.environ[k] for k in os.environ if "ZENODO" in k.upper()); A = {"access_token": TOK}; Z = "https://zenodo.org/api/deposit/depositions"
GH = "https://github.com/hmahaffeyges/IAM-Validation"; CR = [{"name": "Mahaffey, Heath W.", "affiliation": "IAMPerformance Inter-Domain Research Initiative"}]
pkg = dict(title="MethylPhys chain v4: the commissioned IAM cell instruments and the researcher package", version="4.0", upload_type="software",
  publication_date="2026-10-19", creators=CR, license="mit-license",
  keywords=["DNA methylation", "information thermodynamics", "Landauer principle", "EPIC array", "whole-genome bisulfite sequencing", "cell-type deconvolution",
            "HEALPix", "reproducibility", "IAM"],
  description=(
    "<p>One command reads a DNA methylation sample (an Illumina EPIC v1 array, or whole-genome bisulfite sequencing) through five instruments, each "
    "commissioned on its own test, written and sealed before its data was read:</p><ul>"
    "<li><b>Met-A</b>: neutrophils on EPIC v1, read against their own healthy profile.</li>"
    "<li><b>The C-Score</b>: whether a sample's departures from healthy are spread out or clustered, commissioned on whole blood and drawn on a HEALPix sphere in every report.</li>"
    "<li><b>IAM-A</b>: each sample read against its own reference, with no other specimen; commissioned on healthy blood CD8 T cells by sequencing.</li>"
    "<li><b>The IAM-Deconvolver</b>: blood-cell shares on arrays (neutrophils, monocytes, CD4 T and CD8 T cells).</li>"
    "<li><b>IAM-Atlas v2.1</b>: 79 purified healthy human cell types, commissioned use by use (published separately; see the related records).</li></ul>"
    f"<p><b>This record</b> holds the whole repository at commit {S} (IAM-Validation-{S}.zip), because the chain reads files across it. Before upload, "
    "this exact archive was unpacked on a fresh machine, given a new Python 3.11 environment from the chain's requirements file only, and passed "
    f"the chain v4 release check: {RC} (release_check_v4_from_archive.json).</p>"
    "<p><b>Start with README.md</b> (the researcher package: look, run, reproduce). <b>To run it:</b> unzip; in the folder run <code>git init &amp;&amp; git add -A &amp;&amp; git commit -m archive</code> (the release check lists "
    "files with git); create the environment from <code>Biological_Physics/MethylPhys/chain/requirements.txt</code>; then follow START_HERE.md. "
    f"The same commit is on GitHub: {GH}/tree/{C}.</p>"
    f"<p><b>Reproduction.</b> {RP} of {RT} headline results, the commissioning tests among them, were rebuilt from public data (GEO, ENCODE, the NCI "
    "GDC, ENA and Zenodo) with one command each on fresh clones, every compared number identical to the committed result (REPRODUCE_README.md, "
    "reproduce_verified.json). The reproduction runner compares against the committed files with git, so it runs from a clone.</p>"
    "<p><b>Also here, to read without unpacking:</b> Biological_Physics/MethylPhys/QUESTIONS_AND_ANSWERS.md (what a researcher will ask, answered from the record, with the "
    "open questions stated as open), the operating procedure (MethylPhys_CPG_SOP_v4.md), the operations manual (PDF), the five commissioning notes, "
    "and example reports on real healthy whole blood, real healthy CD8 T cells, and a labeled planted example. SHA256SUMS.txt lists every file.</p>"
    f"<p>Results from your own samples, including ones that disagree, are welcome through the repository's issue form: {GH}/issues/new/choose</p>"
    "<p>These are research instruments, not a diagnostic test.</p>"),
  related_identifiers=[
    {"identifier": "10.5281/zenodo.23441999", "relation": "requires", "resource_type": "dataset", "scheme": "doi"},
    {"identifier": "10.5281/zenodo.23422688", "relation": "references", "resource_type": "dataset", "scheme": "doi"},
    {"identifier": D["book"]["doi"], "relation": "isDocumentedBy", "resource_type": "publication-book", "scheme": "doi"},
    {"identifier": f"{GH}/tree/{C}", "relation": "isIdenticalTo", "resource_type": "software", "scheme": "url"}])
book = dict(title="IAM's Law and Order: The Actualization of Reality. The Cost of Recording It, and the Price of Maintaining It", version="2.0.0",
  upload_type="publication", publication_type="book", publication_date="2026-10-19", creators=CR, license="mit-license",
  keywords=["Landauer principle", "horizon thermodynamics", "cosmology", "CMB", "black holes", "quantum information", "DNA methylation",
            "information thermodynamics", "HEALPix"],
  description=(
    "<p>A book that follows one physical cost, k<sub>B</sub>T ln 2 per bit, from the cosmic horizon to the qubit to the genome. IAM starts from the "
    "derivation of Einstein's equations from horizon thermodynamics and adds the cost of actualizing information at the nearest encoding surface. "
    "Seven Parts cover IAM's Law and the virial theorem; cosmology and the CMB; black holes and horizons; particles and quantum records; qubits and "
    "semiconductors; the thermodynamics of the methylome; and the thirty-seven orders of magnitude, in size, that the law spans.</p>"
    "<p><b>Version 2.0.0</b> adds the first commissioned chain of cell instruments (Met-A, the C-Score, IAM-A, the IAM-Deconvolver and IAM-Atlas "
    "v2.1), each with the sealed test it passed, a step-by-step reading of real reports, and the reproduction of the headline results from public data.</p>"
    f"<p>Every equation carries a status label. Every derivation and calculated number is checked by code (docs/book/verify_book.py): {BOOKN:,} "
    f"checks, all passing on the full repository at commit {S} (verify_book_output.txt).</p>"
    f"<p>Read online: https://hmahaffeyges.github.io/IAM-Validation/ . Source: {GH} . Files: the PDF, the EPUB, and the LaTeX source.</p>"),
  related_identifiers=[
    {"identifier": D["package"]["doi"], "relation": "isDocumentedBy", "resource_type": "software", "scheme": "doi"},
    {"identifier": "10.5281/zenodo.23441999", "relation": "references", "resource_type": "dataset", "scheme": "doi"},
    {"identifier": f"{GH}/tree/{C}", "relation": "isSupplementedBy", "resource_type": "software", "scheme": "url"},
    {"identifier": "https://hmahaffeyges.github.io/IAM-Validation/", "relation": "isAlternateIdentifier", "scheme": "url"}])
book["related_identifiers"][0]["relation"] = "isSupplementedBy"
for rec, md in (("package", pkg), ("book", book)):
    r = requests.put(f"{Z}/{D[rec]['id']}", params=A, json={"metadata": md}, timeout=120)
    print(rec, D[rec]["id"], r.status_code, r.text[:200] if r.status_code >= 300 else r.json()["metadata"]["title"][:70])
