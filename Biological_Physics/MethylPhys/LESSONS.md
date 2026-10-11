# MethylPhys — lessons learned (one index)

Every lesson that cost time, a result or trust, once: what happened, the rule, and whether code now enforces it. A lesson that only lives in a
log entry or in memory gets repeated. **Add a row the day a lesson is learned**; when code starts enforcing it, name the check.
Older, topic-specific lists stay where they are and are indexed here: chain (`chain/CPG_Lessons_Learned_2026-06-29.md`), atlas
(`atlas/v2/LESSONS.md`, `atlas/IAMAtlas_FLATNESS_LESSON.md`).

## Reproducibility and the record
| # | what happened | rule | enforced by |
|---|---|---|---|
| R1 | Constants (the eight class H_min; later the holding-energy job PROC-CHANNEL-01) were made in notebook cells and their scripts never committed; the box copy was later wiped | every number comes from a committed script; nothing lives only in a session | `CANON/repro_check.py` rules 1-2 (untracked files; notes with numbers and no script) |
| R2 | A note's numbers were typed from notebook cells even though its folder had scripts (array list, readability bound, a descriptive correlation; 2026-10-10) | every printed number must appear in a committed output of its test folder | `repro_check.py` rule 6 (number traceability) |
| R3 | Status statements ("not commissioned", the dropped 0.20 cut) stayed in the book, chain headers and printed reports after the status changed | each repeated status fact is held once | `CANON/status_facts.json` + `status_check.py` |
| R4 | Results typed by hand into the book and README drifted from their records | the development chapter and README Advancements are generated | `CANON/results_register.json` + `results_to_tex.py` (checked by `status_check.py`) |
| R5 | An outcome went unlogged; a day ended without a summary | every outcome is named in the log in the same push; every day ends with a Day summary | `repro_check.py` record rule |
| R6 | Citation details, author lists and table values were typed from memory (caught by review, more than once) | take citations from CrossRef/Europe PMC and values from the paper's own text or table, never from memory | partly: rule 6 for numbers in notes; bibliography not yet checked by code |
| R7 | A results reader deferred to an archived file instead of the document that superseded it | redirect readers to the current document, never the archive | not enforced by code |
| R8 | API results came back in a different order between runs, so a committed list did not regenerate byte for byte | sort every generated list on a full key | not enforced by code |

## Instruments and tests
| # | what happened | rule | enforced by |
|---|---|---|---|
| T1 | A window was predicted with the simple copy-error formula; Stage Q reads only about half that rise (it counts only isolated errors on mostly-methylated molecules) | predict through the instrument's own measured response (a planted loss on real molecules), never through a formula of what it should read | SOP (simulate first); not enforced by code |
| T2 | A reference-free RRBS reader turned single misreads into fake CpGs and read copy error several times too high; a cross-species reading made with it was withdrawn | a new reader must pass a known-answer test on real molecules before any result is read with it | SOP; not enforced by code |
| T3 | A coincidence noticed after both numbers were known (cells hold about twice the writer's single-step error) looked like a law; the sealed per-context test rejected it | a post-hoc pattern counts only after it is sealed and tested on data it was not drawn from | SOP |
| T4 | Kinetic rates measured in the same steady-state cells return the copy error by construction | check a prediction for circularity in the algebra before designing the test | not enforced by code |
| T5 | A dry run of the fingerprint scorer with healthy runs standing in for cancer falsely called a fingerprint | run every scorer on a no-effect case and on a planted-effect case before the real data | SOP; done for every sealed scorer since |

## Box and compute
| # | what happened | rule | enforced by |
|---|---|---|---|
| B1 | Box Run 9 (2026-10-10): bwa-meth calls bwa with a minimum alignment score of 40; ENCODE RRBS reads are 36 bases, so nothing aligned. The script logged each file "done", deleted the empty alignment and went on: about 80 box-minutes produced nothing | before a run, check read length against the aligner's minimum score; every pipeline step checks its own output and stops the run on failure (mapping rate, file exists) | `boxruns/run9_fp2/session9.sh` guards (mapping rate, .pat produced); not yet in other box scripts |
| B1b | Passing `-T 20` (two tokens) to bwa-meth: its parser took `20` as a second read file, ran paired mode and gave bwa a bare `-T`; every BAM was empty | pass bwa options to bwa-meth as one token (`-T20`) and check the logged bwa command | `synth_align.sh` asserts the logged command |
| B7 | A job ran past its clock because read simulation used one random genome lookup per candidate; the job was killed with nothing written | scan each chromosome once; run long box work detached with a DONE file, and a short waiting job | `synth_align.sh` run detached |
| B8 | A crash guard set without data (stop under 40 % mapped) stopped a sound run: 36-base bisulfite reads map 37-46 % at the chosen score, losing reads to ambiguity and the score floor, independently of methylation | set a guard from what it protects (broken alignment: under 20 %; too little to read: under 100,000 Stage Q opportunities) and record the measured rate | `session9.sh` |
| B2 | A box left running for days | every box session ends with an armed shutdown watcher; stop the box when idle | not enforced by code |
| B3 | A `pkill` pattern sent through a remote command matched its own command line, so the kill did nothing | stop remote processes with a kill-file script | not enforced by code |
| B4 | Installing alignment tools into the chain's environment broke the chain's packages | each tool stack in its own environment | box scripts use separate environments |
| B5 | A sparse checkout silently left out a new folder, so a commit missed files | add new folders to the sparse checkout before writing into them | `repro_check.py` rule 1 catches the resulting untracked files |
| B6 | A scratch-disk cleanup removed a genome file that a symlink still pointed to | pass reference files to scripts explicitly and check them (the per-context reader asserts its CpGs read CG) | per script |

## Code
| # | what happened | rule | enforced by |
|---|---|---|---|
| C1 | pandas read the string "NA" (a category label) as a missing value | read label columns with `keep_default_na=False` | per script |
| C2 | The session database truncates long cell sources (~2,000 characters) | read long sources in pieces | — |
