# Book review: findings for the author (development record; the book is unchanged)
<!-- [RECORD] a review record: superseded values are quoted as history -->

Three reviewers, each with the canon, the commit history of every passage (GitHub API) and the chat history of every session; each item below was raised only after that background search. Limits they declared: no local git clone (the API gives the same commits); reviewer A searched the chats only for its highest-impact items (A1-A5, A7, A8, A14-A16); provenance before 2026-10-10 can show a folder-rename commit instead of the first authoring commit.

**Withdrawn after the history search:** the Met-A floor printed as 1e-7 (p6_06_gauge, p6_24_status, p7_11, p7_08_synthesis) is your ruling of 2026-10-09 ('2.5e-8 bits (1e-7 on the gauge)', commit d167970d); the one-gauge chapter prints the same floor as 8e-8. H_ref = 0.259 bits is correct (canon E_hold_meth 3.084 kT).

## 1. Forbidden language: 'modified Friedmann'

Your rule: IAM enters only through the entropy functional, never the Friedmann equation. Older text, but it breaks a standing rule.

**A7. `appendices/app_S_saturation.tex` line 108** (forbidden language; confidence high; in the book since 2fb600db, 2026-10-05)

> yields the modified Friedmann equation of the matter sector:

Evidence:
- Standing author rule (restated 2026-11-10, twice): "we do NOT ever modify the Friedmann equation"; "modified friedmann is strictly forbidden language because IAM enters in the entropy functional and NEVER in the friedmann".
- The sentence itself, two lines later, correctly states the background is unchanged and the term enters H_m (the matter-sector rate), not the background Friedmann equation -- so the phrase 'modified Friedmann equation' directly contradicts the very mechanism the paragraph describes.
- Not touched by any commit since 2026-10-10 in our diff set, so the forbidden phrase has survived every edit pass to this file.

Recommendation: Replace 'the modified Friedmann equation of the matter sector' with language naming it the matter-sector rate, consistent with the Law chapter's own naming (H_m^2, not a modified Friedmann equation); the background Friedmann equation is untouched.

Proposed text: yields the matter-sector rate seen by matter perturbations:

Question: Per the standing rule that 'modified Friedmann' is forbidden language, should this line (and the appendix's header comment at line 7, same phrase) be reworded to call Eq. eq:sat_Hm the matter-sector rate rather than a modified Friedmann equation?

**A8. `appendices/app_S_saturation.tex` line 7** (forbidden language; confidence high; in the book since 2fb600db, 2026-10-05)

> replaced by the computed ratios (Ch. ch:bridge uses 37 orders for the span of length); the modified Friedmann equation written for the matter sector

Evidence:
- Same forbidden phrase as line 108, in this file's header comment describing what Step 5 does.
<!-- [RECORD] a review record: superseded values are quoted as history -->
- Standing author rule (2026-11-10): 'modified Friedmann' is strictly forbidden language; IAM enters only through the entropy functional, never the Friedmann equation.
- Not touched by any commit since 2026-10-10.

Recommendation: Reword the comment to avoid 'modified Friedmann equation', e.g. 'the matter-sector rate equation (eq:Hm) written with the record term'.

Proposed text: (eq:Hm), with the background unchanged; the matter-sector rate equation carries the record term

Question: Same forbidden-language question as line 108 -- this is a build comment, not rendered prose, but should it still be corrected to avoid 'modified Friedmann' for consistency with the source?

**C4. `part2/p2_03_theory.tex` line 874** (forbidden language; confidence high; in the book since pre-2026-10-10 (baseline text; no commit since 2026-10-10 ha, before 2026-10-10)

> The contracted Bianchi identity ... is therefore satisfied, and the modified Friedmann equation is consistent with the Einstein tensor.

Evidence:
- Author's standing rule (restated twice on 2026-11-10): 'we do NOT ever modify the Friedmann equation' and 'modified friedmann is strictly forbidden language because IAM enters in the entropy functional and NEVER in the friedmann' -- the background H(a) is standard LCDM; the record term enters only the matter-sector rate H_m^2 seen by matter perturbations.
- This is also flagged independently in the lead's candidate list (candidate 6) for this exact file and line.
- The same chapter states the correct rule repeatedly elsewhere, e.g. line 413 'leaving the background Friedmann equation and photon sector unchanged', line 422-423 'the beta_m E(a) term cannot appear as a modification of the background Friedmann equation and instead enters only at the level of matter-sector perturbations', and line 445 'The beta_m E(a) term does not modify the global background' -- so line 874's unqua
- No commit since 2026-10-10 touched any 'Friedmann' text in this file (checked against the full commit/patch history supplied for this file), so this is pre-existing baseline text that the 2026-11-10 ruling newly contradicts.

Recommendation: Rephrase to avoid calling the result 'the modified Friedmann equation'; state instead that the Bianchi identity is satisfied for the matter-sector perturbation equation (H_m^2), consistent with how the rest of the chapter phrases this (e.g. 'the background Friedmann equation... unchanged').

Proposed text: The contracted Bianchi identity $\nabla_\mu G^{\mu\nu}=0$ is therefore satisfied, and the matter-sector perturbation equation is consistent with the Einstein tensor.

Question: This line calls the result 'the modified Friedmann equation', which conflicts with your 'never modify Friedmann' rule and with the rest of this same chapter's careful background-vs-matter-sector-perturbation phrasing -- OK to reword to 'the matter-sector perturbation equation' (or similar) here?

Also the checker's own title for eq. law_F2info in verify_book.py ('modified Friedmann equation including record term'), copied into the generated VERIFY_BOOK_INVENTORY.md and verify_book_output.json. Not a violation: p2_03a's table, where the phrase describes Barrow/Tsallis models.

## 2. Numbers left behind by the holding-energy correction (3.41 to 3.08 kT)

Commit 862930f2 rewrote E_hold, phi, eps0 and H_ref across the book; these passages kept the superseded values. The canon's retired-pattern list names 0.032 explicitly.
<!-- [RECORD] a review record: superseded values are quoted as history -->

**A1. `part6/p6_01_bridge.tex` line 199** (number differs from canon; confidence high; in the book since 862930f2, 2026-10-19)

> Deriving that energy, the cell's efficiency $\phi=0.16$, from the writer's discrimination measured outside the cell is the framework's central open problem

Evidence:
- Canon iam_canon.json: phi = {value: 0.1473, derivation: 'E_hold_meth / M_cell'}.
- Recomputed: 3.084/20.94 = 0.1473, matching canon; 3.41/20.94 = 0.1628, matching the OLD superseded E_hold (3.41 kT).
- Commit 862930f2 (2026-10-19, 'Holding energy on the commissioned counter') edited this exact paragraph, changing E_hold 3.41->3.08 kT and eps0 0.032->0.044 two lines above and below, but left phi=0.16 untouched -- the token was not updated to match the corrected holding energy.
- Chat history (frame 3ebe9d93, 2026-11-10): 'the commissioned counter's value (CANON constants E_hold_meth 3.08 kT, eps0_meth 0.0438, phi 0.147, H_ref_cell 0.259 bits)' confirms 0.147 is the current commissioned value.

Recommendation: Update phi=0.16 to phi=0.147 (or 0.1473) to match the corrected holding energy already printed two lines above in the same paragraph.

Proposed text: cell's efficiency $\phi=0.147$, from the writer's discrimination measured outside the cell is the framework's central open problem

Question: The 10-19 holding-energy correction updated E_hold and eps0 in this paragraph but left phi=0.16 (the old value's ratio). Should phi here read 0.147 to match?

**C1. `appendices/app_N_notation.tex` line 113** (number differs from canon; confidence high; in the book since 862930f2, 2026-10-19)

> $\varepsilon_0$ & healthy reference held error & 0.032 & $1/(1+e^{\varphi\mathcal M})$: form \derived, height \measured

Evidence:
- CANON/iam_canon.json constants.eps0_meth.value = 0.04376 (displayed rounded as 0.044 in CANON/foundation.json entries 5 and 9, and in app_A2_frozen_values.tex: 'E_hold, eps0 & 3.08 kT, 0.044').
- CANON/iam_canon.json /retired/ section explicitly lists the pattern '(varepsilon_0|eps0|...)...0.032' with replaced_by text pointing to the commissioned counter's value (E_hold_meth 3.08 kT, eps0_meth 0.0438, phi 0.147, H_ref_cell 0.259 bits) -- 0.032 is the pre-2026-10-19 superseded value.
- Commit 862930f2 (2026-10-19, 'Holding energy on the commissioned counter') edited the immediately adjacent rows of this same table in app_N_notation.tex -- E_hold 3.41->3.08, phi 0.1628->0.147, Href=H(eps0) 0.2043 bits->0.259 bits -- but left this eps0 row's value at the old 0.032, which is only consistent with the old Href=0.2043 bits, not with the 0.259 bits now printed one row below it (recomputed: H(0.032)=0.2043
- The automated canon retired-pattern regex requires the stale value within 18 characters of 'varepsilon_0'/'eps0'; the gap in this table row ('$ & healthy reference held error & ') exceeds that window, which is consistent with why the stale-check gate did not catch it.

Recommendation: Update eps0 in this row (and the matching row below) from 0.032 to 0.044 (or 0.0438) to match CANON/iam_canon.json eps0_meth and to be internally consistent with the Href=0.259 bits already printed in the next row of the same table.

Proposed text: $\varepsilon_0$ & healthy reference held error & 0.044 & $1/(1+e^{\varphi\mathcal M})$: form \derived, height \measured
<!-- [RECORD] a review record: superseded values are quoted as history -->

Question: Confirm this eps0=0.032 row was simply missed by the 2026-10-19 holding-energy correction (it sits right next to Href=0.259 bits, which only follows from eps0~0.044, not 0.032) -- OK to fix this single targeted entry to 0.044?

**C2. `appendices/app_N_notation.tex` line 274** (number differs from canon; confidence high; in the book since 862930f2, 2026-10-19)

> $\varepsilon_0$ (cell) & healthy reference held error, $1/(1+e^{\varphi\mathcal M})=0.032$ & --- & Ch.~\ref{ch:iama}

Evidence:
- Same superseded value as the Appendix N 'Constants of the cell and of the gauge' table row (line 113): CANON/iam_canon.json eps0_meth.value = 0.04376, and this exact literal pattern is in CANON's /retired/ list as replaced by the commissioned counter's value.
- This is the book's quick-reference notation table (Appendix N, Letters/symbols by chapter); the same symbol is given correctly elsewhere in the book (e.g. app_A2_frozen_values.tex: 'eps0 & ... 0.044') and incorrectly here, so the two cannot both be right.
- Chat history (host.frames, searched across all project frames): the eps0=0.032 / E_hold=3.41 kT / phi=0.1628 / Href=0.2043 bits set is the PRE-2026-10-19 canon value, confirmed present verbatim in CANON/iam_canon.json and book text throughout 2026-10-02/10-03/10-04 sessions (e.g. 'eps0_meth {"value": 0.032 ...}', 'OK eps0 = 1/(1+e^3.41): 0.0319844 (printed 0.032)'); the 2026-10-19 commit set (fc423ca6/862930f2, 'Hold

Recommendation: Update to 0.044 (or 0.0438) to match CANON and the rest of the book.

Proposed text: $\varepsilon_0$ (cell) & healthy reference held error, $1/(1+e^{\varphi\mathcal M})=0.044$ & --- & Ch.~\ref{ch:iama}

Question: Same fix as the Appendix N table row above -- one targeted edit, 0.032 to 0.044, in this second location too?

## 3. The floor and the healthy reference used for each other

Your ruling (d167970d): the floor is H_min from M alone; eps0 = 0.044 is the measured healthy reference, 'not a limit'. The gauge chapter, the IAM-A table and the canon follow it.

**A2. `part6/p6_01_bridge.tex` line 88** (conflicting definitions; confidence high; in the book since 862930f2, 2026-10-19)

> \measured{} The floor is $\varepsilon_0=0.044$, set by the energy that holds a mark at 310\,K (Eq.~\eqref{eq:eps0}).

Evidence:
- CANON/foundation.json entry 'h_min': 'H_min = H(eps_min), the floor set by M and temperature alone ... It is NOT the denominator of A and not 1: it sits below 1 at H_min/healthy reference.'
- CANON/foundation.json entry 'eps0': 'eps_0 = 1/(1+e^{phi M}) = 0.044 ... the holding energy measured across 56 healthy cell types' -- filed separately from h_min, not called a floor.
- CANON/foundation.json entry 'gauge_three_marks': 'BOTH ENDS ARE FIXED BY PHYSICS AND DERIVED: the floor by thermal kicks against one ATP ... THE MIDDLE -- the healthy reference -- is where the instruments differ' (eps0 is explicitly the middle, not an end/floor).
<!-- [RECORD] a review record: superseded values are quoted as history -->
- Author ruling, commit d167970d (2026-10-09, same-file history): 'Floor and healthy reference (author ruling 2026-10-09). The floor H_min is where thermal kicks win against one ATP per site ... 2.5e-8 [bits]' -- establishing H_min, not eps0, as the floor, one line before our review window.

Recommendation: Rename this sentence's subject from 'floor' to 'healthy reference': eps0=0.044 is the measured healthy reference (A=1), not the floor. The floor is H_min (far smaller, derived from M alone).

Proposed text: \measured{} The healthy reference is $\varepsilon_0=0.044$, set by the energy that holds a mark at 310\,K (Eq.~\eqref{eq:eps0}).

Question: This sentence calls eps0=0.044 'the floor', but the 10-09 ruling and p6_08_iama.tex's own table call eps0 the healthy reference and H_min (2.5e-8 bits, derived from M alone) the floor. Should this line say 'healthy reference' instead of 'floor'?

**A3. `part7/p7_08_one_gauge.tex` line 62** (conflicting definitions; confidence medium; in the book since fdf5a2f2, 2026-10-19)

> \paragraph{Where $H_{\min}$ comes from.} For the cell, the form of the floor is IAM's Law: a holding energy $E_{\rm hold}$ against $k_BT$ allows an error $\varepsilon_0=1/(1+e^{E_{\rm hold}/k_BT})$.

Evidence:
- This paragraph is titled 'Where H_min comes from' but gives the formula for eps0 (=1/(1+e^{E_hold/kBT})), not eps_min (=1/(1+e^M)); E_hold=phi*M is the measured holding energy that sets the healthy reference, not the floor.
- The SAME chapter's own symbol table two sections earlier (lines 17-18) correctly separates them: 'H_min & the floor & ... sits below 1' vs 'H_ref & ... healthy reference height ... measured, not a limit' -- an internal inconsistency within one file.
- Commit fdf5a2f2 (2026-10-19) edited this exact paragraph (E_hold 3.41->3.08 kT) without correcting the title/formula mismatch.
- CANON/foundation.json 'three_energies': 'eps_min = 1/(1+e^M) from M and temperature ALONE ... (the floor); eps_0 = 1/(1+e^{phi M}) ... (the middle, measured)'.

Recommendation: Either retitle the paragraph (e.g. 'Where the healthy reference comes from') or give the eps_min=1/(1+e^M) formula for H_min and present eps0's formula as the healthy reference, consistent with the chapter's own symbol table.

Proposed text: \paragraph{Where the healthy reference comes from.} For the cell, the healthy reference follows a holding energy $E_{\rm hold}$ against $k_BT$ allowing an error $\varepsilon_0=1/(1+e^{E_{\rm hold}/k_BT})$.

Question: This paragraph titled 'Where H_min comes from' actually derives eps0 (the healthy reference), not eps_min (the floor) -- conflicting with this same chapter's symbol table. Should the title/wording be corrected to 'healthy reference'?

**A4. `part7/p7_08_one_gauge.tex` line 147** (conflicting definitions; confidence medium; in the book since 0bdde9a8, 2026-10-14)

> \paragraph{A physical floor.} $\varepsilon_0=1/(1+e^{\varphi M})$ comes from the energy budget, not from a reference group.

Evidence:
- Same conflation as line 62 in this file: eps0 (healthy reference) is labeled 'a physical floor', contradicting the chapter's own symbol table (H_min is the floor; H_ref, from eps0, is explicitly 'measured, not a limit').
<!-- [RECORD] a review record: superseded values are quoted as history -->
- CANON/foundation.json 'eps0' entry never calls eps0 a floor; 'h_min' is the only entry labeled the floor.
- This section surrounds a commit (fdf5a2f2, 2026-10-19) that updated the adjacent 'Where H_min comes from' paragraph's numbers without touching this header, so the same conflation persists in two places in one chapter.

Recommendation: Retitle to avoid calling eps0 a 'floor', e.g. 'A physical reference.' -- eps0 comes from the energy budget (phi*M) but is the healthy reference (A=1), not the floor (H_min).

Proposed text: \paragraph{A physical reference.} $\varepsilon_0=1/(1+e^{\varphi M})$ comes from the energy budget, not from a reference group.

Question: Same floor/healthy-reference conflation as the paragraph above it in this chapter -- should this header say 'reference' rather than 'floor'?

**A5. `part3/p3_07_saturation.tex` line 27** (stale since newer result; confidence high; in the book since 8075d004, 2026-10-02)

> cells & chromatin surface, at the cell's architecture position $P$ & to be measured ... & floor $H(\varepsilon_0)$ from the measured holding energy (Part~\ref{part:4}) \measured

Evidence:
- CANON/iam_canon.json P_neutrophil_IAM_A: 'status: retired (2026-10-17): the position form of IAM-A; development flag only.'
- The SAME chapter's figure caption and surrounding prose (just below this table, commit 862930f2, 2026-10-19) were already rewritten to remove the P-form language: 'IAM-A $=\varepsilon/\varepsilon_{\rm pred}(L)$, each sample read against what its own loose share predicts' replacing the old '$P\,H(\varepsilon_0)$' form -- this table row alone was missed in that same-file correction pass.
- This row also calls H(eps0) 'floor', the same floor/healthy-reference conflation found elsewhere (eps0 is the healthy reference; H_min is the floor).

Recommendation: Update this table row to match the rest of the chapter: replace 'architecture position P' with the current self-reference description (eps_pred(L)), and 'floor H(eps0)' with H_min (or 'healthy reference H(eps0)' if that is what is meant).

Proposed text: cells & chromatin surface, read against the sample's own loose-share line $\varepsilon_{\rm pred}(L)$ & to be measured (breach and the cancer region on the gauge) & floor $\Hmin$ from $M$ alone; healthy reference from the measured holding energy (Part~\ref{part:4}) \measured

Question: This table row still uses the retired IAM-A position form (P) and the old 'floor = eps0' wording, even though the figure and prose right below it in the same chapter were updated to the current self-reference form on 2026-10-19. Should this row be updated to match?

## 4. Stale since the atlas-only rule and the new commissionings

Written before Met-A was commissioned against the atlas alone and the same-run tare was retired.

**B1. `part6/p6_14_atlas.tex` line 150** (stale since newer result; confidence medium; in the book since cb8cd31b (2026-10-13, 'Book atlas chapter and atlas README: , 2026-10-13)

<!-- [RECORD] a review record: superseded values are quoted as history -->
> The commissioned Met-A reading on isolated neutrophils does not read the atlas, and IAM-A's healthy position comes from its own sequencing files

Evidence:
- GitHub commits API (path=docs/book/part6/p6_14_atlas.tex): this sentence was introduced verbatim by commit cb8cd31b on 2026-10-13 and has not been touched by any of the 20 commits to this file since (last touch 2026-11-08, commit 09416c44, which only edited the 'two telescopes' table a few lines above, not this sentence).
- host.frames chat search for 'does not read the atlas' (roots_only=False, 140 frames searched) surfaces an EARLIER, DIFFERENT self-contradiction about this same passage from a 2026-10-02 read-through (frame 8aa96196: 'p4_14 says chain v3 does not read the atlas (lines 76 and 87), then says it reads it for neutrophils (lines 214-215)'); that contradiction was resolved by cb8cd31b adding the qualifying clause 'nor its s
- No chat hit (searched 'atlas-only rule', 'two telescopes', 'does not read the atlas') shows the author discussing or ruling on whether this sentence still holds after Met-A's atlas-only commissioning (2026-11-09, commits 9afecf6f/c3666e69, both dated AFTER cb8cd31b and after the last edit to this file); CANON/status_facts.json's metA_commissioned fact (since 2026-11-09) is the newer result this sentence was never che
- CANON/status_facts.json metA_commissioned fact: 'Met-A against IAM-Atlas v2.1 alone is commissioned on sorted neutrophils (450K) and sorted monocytes (EPIC v1 ... ; 450K ...); first commissioned on isolated neutrophils (EPIC v1) against a same-run tare, now history'.

Recommendation: Qualify the sentence to name the specific (now-historical) commissioning it describes, consistent with how p6_24_status.tex and p6_09_cscore.tex already phrase the same transition.

Proposed text: The map is used only where the sky has foregrounds. Met-A's first commissioning, on isolated neutrophils against a same-run tare (now history), did not read the atlas; the current atlas-only commissioning (sorted neutrophils and sorted monocytes) does. IAM-A's healthy position comes from its own sequencing files; a new atlas version therefore leaves the IAM-A commissioning untouched.

Question: This sentence was deliberately revised once already (2026-10-13) to fix a different self-contradiction (whether chain v3 reads the atlas at all), but was never revisited after Met-A's atlas-only commissioning went live on 2026-11-09. Should it now be qualified to say this describes the historical, not the current, Met-A commissioning?

**B3. `part6/p6_12_instrument.tex` line 86** (stale since newer result; confidence medium; in the book since 52c22649 (2026-10-18, 'Book audit pass 2: C-score, translati, 2026-10-18)

> A physical instrument is zeroed against a known input measured the same way. The chain does this for Met-A with healthy reference specimens of the same type, run on the same slide, else in the same batch, through the same chain (Stage~T).

Evidence:
- GitHub commits API (path=docs/book/part6/p6_12_instrument.tex): only 4 commits exist for this file ever (d46e34ce 2026-10-04 folder rename, 52c22649 2026-10-18 wording pass, c1c20a2d 2026-11-04 physics-tie paragraph only). Nothing in this file's history touches the atlas-only rule introduced 2026-11-08/09.
- p6_19_chain.tex's own 'Stage T: the tare' section (same Part) was rewritten by commit 09416c44 (2026-11-08, 'Book, SOP, operating manual, Decoder and chain workflow outline brought to the atlas-only rule: Stage T and the same-laboratory tare marked retired') to open: 'Retired. Healthy comes only from the atlases ... Stage T stays in the code only so the first commissionings' records still rebuild; no new reading uses
- host.frames search for 'Stage T.{0,10}retired' in book prose and for 'instrument chapter' + tare (140 frames) returns zero hits of any author discussion of this chapter specifically; the only hits for the section's own wording ('references run beside the specimen') are from 2026-10-02/10-03, i.e. the chapter's original drafting, before the construction or the atlas-only rule existed.
- CANON/status_facts.json metA_commissioned fact (since 2026-11-09) confirms Met-A's current commissioning is against IAM-Atlas v2.1 alone, not a same-run tare.

Recommendation: Add the same disclosure used in p6_19_chain.tex's Stage T section (same-run tare retired, kept in code only to rebuild earlier commissioning records, superseded by the atlas-only healthy prediction) at the top of this chapter's tare section.

Proposed text: A physical instrument is zeroed against a known input measured the same way. The chain first did this for Met-A with healthy reference specimens of the same type, run on the same slide, else in the same batch, through the same chain (Stage~T); Stage T is now retired for new readings (healthy comes only from the atlases) and is kept in the code only so earlier commissioning records still rebuild (Chapter~\ref{ch:chain}). The chain now divides each reading by its own healthy prediction instead. The historical same-run references were part of the run, not a stored group, and the specimen was neve

Question: This chapter (the book's main description of intake/calibration/tare) has not been touched since 2026-11-04 and still presents Stage T's same-run tare as the chain's current Met-A calibration, unlike the parallel section in the chain chapter which was updated 2026-11-08 to call it retired. Should the same disclosure be added here?

<!-- [RECORD] a review record: superseded values are quoted as history -->
**A14. `part7/p7_08_one_gauge.tex` line 104** (stale since newer result; confidence high; in the book since unknown (not edited since 2026-10-23), on or before 2026-10-23)

> Met-A is commissioned on isolated neutrophils (EPIC v1) and IAM-A on healthy CD8 T cells read by whole-genome sequencing; other cells are added one at a time, each after its own sealed test.

Evidence:
- Standing commissioning status given in this review's own task brief: 'Commissioned now (2026-11-09/10): Met-A against IAM-Atlas v2.1 alone on sorted neutrophils (450K) and sorted monocytes (EPIC v1 and 450K)'.
- p6_07_meta.tex (same Part A fileset, updated 2026-11-10, commit 9afecf6f): 'Met-A is commissioned against IAM-Atlas v2.1 alone on sorted neutrophils (450K) and sorted monocytes (EPIC v1) ... It was first commissioned on isolated neutrophils (EPIC v1), against a same-run healthy tare since retired.'
- p7_08_one_gauge.tex has had no commit since 2026-10-23, predating the 2026-11-09/10 atlas-based re-commissioning milestone by over two weeks; this line still states only the FIRST/historical commissioning as the complete current status.

Recommendation: Update to reflect the current commissioning scope: Met-A against IAM-Atlas v2.1 alone on sorted neutrophils (450K) and sorted monocytes (EPIC v1 and 450K), noting the isolated-neutrophil/EPIC-v1 reading as the earlier, now-superseded commissioning.

Proposed text: Met-A is commissioned against IAM-Atlas v2.1 alone on sorted neutrophils (450K) and sorted monocytes (EPIC v1 and 450K), and IAM-A on healthy CD8 T cells read by whole-genome sequencing; other cells are added one at a time, each after its own sealed test.

Question: This line states Met-A's commissioning scope as only isolated neutrophils (EPIC v1), which is now the superseded first commissioning. Should it be updated to name the current IAM-Atlas v2.1-based commissioning (sorted neutrophils 450K, sorted monocytes EPIC v1 and 450K)?

**A15. `part7/p7_09_open.tex` line 77** (stale since newer result; confidence high; in the book since unknown (not edited since 2026-10-23), on or before 2026-10-23)

> \textbf{I1. Met-A in whole blood.} Met-A is commissioned on isolated neutrophils. On whole blood the chain's C-score and its blood-cell shares are commissioned; Met-A's own whole-blood reading is printed as preliminary.

Evidence:
- Same issue as p7_08_one_gauge.tex line 104: this open-items table states Met-A's commissioning scope as only isolated neutrophils, omitting the current IAM-Atlas v2.1-based commissioning on sorted neutrophils (450K) and sorted monocytes (EPIC v1, 450K).
- p7_09_open.tex has had no commit since 2026-10-23, predating the 2026-11-09/10 commissioning milestone.

Recommendation: Update the description of Met-A's current commissioning scope to match p6_07_meta.tex's current wording before restating the open whole-blood question.

Proposed text: \textbf{I1. Met-A in whole blood.} Met-A is commissioned against IAM-Atlas v2.1 alone on sorted neutrophils (450K) and sorted monocytes (EPIC v1, 450K). On whole blood the chain's C-score and its blood-cell shares are commissioned; Met-A's own whole-blood reading is printed as preliminary.

Question: Same commissioning-scope staleness as p7_08_one_gauge.tex -- should this open-item description be updated to the current atlas-based commissioning before naming what remains open (whole blood)?

**A16. `part6/p6_00b_astrogenetics.tex` line 276** (stale since newer result; confidence medium; in the book since unknown (not edited since before 2026-10-10), pre-2026-10-10)
<!-- [RECORD] a review record: superseded values are quoted as history -->

> The interval on a reading has three parts, the precision of the reference itself, the spread of the same-run tare and the detection limit set by the cell's fraction of the specimen

Evidence:
- Standing rule (this review's task brief): 'Same-run tares and self-tares are RETIRED.'
- Chat history (2026-11-10 ruling) and p6_24_status.tex both describe the same-run tare in the past tense as something already retired: '(Stage T's same-run tare retired)'; p6_07_meta.tex: '...against a same-run healthy tare since retired.'
- This paragraph presents 'the spread of the same-run tare' as a standing, ongoing component of the reading's current uncertainty budget (present tense, general methodology), not as a historical note -- contradicting the retirement.
- Not touched by any commit since 2026-10-10, so it was not updated when the same-run tare was retired (2026-10-1x per other files) or when Met-A was re-commissioned against the atlas alone (2026-11-09/10).

Recommendation: Update this sentence to describe the current uncertainty budget without the retired same-run tare, e.g. replacing it with the atlas-based reference uncertainty that supersedes it.

Proposed text: The interval on a reading has three parts: the precision of the reference itself, the spread of the atlas-measured reference (the same-run tare it replaces, now retired), and the detection limit set by the cell's fraction of the specimen.

Question: This paragraph names the same-run tare as a standing part of the current uncertainty budget, but the same-run tare is retired elsewhere in the book. Should this be updated to describe the atlas-based reference uncertainty that replaced it?

**A18. `part6/p6_06b_toolbox.tex` line 22** (stale since newer result; confidence low; in the book since a3058af2, 2026-10-15)

> against IAM-Atlas v2.1, on cheap and plentiful arrays. Commissioned on isolated neutrophils. Each cell is read at its own identity sites

Evidence:
- In the same Met-A toolbox entry, two lines below (line 24), the 'Status' field was updated on 2026-11-10 (commit 9afecf6f) to: 'Commissioned (tested and commissioned against IAM-Atlas v2.1 alone on sorted neutrophils (450K) and sorted monocytes (EPIC v1) (first commissioned on isolated neutrophils, EPIC v1, against a same-run tare)).'
- The mid-paragraph phrase 'Commissioned on isolated neutrophils' at line 22 was not updated in that same 2026-11-10 commit, so the entry states the current commissioning scope inconsistently within itself.

Recommendation: Update or remove the standalone 'Commissioned on isolated neutrophils' sentence in the 'How it reads' paragraph so it does not read as the complete current commissioning scope, given the fuller and more current Status field two lines below.

Proposed text: against IAM-Atlas v2.1, on cheap and plentiful arrays (commissioned against the atlas alone on sorted neutrophils and monocytes; see Status). Each cell is read at its own identity sites

Question: The Met-A entry's 'How it reads' text still says only 'Commissioned on isolated neutrophils', while the Status field two lines below was updated on 2026-11-10 to the fuller atlas-based scope. Minor inconsistency -- worth a touch-up for consistency?

Also (my check, not a reviewer's): p6_06_gauge L23/L37 and p6_22_leukocyte L16/L19 say the printed state uses 0.95-1.05 until the measured widths replace it. Since the chain change of 11-08, each cell's printed state uses its measured width, but the report's whole-sample gauge picture still draws 0.95-1.05 (report_v4.py line 200). The sentence is half true; the picture is a chain item to fix.
<!-- [RECORD] a review record: superseded values are quoted as history -->

## 5. Retired constructions and terms still described as current

**B2. `part6/p6_19_chain.tex` line 80** (stale since newer result; confidence high; in the book since pre-2026-10-02 (present in the earliest indexed chat/book ve, pre-2026-10-10)

> the variance of 50-site block means, times 50, over the variance of the sites, divided by the healthy baseline 1.1104 (Chapter~\ref{ch:cscore}). It is printed with every reading, with no band.

Evidence:
- GitHub commits API (path=docs/book/part6/p6_19_chain.tex, 16 commits total): commit 8dc258ed (2026-10-18, 'Book audit pass 2: chain (stage table C-score values), report (v4), reach chapters (OTH-072)') updated ONLY the Table~tab:p4_gates row for Stage MC, from 'blocks of 50 sites; enough blocks; baseline 1.1104' to 'blocks of 10 sites; enough blocks; baseline 1.0103; band 0.877--1.152, read one-sided' -- its patch do
- p6_09_cscore.tex (the chapter this paragraph cross-references) was corrected to the 10-site/1.0103 construction on 2026-10-15 (commit e8cbf14d, 'Checkpoint: C-score front-page story; book blocks-of-50 text corrected') and gained the 0.877-1.152 band the same day (commit fe17c068).
- host.frames chat search for 'Stage MC' and 'C-score construction' (140 frames) returns only pre-2026-10-09 snippets describing the original 50-site/1.1104 construction, i.e. the state of the book BEFORE the 2026-10-09 construction change; no later message discusses or defends leaving this specific prose paragraph unupdated while the table beside it was fixed.
- The commit message of 8dc258ed itself names exactly the two places it touched ('stage table C-score values'), confirming the prose paragraph was simply not in scope of that pass rather than deliberately left as-is.

Recommendation: Update the Stage MC prose paragraph to match the table immediately above it (already corrected 2026-10-18) and the C-score chapter (corrected 2026-10-15): blocks of 10 sites, baseline 1.0103, and the commissioned band (0.877-1.152 on whole blood).

Proposed text: The residual map over the identity sites, in genome order, is reduced to the C-score: the variance of 10-site block means, times 10, over the variance of the sites, divided by the healthy baseline 1.0103 (Chapter~\ref{ch:cscore}). It is printed with every reading, against the commissioned band on whole blood (0.877--1.152, one-sided). The IAM-A C-score is defined and not yet built.

Question: Commit 8dc258ed (2026-10-18) fixed the Stage MC row of the stage table in this same chapter but did not touch the Stage MC prose paragraph just below it, which still describes the retired 50-site/1.1104/no-band construction. Should the prose be brought into line with the table in the same pass?

**A13. `part6/p6_06_gauge.tex` line 41** (stale since newer result; confidence medium; in the book since unknown (pre-10-10; trailing clause edited in-window), pre-2026-10-10)

> C-score far end & Met-A C-score $45$ & \calc{} block size 50 over the healthy baseline 1.1104; the sequencing C-score was tried and does not hold on reduced-representation reads

Evidence:
- p6_09_cscore.tex (same book, cross-checked for context): 'Drawn with the construction as it first stood (50-site block means, healthy baseline 1.1104); the commissioned score uses blocks of 10 and the baseline 1.0103.' -- the commissioned C-score construction uses blocks of 10 and baseline 1.0103, not blocks of 50 and baseline 1.1104.
- The row immediately above this one in the same table ('Normal (Met-A)') explicitly flags its own superseded quantity ('The design tolerance 0.95-1.05 ... is superseded, and the printed state uses it until the measured widths replace it') -- showing the author's practice is to flag superseded constructions, which this row omits.
- This row's trailing clause was edited within the window (commit, changing 'the IAM-A C-score is not built' to 'the sequencing C-score was tried and does not hold on reduced-representation reads') without updating the block-size/baseline figures to the commissioned construction.

Recommendation: Either relabel this row's value as the pre-commissioning construction (as p6_09_cscore.tex does), or recompute the far-end value using the commissioned blocks-of-10/baseline-1.0103 construction.

<!-- [RECORD] a review record: superseded values are quoted as history -->
Proposed text: C-score far end & Met-A C-score $45$ (first construction: blocks of 50, baseline 1.1104; commissioned construction uses blocks of 10, baseline 1.0103) & \calc ...

Question: This row's C-score far-end value (45) comes from the pre-commissioning 50-site-block/1.1104-baseline construction, which p6_09_cscore.tex itself calls superseded by the commissioned 10-site-block/1.0103-baseline construction. Should this row note that, or be recomputed?

**A5. `part3/p3_07_saturation.tex` line 27** (stale since newer result; confidence high; in the book since 8075d004, 2026-10-02)

> cells & chromatin surface, at the cell's architecture position $P$ & to be measured ... & floor $H(\varepsilon_0)$ from the measured holding energy (Part~\ref{part:4}) \measured

Evidence:
- CANON/iam_canon.json P_neutrophil_IAM_A: 'status: retired (2026-10-17): the position form of IAM-A; development flag only.'
- The SAME chapter's figure caption and surrounding prose (just below this table, commit 862930f2, 2026-10-19) were already rewritten to remove the P-form language: 'IAM-A $=\varepsilon/\varepsilon_{\rm pred}(L)$, each sample read against what its own loose share predicts' replacing the old '$P\,H(\varepsilon_0)$' form -- this table row alone was missed in that same-file correction pass.
- This row also calls H(eps0) 'floor', the same floor/healthy-reference conflation found elsewhere (eps0 is the healthy reference; H_min is the floor).

Recommendation: Update this table row to match the rest of the chapter: replace 'architecture position P' with the current self-reference description (eps_pred(L)), and 'floor H(eps0)' with H_min (or 'healthy reference H(eps0)' if that is what is meant).

Proposed text: cells & chromatin surface, read against the sample's own loose-share line $\varepsilon_{\rm pred}(L)$ & to be measured (breach and the cancer region on the gauge) & floor $\Hmin$ from $M$ alone; healthy reference from the measured holding energy (Part~\ref{part:4}) \measured

Question: This table row still uses the retired IAM-A position form (P) and the old 'floor = eps0' wording, even though the figure and prose right below it in the same chapter were updated to the current self-reference form on 2026-10-19. Should this row be updated to match?

**A17. `part7/p7_11_status_all.tex` line 123** (stale since newer result; confidence medium; in the book since unknown (not edited since before 2026-10-10), pre-2026-10-10)

> the results use calibrated quantities (the cell references held fixed after one measurement), measured ones (the holding energy $E_{\rm hold}$ and the position $P$), and one factor found numerically

Evidence:
- CANON/iam_canon.json: P_neutrophil_IAM_A 'status: retired (2026-10-17): the position form of IAM-A; development flag only.'
- p7_08_one_gauge.tex (same Part A fileset, line 95): 'The earlier form, $H(\varepsilon)/(P_{\rm cell}H(\varepsilon_0))$ with a frozen per-cell position, is now a development flag.'
- This summary line lists 'the position P' alongside E_hold as one of the book's current measured inputs supporting the 'no free parameter' claim, but P is retired and is no longer part of the commissioned IAM-A reading (which now uses eps_pred(L), the self-reference line).
- Not touched by any commit since 2026-10-10.

Recommendation: Remove 'the position P' from this list of current measured quantities, or replace it with the current self-reference line parameters (the two constants of eps_pred(L)).
<!-- [RECORD] a review record: superseded values are quoted as history -->

Proposed text: the results use calibrated quantities (the cell references held fixed after one measurement), measured ones (the holding energy $E_{\rm hold}$ and the self-reference line $\varepsilon_{\rm pred}(L)$), and one factor found numerically

Question: This summary line cites 'the position P' as a current measured input, but P is retired (development flag only) and the commissioned IAM-A reading now uses the self-reference line eps_pred(L) instead. Should this be updated?

**A10. `appendices/app_S_saturation.tex` line 158** (stale since newer result; confidence medium; in the book since 2fb600db, 2026-10-05)

> It is measured on purified healthy cells of the same type, through chain v3, with nothing fitted

Evidence:
- app_C2_reproduce_cells.tex (cross-checked for context): 'The cell numbers of Part 4 come from chain v4' -- confirms chain v4 is the current chain.
- A deliberate book-wide audit commit (bcff11b8, 2026-10-19, 'Book audit: glossary on chain v4') updated many Part A files (p3_01b_bh_information.tex, p7_08_one_gauge.tex) and other parts to reflect chain v4; a later commit (c15538a2, 2026-11-05) explicitly fixed p6_01_bridge.tex because 'it named the v3 documents' -- confirms naming the current chain 'v3' is treated as an error needing correction.
- app_S_saturation.tex's Step 6 numbers (E_hold=3.08 kT, eps0=0.044, the self-reference figures elsewhere in this same file) are the CURRENT (chain-v4-era) canon values, yet the sentence describing how they were measured still names chain v3.
- This exact phrase was not touched by any commit since 2026-10-10, so it was missed by the chain-v4 audit pass that touched sibling files.

Recommendation: Update 'through chain v3' to 'through chain v4' to match the chain-v4-era numbers this chapter reports and the correction already made to the sibling bridge chapter.

Proposed text: It is measured on purified healthy cells of the same type, through chain v4, with nothing fitted

Question: The book-wide 'chain v3 -> v4' naming audit (and the 2026-11-05 fix to p6_01_bridge.tex) did not reach this line. Since the numbers this chapter reports are chain-v4-era, should 'chain v3' here read 'chain v4'?

**A11. `appendices/app_S_saturation.tex` line 9** (stale since newer result; confidence low; in the book since 2fb600db, 2026-10-05)

> Step 6 is written new from chain v3 (Biological_Physics/MethylPhys/chain, sop/MethylPhys_CPG_SOP_v3.md)

Evidence:
- Same chain-v3/v4 issue as line 158 in this file; p6_01_bridge.tex (commit c15538a2, 2026-11-05) was corrected from naming the v3 SOP/manual to naming the v4 SOP/manual for the identical kind of reference.
- This is a build comment (not rendered prose) but it documents provenance that is now out of date relative to the chain actually used for this chapter's current numbers.

Recommendation: Update the comment to reference the v4 SOP (sop/MethylPhys_CPG_SOP_v4.md), matching the correction already applied to p6_01_bridge.tex.
<!-- [RECORD] a review record: superseded values are quoted as history -->

Proposed text: Step 6 is written new from chain v4 (Biological_Physics/MethylPhys/chain, sop/MethylPhys_CPG_SOP_v4.md)

Question: This is a source comment, not rendered text -- worth updating for consistency with the v3->v4 correction made elsewhere, or left as a historical note since it refers to when Step 6 was first written?

**C3. `appendices/app_F_glossary.tex` line 368** (number differs from canon; confidence high; in the book since 04f71aeb, 2026-11-05)

> The ruler for sequenced samples: 57 healthy cell types at all 28.2 million CpGs of the genome, counted from healthy donors' reads...

Evidence:
- CANON/catalogue.json in_development[0] (IAM Sequencing Atlas): 'for each of 58 healthy cell types (megakaryocytes provisional), at every one of the 28.2 million CpGs...' -- current canon says 58, not 57.
- app_A_canon.tex (the appendix auto-generated directly from CANON/iam_canon.json) currently prints the same entry as '58 healthy cell types (megakaryocytes provisional)' -- matching canon exactly.
- Git history: commit 38be5ebd (2026-11-07, 'Johns Hopkins healthy CD4 test...') explicitly changed app_A_canon.tex's IAM Sequencing Atlas entry from '57 healthy cell types' to '58 healthy cell types (megakaryocytes provisional)', i.e. a megakaryocyte cell type was added to the atlas on 2026-11-07.
- The glossary's separate, hand-written 'IAM Sequencing Atlas' entry (app_F_glossary.tex line 368) was added on 2026-11-05 (commit 04f71aeb) with count 57, and was never touched again -- no commit on this file ever changed it to 58, so it was not updated when the atlas grew by one cell type two days later.

Recommendation: Update the glossary entry to 58 healthy cell types and add the '(megakaryocytes provisional)' qualifier to match app_A_canon.tex and CANON/catalogue.json.

Proposed text: \textbf{IAM Sequencing Atlas} & The ruler for sequenced samples: 58 healthy cell types (megakaryocytes provisional) at all 28.2 million CpGs of the genome, counted from healthy donors' reads, donor by donor, with nothing fitted, two noise maps and a 95~\% range at every CpG. IAM-Atlas v2.1 stays the ruler for arrays. In development; deposited with a DOI. {\footnotesize(Ch.~\ref{ch:seqatlas})}

Question: The Sequencing Atlas grew from 57 to 58 cell types (megakaryocytes, provisional) on 2026-11-07 per app_A_canon.tex, but the glossary's separate 'IAM Sequencing Atlas' entry still says 57 -- OK to bring it to 58 (megakaryocytes provisional) to match?

A5 appears in both 3 and 5 (one table row with both issues).

## 6. Derivation status labels (older text, presumed accurate; question only)

**A9. `appendices/app_S_saturation.tex` line 112** (status label wrong; confidence medium; in the book since 2fb600db, 2026-10-05)

> \derived{} given Assumptions~1 and~2 (Chapter~\ref{ch:iams_law}, Section~\ref{sec:law_caikim}, Eq.~\eqref{eq:Hm}).

Evidence:
<!-- [RECORD] a review record: superseded values are quoted as history -->
- app_C3_derivations.tex (same repo, cross-checked for context) premise table lists the record constraint as a separate item, P8, not covered by 'Assumptions 1 and 2' (Assumption 1 = Jacobson's dQ=TdS; Assumption 2 = S_info addition).
- app_C3_derivations.tex's own 'what the derivations do not yet give' list states explicitly: 'The exponential. rho_info ~ E(a) rests on the record constraint P8 ...; neither follows from the first law, which is linear in S_info. \openprob' -- i.e. the full E(a)-carrying form of Eq. eq:Hm is NOT simply derived from Assumptions 1-2; the exponential shape is an open problem.
- app_S_saturation.tex's own Eq.~eq:Hm (the line right after this one) already carries E(a), the exponential activation function, which per app_C3 needs the separately-tracked P8/record-constraint assumption, not just Assumptions 1-2.

Recommendation: Either cite P8 (the record constraint) alongside Assumptions 1 and 2 for Eq. eq:Hm, or split the status label: the linear structure is \derived given Assumptions 1-2, while the exponential form E(a) is \openprob per app_C3's own premise list.

Proposed text: \derived{} given Assumptions~1 and~2 for the linear structure; the exponential form of $E(a)$ additionally rests on the record constraint (Appendix~\ref{app:derivations}, P8), not yet derived from the first law alone \openprob.

Question: app_C3_derivations.tex lists the E(a) exponential as resting on P8 (the record constraint) and flags it \openprob, separate from Assumptions 1-2. Should this line's status label be split or qualified accordingly?

Not reviewed: app_C3 premise P10 (beta_m = Omega_m/2 for the whole ensemble) is labeled derived in the premise table while item 8 of the open list calls its ensemble use open. app_C3 was outside every reviewer's list (unchanged since 10-10), so it stays a question for you, not a finding.

## 7. Generated chapters: the source file is what would change

Never hand-edit these chapters; their generator rebuilds them.

**B4. `part6/p6_20c_reproduced.tex` line 65** (broken reference or build; confidence high; in the book since 537984a4 (2026-10-18, 'reproduce/: one-command reproduction , 2026-10-18)

> GSE110554; GSE167998; GSE2998xxx neutrophil arrays (listed in reference\_donors.csv)

Evidence:
- GitHub commits API confirms 'GSE2998xxx' was present in tests.json from the moment that file was first added (commit 537984a4, status 'added', patch line '+ "GSE2998xxx neutrophil arrays (listed in reference_donors.csv)"') and has never been edited in any of the 3 commits to that file since, nor in any of the 7 commits to the generated p6_20c_reproduced.tex.
- Biological_Physics/MethylPhys/doors/data/MET_A_FLOOR_V13_REPRO/reference_donors.csv (fetched directly, the file this sentence points the reader to) lists sample accessions GSM2998021, GSM2998057, GSM2998116, GSM2998023, GSM2998143, GSM2998030, ... -- GSM (sample), not GSE (series), accessions, all numbered in the '2998' range.
- host.frames search for 'GSE2998xxx' (140 frames) finds only the same placeholder string being echoed back verbatim in other generated artifacts/summaries, never a discussion of what the real accession is or a decision to use 'xxx' as a deliberate shorthand.
- 'GSE2998xxx' is not a resolvable GEO accession (GSE and GSM are different, non-interchangeable prefixes), so a reader trying to fetch 'the public data' this entry names cannot do so from this string alone.

Recommendation: Fix the accession placeholder in Biological_Physics/MethylPhys/reproduce/tests.json (the source this chapter is generated from, not the .tex file itself) from 'GSE2998xxx' to the correct GSM-prefixed accessions (or their parent GSE), then rebuild the chapter.

Proposed text: GSE110554; GSE167998; GSM2998021 et al. (the 2998-series neutrophil arrays listed in reference_donors.csv)

<!-- [RECORD] a review record: superseded values are quoted as history -->
Question: The reproduction chapter's public-data listing for the Met-A reference rebuild names 'GSE2998xxx', present since the file was first created on 2026-10-18 and never corrected; the actual donor file lists GSM2998021/GSM2998057/etc. (sample, not series, accessions). Should tests.json be corrected and the chapter regenerated?

**A12. `part7/p7_08b_bits.tex` line 123** (conflicting definitions; confidence medium; in the book since 13034791, 2026-10-16)

> \item[Cosmos] one unit of entropy (one nat) per minimum encoding area 4 l\_P\textasciicircum{}2 \hfill(\textit{calculated})

Evidence:
- app_S_saturation.tex (same Part A fileset), 'The floor and the ceiling': 'The floor is H_min ... A horizon has no floor.' and separately: 'The ceiling is the surface saturated at its full capacity ... The black hole is this end: a horizon written to the Bekenstein-Hawking bound.'
- The Bekenstein-Hawking bound (one nat per 4 l_P^2, the minimum encoding area / maximum entropy density) is exactly the quantity app_S_saturation.tex calls the CEILING for a horizon, not the floor -- the two Part A files disagree on which end of the gauge this quantity belongs to.
- This whole chapter is auto-generated ('Generated by CANON/book_bits.py from CANON/four_domains.json; do not edit by hand'); the CANON/four_domains.json source entry id='floor' defines the Cosmos slot as this same Bekenstein-bound quantity, so the inconsistency originates in the canon data file, not the .tex.
- This section ('Floor') was newly added to this file within the review window (commit 13034791, 2026-10-16, which created this whole chapter).

Recommendation: Do not hand-edit the .tex (generator note). Flag CANON/four_domains.json's 'floor' slot for 'cosmos': either remove the Cosmos row from the Floor table (a horizon has no floor, per app_S_saturation.tex) or move this entry to the Saturation/Ceiling table where the Bekenstein bound belongs.

Proposed text: (no direct .tex edit; fix source: CANON/four_domains.json slot id='floor', domain='cosmos' should be removed or reassigned to the ceiling/saturation slot)

Question: This generated table files the Bekenstein-Hawking bound (one nat per 4 l_P^2) under 'Floor' for Cosmos, but app_S_saturation.tex calls that same quantity the ceiling and states a horizon has no floor. Should the four_domains.json source move this entry to Saturation/Ceiling, or should the Floor row for Cosmos simply read '---' (no floor)?
