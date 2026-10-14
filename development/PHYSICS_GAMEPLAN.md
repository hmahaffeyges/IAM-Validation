# Physics game plan: an instrument that needs no atlas (2026-10-14)

Mahaffey and Marcel. KISS: physics first, the whiteboard first, data only to confirm what the math has already fixed.

## The goal in one line

Read a patient's sample against marks that come from physics and from the sample itself, not from a reference library of purified cells.
The instrument still reads the sample. What it drops is the atlas, the purified reference cells and most of the lab-to-lab calibration.

## Why tonight changed the plan

1. **The floor and the ceiling are already derived**, the same for every cell (thermal kicks against one ATP; one bit per site).
2. **The middle is measured** (3.41 kT) and its derivation is open. The half-and-half picture needs one premise proved: that the
   eraser's loss equals the writer's (r = 1).
3. **The cell's operating point follows its architecture.** Across 56 healthy cell types the held error tracks the share of the genome
   the cell holds loosely (rank +0.87, DEV-ARCH-01, exploration). A cell's architecture can be read from the cell's own pattern.
   If the line from architecture to held error is derived, every cell type's reference point comes from the sample itself.

That is the atlas-free instrument: **each sample carries its own reference.** A healthy sample sits on the line; a failing one sits off it,
the way a star off the main sequence tells you something has changed inside it.

## The plan

### Stage 1 · The half (whiteboard)

The question: why would the eraser's loss equal the writer's?
- Euler's rule: the share paid by the record equals the scaling degree. Black hole: energy grows as the square root of its bits,
  share one half (Smarr). Cosmic horizon: energy grows as its bits, share one (our October 2 result).
- For the cell: does a mark's holding come half from itself and half from its neighbors? If held energy grows as the square root of the
  marks held, the half is derived the black hole's way and r = 1 follows.
- Deliverable: the derivation written out in the book's form, each step tagged derived, conjecture or open, checked in code.
- Also settles the rival: "a quarter of the SAM drive" either follows from the same scaling or it does not.

### Stage 2 · The cell's main sequence (whiteboard, then one look at numbers we already have)

The question: what line connects a cell's architecture to its held error, from physics alone?
- Simplest form: a cell holds two kinds of territory, tight and loose. Its held error is the share-weighted sum of the two:
  eps = (1 - L) eps_tight + L eps_loose. Two endpoints, one line, no fitting.
- Derive the two endpoints from the gauge: what a fully held domain and a loosely held domain each cost to hold.
- Then one look, nothing new fetched: do the 56 healthy cells fall on the derived line? (Exploration on numbers already computed.)
- One confirmation worth its hour: the loose share read on another laboratory's data of the same cell types, so the match is not built in
  by reading both numbers from the same molecules.

### Stage 3 · The instrument (math, then code)

- A sample's reading = where it sits against its own line: its held error against what its own architecture says it should hold.
- No atlas, no purified reference cells: the line is physics, the architecture is read from the sample.
- Mixtures (blood, tissue) need a rule for a sample made of several architectures; worked out on the whiteboard first.
- Deliverable: one short function, written on the existing chain, with its planted test.

### Stage 4 · One clean test, then hand it over

- Development on what we already hold: the 219 tumor-normal pairs. Does a tumor sit off its own line, and its normal on it?
- Then one sealed test on a fresh set, rule written first, run once. One set, not a campaign.
- Package for Loyfer and Moss: a two-page method, the derivation, the code, and the one sealed result. Short enough to check in an afternoon.

## Rules we keep while doing it

- Every choice comes from physics (a law, a scaling degree, a derived endpoint). No knob gets turned to hit a measured number.
- The half is held, not copied: writing on SAM, erasing on oxygen and 2-oxoglutarate, independent of division.
- Status tags stay honest: derived, conjecture or open, so a reviewer sees exactly what rests on what.
- The book is the source; the papers point to the section.

## Running in the background (no attention needed)

- Atlas v2.1 fit on methylphys-fit-01 (192 cores, spot). The one-block check runs first, then the full fit, about 13-15 hours.
- DEV-ERASE-01 reads on methylphys-cpu-01. U1 read NOT MET (ratio 0.961); the other four units are recorded as they land.
