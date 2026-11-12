# Why the chain works: IAM's physics

Every reading this chain makes rests on one physical fact. A methyl mark on DNA is a stored record, and keeping a record against heat
costs energy. Landauer's principle sets the least energy that can hold one bit against thermal noise. IAM's Law is that this price
governs every physical record: a qubit, a transistor, the large-scale structure of the universe, and a methyl mark. Two consequences
make the instruments possible.

1. **The scale is derived, not learned.** The energy a cell spends holding a mark sets the least error it can reach: the floor. A site
   whose mark is lost entirely reads as a coin flip, one bit of disorder: the ceiling. Counting single DNA molecules in 56 healthy human
   cell types, the energy that holds a methyl mark is {{canon:E_hold_meth:.2f}} k_B T, about {{canon:phi:.3f}} of the ATP the cell pays to keep each site (M = {{canon:M_cell:.2f}} k_B T).
   A healthy cell sits at a measured place between the floor and the ceiling, so a reading in bits means the same thing in any
   laboratory, for any cell type, before anything has been trained. A candidate derives that place from M and the price of the methyl
   donor (one SAM costs about {{canon:SAM_cost_ATP:.1f}} ATP): restoring paid against thermal kicks gives a held error of
   {{canon:eps0_candidate:.4f}} and {{canon:E_hold_candidate:.2f}} k_B T, against the measured {{canon:eps_healthy_pool:.4f}} (candidate, not yet tested).
2. **Disorder is counted, and counting has exact statistics.** A sample's disorder at a site is the binary entropy of its measured level,
   and the disorder that heat and counting alone would give at that site's number of reads follows exactly. So every instrument sets a
   measured disorder beside an expectation derived from the physics and from the sample's own reads. Nothing is fitted to the sample.

**How each part of the chain uses it.**
- **Met-A** reads a sample's disorder at its cells' identity sites against the disorder healthy cells of the same kind carry there, on
  the floor-to-ceiling scale; 1 is healthy.
- **IAM-A** reads the held error on single molecules against what the cell's own loosely held share predicts, on a line whose form comes
  from the holding physics. The sample is its own reference.
- **The C-Score** asks whether departures from the expected disorder are scattered along the genome, as ordinary wear is, or clustered,
  as when a whole region loses its holding.
- **The territories** make the same comparison at the islands every healthy cell holds unmethylated and the domains some cell types hold
  loosely.
- **Coupling-A** asks whether neighboring marks on one molecule were lost together, against its counted floor and ceiling.
- **The tumor gauge** reads the distance from the measured disorder to the entropy ceiling.
- **Group-A** asks whether a whole group has moved against the healthy samples of its own series.
- **The atlases and the IAM-Deconvolver are the rulers.** They say which healthy cells a sample is made of and where healthy disorder
  sits for each, so the physics can be applied to each person's own mix of cells.
- **The library floor.** Every laboratory's chemistry adds a small rate of false "methylated" calls. The chain measures it at the sites
  no healthy cell marks, and reads it as a property of the library, not of the cell.
