# DEV-SPHERE-01 — The information sphere: the geometry of healthy order (design record, 2026-10-13; nothing read, no prediction sealed)

This is the thread from the conversation of 2026-10-12/13, written down so it drives the whole-genome sky build instead of
living in a chat. The author's framing is quoted, because the correction he made is the whole point of the note.

## What the map is about, in the author's words

> "you are thinking too linearly, to literally, comparing the strand of dna as the same as the 'meaning' of the whole of the
> dna itself. We are not looking at one strand of dna shaped into a sphere, we are looking at the 'information', the 'order'
> that is required to maintain that species. The clusters and distances and 'patterns' are more than just methyl tags on a
> strand of dna."

And the analogy he drew afterwards:

> "The brain is a web of electrical circuits with connections scattered throughout and the map of the brain is not literally
> measuring the hardware so much as its measuring the 'output' the information if you will. A brain is a tiny universe in that
> sense, and so is the genome."

The first version of this map - the DNA address wrapped onto a sphere - is the weak version, and the note says so plainly:
its curvature is our filing system, not the biology. Two stronger versions follow, and they are different claims.

## Version A: the information sphere (established mathematics, not an analogy)

Information already lives on a sphere. This is information geometry, and it sits directly next to what both instruments read:

- For a site with methylation level b, the pair (sqrt(b), sqrt(1-b)) lies on a circle of radius 1 for every b.
- The natural distance between two states of that site - how distinguishable they are **as information** - is the angle
  between them on that circle: the Fisher-Rao distance.
- For a whole methylome, the same square-root step puts the entire pattern on a high-dimensional sphere. The entropy H(b) the
  instruments measure is defined on that same space.

**What that changes about the map.** A site's place would come from its behavior, not its address:
- sites that behave alike across healthy cells, people and cell types sit close together; independent sites sit far apart;
- a cluster is then a group of sites **held in order together** - the coordinated structure the author is describing;
- distances mean informational differences;
- healthy order has a shape, and disorder in aging or cancer shows as that shape blurring or tearing, read with the same
  sphere tools the microwave sky uses: how much structure at each scale, how connected, how big the patches.

In this version the sphere tools measure the order, not our filing system.

## Version B: the nucleus sphere (real curvature, measurable today)

The genome is not only a line. It is folded inside a nucleus that is roughly a ball, and position matters: stretches pressed
against the nuclear wall are mostly switched off, stretches near the center are mostly active, and in many cells including
cancers the wall-side stretches are exactly where large blocks of methylation are lost. Biologists already measure that
position (contact maps, and which stretches touch the wall). A sphere drawn from where DNA actually sits is a sky of the
nucleus with curvature the biology really has.

## Why the microwave sky is drawn on a sphere at all (the part that carries over, and the part that does not)

Carries over: splitting a map into patterns of each angular size (spherical harmonics), and equal-area pixels so noise and
counts compare fairly everywhere. Does not carry over automatically: the acoustic peaks are physics of the early universe,
and nothing says a methylome has peaks. The genome's own version of that question is asked along the DNA - how far along does
the pattern stay alike - and the chain's clustering reading already works that way.

## How it would be built and tested (the author's standing rule: design, then synthetic, then null)

1. **Write the construction down first**: which sites, how behavior is measured, the metric (Fisher-Rao angle), how points are
   placed on the sphere, and the statistic read off it - committed before any healthy map is drawn.
2. **Synthetic test with a known answer**: plant a group of sites held in order together and a group that is independent, at
   known strength, and require the statistic to separate them; and plant nothing, and require it to find nothing.
3. **Null test on held-out healthy cells**: the statistic must not call healthy people disordered, on cells that had no part in
   building the map, before any number it gives is claimed.
4. Only then read aging or cancer, one question at a time.

## Open questions this note exists to keep

- Does the healthy information sphere have a stable shape across people, or is its shape mostly cell type?
- Is the blurring in cancer a loss of structure at **one** scale, or across all of them - and is that the same thing the
  clustering reading already sees along the DNA, or new information?
- Is version A or version B the one that shows the cancer fingerprint more sharply? They can be built from the same data and
  compared, since both are reading the same residual map.
- Has anyone drawn methylation this way? A literature check belongs with step 1, so we do not reinvent a published method or
  miss a published failure.

**In plain words.** Our cancer meters read how a cell's switch pattern differs from healthy. This note asks whether the whole
pattern has a *shape* - not a picture of the DNA strand, but a map of the order a species has to hold, where two sites sit
close together when they behave alike. There is real mathematics that puts exactly that kind of information on a sphere, which
is the same shape the microwave sky is drawn on, so the tools cosmologists built for the sky could measure the order in a
cell. A second version uses where the DNA physically sits inside the nucleus, which is a real ball with a real surface, and
which is where cancers lose the biggest blocks of their pattern. **What it would take.** A written construction, a synthetic
test where the answer is planted and known, and a null test on healthy people's cells that had no part in building the map.
All of it is laptop work and spends no data, which is why it is on the road ahead rather than waiting for funding.

**Not a milestone:** a design record and an open hypothesis; nothing read, nothing predicted, nothing sealed.

## The strand is hardware; the sphere is the outcome (the author, 2026-10-13)

This record is governed by `CANON/foundation.json`, entry **`what_is_read`** (core): what the law reads is informational fidelity - order
against disorder in the held pattern, as errors in individual molecules - and never hardware. The author's framing of the map is the same
statement one scale up:

> The DNA is hardware, the sphere is the outcome, the patterns of order and disorder... the information.

That settles a choice the sky build would otherwise have to make twice:

| version | coordinates | what it maps | status |
|---|---|---|---|
| the DNA address wrapped on a ball | chromosome and position | the hardware - our filing system | **weak, named as weak**; its curvature is an artifact of how we index, not a property of the cell |
| the information sphere | properties of the pattern itself (two cells near each other when their order is alike) | the outcome - what the law reads | the version to build |
| the nucleus | measured physical positions of the sites | the hardware's real geometry, which the pattern lives on | a separate, later question; it asks where in space the disorder sits, not what the disorder is |

So the first build uses pattern coordinates, not addresses, and the sky tools are borrowed only where they act on a field of information:
the angular spectrum of a pattern is meaningful; a spectrum over base-pair position is bookkeeping. The nucleus version stays on the list
because "where does the disorder sit in the folded genome" is a real question - but it is a question about the medium (CANON/foundation.json,
`_layers`), and an answer there cannot confirm or break the law.

**In plain words.** Think of a page of text. The paper, the ink and the order of the letters on the line are the hardware. Whether the page
still says what it is supposed to say - and whether the damage is one smudged letter or a whole wiped passage - is the information. We are
mapping the second thing. A map drawn by paper position would be a map of the book's binding.

**What it would take to finish.** A built map whose construction was committed before any map was drawn, a planted test that recovers a known
blur, and a null test on healthy cells that had no part in building it. All three are laptop work and spend no data.

## Step 0, the literature check (2026-10-13, three searches; metadata only)

The per-site square-root geometry is already used on methylation, so it is not ours to claim:
- Sanchez and Mackenzie, PLOS ONE 11, e0150427 (2016), and Sanchez, Yang, Maher and Mackenzie, IJMS 20, 5343 (2019), the MethylIT
  method: the Hellinger divergence between methylation levels, which is the chord on the same square-root circle. Credited here for that
  distance only. They are credited in the book's Landauer chapter for putting Landauer's bound on methylation; they are NOT prior art for
  IAM's law (author, 2026-10-13): they treat the background as noise to be removed, where IAM reads against the healthy reference as the
  ruler, and the cost per bit is Landauer's, not theirs.
- Jenkinson, Pujadas, Goutsias and Feinberg, Nature Genetics 49, 719-729 (2017): methylation as a potential-energy landscape, entropy
  read along the genome (informME).
- Amari, Information Geometry and Its Applications (Springer, 2016): the Fisher-Rao geometry itself.
Not found in these searches: methylation sites PLACED on a sphere by how they behave together, or methylation read with sky-map scale
statistics. Three searches is not a review; the check continues before step 1, and any paper found later is added here.

