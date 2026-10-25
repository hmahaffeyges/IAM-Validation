# Field notes

*Every update to every experiment on the Now running board, in plain words, oldest first. Notes are never edited or removed;
a correction is a new note. Generated from CANON/now_running.json by CANON/now_running.py; do not edit by hand.*

<a id="atlas-v21"></a>
## Building a bigger map of healthy cells: IAM-Atlas v2.1

*Result is in.* Every IAM reading compares a sample with healthy cells. The atlas is that comparison: a map of what 74 kinds of healthy human cells look like. Version 2.1 more than doubles the samples behind it and adds second and third laboratories for key cells, so the map stops depending on any one lab's habits.

### · Version 2 published with its own DOI, so anyone can download the map and check it.

Record: [README.md](atlas/README.md)

### · The plan is locked before any new data is opened: which samples build the map and which are kept back to test it.

Record: [DEV_ATLAS_V21_01.md](doors/DEV_ATLAS_V21_01.md)

### · First step running: measuring how much growing cells in a dish changes them.

Why the dish matters: every public sample of lung support cells (the fibers and muscle that hold the lung together) was grown in a dish first. Growing in a dish slowly changes a cell's methylation, a little more with each division, because the cell has lost its neighbors, its blood supply and the signals of the body around it. If we put those samples straight into the atlas, we would be describing "lung cells after weeks in a dish", not lung cells in a person. So we first measure how much the dish changes cells, using cells a laboratory grew until they stopped dividing.

Record: [DEV_ATLAS_V21_01.md](doors/DEV_ATLAS_V21_01.md)

### · Culture check, first result: growing cells in a dish changes them a lot. In the cleanest comparison, about 1 in 9 of the sites we read moves steadily as the cells keep dividing. Those sites will be set aside before any dish-grown cell is compared with the atlas. Two more laboratories' data are next.

A laboratory grew seven kinds of human cells in dishes until they stopped dividing, and measured them along the way. We looked for the switches that move steadily the longer cells grow. By the rule we wrote down first, 44.8 % of the sites we read move with time in the dish. Part of that turned out to be a lab artifact, not the dish: in five of the seven lines the early samples were measured on one date and the late ones on another, so "later in the dish" and "later on the machine" were mixed together. We found that on review and wrote it into the record. In the two cleanest lines, measured on a single date, about 1 in 9 sites (11.1 %) drifts in both, mostly in the same direction. That is the real dish effect as best we can measure it today. Before any dish-grown cell, such as lung support cells, is compared with the atlas, those sites are set aside. We keep the larger set as the official mask, as the rule said: a mask can only remove sites, so an oversized one keeps the test honest and only costs coverage. Next, two more laboratories' dish-growth series, to see whether the drift looks the same elsewhere.

Record: [DEV_ATLAS_V21_01.md](doors/DEV_ATLAS_V21_01.md)

### · Culture check, second and third laboratories: much of the dish effect shows up again, in different cells grown in different labs.

We checked whether the sites that drift in a dish at the first laboratory drift the same way elsewhere. At Columbia, skin and breast support cells grown for months: about half of those sites drift the same way there, against about a third if it were chance. At Aachen, blood-vessel cells from three newborns grown from early to late passage: those sites move the same way in all three donors more than twice as often as sites in general. So the dish leaves a shared mark on dividing cells, and also a mark that differs by cell type and by laboratory. Our mask, the larger set we fixed before looking, stays as it is: it removes the shared part and more. Next we measure the dish effect directly, on cells that exist both fresh and dish-grown.

Record: [DEV_ATLAS_V21_01.md](doors/DEV_ATLAS_V21_01.md)

### · Not running. The fit itself needs a rented 128-core machine with 246 GB of memory - the same kind that built version 2 - so it is queued behind the free work. What moved instead: the four laboratories we had planned to TEST the new map against turned out to have been read already, so they can help build it but cannot judge it. One unread laboratory has been chosen as the test, with its two weaknesses written down first.

Record: [DEV_ATLAS_V21_01.md](doors/DEV_ATLAS_V21_01.md)

### · The fit is running. It started at 11:26 PM Pacific on a rented 128-core machine, and 164 of its 700 blocks are done. Every block so far has passed the sampler's own health checks: no divergent draws, and in every block 99 percent of sites show an R-hat (agreement between independent chains) of 1.009 or less. It is expected to finish Friday about 9 AM Pacific, and the live count is on the front page.

What it took to get here: 22 checks on a launch list, every one passed before the machine started. The new map's identity and sex checks; a test that no new cell is a duplicate of an old one; each laboratory's offset; a dry run; a cost sheet; and a speed fix checked against the old model on 60 real test sites before it was used. Sealed BEFORE the map exists: the bars it will be judged by (how often new samples fall inside its stated range, the blind sites the tissue tests rely on, blood composition), and a showcase test on the two jobs Loyfer and Moss built their atlases for: finding each cell's marker sites, and working out which cells are in a mixed sample, including cells their atlases do not hold.

Record: [DEV_ATLAS_V21_01.md](doors/DEV_ATLAS_V21_01.md)

### · Commissioned. IAM-Atlas v2.1 met its sealed tests, and it outperformed both published reference atlases of human cell types on the jobs each was built for.

Record: [COMMISSIONING_NOTE_ATLAS_V21.md](doors/COMMISSIONING_NOTE_ATLAS_V21.md)

<a id="iama-commission"></a>
## Does our second instrument read the same DNA the same way everywhere?

*On hold.* Before anyone trusts an instrument, it has to give the same answer on the same sample. We read identical reference DNA prepared with five different kits in two laboratories.

### · Four of five checks passed: repeats agree, kits agree, and a planted change was seen every time. The lab-to-lab check failed on three files, so by our own rule nothing is declared until the cause is found.

The goal: before anyone trusts an instrument, it must give the same answer on the same DNA. A public project sent identical reference DNA from 7 people to be prepared with 5 different kits, in 2 laboratories, each twice. We downloaded 78 of those sequencing files, lined each up against the human genome on a rented computer, and read them with IAM-A. The rented machines kept being taken back by Amazon when demand rose, which is why it took so long; every finished file was saved, so nothing was lost. Results: the same sample prepared twice agrees, mostly within about 1 %. Different kits agree. A planted 1 % change was seen every time. The intake check refused two kits whose chemistry did not convert the DNA well enough, which is the instrument protecting itself, as designed. One check failed: the same kit in two laboratories. Four of seven samples agreed; three files from the second laboratory read about 15 % low, and those three are the most heavily sequenced files in the set. Our own rule says that when any check fails, we stop and find the cause before declaring anything. So the instrument is on hold, not failed.

Record: [DEV_IAMA_COMMISSION_01_OUTCOME.md](doors/DEV_IAMA_COMMISSION_01_OUTCOME.md)

### · The cause test is written and listed in the run ledger. It starts when rented-computer prices and availability allow.

The test: read those three large files, and two normal ones, from the start, the middle and the end, and compare. If the low readings come only from the start of the big files, it is how we sample a file, which we can fix and then redo the test under the fixed rule. If they read low everywhere, the difference is real and the lab-to-lab check stays failed for that kit, and the record says so. The rules for deciding were written down and tested on made-up data before any of the files is read.

Record: [DEV_IAMA_CAUSE_01.md](doors/DEV_IAMA_CAUSE_01.md)

### · The cause test is running on a rented computer: the three low files and two normal ones, each read from its start, middle and end.

Rented computers came back into reach tonight, so the cause test started. It reads the three files that came out low, and two from the same laboratory that came out normal, each from three places in the file: the start, the middle and the end. If the low reading follows where in the file we read, the fault is in our sampling and we fix the method and run the check again. If it stays low wherever we read, the laboratory really does differ, and the record says so. The rule deciding between the two was written before the first file was read.

Record: [DEV_IAMA_CAUSE_01.md](doors/DEV_IAMA_CAUSE_01.md)

### · The cause of the failed laboratory check is now known, and it was not the laboratories. The reading had been taken from the very beginning of each sequencing file, and the beginning reads differently - the three disagreeing libraries sit 11 to 14 percent below the controls read at the same spot, while both controls read flat. Our first fix, reading several places and averaging, failed its own test on made-up data: averaging shrinks a bad measurement, it does not remove it. The replacement skips the unsettled beginning, and how much to skip is decided by the sequencing chemistry itself rather than by whether our answer improves. It passed six tests on made-up files, including recovering a planted 12 percent error to within a fifth of a percent. Nothing real has been trimmed yet.

Record: [DEV_IAMA_TRIM_01.md](doors/DEV_IAMA_TRIM_01.md)

### · Paused so the commissioned chain reaches researchers first; this picks up after that.

Record: [LIMITS_STUDY_2026-10-17.md](../../development/LIMITS_STUDY_2026-10-17.md)

### · Still paused while we finish the chain researchers will receive first. Nothing in this test has been read since it was paused, so it resumes exactly where it stopped.

Record: [LIMITS_STUDY_2026-10-17.md](../../development/LIMITS_STUDY_2026-10-17.md)

<a id="neurons"></a>
## Brain cells: does healthy order hold in cells that never divide?

*On hold.* Neurons stop dividing for life, so they can never refresh their switch pattern by making a new copy. If they still sit where physics says healthy cells should, that tells us how cells hold their information. We built a healthy brain-cell reference from one laboratory's neurons, using a rule written before we looked.

### · Healthy brain-cell reference built by its rule, from seven samples of five donors. Cleanly sorted cells read tightly together; unsorted cells read higher.

We read ten public neuron samples from one laboratory whole, and built the healthy reference from the cortex samples the rule allowed: seven samples from five donors. Then we looked at the spread. The neurons that had been cleanly sorted from the rest of the brain tissue read tightly together; the samples marked unsorted read higher, whatever the donor's age or brain region. So preparation matters more than we expected. We keep the reference exactly as built, and add a second one from sorted cells only, labeled as a change made after looking. Both independent neuron laboratories sorted their cells, and we have opened neither, so that second reference gets a fair test: it has to read their healthy donors as normal, with the rule sealed before we look.

Record: [DEV_IAMA_NEURON_01.md](doors/DEV_IAMA_NEURON_01.md)

### · The rule for what counts as a healthy brain-cell reference is now sealed: cleanly sorted cells only. That was a decision made after looking at the data, so it is labeled as one, and it was sealed before either independent laboratory was opened. Neither has been read.

Record: [DEV_IAMA_NEURON_01.md](doors/DEV_IAMA_NEURON_01.md)

### · Paused so the commissioned chain reaches researchers first; this picks up after that.

Record: [LIMITS_STUDY_2026-10-17.md](../../development/LIMITS_STUDY_2026-10-17.md)

### · Still paused. Neither independent laboratory's neurons has been opened, so the test is still blind when it resumes.

Record: [DEV_IAMA_NEURON_01.md](doors/DEV_IAMA_NEURON_01.md)

<a id="lung"></a>
## Lung cancer: waiting for a better map of the lung

*On hold.* We compare each patient's tumor with the same patient's normal lung. Before any tumor is opened, made-up tests on normal lungs must show that ordinary differences between people's lungs cannot fake a cancer signal.

### · Four lung sets tried: one read with no clear cancer signal; three stopped by their made-up tests before any tumor was opened, so none was wasted.

A normal lung is a mix of many cell types: air-sac cells, airway cells, blood-vessel lining, immune cells, and the support tissue that holds it together. How much of each varies a lot from person to person. Our reading only counts if a change in that mix cannot fake a cancer signal, so before any tumor is opened we build made-up lungs from the healthy ones and check. The first lung set was read and showed no clear signal. The next three failed those made-up tests: in the last one, a lung with more blood-vessel lining than usual could fake a result, even at the sites where all known lung cells look alike, because healthy lungs match each other so closely that the bar is very tight. The tests refused, which is their job, and no tumor was read. The fix is a better picture of the lung's support cells in the atlas, which is why the dish check comes first.

Record: [DEV_TISSUE_LUNG_04.md](doors/DEV_TISSUE_LUNG_04.md)

### · Still on hold, and now with a named condition for coming off hold: measured lung support cells in the map. Four lung attempts failed for one reason - a healthy lung whose cell mixture leans one way can imitate the signal - and in three of the four no tumor was ever opened.

Record: [DEV_TISSUE_LUNG_03.md](doors/DEV_TISSUE_LUNG_03.md)

### · Paused so the commissioned chain reaches researchers first; this picks up after that.

Record: [LIMITS_STUDY_2026-10-17.md](../../development/LIMITS_STUDY_2026-10-17.md)

### · Still on hold for the same reason: the map needs measured lung support cells first. The tumors stay unread until then.

Record: [DEV_TISSUE_LUNG_03.md](doors/DEV_TISSUE_LUNG_03.md)

<a id="blood-group"></a>
## Can a tube of blood tell us about a GROUP of people?

*On hold.* Reading cancer from one person's blood on these arrays is out of reach: healthy people differ from each other by more than a tumor-sized signal. But comparing GROUPS is a different question, and that is within reach today.

### · First test done, and it is the honesty test rather than the detection one. Two groups of HEALTHY people, read in two different laboratories: in the raw reading they differ by 3.69 percent - as much as a tumor would - and after the chain's correction, which compares each person against healthy people measured in the same run, they agree to 0.16 percent. The rule this forces, written before any patient data is opened: patients and their healthy comparisons must be measured in the same run, or the comparison is measuring the laboratories.

Record: [DEV_GROUP_01.md](doors/DEV_GROUP_01.md)

### · Paused so the commissioned chain reaches researchers first; this picks up after that.

Record: [LIMITS_STUDY_2026-10-17.md](../../development/LIMITS_STUDY_2026-10-17.md)

### · Still paused. Meanwhile the same chain read whole series of tissues: in prostate tumors the CpG islands read higher than in the neighboring normal tissue, across 237 samples, the early change this reading was built to catch.

Record: [DEV_REPLICATE_01.md](doors/DEV_REPLICATE_01.md)

<a id="levers"></a>
## Testing the physics itself: the levers

*On hold.* A lever is a handle that moves one term in the law. If the law is real, turning a handle moves the reading by the amount the equation says - and if it does not, the law is wrong. Every lever carries, in writing and before any data is opened, the reading that would break it.

### · The three-channel test came back: a cell has three jobs at a methylation site - copy the pattern, write a new mark, remove a mark - and we asked whether our readings can tell which one has been damaged. Damaging the removal job is the only one that makes the reading go DOWN, so that one is identifiable on its own. The other two both push it up and can only be told apart if the damage size is already known, which in a patient it would not be. That limit is why a third reading is being built. All of this is on made-up data; no lever has yet been read against the world except one mouse result.

Record: [DEV_LEVERS_01.md](doors/DEV_LEVERS_01.md)

### · A healthy cell can be its own reference. Across 56 kinds of healthy human cells, the rate of tagging mistakes follows one straight line set by how loosely the cell holds its own DNA (r 0.87; 53 of 56 cells within 0.003 of the line). That means a sample's expected healthy reading can come from the sample itself, instead of from a library of other people's cells.

Why it matters: the hardest thing in this field is the healthy reference, cells of the same kind, often from the same lab. A test that carries its own reference can be pointed at any tissue and any lab. Same day: the line held when the cut was moved to the physics value of one Landauer bit (2/3); it came back, weaker but real, from each sample's own molecules with no array at all (r 0.79); and one prediction did not hold (removing the eraser enzyme did not raise the island errors as predicted, which points to a shield rather than the eraser). One laboratory so far; the second-laboratory test and a sealed grand test on fresh data come next. Development, not commissioned.

Record: [loose_end_line.py](../../development/loose_end_2026-10-15/loose_end_line.py)

### · Paused so the commissioned chain reaches researchers first; this picks up after that.

Record: [LIMITS_STUDY_2026-10-17.md](../../development/LIMITS_STUDY_2026-10-17.md)

### · Two new tests of where a cell's tagging mistakes come from. When we knocked out a protein that keeps the tags in repair, mistakes in the molecules that are still held nearly tripled. When we compared dividing and resting cells of one kind, dividing added only about a twentieth. Most of the mistakes come from the upkeep itself, not from copying the DNA.

Record: [DEV_ERASE_01.md](doors/DEV_ERASE_01.md)
