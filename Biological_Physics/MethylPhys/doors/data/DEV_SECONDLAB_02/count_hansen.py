"""DEV-SECONDLAB-02 step 3: each Hansen donor's deduplicated Bismark alignment counted by chain Stage Q2 count_bam on the reference the
alignment used (the Bismark hg38 genome folder of the tumor-work tools bundle), stamped with its manifest rows. Also records the read length
(from the first 20,000 alignments) and the input file's sha256. Usage: python count_hansen.py <bam> <ref.fa> <donor> <gsm> <srr> <workers>"""
import os, sys, json, collections
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "chain")); import stage_q2_selfref as Q2
import pysam
bam, fa, donor, gsm, srr, w = sys.argv[1:7]
if not (os.path.exists(bam + ".bai") or os.path.exists(bam[:-4] + ".bai")): pysam.index(bam, "-@", str(w))   # run 1 of the count: the machine fetched no index, every region read failed (2026-10-16)
L = collections.Counter()
for i, a in enumerate(pysam.AlignmentFile(bam, "rb")):
    if i >= 20000: break
    L[a.query_length] += 1
o = Q2.count_bam(bam, fa, int(w), "hg38 (Bismark genome of the tumor-work bundle)")
o.update(sample=gsm, donor=donor, run=srr, cell="b cells", reference_cell="b cells", read_length_mode=L.most_common(1)[0][0], read_lengths=dict(L.most_common(5)))
Q2.write_json(o, f"out/{srr}.json", inputs=[dict(name=os.path.basename(bam), sha256=Q2.file_sha(bam), size=os.path.getsize(bam)), dict(name=os.path.basename(fa), sha256=Q2.file_sha(fa))],
              manifest=["PHY-047", "PHY-050", "COM-IAMA-054"]); print(srr, "ok", o["loose_share"], o["held_err"], o["sub_err"], flush=True)
