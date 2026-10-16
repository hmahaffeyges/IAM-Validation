"""DEV-IAMA-GRAND-01: one CD8 sample's deduplicated Bismark alignment counted by chain Stage Q2 count_bam on the reference the alignment used
(the Bismark hg38 genome of the tumor-work tools bundle, as for Hansen), stamped with COM-IAMA-052. Records read lengths and the input's sha256.
Usage: python count_grand.py <bam> <ref.fa> <gsm> <sample_id> <srr> <subset> <shape_type> <workers>"""
import os, sys, json, collections
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "chain")); import stage_q2_selfref as Q2
import pysam
bam, fa, gsm, sid, srr, subset, shape_type, w = sys.argv[1:9]
if not os.path.exists(bam + ".bai"): pysam.index(bam, "-@", str(w))
L = collections.Counter()
for i, a in enumerate(pysam.AlignmentFile(bam, "rb")):
    if i >= 20000: break
    L[a.query_length] += 1
o = Q2.count_bam(bam, fa, int(w), "hg38 (Bismark genome of the tumor-work bundle)")
o.update(sample=gsm, sample_id=sid, run=srr, cell=f"cd8 t cells ({subset})", subset=subset, shape_type=shape_type, lab="Youngblood (St. Jude) GSE144691",
         read_length_mode=L.most_common(1)[0][0], read_lengths={str(k): v for k, v in L.most_common(5)})
Q2.write_json(o, f"out/{srr}.json", inputs=[dict(name=os.path.basename(bam), sha256=Q2.file_sha(bam), size=os.path.getsize(bam)), dict(name=os.path.basename(fa), sha256=Q2.file_sha(fa))],
              manifest=["COM-IAMA-052"]); print(srr, "ok", o["loose_share"], o["held_err"], o["sub_err"], flush=True)
