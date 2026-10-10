"""DEV-FINGERPRINT-02 readability bound (normal side): share of RRBS reads carrying >= 6 methylated CG (the most Stage Q could read), on the first
80 MB of the compressed fastq ENCFF000MLS (prostate epithelial cells, ENCSR000DDU rep 2), unaligned. Usage: python3 readability_check.py"""
import zlib, collections, requests, numpy as np
x = requests.get("https://www.encodeproject.org/experiments/ENCSR000DDU/", params={"format": "json"}, headers={"Accept": "application/json"}, timeout=60).json()
f = [f for f in x["files"] if f["accession"] == "ENCFF000MLS"][0]
raw = requests.get("https://www.encodeproject.org" + f["href"], headers={"Range": "bytes=0-80000000"}, timeout=300).content
txt = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(raw).decode(errors="replace").split("\n")
seqs = [txt[i] for i in range(1, len(txt) - 4, 4)]; cg = np.array([s.count("CG") for s in seqs])
print("reads", len(seqs), "| read length", collections.Counter(len(s) for s in seqs).most_common(1)[0][0], "| share with >= 6 methylated CG", round(float((cg >= 6).mean()), 5))
