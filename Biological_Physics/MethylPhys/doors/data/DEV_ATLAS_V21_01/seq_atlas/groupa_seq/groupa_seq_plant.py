"""Group-A on sequencing (checklist B9): the planted group shift, written into a sample's own read counts (TESTS ONLY). At every identity
site of the sample's own cell (the Met-A reference, metaa_seq_reference_v1.parquet: hg19 CpG index and the cell's level mu), the methylated
count m of n reads moves by LOSS (0.5 - mu) n, rounded without bias (the whole part, and one read more with the
fractional part's chance; seeded by the sample's number), held inside 0 and n: the arrays' commissioning plant (x + loss (0.5 - mu)) on counts.
LOSS 0 writes the file back unchanged, so both groups take the same path. Usage: python3 groupa_seq_plant.py IN.beta OUT.lbeta CELL LOSS"""
import os, sys, numpy as np, pyarrow.parquet as pq, pyarrow.compute as pc
IN, OUT, CELL, LOSS = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]); HG19 = 28217448
CH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "..", "chain")
T = pq.read_table(os.path.join(CH, "Runtime Matrices", "Seq_Atlas", "metaa_seq_reference_v1.parquet"), columns=["cpg", "cell", "level"])
T = T.filter(pc.equal(T["cell"], CELL)).to_pandas(); assert len(T), f"no identity sites for {CELL!r}"
T = T[T.level != 65535].drop_duplicates("cpg"); cpg = T.cpg.to_numpy(np.int64); mu = T.level.to_numpy() / 1e4
a = np.fromfile(IN, np.uint8).reshape(-1, 2).astype(np.uint16); assert len(a) == HG19, len(a)
m = a[cpg - 1, 0].astype(float); n = a[cpg - 1, 1].astype(float)
d = LOSS * (0.5 - mu) * n; rng = np.random.default_rng(int("".join(c for c in os.path.basename(IN) if c.isdigit())[:9] or 0) + 20261102)
d = np.floor(d) + (rng.random(len(d)) < (d - np.floor(d)))   # unbiased rounding: the expected shift is exactly LOSS (0.5 - mu) n; LOSS 0 moves nothing
a[cpg - 1, 0] = np.clip(m + d, 0, n).astype(np.uint16); a.tofile(OUT)
print(f"{CELL}: {len(cpg)} identity sites, {int((a[cpg - 1, 0] != m).sum())} counts moved (loss {LOSS})")
