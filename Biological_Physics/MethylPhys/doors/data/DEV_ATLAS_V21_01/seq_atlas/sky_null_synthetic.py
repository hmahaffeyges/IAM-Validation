"""The sky null test on the sphere (stage_seq_atlas.sky_null; checklist D6 on the sphere, D10): synthetic, written and committed before it
runs. The instrument's job: a map with no spatial structure stays inside its shuffled null; a map with real large-scale structure does not.
Known answer (seed 20261114): at every IAM-Atlas v2.1 site, independent noise N(0, 0.06) (the size of a noise map's values), five times;
and the same noise plus a planted change of 0.02 that rises and falls five times along the genome (sin, genome order), five times. The
masks of sky_mask_sites (sex chromosomes, poorly mapping probes), 20 shuffles each. Bars: the plain noise has at most one band above its
null in each of the 5 maps; the planted change puts the lowest band (multipoles 2 to 8) above its null in all 5.
Usage: python3 sky_null_synthetic.py OUT.txt   (needs the array manifests, as the sphere build does)"""
import os, sys, numpy as np
CH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "chain"); sys.path[:0] = [CH, os.path.join(CH, "MethylPhys_Interface")]
import pyarrow.parquet as pq, stage_seq_atlas as Q, stage_metaa_atlas as SA, reference_globes as RG
sys.path.insert(0, os.path.join(CH, "..", "atlas", "tools")); import sky_maps_v21 as SM
ids = pq.read_table(os.path.join(SA.ATLAS, "sites.parquet")).column("cpg").to_pylist(); order, _ = SM.genome_order(RG.manifests(), ids); msk = Q.sky_mask_sites(ids)
rng = np.random.default_rng(20261114); L = [f"sites {len(ids)}, placed {len(order)}, masked {int(msk.sum())}"]; plain, planted = [], []
for k in range(5):
    v = rng.normal(0, 0.06, len(ids)); r = Q.sky_null(v, order, msk, rng=np.random.default_rng(100 + k)); plain.append(len(r["bands_above_null"]))
    L.append(f"plain {k + 1}: bands above null {r['bands_above_null']}, ratio {r['ratio']}")
    w = v.copy(); w[order] += 0.02 * np.sin(2 * np.pi * 5 * np.arange(len(order)) / len(order)); r = Q.sky_null(w, order, msk, rng=np.random.default_rng(200 + k)); planted.append(0 in r["bands_above_null"])
    L.append(f"planted {k + 1}: bands above null {r['bands_above_null']}, ratio {r['ratio']}")
b1 = all(n <= 1 for n in plain); b2 = all(planted)
L += [f"bar 1 plain noise, at most one band above its null in each: {plain}: {'met' if b1 else 'not met'}",
      f"bar 2 planted change, lowest band above its null in all five: {planted}: {'met' if b2 else 'not met'}", f"synthetic: {'met' if b1 and b2 else 'not met'}"]
open(sys.argv[1], "w").write("\n".join(L) + "\n"); print("\n".join(L))
