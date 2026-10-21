"""The sky layout of IAM-Atlas v2.1, frozen for the chain (2026-10-21). Every report globe lays a sample, its healthy reference and their
difference on the same pixels as the atlas's cell pages and the public globe. That layout puts every atlas site in genome order (chr1..chr22,
chrX, chrY, then position; Illumina's EPIC v1 B4 and 450K manifests, hg19) and site i of n at NESTED pixel floor(i * NPIX / n), NSIDE 256
(sky_maps_v21.genome_order and .to_sky, unchanged). The chain should not download manifests to draw a report, so the order is written once:
OUT/sky_order_v21.npz holds `order` (atlas row numbers in genome order), `unplaced`, and the checksums of the atlas site list and of both
manifests. Checked before it is written: the neutrophil map from this order has the measured-pixel count of the cell pages.
Usage: python3 sky_order_v21.py OUT"""
import os, sys, json, hashlib, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.abspath(os.path.join(H, "..", "..", "..", ".."))
sys.path[:0] = [H, os.path.join(R, "website", "art"), os.path.join(R, "Biological_Physics", "MethylPhys", "chain")]
import sky_maps_v21 as SM, make_sphere_3d as MS, pyarrow.parquet as pq
A = os.path.join(R, "Biological_Physics", "MethylPhys", "atlas", "v2.1"); out = sys.argv[1]; os.makedirs(out, exist_ok=True)
man = MS.manifests(); ids = pq.read_table(os.path.join(A, "sites.parquet")).column("cpg").to_pylist(); order, unplaced = SM.genome_order(man, ids)
part = next(k for k in (1, 2, 3) if "neutrophils" in pq.read_schema(os.path.join(A, f"mu_part{k}.parquet")).names)
mu = SM.to_sky(SM.decode(pq.read_table(os.path.join(A, f"mu_part{part}.parquet"), columns=["neutrophils"]).column("neutrophils").to_numpy())[order])
ix = {c["slug"]: c.get("pixels_measured") for c in json.load(open(os.path.join(R, "website", "art", "out", "cells", "index.json")))["cells"]}
got = int(np.isfinite(mu).sum()); print("neutrophil measured pixels", got, "| cell page", ix.get("neutrophils"))
if got != ix.get("neutrophils"): sys.exit("sky_order_v21: the order does not reproduce the cell pages' layout; not written")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
np.savez_compressed(os.path.join(out, "sky_order_v21.npz"), order=order.astype(np.int32), unplaced=np.int64(unplaced), nside=np.int64(SM.NSIDE),
                    sites_sha256=sha(os.path.join(A, "sites.parquet")),
                    epic_manifest_sha256=sha(os.path.join(man, "MethylationEPIC_v-1-0_B4.CoreColumns.csv.gz")),
                    hm450_manifest_sha256=sha(os.path.join(man, "HumanMethylation450k_15017482_v3.csv.gz")))
print("written", len(order), "sites in genome order;", unplaced, "unplaced")
