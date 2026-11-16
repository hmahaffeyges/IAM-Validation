#!/usr/bin/env python3
"""Synthetic for the per-tile line (TERR/GLB, 2026-11-16). Molecules every 2 kb on chr1-chr4, in 10 Mb blocks that alternate FIRM (even-half
mean call 0.9, held loss 0.03) and LOOSE (mean call 0.4, held loss 0.08). Read through the one command (count only). The synthetic counts the
lone losses it planted itself, block by block, with the count's own rule (a lone unmethylated call between two methylated ones), and checks
that held_err_firm_tiles and held_err_loose_tiles are within 10 % of its own firm and loose counts (tiles straddling a block edge mix). Exit 1."""
import os, sys, json, gzip, random, subprocess, tempfile
H = os.path.dirname(os.path.abspath(__file__)); IF = os.path.abspath(os.path.join(H, "..", "..", "..", "..", "chain", "MethylPhys_Interface"))
W = tempfile.mkdtemp(); rng = random.Random(46); pat = os.path.join(W, "tiles.pat.gz"); own = {"firm": [0, 0], "loose": [0, 0]}
with gzip.open(pat, "wt") as f:
    for chrom in ("chr1", "chr2", "chr3", "chr4"):
        for pos in range(1_000_000, 120_000_000, 2000):
            kind = "firm" if (pos // 10_000_000) % 2 == 0 else "loose"
            mm, ph = (0.9, 0.03) if kind == "firm" else (0.4, 0.08)
            ev = "".join("C" if rng.random() < mm else "T" for _ in range(10))
            od = ["T" if rng.random() < ph else "C" for _ in range(12)]
            if od.count("C") >= 0.8 * 12:
                own[kind][0] += sum(1 for q in range(1, 11) if od[q] == "T" and od[q - 1] == "C" and od[q + 1] == "C"); own[kind][1] += 10
            f.write(f"{chrom}\t{pos}\t{ev}\t1\n{chrom}\t{pos + 1000}\t{''.join(od)}\t1\n")
b = os.path.join(W, "tiles.json")
p = subprocess.run([sys.executable, "run_sample.py", "--pat", pat, "--specimen", "cultured cells", "--seq-cell", "syn", "--id", "syn",
                    "--dev-count-only", "--dev-allow-partial-genome", "--dev-count-genome", "hg19",
                    "--out", os.path.join(W, "t.html"), "--bundle", b], cwd=IF, capture_output=True, text=True)
if p.returncode != 0: print("::error::run failed " + (p.stdout[-300:] + p.stderr[-500:]).replace("\n", " | ")); sys.exit(1)
c = json.load(open(b))["iam_a_count_only"]["count"]
ef, el = own["firm"][0] / own["firm"][1], own["loose"][0] / own["loose"][1]
gf, gl = c["held_err_firm_tiles"], c["held_err_loose_tiles"]
ok = gf is not None and gl is not None and abs(gf - ef) <= 0.10 * ef and abs(gl - el) <= 0.10 * el
msg = f"tile-line synthetic: firm read {gf} vs planted {ef:.5f}; loose read {gl} vs planted {el:.5f}; tiles {c.get('tile_line_n')}; bands {c.get('held_err_by_tile_call')} -> {'met' if ok else 'NOT met'}"
os.makedirs("xout", exist_ok=True); open("xout/tile_line_synthetic_summary.txt", "w").write(msg + "\n"); print("::notice::" + msg[:1800])
sys.exit(0 if ok else 1)
