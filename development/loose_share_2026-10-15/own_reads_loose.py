"""The loose share from each sample's own molecules (2026-10-15), in place of the arrays. KISS.
Reads: Loyfer 2023 .pat files cut to PROC-CHANNEL-01's 399 random 2,000-CpG windows (windows_hg19_cpgidx.bed; cached on the box image at
~/data/lines/loyfer_pat_windows). Each sample's read lines are split by parity so no molecule is used twice:
 EVEN lines -> loose share: a window is loose if the mean methylation of its molecules' CpG calls is below CUT (0.70, DEV-ARCH-01's cut,
               and 2/3, the one-Landauer-bit cut); loose share = loose windows / windows with >= 200 CpG calls.
 ODD lines  -> held error: PROC-CHANNEL-01's rule (reads >= 6 CpGs, >= 80 % methylated; isolated unmethylated call per interior CpG).
Usage: python own_reads_loose.py synthetic | real <pat_dir> <windows.bed>"""
import os, sys, glob, json, numpy as np, pandas as pd
CUTS = (0.70, 2 / 3)
def windows(bed):
    W = pd.read_csv(bed, sep="\t", header=None, names=["chr", "s", "e"]); return {c: g[["s", "e"]].values for c, g in W.groupby("chr")}
def one(lines, WIN):
    calls = {}; e_iso = e_opp = 0
    for k, line in enumerate(lines):
        p = line.rstrip("\n").split("\t")
        if len(p) < 4: continue
        c, i, pat, n = p[0], int(p[1]), p[2], int(p[3]); s = [x for x in pat if x != "."]
        if not s: continue
        if k % 2 == 0:
            w = WIN.get(c)
            if w is None: continue
            j = np.searchsorted(w[:, 0], i, side="right") - 1
            if j < 0 or i > w[j, 1]: continue
            a = calls.setdefault((c, j), [0, 0]); a[0] += n * s.count("C"); a[1] += n * len(s)
        else:
            if len(s) >= 6 and s.count("C") >= 0.8 * len(s):
                e_opp += n * (len(s) - 2); e_iso += n * sum(1 for q in range(1, len(s) - 1) if s[q] == "T" and s[q-1] == "C" and s[q+1] == "C")
    m = np.array([a[0] / a[1] for a in calls.values() if a[1] >= 200])
    out = {f"loose_{cut:.4f}": float((m < cut).mean()) if len(m) else np.nan for cut in CUTS}
    out.update(windows=int(len(m)), held_err=e_iso / max(e_opp, 1), held_opp=int(e_opp)); return out
if __name__ == "__main__" and sys.argv[1] == "synthetic":
    rng = np.random.default_rng(7); WIN = {"chr1": np.array([[w * 3000, w * 3000 + 2000] for w in range(100)])}; L = []
    for share in (0.2, 0.5, 0.8):
        loose = set(rng.choice(100, int(share * 100), replace=False)); lines = []
        for w in range(100):
            beta = 0.45 if w in loose else 0.85
            for r in range(120):
                i = w * 3000 + int(rng.integers(0, 1990)); calls = "".join("C" if rng.random() < beta else "T" for _ in range(8))
                lines.append(f"chr1\t{i}\t{calls}\t1")
        o = one(lines, WIN); L.append(f"planted loose share {share}: counted {o['loose_0.7000']:.3f} at 0.70, {o['loose_0.6667']:.3f} at 2/3 ({o['windows']} windows)")
    for eps in (0.02, 0.05):
        lines = []
        for r in range(40000):
            calls = "".join("T" if rng.random() < eps else "C" for _ in range(10)); lines.append(f"chr1\t{r}\t{calls}\t1")
        o = one(lines, WIN); exp = eps * (1 - eps) ** 2
        L.append(f"planted held error {eps}: counted isolated rate {o['held_err']:.5f} (expected {exp:.5f}, isolated losses only)")
    open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "own_reads_loose_synthetic.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
elif __name__ == "__main__":
    WIN = windows(sys.argv[3]); R = []
    for f in sorted(glob.glob(os.path.join(sys.argv[2], "*.pat.txt"))):
        with open(f) as fh: o = one(fh, WIN)
        R.append(dict(sample=os.path.basename(f).split(".")[0], **o)); print(R[-1], flush=True)
    pd.DataFrame(R).to_csv("own_reads_loose.csv", index=False)
