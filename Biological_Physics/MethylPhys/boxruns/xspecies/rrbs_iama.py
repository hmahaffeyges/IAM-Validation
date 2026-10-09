"""DEV-XSPECIES-TEMP-01 reader: reference-free copy error from directional RRBS reads (fastq on stdin or a file).
Fragment = fully converted read (C->T); a position is a CpG of the fragment if any of its reads has C then G there; calls C=methylated,
T=unmethylated at those positions; bases 0-3 and the last 2 not called. Qualifying read: >= MIN_CALLS calls, >= 80 % methylated.
eps = isolated unmethylated interior calls / interior calls (Stage Q definition). Prints one JSON line."""
import sys, json, gzip, numpy as np
MIN_CALLS = 4; SKIP5 = 4; SKIP3 = 2

def masks(seq):
    L = len(seq); cg = tg = 0
    for j in range(SKIP5, L - SKIP3 - 1):
        if seq[j + 1] == "G":
            b = seq[j]
            if b == "C": cg |= 1 << j
            elif b == "T": tg |= 1 << j
    return cg, tg

def run(lines):
    H = []; CG = []; TG = []
    for i, l in enumerate(lines):
        if i % 4 != 1: continue
        s = l.strip()
        if len(s) < 20: continue
        s = s[:63]
        cg, tg = masks(s)
        H.append(hash(s.replace("C", "T"))); CG.append(cg); TG.append(tg)
    H = np.array(H, dtype=np.int64); CG = np.array(CG, dtype=np.uint64); TG = np.array(TG, dtype=np.uint64)
    o = np.argsort(H, kind="stable"); Hs = H[o]; st = np.r_[0, np.flatnonzero(np.diff(Hs)) + 1]
    grp = np.bitwise_or.reduceat(CG[o], st); size = np.diff(np.r_[st, len(Hs)])
    gmask = np.repeat(grp, size); gsize = np.repeat(size, size)
    keep = gsize >= 2
    cgk = CG[o][keep] & gmask[keep]; tgk = TG[o][keep] & gmask[keep]
    nm = np.array([bin(int(x)).count("1") for x in cgk]); nu = np.array([bin(int(x)).count("1") for x in tgk])
    q = ((nm + nu) >= MIN_CALLS) & (nm >= 0.8 * (nm + nu))
    err = opp = 0
    for m_, u_ in zip(cgk[q], tgk[q]):
        m_ = int(m_); u_ = int(u_); allc = m_ | u_
        pos = [j for j in range(64) if allc >> j & 1]
        c = ["C" if m_ >> j & 1 else "T" for j in pos]
        for k in range(1, len(c) - 1):
            opp += 1; err += c[k] == "T" and c[k - 1] == "C" and c[k + 1] == "C"
    return dict(reads=int(len(H)), fragments=int(len(st)), reads_in_groups=int(keep.sum()), qualifying=int(q.sum()),
                opportunities=int(opp), errors=int(err), eps=(err / opp if opp else None))

if __name__ == "__main__":
    f = sys.stdin if len(sys.argv) < 2 else (gzip.open(sys.argv[1], "rt") if sys.argv[1].endswith(".gz") else open(sys.argv[1]))
    print(json.dumps(run(f)))
