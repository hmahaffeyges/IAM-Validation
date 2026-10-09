"""DEV-IAMA-KIT-01 part 3: split each run's copy errors by the molecule's background. Molecules with >= 6 calls; the background is the
majority state; errors are minority calls. Errors in UNMETHYLATED-background molecules (a C among Ts) are what incomplete bisulfite
conversion makes; errors in METHYLATED-background molecules (a T among Cs) are what conversion cannot make. Runs: donor 6 Swift / TruSeq."""
import gzip, json, collections
out = {}
for r in ("SRR9888332", "SRR9888333"):
    c = collections.Counter()
    with gzip.open(f"/mnt/scratch/clip/pat_c0/{r}.pat.gz", "rt") as f:
        for l in f:
            p = l.rstrip("\n").split("\t"); s = p[2]; n = int(p[3])
            m = s.count("C"); u = s.count("T"); k = m + u
            if k < 6 or m == u: continue
            if m > u: c["M_calls"] += k * n; c["M_err"] += u * n; c["M_mol"] += n
            else: c["U_calls"] += k * n; c["U_err"] += m * n; c["U_mol"] += n
    out[r] = dict(c, eps_M=c["M_err"] / c["M_calls"], eps_U=c["U_err"] / c["U_calls"], share_U_molecules=c["U_mol"] / (c["U_mol"] + c["M_mol"]))
    print(r, json.dumps(out[r]))
json.dump(out, open("bg_split.json", "w"), indent=1)
