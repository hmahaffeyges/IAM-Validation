"""Whole-file copy error of a wgbstools .pat(.gz) by Stage Q's rule (stage_q_iam_a.pat_site_table), streamed: '.' calls dropped,
molecules with >= 6 calls and >= 80 % methylated qualify, interior call = opportunity, unmethylated interior call with both neighbouring
calls methylated = isolated error; each line weighted by its molecule count. Usage: pat_eps.py FILE [label] -> one JSON line."""
import sys, gzip, json
def eps(path):
    err = opp = mol = qmol = 0
    with gzip.open(path, "rt") as f:
        for l in f:
            q = l.split("\t")
            if len(q) < 4: continue
            c = q[2].replace(".", ""); n = int(q[3]); mol += n
            L = len(c)
            if L < 6 or c.count("C") < 0.8 * L: continue
            qmol += n; opp += (L - 2) * n
            e = 0
            for i in range(1, L - 1):
                if c[i] == "T" and c[i - 1] == "C" and c[i + 1] == "C": e += 1
            err += e * n
    return dict(molecules=mol, qualifying=qmol, opportunities=opp, errors=err, eps=err / opp if opp else None)
if __name__ == "__main__":
    o = eps(sys.argv[1]); o["file"] = sys.argv[1].split("/")[-1]
    if len(sys.argv) > 2: o["label"] = sys.argv[2]
    print(json.dumps(o))
