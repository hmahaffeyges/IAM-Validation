#!/usr/bin/env python3
"""run_draw.py - one blood draw through the one command (DEVELOPMENT, 2026-10-22, to-do step 40; author: "EVERY instrument scoring on one whole
blood sample that is separated between Buffy coat and cfDNA ... reproducible with the push of a button").

Runs run_sample.py on the draw's buffy coat and on its plasma, each as given (IDAT pair, values, a .pat file or a GEO sample), with
--cell-panel atlas, and for sequencing --dev-seq-values, so every instrument that has enough to read reads: the IAM-Deconvolver's shares on
IAM-Atlas v2.1, Met-A for each cell or source found, IAM-A on the long molecules, the globes. Then writes the draw's joint record, read only
from the two bundles: each half's sources, the plasma's sources that the person's own buffy coat does not hold (non-blood sources, and blood
sources at a higher share than in the buffy coat), and each source's Met-A. Every reading keeps its own flags; nothing here is commissioned.

  run_draw.py --id DRAW --out OUTDIR --buffy-grn F --buffy-red F | --buffy-betas F | --buffy-pat F | --buffy-geo-pat GSM
                                     --plasma-grn F --plasma-red F | --plasma-betas F | --plasma-pat F | --plasma-geo-pat GSM
              [--min-calls N] [-- extra run_sample.py flags]
Writes OUTDIR/buffy_coat/ and OUTDIR/plasma/ (each the one command's report and bundle) and OUTDIR/draw.json and draw_summary.txt.
"""
import os, sys, json, argparse, subprocess

HERE = os.path.dirname(os.path.abspath(__file__)); RS = os.path.join(HERE, "run_sample.py")
BLOOD_WORDS = ("neutrophil", "monocyte", "b cell", "t cell", "nk cell", "macrophage", "dendritic", "erythro", "megakaryo", "hsc", "cmp", "gmp",
               "mep", "mpp", "plasma cell", "regulatory t", "bone marrow")
def is_blood(c): return any(w in c.lower() for w in BLOOD_WORDS)

def half_args(a, half):
    g = lambda k: getattr(a, f"{half}_{k}")
    if g("grn") and g("red"): return ["--grn", g("grn"), "--red", g("red")]
    if g("betas"): return ["--betas", g("betas")]
    if g("pat"): return ["--pat", g("pat"), "--dev-seq-values", str(a.min_calls)]
    if g("geo_pat"): return ["--geo-pat", g("geo_pat"), "--dev-seq-values", str(a.min_calls)]
    return None

def read_half(a, half, specimen, extra):
    src = half_args(a, half)
    if src is None: return None, f"no {half.replace('_', ' ')} given"
    d = os.path.join(a.out, half); os.makedirs(d, exist_ok=True); b = os.path.join(d, "bundle.json")
    cmd = [sys.executable, RS] + src + ["--specimen", specimen, "--id", f"{a.id}_{half}", "--cell-panel", "atlas",
                                        "--out", os.path.join(d, "report.html"), "--bundle", b] + extra
    print("run_draw:", " ".join(cmd[1:]), flush=True)
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode or not os.path.exists(b): return None, f"run_sample.py exit {r.returncode}"
    return json.load(open(b)), None

def sources(B):
    sh = {x["cell"]: x for x in (B or {}).get("cell_shares", {}).get("cells", []) or []}
    me = {x["cell"]: x for x in ((B or {}).get("met_a_each_cell") or {}).get("cells", []) or []}
    return sh, me

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", required=True); ap.add_argument("--out", required=True); ap.add_argument("--min-calls", type=int, default=1)
    for h in ("buffy", "plasma"):
        for k in ("grn", "red", "betas", "pat", "geo-pat"): ap.add_argument(f"--{h}-{k}", default=None)
    argv = sys.argv[1:]; extra = argv[argv.index("--") + 1:] if "--" in argv else []; argv = argv[:argv.index("--")] if "--" in argv else argv
    a = ap.parse_args(argv); a.out = os.path.abspath(a.out); os.makedirs(a.out, exist_ok=True)
    for h in ("buffy", "plasma"):   # run_sample.py runs from its own folder, so file paths are made absolute
        for k in ("grn", "red", "betas", "pat"):
            v = getattr(a, f"{h}_{k}")
            if v: setattr(a, f"{h}_{k}", os.path.abspath(v))
    BC, ebc = read_half(a, "buffy", "buffy coat", extra); PL, epl = read_half(a, "plasma", "plasma", extra)
    sb, mb = sources(BC); sp, mp = sources(PL)
    rec = {"draw": a.id, "flag": "DEVELOPMENT - not commissioned (to-do step 40): every reading keeps its own flags",
           "buffy_coat": {"read": BC is not None, "why_not": ebc, "iam_a": (BC or {}).get("iam_a"),
                          "sources": {c: dict(share=x["share"], interval_95=x.get("interval_95"), met_a=(mb.get(c) or {}).get("A")) for c, x in sb.items()}},
           "plasma": {"read": PL is not None, "why_not": epl, "iam_a": (PL or {}).get("iam_a"),
                      "sources": {c: dict(share=x["share"], interval_95=x.get("interval_95"), met_a=(mp.get(c) or {}).get("A")) for c, x in sp.items()}}}
    if BC is not None and PL is not None:   # what the plasma holds that the person's own blood cells do not
        un = {}
        for c, x in sp.items():
            own = (sb.get(c) or {}).get("share", 0.0)
            if (not is_blood(c) and x["share"] > 0) or (is_blood(c) and x.get("interval_95") and x["interval_95"][0] > own):
                un[c] = dict(plasma_share=x["share"], buffy_coat_share=own, met_a=(mp.get(c) or {}).get("A"), blood_cell=is_blood(c))
        rec["plasma_not_explained_by_own_blood_cells"] = dict(sorted(un.items(), key=lambda kv: -kv[1]["plasma_share"]))
    json.dump(rec, open(os.path.join(a.out, "draw.json"), "w"), indent=1, default=str)
    L = [f"draw {a.id}: {rec['flag']}"]
    for h in ("buffy_coat", "plasma"):
        R = rec[h]
        if not R["read"]: L.append(f"{h.replace('_', ' ')}: not read ({R['why_not']})"); continue
        top = sorted(R["sources"].items(), key=lambda kv: -kv[1]["share"])
        L.append(f"{h.replace('_', ' ')}: {sum(1 for _, v in top if v['share'] >= 0.001)} sources above 0.001; " + ", ".join(
            f"{c} {v['share']:.3f}" + (f" (Met-A {v['met_a']:.3f})" if v.get("met_a") is not None else "") for c, v in top[:8]))
        q = R.get("iam_a") or {}
        L.append(f"   IAM-A: {q.get('A')} {q.get('state', '')}" + (f", counting range {q['counting']['A_95']}" if (q.get('counting') or {}).get('A_95') else "")
                 + (f" ({str(q.get('refusal'))[:120]})" if q.get("A") is None and q else ""))
    if "plasma_not_explained_by_own_blood_cells" in rec:
        U = rec["plasma_not_explained_by_own_blood_cells"]
        L.append("plasma sources the person's own blood cells do not explain: " + (", ".join(f"{c} {v['plasma_share']:.3f}" for c, v in list(U.items())[:10]) or "none"))
    open(os.path.join(a.out, "draw_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
    return 0 if (BC is not None or PL is not None) else 1

if __name__ == "__main__": sys.exit(main())
