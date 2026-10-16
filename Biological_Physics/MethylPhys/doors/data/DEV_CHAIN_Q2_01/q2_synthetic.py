"""DEV-CHAIN-Q2-01 synthetic: chain Stage Q2 (IAM-A self-reference) on planted data, read ONLY through the chain.
 T1 the .pat reader equals the development counter (own_reads_loose.one) exactly on the same lines, and recovers planted loose shares
 T2 the aligned-read reader recovers planted loose share, held error (truth = the counter's rule on the planted molecules) and sequencing
    error, for gemBS (XB) on both strands and for Bismark (XM)
 T5 the tile percentiles (shape check) equal those of the planted tile means; under-covered tiles are left out
 T6 the sample-shape check: margin from healthy spread, read/refuse at the limit, unknown type and edited reference refused
 T3 build_line recovers a planted line; residual() reads a sample against it; an edited line file is refused
 T4 every output carries a chain stamp whose payload hash verifies
Writes q2_synthetic.json (stamped)."""
import os, sys, json, tempfile, importlib.util, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); CHAIN = os.path.abspath(os.path.join(H, "..", "..", "..", "chain")); sys.path.insert(0, CHAIN)
import stage_q2_selfref as Q2
rng = np.random.default_rng(15); R = {}; ok = {}
# ---- T1
spec = importlib.util.spec_from_file_location("orl", os.path.abspath(os.path.join(H, "..", "..", "..", "..", "..", "development", "loose_share_2026-10-15", "own_reads_loose.py")))
ORL = importlib.util.module_from_spec(spec); spec.loader.exec_module(ORL)
tiles = {"chr1": np.array([[w * 3000, w * 3000 + 2000] for w in range(100)])}; t1 = []
for share in (0.2, 0.5, 0.8):
    loose = set(rng.choice(100, int(share * 100), replace=False)); lines = []
    for w in range(100):
        beta = 0.45 if w in loose else 0.85
        for r in range(120):
            i = w * 3000 + int(rng.integers(0, 1990)); calls = "".join("C" if rng.random() < beta else "T" for _ in range(8)); lines.append(f"chr1\t{i}\t{calls}\t1")
    for r in range(4000): lines.append(f"chr2\t{r}\t{''.join('T' if rng.random() < 0.03 else 'C' for _ in range(10))}\t1")   # off the tiles (first version put them on chr1 tiles 0-1)
    a = Q2.count_pat(lines, tiles); b = ORL.one(lines, tiles)
    same = abs(a["loose_share"] - b["loose_0.7000"]) < 1e-12 and abs(a["held_err"] - b["held_err"]) < 1e-12 and a["held_opp"] == b["held_opp"]
    t1.append(dict(planted=share, chain=a["loose_share"], development=b["loose_0.7000"], held_chain=a["held_err"], held_dev=b["held_err"], identical=same, recovered=abs(a["loose_share"] - share) < 0.02))
ok["T1"] = all(x["identical"] and x["recovered"] for x in t1); R["T1"] = t1
# ---- T2
import pysam
W = tempfile.mkdtemp(); seq = []
for i in range(4000):
    b = list(rng.choice(list("AT"), 25)); b[12], b[13] = "C", "G"; seq.append("".join(b))
seq = "".join(seq); open(f"{W}/ref.fa", "w").write(">chr1\n" + "\n".join(seq[i:i + 60] for i in range(0, len(seq), 60)) + "\n"); pysam.faidx(f"{W}/ref.fa")
cp = Q2._cpg_index(seq)
def case(beta, eps, suberr, fmt):
    hdr = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": "chr1", "LN": len(seq)}]}; reads = []; TR = [0, 0]
    for n in range(6000):
        st = 25 * int(rng.integers(0, (len(seq) - 300) // 25)) + 13; tile = min(int(np.searchsorted(cp, st)) // Q2.TILE, 1); top = n % 2 == 0
        meth = {int(p): (rng.random() < beta[tile]) for p in cp[(cp >= st + 5) & (cp < st + 245)]}
        if eps is not None:
            meth = {p: (rng.random() >= eps) for p in meth}; cl = [v for _, v in sorted(meth.items())]
            if len(cl) >= 6 and sum(cl) >= 0.8 * len(cl): TR[1] += len(cl) - 2; TR[0] += sum(1 for i in range(1, len(cl) - 1) if not cl[i] and cl[i - 1] and cl[i + 1])
        for mate, (a, ln) in enumerate(((st, 150), (st + 100, 150))):
            s = list(seq[a:a + ln]); xm = ["."] * ln
            for j in range(ln):
                g = a + j
                if top and seq[g] == "C":
                    if g in meth: s[j] = "C" if meth[g] else "T"; xm[j] = "Z" if meth[g] else "z"
                    else: s[j] = "T"; xm[j] = "h"
                if (not top) and seq[g] == "G":
                    if (g - 1) in meth: s[j] = "G" if meth[g - 1] else "A"; xm[j] = "Z" if meth[g - 1] else "z"
                    else: s[j] = "A"; xm[j] = "h"
                if s[j] in "AT" and seq[g] in "AT" and rng.random() < suberr: s[j] = "C" if s[j] == "A" else "G"
            r = pysam.AlignedSegment(); r.query_name = f"m{n}"; r.query_sequence = "".join(s); r.reference_id = 0; r.reference_start = a
            r.cigarstring = f"{ln}M"; r.mapping_quality = 40; r.flag = 1 | 2 | (64 if mate == 0 else 128) | (16 if mate else 32)
            r.next_reference_id = 0; r.next_reference_start = st + 100 if mate == 0 else st; r.query_qualities = pysam.qualitystring_to_array("I" * ln)
            if fmt == "XB": r.set_tag("XB", "C" if top else "G")
            else: r.set_tag("XM", "".join(xm)); r.set_tag("XG", "CT" if top else "GA")
            reads.append(r)
    reads.sort(key=lambda r: r.reference_start)
    with pysam.AlignmentFile(f"{W}/t.bam", "wb", header=hdr) as o:
        for r in reads: o.write(r)
    pysam.index(f"{W}/t.bam")
    S = Q2.combine_bam([Q2.count_bam_region((f"{W}/t.bam", f"{W}/ref.fa", "chr1", 0, len(seq)))], "synthetic")
    truth = TR[0] / TR[1] if TR[1] else None; se = (truth / max(S["held_opp"], 1)) ** 0.5 if truth else None
    return S, truth, se
t2 = []
for fmt in ("XB", "XM"):
    for beta, eps, sub, wantL in (((0.40, 0.92), None, 0.002, 0.5), ((0.92, 0.92), None, 0.002, 0.0), ((1.0, 1.0), 0.03, 0.004, None)):
        S, truth, se = case(beta, eps, sub, fmt)
        good = (wantL is None or abs(S["loose_share"] - wantL) < 1e-9) and (truth is None or abs(S["held_err"] - truth) < 3 * se) and abs(S["sub_err"] - sub) < 0.5 * sub + 2e-4
        t2.append(dict(format=fmt, planted_loose=wantL, loose=S["loose_share"], planted_held_truth=truth, held=S["held_err"], planted_sub=sub, sub=S["sub_err"], ok=good))
ok["T2"] = all(x["ok"] for x in t2); R["T2"] = t2
# ---- T3
firm, slope = 0.040, 0.035; rows = []
for c in range(30):
    L = rng.uniform(0.02, 0.4)
    for s in range(3): rows.append(dict(cell=f"c{c}", loose_share=L + rng.normal(0, 0.002), held_err=firm + slope * L + rng.normal(0, 0.0005), held_opp=1_000_000))
J, C = Q2.build_line(pd.DataFrame(rows)); t3 = dict(planted=(firm, slope), built=(round(J["firm"], 5), round(J["slope"], 5)))
t3["recovered"] = abs(J["firm"] - firm) < 0.001 and abs(J["slope"] - slope) < 0.005
LD = Q2.LINE_DIR; Q2.LINE_DIR = tempfile.mkdtemp(); J["version"] = 0
Q2.write_json(J, os.path.join(Q2.LINE_DIR, "selfref_line_v0.json"))
rd = Q2.residual(dict(loose_share=0.2, held_err=firm + slope * 0.2 + 0.001)); t3["residual_of_planted_plus_0.001"] = round(rd["residual"], 5)
t3["residual_ok"] = abs(rd["residual"] - 0.001) < 0.0005
X = json.load(open(os.path.join(Q2.LINE_DIR, "selfref_line_v0.json"))); X["firm"] = X["firm"] * 0.7412824486
json.dump(X, open(os.path.join(Q2.LINE_DIR, "selfref_line_v0.json"), "w"))
try: Q2.predict(0.2); t3["edited_line_refused"] = False
except RuntimeError as e: t3["edited_line_refused"] = True; t3["refusal"] = str(e)
Q2.LINE_DIR = LD
ok["T3"] = t3["recovered"] and t3["residual_ok"] and t3["edited_line_refused"]; R["T3"] = t3
# ---- T4
o = Q2.stamp({"x": 1}); ok["T4"] = Q2.payload_sha(o) == o["chain_stamp"]["payload_sha256"] and len(o["chain_stamp"]["chain_sha"]) == 64
# ---- T5 (2026-10-15): the tile distribution's shape. 900 tiles at 0.85 and 100 at 0.50 with 400 calls each, plus 50 tiles at 0.10 with
# 150 calls (below MIN_TILE_CALLS, must be left out); the chain's percentiles must equal those of the 1,000 planted tile means.
tc = {}
for i in range(900): tc[("chr1", i)] = (0.85 * 400, 400)
for i in range(900, 1000): tc[("chr1", i)] = (0.50 * 400, 400)
for i in range(1000, 1050): tc[("chr1", i)] = (0.10 * 150, 150)
o5 = Q2._finish(tc, 0, 1); want = np.array([0.85] * 900 + [0.50] * 100)
t5 = dict(tile_p10=o5["tile_p10"], want_p10=float(np.quantile(want, 0.1)), tile_p50=o5["tile_p50"], want_p50=float(np.quantile(want, 0.5)),
          tile_p90=o5["tile_p90"], want_p90=float(np.quantile(want, 0.9)), tiles=o5["tiles"], loose_share=o5["loose_share"])
ok["T5"] = (o5["tiles"] == 1000 and abs(o5["loose_share"] - 0.1) < 1e-12
            and all(abs(t5[k] - t5["want_" + k[5:]]) < 1e-12 for k in ("tile_p10", "tile_p50", "tile_p90"))); R["T5"] = t5
# ---- T6 (2026-10-15): the sample-shape check. Planted reference: type X healthy p10 0.70, 0.72, 0.74 (spread 0.04); type Y 0.60, 0.61
# (spread 0.01); type Z one sample 0.80. Margin = 90% quantile of [0.04, 0.01] = 0.037. Checks: X at 0.67 read (limit 0.663), X at 0.65 refused,
# an unknown type not read, a record without a shape not read, and an edited stored reference refused.
recs = [dict(cell="X", tile_p10=v) for v in (0.70, 0.72, 0.74)] + [dict(cell="Y", tile_p10=v) for v in (0.60, 0.61)] + [dict(cell="Z", tile_p10=0.80)]
ref = Q2.build_shape_reference(recs); want_m = float(np.quantile([0.04, 0.01], 0.9))
a6 = Q2.shape_check(dict(tile_p10=0.67), "X", ref=ref); b6 = Q2.shape_check(dict(tile_p10=0.65), "X", ref=ref)
c6 = Q2.shape_check(dict(tile_p10=0.9), "W", ref=ref); d6 = Q2.shape_check(dict(), "X", ref=ref)
with tempfile.TemporaryDirectory() as td:
    keep = Q2.LINE_DIR; Q2.LINE_DIR = td
    Q2.write_json(ref, os.path.join(td, "shape_reference_v1.json")); J6 = json.load(open(os.path.join(td, "shape_reference_v1.json"))); J6["margin"] = 0.5
    json.dump(J6, open(os.path.join(td, "shape_reference_v1.json"), "w"))
    try: Q2.shape_reference(1); e6 = False
    except RuntimeError: e6 = True
    Q2.LINE_DIR = keep
t6 = dict(margin=ref["margin"], want_margin=want_m, x_067=a6["read"], x_065=b6["read"], unknown=c6["read"], no_shape=d6["read"], edited_refused=e6)
ok["T6"] = (abs(ref["margin"] - want_m) < 1e-12 and a6["read"] and not b6["read"] and not c6["read"] and not d6["read"] and e6); R["T6"] = t6
# ---- T7 the reading and the band (COM-IAMA-054): planted samples exactly on a planted line read A = 1; a planted spread gives the known percentiles
import numpy as _np
_J = {"firm": 0.04, "slope": 0.03, "version": 99, "axis": "raw"}
_orig = Q2.line; Q2.line = lambda version=None: _J
_on = [dict(loose_share=x, held_err=0.04 + 0.03 * x, lab="a") for x in _np.linspace(0.1, 0.9, 9)]
_k = _np.linspace(0.8, 1.2, 401); _pl = [dict(loose_share=0.5, held_err=float(k) * 0.055, lab=("a" if i % 2 else "b")) for i, k in enumerate(_k)]
t7 = dict(on_line=max(abs(Q2.reading(o)["A"] - 1) for o in _on), band=Q2.build_band(_pl))
Q2.line = _orig
ok["T7"] = t7["on_line"] < 1e-12 and abs(t7["band"]["lo"] - _np.percentile(_k, 2.5)) < 1e-9 and abs(t7["band"]["hi"] - _np.percentile(_k, 97.5)) < 1e-9 and t7["band"]["labs"] == {"a": 200, "b": 201}
R["T7"] = t7
# ---- T8 a result with number keys verifies after it is written and read back (the Hansen count run 2 failure, 2026-10-16)
import tempfile as _tf
_p = os.path.join(_tf.mkdtemp(), "t8.json"); Q2.write_json({"read_lengths": {101: 5, 99: 2, 100: 7}, "x": 0.1}, _p, manifest=["OTH-057"])
_o = json.load(open(_p)); ok["T8"] = Q2.payload_sha(_o) == _o["chain_stamp"]["payload_sha256"]; R["T8"] = ok["T8"]
R["ok"] = ok; R["verdict"] = "MET" if all(ok.values()) else "NOT MET"
Q2.write_json(R, os.path.join(H, "q2_synthetic.json"), manifest=["OTH-057", "COM-IAMA-053", "COM-IAMA-054"])
print(json.dumps(ok), R["verdict"]); print(json.dumps(t1)[:400]); [print(x) for x in t2]; print(t3)
