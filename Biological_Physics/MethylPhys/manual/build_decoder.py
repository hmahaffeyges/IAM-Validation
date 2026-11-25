"""build_decoder.py - the Chain Decoder of the Official IAM Operations Manual (to-do step 25, 2026-10-20).
Merges the hand-written explanations (manual/decoder_text.md: '## ' stage sections and one '### file::function' entry per function)
with the facts read from the code (manual/decoder_inventory.json, made by decoder_inventory.py from the chain itself and the traced
runs) into manual/OM_v4_decoder.md, in the order the traced run first entered each file.
  python3 build_decoder.py            write manual/OM_v4_decoder.md
  python3 build_decoder.py --check    refuse when a function the chain loads has no entry, an entry names a function that does not exist,
                                      or the committed output differs from what this writes"""
import json, os, re, sys
import sys as _cs; _cs.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "CANON")); from canon_fill import fill as _canon_fill   # 2026-11-06: physics values only from the canon
H = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.join(H, "decoder_text.md"); INV = os.path.join(H, "decoder_inventory.json")
OUT = os.path.join(H, "OM_v4_decoder.md")

def parse():
    t = _canon_fill(open(SRC, encoding="utf-8").read()); t = re.sub(r"<!--.*?-->\n?", "", t, flags=re.S)
    head, *rest = re.split(r"(?m)^(?=## )", t); stages = []
    for block in rest:
        title = block.split("\n", 1)[0][3:].strip(); body = block.split("\n", 1)[1] if "\n" in block else ""
        prose, *ents = re.split(r"(?m)^### (?=\S+\.(?:py|R)::)", body); E = {}   # an entry is '### file::function'; other ### headings are prose
        for e in ents:
            k, _, txt = e.partition("\n"); E[k.strip()] = txt.strip()
        stages.append(dict(title=title, prose=prose.strip(), entries=E))
    return head.strip(), stages

def facts(x):
    calls = ", ".join(f"{r} run {n:,}" for r, n in x["traced_calls"].items()) or "not called on the traced runs"
    L = [f"`{x['file']}` line {x['line']} ({x['lines']} lines) · arguments: {', '.join(x['args']) or 'none'} · calls: {calls}"]
    if x["constants"]: L.append("constants read: " + ", ".join(f"`{c}`" for c in x["constants"]))
    if x["data_files"]: L.append("files named: " + ", ".join(f"`{d}`" for d in x["data_files"]))
    return "*Code:* " + " · ".join(L)

RS = os.path.join(H, "..", "chain", "MethylPhys_Interface", "run_sample.py")

def options():
    """2026-11-09 (author: 'any and all changes we make to the chain is also updated in the decoder'): every option of the one command, read
    from run_sample.py's own definitions at build time (its help text, its default), and whether the default chain turns it on (the block
    under 'if a.full_chain:'), so an option, a help text or a default cannot change without the Decoder changing with it."""
    import ast
    src = open(RS, encoding="utf-8").read(); tree = ast.parse(src); rows = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "add_argument" and n.args and isinstance(n.args[0], ast.Constant) and str(n.args[0].value).startswith("--"):
            kw = {k.arg: k.value for k in n.keywords}; h = kw.get("help")
            try: h = ast.literal_eval(h) if h is not None else ""
            except Exception: h = ast.unparse(h)
            d = kw.get("default"); act = kw.get("action")
            d = ("off" if act is not None and ast.unparse(act) == "'store_true'" else (ast.unparse(d) if d is not None else "none"))
            rows.append((n.args[0].value, " ".join(str(h).split()), d))
    i = src.find("    if a.full_chain:"); j = src.find("\n    if ", i + 10) if i >= 0 else -1
    blk = src[i:j] if i >= 0 else ""; on = set(re.findall(r"a\.([a-z_]+)\s*=", blk))
    return sorted(rows), on

sys.path.insert(0, H)
def build():
    head, stages = parse(); inv = json.load(open(INV, encoding="utf-8"))
    key = {f"{f['file'].split('/')[-1]}::{x['function']}": x for f in inv["files"] for x in f["functions"]}
    have = {k for s in stages for k in s["entries"]}
    missing = [k for k in key if k not in have]; unknown = sorted(have - set(key))
    out = [head, "", (lambda _p: "#" + _canon_fill(open(_p, encoding="utf-8").read()).strip())(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "CANON", "physics_root.md")), "",   # 2026-11-03: the physics every instrument rests on
           "## The chain in numbers", "",
           f"Read from the code at build time: the one command can load **{inv['n_files']} Python files** holding **{inv['n_functions']} functions** "
           f"(nested ones included); {len(key) - len(missing)} of them have their entry in this Decoder, and a push is refused when one does not. "
           "Called: how many of a file's functions each traced run used (the array run on a healthy whole blood; the sequencing run on the "
           "release check's planted BAM).", "",
           "| file | lines | functions | called, array run | called, sequencing run |", "|---|---|---|---|---|"]
    for f in inv["files"]:
        fn = f["functions"]; out.append(f"| `{f['file']}` | {f['lines']:,} | {len(fn)} | {sum(1 for x in fn if x['traced_calls'].get('array'))} | {sum(1 for x in fn if x['traced_calls'].get('bam'))} |")
    out.append("")
    rows, on = options(); undoc = [r[0] for r in rows if not r[1]]
    out += ["## The one command's options", "", f"Read from `run_sample.py`'s own option definitions at build time: **{len(rows)} options**. Without `--commissioned-only` "
            "the full development chain runs on every sample (each development reading flagged); the last column says whether that default turns the option on. "
            "A push is refused when an option has no description.", "", "| option | default | on in the default chain | what it does |", "|---|---|---|---|"]
    for o, h, d in rows:
        out.append(f"| `{o}` | {d} | {'yes' if o[2:].replace('-', '_') in on else ''} | {h.replace('|', '/')} |")
    out.append("")
    for s in stages:
        out += ["## " + s["title"], "", s["prose"], ""]
        for k, txt in s["entries"].items():
            if k in key: out += [f"### `{k}`", "", txt, "", facts(key[k]), ""]
    import instrument_records as _IR   # edit 243.7: every instrument's full record, inputs and outputs included
    _RM = os.path.join(H, "..", "chain", "Runtime Matrices")   # edit 255: every reference folder, its README and its files, read at build time
    import reference_inventory as _RI   # edit 257
    out += _RI.markdown(level=2, counts=True)
    out += ["## Reference folders and how each was built", "", "Every folder under `chain/Runtime Matrices`, with its README in full and its files, read when this Decoder is built "
            "(a reference built before any chain code reads it is documented here from the day it lands).", ""]
    for _d in sorted(x for x in os.listdir(_RM) if os.path.isdir(os.path.join(_RM, x)) and not x.startswith((".", "archive"))):
        _rd = os.path.join(_RM, _d, "README.md"); _t = open(_rd, encoding="utf-8").read().strip() if os.path.exists(_rd) else "No README yet."
        _t = re.sub(r"(?m)^(#+) ", lambda m: "#" * min(6, len(m.group(1)) + 3) + " ", _t)
        _f = sorted(f for f in os.listdir(os.path.join(_RM, _d)) if not f.startswith(".") and os.path.isfile(os.path.join(_RM, _d, f)))
        out += [f"### {_d}", "", _t, "", "Files: " + (", ".join(f"`{f}`" for f in _f[:60]) + (f" and {len(_f) - 60} more" if len(_f) > 60 else "")), ""]
    out += ["## Every sequencing instrument: full record, inputs and outputs", "", "Generated from the instruments database (manual/instrument_records/), never typed here.", ""] + _IR.markdown(level=3, io=True)
    return "\n".join(out).rstrip() + "\n", missing, unknown, undoc

# ---- 2026-11-16 (author: the Decoder 'IS the CANON document for the chain code ... a researcher can take it and work out the chain and math
# on his own'; 'make the code demand it'). Every input the chain reads must have its own Decoder entry with its physics: each data file under
# chain/Runtime Matrices or chain/FROZEN_INPUTS named in chain code (a family of per-type files counts once, by its folder), and every canon key
# chain code reads. An entry is a '#### Input: <name>' block in decoder_text.md with all seven labeled lines filled. For a canon key the entry's
# Kind must agree with the canon's own status for that key.
INPUT_FIELDS = ("Kind:", "Register:", "What it is:", "Physics / equation:", "How the chain uses it:", "Reproduce it:", "Status and open parts:")
KINDS = ("derived", "calculated", "measured", "calibrated", "development")
def chain_inputs():
    CHD = os.path.join(H, "..", "chain"); code = ""
    for root, _, fs in os.walk(CHD):
        if "archive" in root or "__pycache__" in root or "trace" in root: continue
        for f in fs:
            if f.endswith(".py"): code += open(os.path.join(root, f), encoding="utf-8", errors="replace").read() + "\n"
    files = set()
    for top in ("Runtime Matrices", "FROZEN_INPUTS"):
        base = os.path.join(CHD, top)
        if not os.path.isdir(base): continue
        for root, _, fs in os.walk(base):
            if "archive" in root: continue
            for f in fs:
                if f.endswith(".stamp.json") or f.startswith("."): continue
                if f in code or os.path.splitext(f)[0] in code:
                    files.add(f)
                elif os.path.basename(root) in code and root != base:
                    files.add(os.path.basename(root) + "/")      # a family read by folder
    keys = set(re.findall(r"""\[["']constants["']\]\[["']([A-Za-z0-9_]+)["']\]|canon_value\(["']([A-Za-z0-9_]+)["']""", code))
    keys = {a or b for a, b in keys}
    return sorted(files), sorted(keys)
def input_entries(src):
    out = {}
    for m in re.finditer(r"^#### Input: (.+?)\n(.*?)(?=^#### |^### |^## |\Z)", src, re.S | re.M):
        out[m.group(1).strip().strip("`")] = m.group(2)
    return out
def check_inputs():
    src = open(SRC, encoding="utf-8").read(); E = input_entries(src); bad = []
    files, keys = chain_inputs()
    try:
        C = json.load(open(os.path.join(H, "..", "..", "..", "CANON", "iam_canon.json"), encoding="utf-8"))["constants"]
    except Exception:
        C = {}
    def entry_for(name):
        for k, body in E.items():
            if name == k or name in [x.strip().strip("`") for x in re.split(r"[,;]| and ", k)] or (name.endswith("/") and k.rstrip("/").endswith(name.rstrip("/"))):
                return body
        return None
    for name in files + keys:
        body = entry_for(name)
        if body is None: bad.append(f"no Decoder input entry for {name}"); continue
        for fld in INPUT_FIELDS:
            m = re.search(r"^\*?\*?" + re.escape(fld) + r"\*?\*?\s*(.+)$", body, re.M)
            if not m or len(m.group(1).strip()) < 3: bad.append(f"{name}: '{fld}' missing or empty")
        k = re.search(r"^\*?\*?Kind:\*?\*?\s*([a-z]+)", body, re.M)
        if k and k.group(1) not in KINDS: bad.append(f"{name}: Kind '{k.group(1)}' is not one of {KINDS}")
        if name in C and k:
            st = str(C[name].get("status", "")).lower()
            if ("derived" in st and k.group(1) == "measured") or (("measured" in st or "calibrated" in st) and "derived" not in st and k.group(1) == "derived"):
                bad.append(f"{name}: Decoder says {k.group(1)}, canon status says '{st[:40]}'")
    return bad, len(files), len(keys)

if __name__ == "__main__":
    txt, missing, unknown, undoc = build()
    if "--check" in sys.argv:
        _ib, _nf, _nk = check_inputs(); print(f"build_decoder: inputs: {_nf} input files or families, {_nk} canon keys; {len(_ib)} without a complete entry{(': ' + '; '.join(_ib[:8])) if _ib else ''}")
        bad = []
        if missing: bad.append(f"{len(missing)} chain function(s) have no Decoder entry, e.g. {missing[:4]}")
        if undoc: bad.append(f"{len(undoc)} option(s) of the one command have no description (help text): {undoc[:6]}")
        if unknown: bad.append(f"entries name functions the chain does not have: {unknown[:4]}")
        if not os.path.exists(OUT) or open(OUT, encoding="utf-8").read() != txt: bad.append("manual/OM_v4_decoder.md differs (run python3 Biological_Physics/MethylPhys/manual/build_decoder.py)")
        print("build_decoder: " + ("ok (every chain function has its entry)" if not bad else "REFUSED: " + "; ".join(bad))); sys.exit(1 if bad else 0)
    open(OUT, "w", encoding="utf-8").write(txt)
    print(f"build_decoder: wrote manual/OM_v4_decoder.md; {len(missing)} function(s) still without an entry{(' (' + ', '.join(map(str, missing[:5])) + ')') if missing else ''}; unknown entries: {unknown}")
