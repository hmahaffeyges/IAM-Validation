"""build_decoder.py - the Chain Decoder of the Official IAM Operations Manual (to-do step 25, 2026-10-20).
Merges the hand-written explanations (manual/decoder_text.md: '## ' stage sections and one '### file::function' entry per function)
with the facts read from the code (manual/decoder_inventory.json, made by decoder_inventory.py from the chain itself and the traced
runs) into manual/OM_v4_decoder.md, in the order the traced run first entered each file.
  python3 build_decoder.py            write manual/OM_v4_decoder.md
  python3 build_decoder.py --check    refuse when a function the chain loads has no entry, an entry names a function that does not exist,
                                      or the committed output differs from what this writes"""
import json, os, re, sys
H = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.join(H, "decoder_text.md"); INV = os.path.join(H, "decoder_inventory.json")
OUT = os.path.join(H, "OM_v4_decoder.md")

def parse():
    t = open(SRC, encoding="utf-8").read(); t = re.sub(r"<!--.*?-->\n?", "", t, flags=re.S)
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

def build():
    head, stages = parse(); inv = json.load(open(INV, encoding="utf-8"))
    key = {f"{f['file'].split('/')[-1]}::{x['function']}": x for f in inv["files"] for x in f["functions"]}
    have = {k for s in stages for k in s["entries"]}
    missing = [k for k in key if k not in have]; unknown = sorted(have - set(key))
    out = [head, "", "## The chain in numbers", "",
           f"Read from the code at build time: the one command can load **{inv['n_files']} Python files** holding **{inv['n_functions']} functions** "
           f"(nested ones included); {len(key) - len(missing)} of them have their entry in this Decoder, and a push is refused when one does not. "
           "Called: how many of a file's functions each traced run used (the array run on a healthy whole blood; the sequencing run on the "
           "release check's planted BAM).", "",
           "| file | lines | functions | called, array run | called, sequencing run |", "|---|---|---|---|---|"]
    for f in inv["files"]:
        fn = f["functions"]; out.append(f"| `{f['file']}` | {f['lines']:,} | {len(fn)} | {sum(1 for x in fn if x['traced_calls'].get('array'))} | {sum(1 for x in fn if x['traced_calls'].get('bam'))} |")
    out.append("")
    for s in stages:
        out += ["## " + s["title"], "", s["prose"], ""]
        for k, txt in s["entries"].items():
            if k in key: out += [f"### `{k}`", "", txt, "", facts(key[k]), ""]
    return "\n".join(out).rstrip() + "\n", missing, unknown

if __name__ == "__main__":
    txt, missing, unknown = build()
    if "--check" in sys.argv:
        bad = []
        if missing: bad.append(f"{len(missing)} chain function(s) have no Decoder entry, e.g. {missing[:4]}")
        if unknown: bad.append(f"entries name functions the chain does not have: {unknown[:4]}")
        if not os.path.exists(OUT) or open(OUT, encoding="utf-8").read() != txt: bad.append("manual/OM_v4_decoder.md differs (run python3 Biological_Physics/MethylPhys/manual/build_decoder.py)")
        print("build_decoder: " + ("ok (every chain function has its entry)" if not bad else "REFUSED: " + "; ".join(bad))); sys.exit(1 if bad else 0)
    open(OUT, "w", encoding="utf-8").write(txt)
    print(f"build_decoder: wrote manual/OM_v4_decoder.md; {len(missing)} function(s) still without an entry; unknown entries: {unknown}")
