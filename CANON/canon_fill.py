"""CANON/canon_fill.py - fills {{canon:KEY}} and {{canon:KEY:FORMAT}} in any hand-written source from CANON/iam_canon.json, so no physics
value is ever typed by hand (author 2026-11-06: "all of the physics comes from a data base document in the repo ... so NOTHING should be hand
written"; the chain, the Decoder and the workflow flowchart use ONLY the canon for their physics).

  fill(text)             returns the text with every token replaced by the canon's constants[KEY]["value"], formatted (e.g. :.2f);
                         a key the canon does not hold stops the build (CanonKeyError), so a typo cannot print a wrong number.
  python3 CANON/canon_fill.py FILE   prints FILE filled (a check by eye).
Used by CANON/catalogue.py, CANON/physics_root_tex.py, sop/build_sop_v4.py and manual/build_decoder.py."""
import os, re, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
C = json.load(open(os.path.join(HERE, "iam_canon.json"), encoding="utf-8"))["constants"]
TOKEN = re.compile(r"\{\{canon:([A-Za-z0-9_]+)(?::([^}]+))?\}\}")
class CanonKeyError(KeyError): pass
def value(key, fmt=None):
    if key not in C: raise CanonKeyError(f"canon_fill: '{key}' is not in CANON/iam_canon.json constants")
    v = C[key]["value"]
    if fmt:
        try: return format(float(v), fmt)
        except (TypeError, ValueError): raise CanonKeyError(f"canon_fill: '{key}' is not a number, so format '{fmt}' cannot apply")
    return str(v)
def fill(text): return TOKEN.sub(lambda m: value(m.group(1), m.group(2)), text)
if __name__ == "__main__": print(fill(open(sys.argv[1], encoding="utf-8").read()))
