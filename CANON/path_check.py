"""path_check.py - every folder or file a script finds from its own location must exist (2026-10-20, after the deconv reproduction broke).
reproduce/heldout_arrays.py counted two levels up to the repository root; when reproduce/ moved into Biological_Physics/MethylPhys
(2026-10-19) that pointed at a folder that does not exist, and the reproduction failed on GitHub. The document link gate could not see it.
For each script below, every module-level assignment whose value is built from __file__ (os.path.dirname / join / abspath / normpath
and literals) is evaluated with __file__ set to the script, in order; each resulting path must exist. Refuses with the script, the line,
the name and the path it computed."""
import ast, glob, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GLOBS = ["Biological_Physics/MethylPhys/reproduce/*.py", "Biological_Physics/MethylPhys/chain/*.py", "Biological_Physics/MethylPhys/chain/MethylPhys_Interface/*.py",
         "Biological_Physics/MethylPhys/manual/*.py", "Biological_Physics/MethylPhys/kit/*.py"]

def uses_file(node): return any(isinstance(n, ast.Name) and n.id == "__file__" for n in ast.walk(node))
CREATED = (".venv", "/out", "/tmp")   # made by setup or at run time, not committed

def pure_path(node):
    """Only os.path.<fn>(...) calls, names, string literals and + of those: a path, not a file read or anything else."""
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            if not (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Attribute) and isinstance(f.value.value, ast.Name)
                    and f.value.value.id == "os" and f.value.attr == "path"): return False
        elif not isinstance(n, (ast.Name, ast.Constant, ast.Attribute, ast.Load, ast.BinOp, ast.Add, ast.Starred, ast.keyword, ast.Expression)): return False
    return True

def check(path):
    bad, ns = [], {"os": os, "__file__": path}
    try: tree = ast.parse(open(path, encoding="utf-8").read())
    except SyntaxError: return bad
    for st in tree.body:
        if not isinstance(st, ast.Assign) or not pure_path(st.value): continue
        tg = [t.id for t in st.targets if isinstance(t, ast.Name)]
        try: v = eval(compile(ast.Expression(st.value), path, "eval"), ns)
        except Exception: continue
        for t in tg: ns[t] = v
        if uses_file(st.value) or any(isinstance(n, ast.Name) and n.id in ns and n.id not in ("os",) and isinstance(ns.get(n.id), str) for n in ast.walk(st.value)):
            vals = [v] if isinstance(v, str) else []
            for x in vals:
                if (os.sep in x or "/" in x) and len(x) < 400 and not os.path.exists(x) and not any(c in x for c in CREATED):
                    bad.append(f"{os.path.relpath(path, ROOT)} line {st.lineno}: {', '.join(tg)} = {os.path.relpath(x, ROOT)} does not exist")
    return bad

if __name__ == "__main__":
    B = [b for g in GLOBS for p in sorted(glob.glob(os.path.join(ROOT, g))) for b in check(p)]
    print("path_check: " + ("ok (every self-located path exists)" if not B else f"REFUSED: {len(B)} path(s) do not exist:\n  " + "\n  ".join(B)))
    sys.exit(1 if B else 0)
