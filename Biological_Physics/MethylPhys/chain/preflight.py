"""chain/preflight.py - run before any long job (2026-10-23; author, after a 4-hour alignment ended on a missing package: "You MUST make sure this
does not continue to happen"). Reads every chain file's imports (ast, nothing executed) and checks that each outside package is installed. An
import inside a try block that catches ImportError is optional and skipped. Exit 1 naming each missing package and the file that needs it, so a
job stops in seconds, not after its long part.  Usage: python3 chain/preflight.py"""
import ast, os, sys, importlib.util
H = os.path.dirname(os.path.abspath(__file__)); D = [H, os.path.join(H, "MethylPhys_Interface")]
T = [os.path.join(H, "..", "atlas", "tools"), os.path.join(H, "..", "manual"), os.path.join(H, "..", "sop")]   # repository folders the chain adds to its path
local = {f[:-3] for d in D + [t for t in T if os.path.isdir(t)] for f in os.listdir(d) if f.endswith(".py")} | {"atlas", "Runtime Matrices"}
def optional(tree):
    s = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Try) and any(isinstance(h.type, ast.Name) and h.type.id in ("ImportError", "ModuleNotFoundError", "Exception") or h.type is None for h in n.handlers):
            for m in ast.walk(n):
                if isinstance(m, (ast.Import, ast.ImportFrom)): s.add(id(m))
    return s
need = {}
for d in D:
    for f in sorted(os.listdir(d)):
        if not f.endswith(".py") or "RETIRED" in f: continue
        p = os.path.join(d, f)
        try: t = ast.parse(open(p, encoding="utf-8").read())
        except SyntaxError as e: print(f"preflight: {f} does not compile: {e}"); sys.exit(1)
        opt = optional(t)
        for n in ast.walk(t):
            if id(n) in opt: continue
            names = [a.name for a in n.names] if isinstance(n, ast.Import) else ([n.module] if isinstance(n, ast.ImportFrom) and n.module and n.level == 0 else [])
            for m in names:
                top = m.split(".")[0]
                if top in local or top in sys.stdlib_module_names: continue
                need.setdefault(top, set()).add(f)
miss = {m: fs for m, fs in need.items() if importlib.util.find_spec(m) is None}
for m, fs in sorted(miss.items()): print(f"preflight: MISSING package '{m}', needed by {', '.join(sorted(fs)[:4])}")
print(f"preflight: {'REFUSED' if miss else 'ok'} ({len(need)} outside packages checked, {len(miss)} missing)"); sys.exit(1 if miss else 0)
