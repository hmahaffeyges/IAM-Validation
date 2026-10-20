"""trace_chain.py - the order the chain actually runs (2026-10-20, for the Chain Decoder, to-do step 25). Runs run_sample.py unchanged in this
process with a profiler attached and records, in first-call order, every chain function called (file, function, line, how many calls) and
every file the chain opens (data, matrices, references), in the order they are first opened. Nothing in the chain is changed or skipped.
Usage: python3 trace_chain.py OUT_PREFIX -- <the run_sample.py arguments>    -> OUT_PREFIX_calls.csv, OUT_PREFIX_files.csv"""
import os, sys, csv, builtins, runpy, collections
H = os.path.dirname(os.path.abspath(__file__)); OUT = sys.argv[1]; ARGS = sys.argv[sys.argv.index("--") + 1:]
first, count, files = {}, collections.Counter(), {}
def prof(frame, event, arg):
    if event != "call": return
    f = os.path.abspath(frame.f_code.co_filename)
    if not f.startswith(H) or f == os.path.abspath(__file__): return
    k = (os.path.relpath(f, H), frame.f_code.co_name, frame.f_code.co_firstlineno); count[k] += 1
    if k not in first: first[k] = len(first)
_open = builtins.open
def traced_open(p, *a, **k):
    try:
        q = os.path.abspath(os.fspath(p))
        if q not in files and not q.endswith(".py"): files[q] = (len(files), a[0] if a else k.get("mode", "r"))
    except Exception: pass
    return _open(p, *a, **k)
builtins.open = traced_open
sys.argv = [os.path.join(H, "MethylPhys_Interface", "run_sample.py")] + ARGS; sys.path[:0] = [H, os.path.join(H, "MethylPhys_Interface")]
code = 0
sys.setprofile(prof)
try: runpy.run_path(sys.argv[0], run_name="__main__")
except SystemExit as e: code = e.code
finally: sys.setprofile(None); builtins.open = _open
with _open(OUT + "_calls.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["order", "file", "function", "line", "calls"])
    for k, o in sorted(first.items(), key=lambda x: x[1]): w.writerow([o + 1, k[0], k[1], k[2], count[k]])
root = os.path.abspath(os.path.join(H, "..", "..", ".."))
with _open(OUT + "_files.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["order", "file", "mode"])
    for p, (o, m) in sorted(files.items(), key=lambda x: x[1][0]): w.writerow([o + 1, os.path.relpath(p, root) if p.startswith(root) else p, m])
print(f"trace: exit {code}; {len(first)} chain functions called ({sum(count.values())} calls); {len(files)} files opened")
