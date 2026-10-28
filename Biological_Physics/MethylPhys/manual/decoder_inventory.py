"""decoder_inventory.py - the mechanical layer of the Chain Decoder (Operations Manual, to-do step 25, 2026-10-20).
Reads the chain's own code (never a description of it): for every Python file the one command loads, in the order a traced run first
entered it (chain/trace/), every function, nested ones included, with its line, arguments, length, first docstring line, the module
constants it reads, the data files it names, and how many times each traced run called it. Writes manual/decoder_inventory.json.
Usage: python3 decoder_inventory.py [--check]   (--check: refuse when the committed inventory differs from the code)"""
import ast, csv, json, os, re, sys, collections
H = os.path.dirname(os.path.abspath(__file__)); CH = os.path.join(H, "..", "chain"); OUT = os.path.join(H, "decoder_inventory.json")
TRACES = [("array", os.path.join(CH, "trace", "trace_calls.csv")), ("bam", os.path.join(CH, "trace", "trace_bam_calls.csv"))]
LOADED = ["MethylPhys_Interface/run_sample.py", "MethylPhys_Interface/info_sphere_viewer.py", "stage_0_intake.py", "stage_0_1_qc_handoff.py", "stage_1_idat_calibration.py", "conductor_v3.py",
          "stage_m_met_a.py", "stage_t_selftare.py", "stage_seq_values.py", "dev_stages.py", "dev_comp_methods.py", "stage_v4.py", "deconv_v2.py", "stage_deconv_v21.py", "sky_statistics.py",
          "stage_atlas_commission.py", "stage_q0_intake.py", "stage_q_iam_a.py", "stage_q2_selfref.py", "stage_coupling.py", "stage_provisional.py", "serial_mode.py", "stage_groupa.py",
          "MethylPhys_Interface/report_v3.py", "MethylPhys_Interface/report_v4.py", "MethylPhys_Interface/report_figs_v4.py", "MethylPhys_Interface/report_pdf_v4.py",
          "MethylPhys_Interface/report_globe_v4.py", "MethylPhys_Interface/globe_viewer.py",
          # 2026-10-22: loaded by the one command's development flags and build mode (--dev-metaa-atlas, --dev-genome-globe, --build), and the series runner
          "stage_metaa_atlas.py", "stage_q2_globe.py", "stage_info_sphere.py", "MethylPhys_Interface/reference_globes.py", "MethylPhys_Interface/run_series.py", "MethylPhys_Interface/run_draw.py"]
DATA = re.compile(r"[\w./ -]+\.(?:json|csv|parquet|txt|tsv|npz|pkl|xz|gz)\b")

def traced():
    calls, order = collections.defaultdict(dict), {}
    for route, p in TRACES:
        if not os.path.exists(p): continue
        for r in csv.DictReader(open(p)):
            k = (r["file"], r["function"], int(r["line"])); calls[k][route] = int(r["calls"])
            order.setdefault(r["file"], (route != "array", int(r["order"])))
    return calls, order

def functions(path):
    src = open(os.path.join(CH, path), encoding="utf-8").read(); t = ast.parse(src)
    consts = {n.targets[0].id for n in t.body if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id.isupper()}
    out = []
    def walk(node, prefix):
        for n in ast.iter_child_nodes(node):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                q = prefix + n.name; seg = ast.get_source_segment(src, n) or ""
                doc = (ast.get_docstring(n) or "").strip().split("\n\n")[0].replace("\n", " ")
                names = {x.id for x in ast.walk(n) if isinstance(x, ast.Name)}
                out.append(dict(file=path, function=q, name=n.name, line=n.lineno, end=n.end_lineno, args=[a.arg for a in n.args.args],
                                lines=n.end_lineno - n.lineno + 1, doc=doc[:400], constants=sorted(names & consts),
                                data_files=sorted(set(m.strip() for m in DATA.findall(seg) if "/" in m or m.count(".") == 1))[:12]))
                walk(n, q + ".")
            elif isinstance(n, ast.ClassDef): walk(n, prefix + n.name + ".")
            elif not isinstance(n, ast.Lambda): walk(n, prefix)   # a def inside an if, for, with or try block
    walk(t, ""); return out, len(src.split("\n"))

def build():
    calls, order = traced(); files = []
    for f in sorted(LOADED, key=lambda f: order.get(f, (2, LOADED.index(f)))):
        F, n = functions(f)
        for x in F:   # by name, each traced run on its own; when a name repeats in a file, that run's traced line nearest this one
            same = [y for y in F if y["name"] == x["name"]]; tc = {}   # (lines shift when a file is edited between traces)
            for route in sorted({r for v in calls.values() for r in v}):
                c = [(abs(k[2] - x["line"]), v[route]) for k, v in calls.items() if k[0] == f and k[1] == x["name"] and route in v]
                if c:
                    dist, cnt = min(c, key=lambda z: z[0])
                    if len(same) == 1 or dist < 15: tc[route] = cnt
            x["traced_calls"] = tc
        files.append(dict(file=f, lines=n, functions=F, first_entered=dict(zip(("route", "order"), ("bam" if order[f][0] else "array", order[f][1]))) if f in order else None))
    return dict(_about=__doc__.split("\n")[0], files=files, n_files=len(files), n_functions=sum(len(x["functions"]) for x in files))

if __name__ == "__main__":
    inv = build(); txt = json.dumps(inv, indent=1, ensure_ascii=False) + "\n"
    if "--check" in sys.argv:
        ok = os.path.exists(OUT) and open(OUT, encoding="utf-8").read() == txt
        print("decoder_inventory: " + ("ok" if ok else "REFUSED: manual/decoder_inventory.json differs from the chain's code (run python3 Biological_Physics/MethylPhys/manual/decoder_inventory.py)")); sys.exit(0 if ok else 1)
    open(OUT, "w", encoding="utf-8").write(txt); print(f"decoder_inventory: {inv['n_files']} files, {inv['n_functions']} functions")
