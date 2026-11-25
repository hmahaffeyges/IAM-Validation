"""Reference inventory, generated from the reference files themselves when the SOP and Decoder are built.

For every reference folder the chain reads (chain/Runtime Matrices/<folder> and atlas/<version>), it reports: its files and total size, whether a table is
split into row parts, how a table holds cell types (one column per cell type and field, or one row per cell type), how many cell types it holds, and
which chain files name the folder (its readers). Nothing here is typed by hand, so it cannot fall behind the files.
"""
import os, re, json

HERE = os.path.dirname(os.path.abspath(__file__))
MP = os.path.normpath(os.path.join(HERE, ".."))
ROOTS = (("chain/Runtime Matrices", os.path.join(MP, "chain", "Runtime Matrices")), ("atlas", os.path.join(MP, "atlas")))
CHAIN = os.path.join(MP, "chain")


def _mb(b):
    return f"{b / 1e6:,.1f} MB" if b >= 1e5 else f"{b / 1e3:,.0f} kB"


def _parquet_shape(p):
    """Columns of a parquet table and how it holds cell types; read from its schema only (no data loaded)."""
    try:
        import pyarrow.parquet as pq
        sch = pq.read_schema(p); cols = list(sch.names); rows = pq.ParquetFile(p).metadata.num_rows
    except Exception:
        return None
    cellcols = sorted({c.split("|")[0] for c in cols if "|" in c})
    if cellcols:
        return dict(rows=rows, cols=len(cols), layout="one column per cell type and field", cells=len(cellcols))
    for k in ("cell", "cell_type", "celltype"):
        if k in cols:
            try:
                import pyarrow.parquet as pq
                n = len(set(pq.read_table(p, columns=[k]).column(0).to_pylist()))
            except Exception:
                n = None
            return dict(rows=rows, cols=len(cols), layout="one row per cell type and site", cells=n)
    return dict(rows=rows, cols=len(cols), layout="not by cell type", cells=None)


def _readers():
    """Chain files and the folder names they mention."""
    txt = {}
    for sub in ("", "MethylPhys_Interface"):
        d = os.path.join(CHAIN, sub)
        for f in sorted(os.listdir(d)):
            if f.endswith(".py"):
                txt[(sub + "/" if sub else "") + f] = open(os.path.join(d, f), encoding="utf-8", errors="replace").read()
    return txt


def inventory():
    T = _readers(); out = []
    for label, root in ROOTS:
        if not os.path.isdir(root):
            continue
        for d in sorted(x for x in os.listdir(root) if os.path.isdir(os.path.join(root, x)) and not x.startswith((".", "archive", "_"))):
            files = []
            for dp, _, fs in os.walk(os.path.join(root, d)):
                for f in fs:
                    if not f.startswith("."):
                        files.append(os.path.join(dp, f))
            size = sum(os.path.getsize(f) for f in files)
            parts = sorted({re.sub(r"_?part\d+", "_part*", os.path.basename(f)) for f in files if re.search(r"part\d+\.", os.path.basename(f))})
            shapes = {}
            for f in files:
                if f.endswith(".parquet"):
                    s = _parquet_shape(f)
                    if s:
                        shapes.setdefault(re.sub(r"_?part\d+", "_part*", os.path.basename(f)), s)
            pat = re.compile(r"""(?:["'/])""" + re.escape(d) + r"""(?=["'/])""")   # the folder named in a path or a quoted string, not in prose
            readers = sorted(k for k, v in T.items() if pat.search(v))
            picks = sorted(k for k in readers if re.search(r"columns=\s*\[", T[k]) or "read_table(" in T[k] and "columns=" in T[k])
            out.append(dict(root=label, folder=d, files=len(files), bytes=size, parts=parts, shapes=shapes, readers=readers, picks=picks))
    return out


def markdown(level=2, title="How the references are split and loaded", counts=True):
    H = "#" * level; I = inventory()
    tot = {}
    for r in I:
        tot[r["root"]] = tot.get(r["root"], 0) + r["bytes"]
    L = [f"{H} {title}", "",
         "Generated from the reference files when this document is built. "
         + "; ".join(f"`{k}/` holds {_mb(v)}" for k, v in tot.items()) + ". "
         "A copy of the repository carries every reference. A table that holds cell types as columns "
         "can be read for the cell types a sample needs only; the last column lists the readers whose code reads selected columns of a table. A large table is split "
         "into row parts so that each file stays under the repository's 100 MB file limit.", "",
         "| Folder | Files | Size | Row parts | Tables and how they hold cell types | Read by | Readers that read selected columns |", "|---|---|---|---|---|---|---|"]
    for r in I:
        sh = "; ".join(f"`{k}`: " + (f"{v['rows']:,} rows, {v['cols']} columns, " if counts else "") + f"{v['layout']}" + (f" ({v['cells']} cell types)" if v.get("cells") else "")
                       for k, v in sorted(r["shapes"].items())[:4]) + (f"; and {len(r['shapes']) - 4} more tables" if len(r["shapes"]) > 4 else "")
        pk = ", ".join(f"`{x}`" for x in r["picks"]) or "none"
        rd = ", ".join(f"`{x}`" for x in r["readers"][:6]) + (f" and {len(r['readers']) - 6} more" if len(r["readers"]) > 6 else "")
        L.append(f"| `{r['root']}/{r['folder']}` | {r['files']} | {_mb(r['bytes'])} | {', '.join(f'`{p}`' for p in r['parts']) or 'no'} | {sh or 'no tables'} | {rd or 'none found by name'} | {pk} |")
    L.append("")
    return L


if __name__ == "__main__":
    print("\n".join(markdown()))
