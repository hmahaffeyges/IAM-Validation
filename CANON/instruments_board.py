"""The candidate-instrument board: writes INSTRUMENTS.md from CANON/instruments.json, so the priority list has one source.

Checks (run by status_check): every record named exists, every entry has a falsifier ("breaks") and a next step, and the generated
board matches the register. Usage: python3 CANON/instruments_board.py [--check]
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
SRC = os.path.join(HERE, "instruments.json"); OUT = os.path.join(ROOT, "Biological_Physics", "MethylPhys", "INSTRUMENTS.md")

def build(base=None):
    J = json.load(open(SRC, encoding="utf-8"))
    P = []
    for x in J["instruments"]:
        for k in ("breaks", "next"):
            if not x.get(k): P.append(f"[instruments] {x['id']}: no {k}")
        if x.get("rec") and not os.path.isfile(os.path.join(ROOT, x["rec"])): P.append(f"[instruments] {x['id']}: record {x['rec']} does not exist")
    if J["gate"].get("rec") and not os.path.isfile(os.path.join(ROOT, J["gate"]["rec"])): P.append("[instruments] gate: record does not exist")
    L = [f"# {J['title']}", "", "_Generated from `CANON/instruments.json` by `CANON/instruments_board.py`; do not edit by hand._", "",
         J["_rule"].split(". Written")[0] + ".", ""]
    for x in sorted(J["instruments"], key=lambda y: y["p"]):
        L += [f"## {x['p']}. {x['name']}", "",
              f"- **Reads:** {x.get('reads', 'MISSING')}",
              f"- **Substrate:** {x.get('substrate', 'MISSING')} · **Cost:** {x.get('cost', 'MISSING')}",
              f"- **Where it stands:** {x.get('status', 'MISSING')}",
              f"- **Still open:** {x.get('open', 'MISSING')}",
              f"- **Advantage:** {x.get('advantage', 'MISSING')}",
              f"- **Blind spot:** {x.get('blind', 'MISSING')}",
              f"- **What would break it:** {x.get('breaks') or 'MISSING - this is a demonstration, not a test'}",
              f"- **Next step:** {x.get('next') or 'MISSING'}"]
        if x.get("rec"): L.append(f"- **Record:** [{os.path.basename(x['rec'])}]({os.path.relpath(os.path.join(ROOT, x['rec']), base or os.path.dirname(OUT))})")
        L.append("")
    L += [f"## {J['gate']['name']}", "", J["gate"]["text"], "", f"_{J['closing']}_", ""]
    return "\n".join(L), P

if __name__ == "__main__":
    sys.path.insert(0, HERE); import pointer
    _, P = build(); md = pointer.text(OUT, os.path.join(ROOT, "INSTRUMENTS.md"), "The candidate instruments board")   # merged into the catalogue 2026-10-16
    if "--check" in sys.argv:
        cur = open(OUT, encoding="utf-8").read() if os.path.isfile(OUT) else ""
        if cur.strip() != md.strip(): P.append("[instruments] Biological_Physics/MethylPhys/INSTRUMENTS.md is not the pointer: run python3 CANON/instruments_board.py")
        print("instruments_board: " + ("ok" if not P else "REFUSED:\n  " + "\n  ".join(P))); sys.exit(1 if P else 0)
    open(OUT, "w", encoding="utf-8").write(md)
    print("instruments_board: wrote INSTRUMENTS.md" + ("" if not P else " WITH PROBLEMS:\n  " + "\n  ".join(P)))
