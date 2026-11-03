"""The IAM Sequencing Atlas, 57 cells: the granulocytes' rows, counted by the one command from their own three inputs (--build
seq-metaa-donors, seq-metaa-counts, seq-terr-halves on gran_inputs.csv), added to the 56 cells' per-input tables. Each of these tables is
counted input by input and cell by cell, so a new cell's rows are added without recounting the others; the 56 cells' rows are kept byte for
byte (checked). Usage: python3 atlas57_add_rows.py GRAN_DIR (holding metaa_donor_counts_gran.parquet, metaa_half_counts_gran.parquet,
terr_half_counts_gran.parquet)"""
import os, sys, json, pandas as pd
CH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "chain")); RM = os.path.join(CH, "Runtime Matrices", "Seq_Atlas")
T = {"metaa_donor_counts_gran.parquet": os.path.join(RM, "metaa_donor_counts", "metaa_donor_counts_v1.parquet"),
     "metaa_half_counts_gran.parquet": os.path.join(RM, "metaa_half_counts", "metaa_half_counts_v1.parquet"),
     "terr_half_counts_gran.parquet": os.path.join(RM, "terr_half_counts", "terr_half_counts_v1.parquet")}
rep = {}
for g, t in T.items():
    A = pd.read_parquet(t); G = pd.read_parquet(os.path.join(sys.argv[1], g))
    assert list(G.columns) == list(A.columns), (g, list(G.columns), list(A.columns))
    O = G[G.cell != "granulocytes"]; cnt = [c for c in G.columns if c in ("m", "n", "mA", "nA", "mB", "nB")]   # a builder that writes every cell's
    assert (O[cnt].astype("int64") == 0).all().all(), (g, "a cell other than granulocytes has reads")    # rows writes zeros for cells not given
    G = G[G.cell == "granulocytes"]; assert len(G), g
    A = A[A.cell != "granulocytes"]; n0 = len(A); B = pd.concat([A, G.astype(A.dtypes.to_dict())], ignore_index=True)
    assert B.iloc[:n0].equals(A.reset_index(drop=True)), g
    B.to_parquet(t, index=False); rep[os.path.basename(t)] = dict(rows_kept=n0, rows_added=len(G), cells=int(B.cell.nunique()))
print(json.dumps(rep, indent=1)); json.dump(rep, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "atlas57_add_rows.json"), "w"), indent=1)
