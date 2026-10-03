#!/usr/bin/env python3
"""Recompute every number printed in docs/book/part4/p4_00b_astrogenetics.tex
and in its insertion blocks (MANIFEST). Run from the repository root:
    python3 docs/verification/scripts/verify_astrogenetics_book.py
Records read (repository paths):
  Biological_Physics/MethylPhys/doors/PROC_TUMOUR_01_OUTCOME.md         (tumour/normal copy error, table rows)
  Biological_Physics/MethylPhys/doors/PROC_DNMT_01_PARTB/dnmt_b_pairs.csv (IAM-A under DNMT1 block)
  Biological_Physics/MethylPhys/doors/PROC_LINES_02_channels/imr90_channels.csv (IMR90 channels)
Constants that are not in a record carry their book source in a comment.
"""
import csv, math, os, re, sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
D = os.path.join(ROOT, "Biological_Physics/MethylPhys/doors")
fails = []
def check(name, got, want, tol):
    ok = abs(got - want) <= tol
    print(f"{'OK  ' if ok else 'FAIL'} {name}: {got:.6g} (printed {want})")
    if not ok: fails.append(name)

H = lambda x: -x*math.log2(x) - (1-x)*math.log2(1-x)

# --- gauge points (book: p4_05_floorbreach, p4_06_gauge, p3_08_one_gauge) ---
Href = 0.330263          # metA_floors_v1_3.json floor, bits (p4_14_atlas l.20)
P = 1.099                # neutrophil position (p4_24_status)
Ehold = 3.41             # k_BT, measured holding energy (p4_06 l.129)
M = 54000/(8.314*310.15) # Delta G_ATP / R T_body (p3_08 l.138-139)
eps0 = 1/(1+math.exp(Ehold))
check("Met-A full surface 1/0.330263", 1/Href, 3.03, 0.005)
check("eps0 = 1/(1+e^3.41)", eps0, 0.032, 0.0005)
check("H(eps0) at 0.032, bits", H(0.032), 0.2043, 0.00005)
check("IAM-A full surface 1/(P H(eps0))", 1/(P*H(0.032)), 4.45, 0.005)
check("IAM-A floor 1/P", 1/P, 0.910, 0.0005)
check("M = 54000/(8.314*310.15)", M, 20.94, 0.005)
check("holding energy in Landauer units 3.41/ln2", Ehold/math.log(2), 4.9, 0.05)
check("phi = 3.41/20.94", Ehold/M, 0.163, 0.0005)
kB = 1.380649e-23
check("Landauer cost at 310.15 K (1e-21 J)", kB*310.15*math.log(2)/1e-21, 2.97, 0.005)

# --- tumour against the same patient's normal (PROC-TUMOUR-01 table) ---
rows = []
for line in open(os.path.join(D, "PROC_TUMOUR_01_OUTCOME.md")):
    m = re.match(r"\|\s*CRC(\d)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|", line)
    if m: rows.append((float(m.group(2)), float(m.group(3)), float(m.group(4))))
assert len(rows) == 6, rows
ratios = sorted(t/n for n, t, _ in rows)
med = (ratios[2]+ratios[3])/2
print("tumour pairs:", len(rows), "tumour > normal in", sum(t > n for n, t, _ in rows))
check("tumour/normal copy-error ratio, median", med, 1.148, 0.0005)
check("tumour/normal ratio, lowest", ratios[0], 1.067, 0.0005)
check("tumour/normal ratio, highest", ratios[-1], 1.328, 0.0005)
check("percent excess, lowest", 100*(ratios[0]-1), 7, 0.5)
check("percent excess, highest", 100*(ratios[-1]-1), 33, 0.5)

# --- DNMT1 block on single molecules (PROC-DNMT-01 Part B) ---
A = [float(r["A"]) for r in csv.DictReader(open(os.path.join(D, "PROC_DNMT_01_PARTB/dnmt_b_pairs.csv")))]
print("DNMT1 pairs:", len(A), "above 1.05:", sum(a > 1.05 for a in A))
check("IAM-A under DNMT1 block, lowest", min(A), 1.65, 0.005)
check("IAM-A under DNMT1 block, highest", max(A), 1.97, 0.005)

# --- IMR90 channels (PROC_LINES_02) ---
im = list(csv.DictReader(open(os.path.join(D, "PROC_LINES_02_channels/imr90_channels.csv"))))
def rng(state, col):
    v = [float(r[col]) for r in im if r["state"] == state]; return min(v), max(v)
for st, col, lo, hi in [("Senescent","A_unmeth",0.685,0.695),("Senescent","A_meth",0.941,1.016),
                        ("SV40","A_meth",1.077,1.120),("SV40","A_unmeth",0.587,0.664),("SV40","A_both",0.965,0.975),
                        ("Proliferating","A_both",0.997,1.002)]:
    a, b = rng(st, col)
    check(f"IMR90 {st} {col} low", a, lo, 0.0005); check(f"IMR90 {st} {col} high", b, hi, 0.0005)

# --- compact stars on their own gauge (p2_01_blackholes table; masses from its sources) ---
check("Sun's future white dwarf 0.54/0.6", 0.54/0.6, 0.90, 0.005)
check("Chandrasekhar 1.44/0.6", 1.44/0.6, 2.40, 0.005)
check("TOV 2.3/1.4", 2.3/1.4, 1.64, 0.005)
check("PSR J0740+6620 2.08/1.4", 2.08/1.4, 1.49, 0.005)

# --- cosmology, as printed in ch:baryon_chain and ch:lambda ---
eta, s_eta = 6.113, 0.037       # 18th chain, CMB only (p2_13b l.117)
eta_pl = 6.127                  # Planck 2018 (p2_13b l.140)
eta_bbn, s_bbn = 6.180, 0.195   # Cyburt 2016 Table IV, BBN+D (p2_13b l.148)
check("eta below Planck, percent", 100*(eta_pl-eta)/eta_pl, 0.2, 0.05)
check("eta vs nucleosynthesis, sigma", abs(eta_bbn-eta)/math.hypot(s_eta, s_bbn), 0.3, 0.05)
check("Lambda expression over measured, percent", 100*(1.142/1.133-1), 0.79, 0.01)
check("beta_m = Omega_m/2 (Omega_m 0.3153)", 0.3153/2, 0.15765, 0.000005)

print("FAILURES:", len(fails), fails)
sys.exit(1 if fails else 0)
