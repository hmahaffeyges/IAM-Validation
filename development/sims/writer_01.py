"""DEV-WRITER-01 step 1 (2026-10-10, before any enzyme measurement is looked up): the copy error a writer of discrimination D can hold.
D = (k_cat/K_M on hemimethylated CpG) / (k_cat/K_M on unmethylated CpG), measured on the enzyme outside the cell.
Model A, one discriminating selection per site per copy (the book's form eps0 = 1/(1+e^{E_hold/k_BT}); Hopfield's error of a single
   discrimination step, no proofreading): eps = 1/(1+D), so E_hold = k_BT ln D.
Model B, independent sites, kinetic window: opposite a methylated parent the site is written at rate D*k, opposite an unmethylated parent at
   rate k, for a window x = k*t. Per copy f = exp(-D x), g = 1 - exp(-x). The steady state of an independent site is UNIQUE,
   pi_M = g/(f+g) whatever the territory: one writer with these rates cannot hold a methylated territory (pi_M near 1) and an unmethylated one
   (pi_M near 0) at once. Shown below: pi_M at the same rates is the same in both territories, so 'error' in one is 'correctness' lost in the other.
   Holding both needs neighbour coupling (gain guided by methylated neighbours), so Model B with independent sites is not a model of the cell.
Usage: python3 writer_01.py"""
import numpy as np
E = lambda e: np.log((1 - e) / e)
print("Model A:   D     eps      E_hold (k_BT)")
for D in (3, 10, 20, 30, 40, 60, 100, 300):
    print(f"        {D:5d}   {1 / (1 + D):.4f}   {E(1 / (1 + D)):.2f}")
lo, hi = 1.0, 1e6
for _ in range(80):
    mid = np.sqrt(lo * hi); (lo, hi) = (mid, hi) if E(1 / (1 + mid)) < 3.41 else (lo, mid)
print(f"Model A: E_hold = 3.41 k_BT (canon) needs D = {lo:.1f}; factor-2 band D {lo / 2:.1f}-{lo * 2:.1f} = E_hold {np.log(lo / 2):.2f}-{np.log(lo * 2):.2f}")
print("Model B (independent sites): D=30; window x -> pi_M (same in methylated and unmethylated territory)")
for x in (0.001, 0.01, 0.05, 0.1, 0.3, 1.0):
    f, g = np.exp(-30 * x), 1 - np.exp(-x); print(f"   x {x:5.3f}  f {f:.4f}  g {g:.4f}  pi_M {g / (f + g):.4f}")
