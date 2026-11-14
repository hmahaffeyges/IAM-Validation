"""X7, third reading (desk check, 2026-11-14). Hypothesis: the rate accounting (eq:rate, eq:Sdot with n = 7/2) fixes the RELATIVE growth of the
record, d ln rho_info / d ln a, not the entropy written. Then it is the record constraint itself (d ln E/d ln a = 1/a in the matter era), the first law
supplies the writing rate (eq:law_Sneed), and the two premises collapse into one with no conflict. Matter era: exact. Full LCDM (Planck Omega_m 0.3153):
integrate g(a) = rho_m D^n f H (per e-fold, eq:rate/eq:Sdot) and compare ln rho with ln E = 1 - 1/a. One normalization is free (the coefficient C)."""
import math, os
H = os.path.dirname(os.path.abspath(__file__)); Om, OL = 0.3153, 0.6847
Hh = lambda x: math.sqrt(Om * x**-3 + OL)
def D(x, N=3000):
    return 2.5 * Om * Hh(x) * sum(((((k + 0.5) / N) * x) * Hh(((k + 0.5) / N) * x)) ** -3 for k in range(N)) * x / N
def f(x, h=1e-4): return (math.log(D(x * (1 + h))) - math.log(D(x * (1 - h)))) / (2 * h)
g = lambda x, n: Om * x**-3 * D(x)**n * f(x) * Hh(x)
grid = [0.05 + 0.0095 * i for i in range(101)]                  # a from 0.05 to 1.0
out = ["Matter era, exact: g ~ a^(n - 9/2) = 1/a for n = 7/2, and d ln E/d ln a = 1/a: the relative-growth reading IS the record constraint."]
for n in (3.0, 3.5, 4.0):
    G = [g(x, n) for x in grid]
    I = [0.0]
    for i in range(1, len(grid)): I.append(I[-1] + 0.5 * (G[i] + G[i - 1]) * math.log(grid[i] / grid[i - 1]))
    lnrho = [v - I[-1] for v in I]                                  # zero today
    lnE = [1 - 1 / x for x in grid]
    C = sum(a * b for a, b in zip(lnrho, lnE)) / sum(a * a for a in lnrho)   # best single coefficient
    worst = max(abs(C * a - b) for a, b in zip(lnrho, lnE) if b > -3)      # over E > e^-3 (the part the data see)
    out.append(f"full LCDM, n = {n}: best coefficient C = {C:.3f}; largest |ln rho - ln E| where E > 0.05: {worst:.3f}; at a = 0.5: "
               f"{C * lnrho[grid.index(min(grid, key=lambda x: abs(x - 0.5)))]:.3f} vs ln E(0.5) = {1 - 2:.3f}")
open(os.path.join(H, "x7_relative_growth.md"), "w").write("# X7, third reading: the rate accounting as the record's relative growth\n\n" + "\n".join("- " + l for l in out) + "\n")
print("\n".join(out))
