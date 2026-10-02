#!/usr/bin/env python3
"""Recomputes every number in Part 1's opening chapter (the encoding-surface ladder). CODATA 2018 via scipy.constants."""
import numpy as np, scipy.constants as C
hbar,c,G,k=C.hbar,C.c,C.G,C.k; Msun=1.98847e30; Mpc=3.0857e22; ln2=np.log(2)
lP=np.sqrt(hbar*G/c**3)
print(f"Planck length {lP:.4e} m; area per bit (Bekenstein-Hawking, in bits) 4 lP^2 ln2 = {4*lP**2*np.log(2):.3e} m^2")
rows=[]
# 1 cell: CpG methylation record, body temperature
T=310.15; N=19.6e6; e=k*T*ln2; rows.append(("cell (CpG record)",T,N,e,N*e))
atp=50e3/C.N_A   # ~50 kJ/mol in vivo free energy of ATP hydrolysis
print(f"cell: k_B T ln2 = {e:.3e} J/bit; N x = {N*e:.3e} J; = {N*e/atp:.2e} ATP at 50 kJ/mol")
# 2 room-temperature electronics and the Landauer experiments
T=300; rows.append(("device at 300 K",T,1,k*T*ln2,k*T*ln2))
print(f"300 K: k_B T ln2 = {k*300*ln2:.3e} J = {k*300*ln2/C.e*1e3:.2f} meV (Berut 2012 measured erasure heat approaching this bound)")
# 3 superconducting qubit at a dilution-refrigerator stage
T=0.015; rows.append(("qubit at 15 mK",T,1,k*T*ln2,k*T*ln2))
print(f"15 mK: k_B T ln2 = {k*T*ln2:.3e} J")
# 4 black holes
for m in (1.0,4.3e6):
    M=m*Msun; T=hbar*c**3/(8*np.pi*G*M*k); A=16*np.pi*(G*M/c**2)**2; N=A/(4*lP**2)/ln2
    rows.append((f"black hole {m:g} Msun",T,N,k*T*ln2,N*k*T*ln2))
    print(f"BH {m:g} Msun: T_H {T:.3e} K, bits {N:.3e}, N k T ln2/(M c^2) = {N*k*T*ln2/(M*c**2):.6f}")
# 5 cosmic horizon
H0=67.36e3/Mpc; T=hbar*H0/(2*np.pi*k); R=c/H0; A=4*np.pi*R**2; N=A/(4*lP**2)/ln2; MH=3*H0**2/(8*np.pi*G)*4/3*np.pi*R**3
rows.append(("cosmic horizon",T,N,k*T*ln2,N*k*T*ln2))
print(f"cosmic horizon: T_GH {T:.3e} K, bits {N:.3e}, N k T ln2/(M_H c^2) = {N*k*T*ln2/(MH*c**2):.6f}")
print("\nladder: surface | T [K] | bits N | cost per bit [J] | N x cost [J]")
for r in rows: print(f"  {r[0]:22s} {r[1]:10.3e} {r[2]:10.3e} {r[3]:10.3e} {r[4]:10.3e}")
print("\nspans")
Tc,Tb=310.15,hbar*c**3/(8*np.pi*G*Msun*k); print(f"  T cell / T_H(1 Msun) = {Tc/Tb:.2e}   (paper: 10^10)")
Nb=16*np.pi*(G*Msun/c**2)**2/(4*lP**2)/ln2; print(f"  N bits BH(1 Msun) / N_CpG = {Nb/19.6e6:.2e}   (paper: 10^27)")
print(f"  length: atom (Bohr radius {C.physical_constants['Bohr radius'][0]:.2e} m) to Hubble radius ({R:.2e} m): {np.log10(R/C.physical_constants['Bohr radius'][0]):.1f} orders")
print(f"  cost per bit: cell / cosmic horizon = {e/(k*T*ln2):.2e}  ({np.log10(310.15/T):.1f} orders)")
