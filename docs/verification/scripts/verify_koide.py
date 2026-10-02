#!/usr/bin/env python3
"""Koide paper checks: Q from PDG pole masses; which n survive positivity with y/x = sqrt2 for any phase offset; the Brannen offset; delta = 0 masses."""
import numpy as np
from scipy.optimize import brentq
me,mm,mt=0.51099895,105.6583755,1776.86; s=np.sqrt([me,mm,mt]); x=s.sum()/3
print("Q =",round((me+mm+mt)/s.sum()**2,8),"  x^2 =",round(x**2,3),"MeV")
dl=np.linspace(0,2*np.pi,200001)
for n in range(2,8):
    mn=(1+np.sqrt(2)*np.cos(dl[:,None]+2*np.pi*np.arange(n)/n)).min(1)
    print(f"n {n}: positive for some offset {bool((mn>1e-9).any())}  (fraction of offsets {np.mean(mn>1e-9):.3f})")
d=brentq(lambda t:(1+np.sqrt(2)*np.cos(t))-s[2]/x,0,1); print("offset delta =",round(d,5),"rad; 2/9 =",round(2/9,5))
print("masses from delta:",sorted(round((x*(1+np.sqrt(2)*np.cos(d+2*np.pi*k/3)))**2,3) for k in range(3)))
print("delta = 0 masses:",sorted(round((x*(1+np.sqrt(2)*np.cos(2*np.pi*k/3)))**2,3) for k in range(3)))
