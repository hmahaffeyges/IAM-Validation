"""Temperature ladder of the encoding surfaces in Part 3, against the cosmic microwave background.
Numbers: computed below from CODATA constants. Output: figures/part5/fig_p3_temperature_ladder.pdf
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _bookstyle as bs

bs.apply()
kB=1.380649e-23; hbar=1.054571817e-34; c=2.99792458e8; G=6.67430e-11; Msun=1.98847e30
TBH=hbar*c**3/(8*math.pi*G*Msun*kB)
TCMB=2.7255
rows=[("1 M$_\\odot$ black-hole horizon", TBH, bs.GR),
      ("qubit mixing chamber", 0.015, bs.IAM),
      ("Al critical temperature $T_c$", 1.20, bs.SKY),
      ("Al gap scale $\\Delta/k_B$", 2.11, bs.IAM),
      ("4 K cryostat stage", 4.0, bs.LIGHT),
      ("50 K cryostat stage", 50.0, bs.LIGHT),
      ("cell, 37 $^\\circ$C", 310.15, bs.ALT),
      ("transistor junction, 75 $^\\circ$C", 348.15, bs.GOLD)]
fig,ax=plt.subplots(figsize=(bs.TEXTW,2.6))
ax.axvline(TCMB,color=bs.DATA,lw=1.0,zorder=1)
ax.text(TCMB*1.15,len(rows)-0.35,"CMB 2.7255 K",color=bs.DATA,fontsize=7,va='center')
for i,(lab,T,col) in enumerate(rows):
    ax.plot([T],[i],'o',color=col,ms=5,zorder=3,mec='k',mew=0.3)
    r=T/TCMB
    rs=f"{1/r:.3g}× colder" if r<1 else f"{r:.3g}× hotter"
    if T<1e-3: rs=bs.sci(1/r,2)+"× colder"
    right = T<TCMB or T<10
    ax.text(T*1.5 if right else T/1.5, i, f"{lab}  ({rs})" if right else f"({rs})  {lab}",
            ha='left' if right else 'right', va='center', fontsize=6.5, zorder=4, bbox=dict(fc='white', ec='none', pad=0.4))
ax.set_xscale('log'); ax.set_xlim(1e-8,3e3); ax.set_ylim(-0.7,len(rows)-0.1)
ax.set_yticks([]); ax.set_xlabel("temperature (K)")
for s in ('left','right','top'): ax.spines[s].set_visible(False)
bs.save(fig,"part5","fig_p3_temperature_ladder")
