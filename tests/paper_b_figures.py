"""Paper B — Figures"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

hbar=1.0546e-34; k_B=1.3806e-23; G=6.674e-11; ln2=np.log(2); eV=1.602e-19; c=2.998e8
m_e=9.109e-31; m_p=1.673e-27; amu=1.661e-27
R_func = lambda m: max((3*m/(4*np.pi*2000))**(1./3), 1e-15)

plt.rcParams.update({'font.family':'serif','font.size':11,'axes.labelsize':13,
    'axes.titlesize':13,'legend.fontsize':10,'figure.dpi':150,'savefig.dpi':300,'savefig.bbox':'tight'})
IAM_C='#2563EB'; STD_C='#DC2626'; PHOTON_C='#F59E0B'; MATTER_C='#7C3AED'; THRESH_C='#059669'

# FIG B1: Sector crossing diagram
fig1, ax = plt.subplots(1,1,figsize=(14,9))
ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis('off')
ax.text(5,9.5,'The Sector Crossing: What "Measurement" Actually Is',fontsize=16,fontweight='bold',ha='center')

sigma_box = FancyBboxPatch((0.3,4.5),3.5,4.0,boxstyle="round,pad=0.2",facecolor='#EFF6FF',edgecolor=IAM_C,linewidth=2.5)
ax.add_patch(sigma_box)
ax.text(2.05,8.1,'Σ = 1 Sector',fontsize=14,fontweight='bold',color=IAM_C,ha='center')
ax.text(2.05,7.5,'(Coherent / Reversible)',fontsize=11,color=IAM_C,ha='center',fontstyle='italic')
for i,t in enumerate(['Photons (always)','Superpositions (until crossing)','Entangled pairs in transit',
    'Unitary evolution','Q < k_BT ln 2','Erasure possible']):
    ax.text(2.05,6.8-i*0.45,f'• {t}',fontsize=10,color=IAM_C,ha='center')

mu_box = FancyBboxPatch((6.2,4.5),3.5,4.0,boxstyle="round,pad=0.2",facecolor='#FFF1F2',edgecolor=MATTER_C,linewidth=2.5)
ax.add_patch(mu_box)
ax.text(7.95,8.1,'μ < 1 Sector',fontsize=14,fontweight='bold',color=MATTER_C,ha='center')
ax.text(7.95,7.5,'(Decohered / Irreversible)',fontsize=11,color=MATTER_C,ha='center',fontstyle='italic')
for i,t in enumerate(['Matter (timelike worldlines)','Classical states','Measured particles',
    'Entropy: ΔS ≥ k_B ln 2','Q ≥ k_BT ln 2','Erasure impossible']):
    ax.text(7.95,6.8-i*0.45,f'• {t}',fontsize=10,color=MATTER_C,ha='center')

ax.annotate('',xy=(6.1,6.5),xytext=(3.9,6.5),arrowprops=dict(arrowstyle='->',lw=3,color=THRESH_C))
ax.text(5,7.0,'SECTOR CROSSING',fontsize=12,fontweight='bold',color=THRESH_C,ha='center')
ax.text(5,6.65,'"Measurement"',fontsize=11,color=THRESH_C,ha='center',fontstyle='italic')

crit_box = FancyBboxPatch((1.5,1.0),7.0,2.8,boxstyle="round,pad=0.3",facecolor='#F0FDF4',edgecolor=THRESH_C,linewidth=2)
ax.add_patch(crit_box)
ax.text(5,3.4,'THE LANDAUER CRITERION',fontsize=13,fontweight='bold',color=THRESH_C,ha='center')
ax.text(5,2.8,r'Sector crossing occurs when: $Q_{interaction} \geq k_B T \ln 2$',fontsize=12,ha='center')
ax.text(5,2.2,'Q = energy dissipated into environment by the interaction\nT = temperature of the environment',
    fontsize=10,ha='center',color='#444',fontstyle='italic')
ax.text(5,1.4,'No observer required. No consciousness. Just thermodynamics.',fontsize=11,ha='center',fontweight='bold')

fig1.savefig('/home/claude/fig_b1_sector_crossing.png'); print("Fig B1 saved")
plt.close()

# FIG B2: Decoherence landscape
fig2, ax = plt.subplots(1,1,figsize=(12,8))
masses = np.logspace(-30,5,500)
tau_iam=[]; tau_pd=[]
for m in masses:
    R=R_func(m); EG=G*m**2/R
    tau_iam.append(hbar*k_B**2*300**2*ln2/EG**3)
    tau_pd.append(hbar/EG)

ax.loglog(masses,tau_iam,color=IAM_C,lw=2.5,label=r'IAM: $\tau_{IAM}$')
ax.loglog(masses,tau_pd,color=STD_C,lw=2.5,ls='--',label=r'Penrose-Diósi: $\tau_{PD}$')

markers=[("e⁻",m_e),("p",m_p),("C₆₀",60*12*amu),("Virus",1e-18),("Bact.",1e-15),
    ("Dust",1e-12),("Sand",1e-6),("Cat",4.0),("Human",70.0)]
for label,m in markers:
    R=R_func(m); EG=G*m**2/R; t=hbar*k_B**2*300**2*ln2/EG**3
    ax.plot(m,t,'o',color=MATTER_C,ms=8,zorder=5)
    ax.annotate(label,xy=(m,t),xytext=(8,8),textcoords='offset points',fontsize=9,color=MATTER_C,fontweight='bold')

ax.axhline(y=1,color='gray',lw=0.8,ls=':',alpha=0.5); ax.text(1e-28,2,'1 second',fontsize=9,color='gray')
ax.axhline(y=5.4e-44,color='gray',lw=0.8,ls=':',alpha=0.5); ax.text(1e-28,1e-43,'Planck time',fontsize=9,color='gray')
ax.axhline(y=4.35e17,color='gray',lw=0.8,ls=':',alpha=0.5); ax.text(1e-28,8e17,'Age of universe',fontsize=9,color='gray')
ax.axvspan(1e-15,1e-10,alpha=0.08,color=THRESH_C)
ax.text(3e-13,1e50,'Mesoscopic\nfrontier',fontsize=11,color=THRESH_C,ha='center',fontweight='bold')

ax.set_xlabel('Mass (kg)'); ax.set_ylabel('Decoherence timescale (s)')
ax.set_title("The Quantum-to-Classical Transition\nWhy Schrödinger's Cat Was Never in Superposition",fontweight='bold')
ax.legend(loc='upper right',fontsize=11); ax.set_xlim(1e-30,1e5); ax.set_ylim(1e-60,1e110); ax.grid(True,alpha=0.15,which='both')
fig2.savefig('/home/claude/fig_b2_decoherence_landscape.png'); print("Fig B2 saved")
plt.close()

# FIG B3: Delayed choice timeline
fig3, ax = plt.subplots(1,1,figsize=(14,6))
ax.set_xlim(0,10); ax.set_ylim(0,6); ax.axis('off')
ax.text(5,5.7,"Wheeler's Delayed Choice — IAM Interpretation",fontsize=15,fontweight='bold',ha='center')
ax.plot([1,9],[3,3],'k-',lw=2)

for x,label,col in [(2,'Photon\nemitted',IAM_C),(4,'Through\nslits',IAM_C),(6,'Experimenter\nchooses','#888'),(8,'Hits\ndetector',MATTER_C)]:
    ax.plot(x,3,'o',color=col,ms=15,zorder=5)
    ax.text(x,3.6,label,fontsize=10,ha='center',fontweight='bold',color=col)

ax.fill_between([1.5,7.5],2.3,2.7,color=IAM_C,alpha=0.15)
ax.text(4.5,2.5,'Σ = 1 (coherent, no decoherence)',fontsize=10,ha='center',color=IAM_C,fontweight='bold')
ax.fill_between([7.5,8.5],2.3,2.7,color=MATTER_C,alpha=0.15)
ax.text(8,2.1,'μ<1',fontsize=9,ha='center',color=MATTER_C,fontweight='bold')

ax.text(5,4.8,'Standard QM: photon "decides" retroactively based on future choice',fontsize=10,ha='center',color=STD_C,fontstyle='italic')
ax.annotate('',xy=(2,4.5),xytext=(6,4.5),arrowprops=dict(arrowstyle='<-',lw=1.5,color=STD_C,ls='--'))
ax.text(4,4.6,'???',fontsize=12,color=STD_C,ha='center',fontweight='bold')

ax.text(5,1.5,'IAM: photon ALWAYS coherent during flight. No retrocausality.',fontsize=11,ha='center',color=IAM_C,fontweight='bold')
ax.text(5,0.9,'Detector configuration determines WHERE crossing happens, not WHETHER photon was coherent.',fontsize=10,ha='center',color=THRESH_C)

fig3.savefig('/home/claude/fig_b3_delayed_choice.png'); print("Fig B3 saved")
plt.close()

# FIG B4: Entanglement survival
fig4, (ax1,ax2) = plt.subplots(1,2,figsize=(14,6))
distances = np.logspace(0,6,500)

mass_cases = [(0,"Photons",PHOTON_C),(1e-18,"Virus",'#22C55E'),(1e-15,"Bacterium",'#EAB308'),
    (1e-12,"Dust grain",'#F97316'),(1e-9,"Microsphere",STD_C)]

for m_c,label,col in mass_cases:
    if m_c == 0:
        F_arr = np.ones_like(distances)
    else:
        R=R_func(m_c); EG=G*m_c**2/R; tau=hbar*k_B**2*0.01**2*ln2/EG**3
        F_arr=[]
        for d in distances:
            eta = d / tau
            if eta<0.01: F_arr.append(1.0)
            elif eta>20: F_arr.append(0.0)
            else: F_arr.append(max(0, 1.0-np.exp(1-1/eta)/np.e))
        F_arr = np.array(F_arr)
    ax1.semilogx(distances,F_arr,color=col,lw=2.5,label=label)

ax1.set_xlabel('Separation (m)'); ax1.set_ylabel('Entanglement fidelity')
ax1.set_title('(a) Entanglement Survival vs Distance\n(T=10 mK, matter at 1 m/s)')
ax1.legend(loc='center left',fontsize=9); ax1.set_xlim(1,1e6); ax1.set_ylim(-0.05,1.1); ax1.grid(True,alpha=0.2)

records = [("Micius photons\n(2017)",1200e3,PHOTON_C),("Fiber photons\n(2022)",33e3,PHOTON_C),
    ("Ion trap\n(2019)",1.3,MATTER_C),("Solid-state\n(2023)",0.002,MATTER_C)]
for label,dist,col in records:
    ax2.barh(label,dist,color=col,alpha=0.7,edgecolor=col,lw=1.5)

ax2.set_xscale('log'); ax2.set_xlabel('Max entanglement distance (m)')
ax2.set_title('(b) Distance Records:\nPhotons vs Matter')
ax2.set_xlim(1e-4,1e7); ax2.grid(True,alpha=0.2,axis='x')

fig4.suptitle('Why Photon Entanglement Survives: Σ=1 Sector Immunity',fontsize=13,fontweight='bold',y=1.03)
plt.tight_layout(); fig4.savefig('/home/claude/fig_b4_entanglement.png'); print("Fig B4 saved")
plt.close()

print("\nAll 4 Paper B figures generated.")
