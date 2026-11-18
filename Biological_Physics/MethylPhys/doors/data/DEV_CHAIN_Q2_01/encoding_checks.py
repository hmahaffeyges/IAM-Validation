import subprocess as _sp, sys as _sy
_sp.run([_sy.executable, '-m', 'pip', 'install', '-q', 'sympy', 'numpy', 'scipy', 'mpmath'], check=True)
# SPIDER_ENCODING 2026-11-17: sympy checks for the encoding half of IAM's Law.
# Every check prints PASS/FAIL with an id; counts at the end. Private exploration file.
import sympy as sp
from sympy import Rational as R, pi, sqrt, symbols, simplify, Eq, solve, integrate, oo, cos, sin, exp, Matrix, nsimplify
res = []
def chk(cid, ok, note=""):
    ok = bool(ok); res.append((cid, ok, note)); print(("PASS " if ok else "FAIL ") + cid + ("  | " + note if note else ""))

hbar, c, G, kB, kap, eta, M, A, E, Rr, T, H = symbols('hbar c G k_B kappa eta M A E R T H', positive=True)
lP2 = hbar*G/c**3

# ---------------- 1. Jacobson chain ----------------
# J1 Unruh: Euclidean period P = 2 pi c/kappa  ->  T = hbar/(k_B P)
P = 2*pi*c/kap
chk("J1_unruh_from_period", simplify(hbar/(kB*P) - hbar*kap/(2*pi*kB*c)) == 0)
# J2 Clausius: T_ab k k = (hbar c eta/2pi) R_ab k k ; match to c^4/(8 pi G) -> G = c^3/(4 hbar eta)
Gsol = solve(Eq(hbar*c*eta/(2*pi), c**4/(8*pi*G)), G)[0]
chk("J2_G_from_eta", simplify(Gsol - c**3/(4*hbar*eta)) == 0, str(Gsol))
etasol = solve(Eq(hbar*eta/(2*pi), c**3/(8*pi*G)), eta)[0]
chk("J3_eta_quarter", simplify(etasol - 1/(4*lP2)) == 0, "eta = 1/(4 l_P^2) given G (algebra of Jacobson's own relation)")
# J4 lemma: symmetric X with X_ab k^a k^b = 0 for all null k is proportional to g (Minkowski)
xs = symbols('x0:10'); X = Matrix(4,4, lambda i,j: 0)
idx = {}; n=0
for i in range(4):
    for j in range(i,4):
        X[i,j]=X[j,i]=xs[n]; n+=1
import itertools
nulls=[]
for s in itertools.product([1,-1],[1,-1],[1,-1]):
    pass
cands=[Matrix([1,1,0,0]),Matrix([1,-1,0,0]),Matrix([1,0,1,0]),Matrix([1,0,-1,0]),Matrix([1,0,0,1]),Matrix([1,0,0,-1]),
       Matrix([sqrt(2),1,1,0]),Matrix([sqrt(2),1,0,1]),Matrix([sqrt(2),0,1,1]),Matrix([sqrt(2),1,-1,0]),Matrix([sqrt(2),1,0,-1]),Matrix([sqrt(2),0,1,-1])]
eqs=[(k.T*X*k)[0] for k in cands]
sol=sp.linsolve(eqs, xs)
gm=sp.diag(-1,1,1,1)
Xs=X.subs(dict(zip(xs,list(sol)[0])))
free=list(Xs.free_symbols)
chk("J4_null_lemma_X_prop_g", len(free)==1 and simplify(Xs - Xs[1,1]*gm)==sp.zeros(4), "solution space 1-dim, X = f g")
# J5 Bianchi: T = C(R_ab + f g); div: C(R'/2 + f') = 0 -> f = -R/2 + Lambda
x=symbols('x'); Rf=sp.Function('Rs')(x); f=sp.Function('f')
fs=sp.dsolve(sp.Eq(sp.Rational(1,2)*Rf.diff(x)+f(x).diff(x),0), f(x))
chk("J5_bianchi_f", simplify(fs.rhs + Rf/2).free_symbols <= {symbols('C1')}, str(fs))
# J6 minimum area per nat, kappa as acceleration (book p3_01a)
dAmin = (8*pi*G/(kap*c**2))*(hbar*kap/(2*pi*c))
chk("J6_dAmin_4lP2", simplify(dAmin - 4*lP2) == 0)
chk("J6b_eta_dAmin_one_nat", simplify(dAmin/(4*lP2)) == 1)
# J7 Schwarzschild first law: c^2 dM/dA = kappa c^2/(8 pi G), kappa = c^4/(4GM), A = 16 pi G^2 M^2/c^4
Ab = 16*pi*G**2*M**2/c**4; kb = c**4/(4*G*M)
chk("J7_BH_first_law", simplify(c**2/sp.diff(Ab,M) - kb*c**2/(8*pi*G)) == 0)
# J8 hoop: S_BH / A at r_s = k_B/(4 l_P^2)
SBH = 4*pi*G*kB*M**2/(hbar*c)
chk("J8_holo_saturated_at_rs", simplify(SBH/Ab - kB/(4*lP2)) == 0)
# J9 Smarr Schwarzschild
TBH = hbar*c**3/(8*pi*G*kB*M)
chk("J9_smarr_half", simplify(TBH*SBH/(M*c**2)) == R(1,2))
# J10 Bekenstein bound read as S <= E / T_R with k_B T_R = hbar c/(2 pi R); BH saturates it
TR = hbar*c/(2*pi*kB*Rr)
chk("J10_bekenstein_as_E_over_TR", simplify(2*pi*kB*Rr*E/(hbar*c) - E/TR) == 0)
chk("J10b_BH_saturates_bekenstein", simplify((2*pi*kB*Rr*E/(hbar*c)).subs({Rr:2*G*M/c**2, E:M*c**2}) - SBH) == 0)
# J11 Cai-Kim with S_info: Hdot = -4 pi G (rho+P) + G H^3 Sdot/(2 pi)  (hbar=c=kB=1)
rho,Pp,Hd,Sd=symbols('rho P Hdot Sdot_info')
sol=solve(Eq(4*pi*(rho+Pp)/H**2, -Hd/(G*H**2) + H/(2*pi)*Sd), Hd)[0]
chk("J11_caikim_info", simplify(sol - (-4*pi*G*(rho+Pp) + G*H**3*Sd/(2*pi))) == 0)

# numbers (SI)
import math
hb=1.054571817e-34; cc=2.99792458e8; GG=6.67430e-11; kk=1.380649e-23
lp2=hb*GG/cc**3
TU_g = hb*9.80665/(2*math.pi*kk*cc)
chk("N1_unruh_at_g", 3e-20 < TU_g < 5e-20, f"T_U(g=9.81) = {TU_g:.3e} K vs 310 K")
Rc=5e-6; mcell=1000*4/3*math.pi*Rc**3; Ecell=mcell*cc**2
S_bek_nats = 2*math.pi*Rc*Ecell/(hb*cc); S_holo_nats = 4*math.pi*Rc**2/(4*lp2); N_cpg_bits=2.8217448e7
chk("N2_cell_far_below_bounds", S_bek_nats > 1e25 and S_holo_nats > 1e59,
    f"cell R=5um m={mcell:.2e} kg: Bekenstein {S_bek_nats:.2e} nats, holographic {S_holo_nats:.2e} nats, CpG record {N_cpg_bits:.2e} bits; log10 gaps {math.log10(S_bek_nats/N_cpg_bits):.1f}, {math.log10(S_holo_nats/N_cpg_bits):.1f}")
R310 = hb*cc/(2*math.pi*kk*310.15)
chk("N3_TR_310K_radius", 1.0e-6 < R310 < 1.3e-6, f"T_R = 310.15 K at R = {R310:.3e} m (thermal-photon wavelength scale; dimensional coincidence, no bound content)")

# ---------------- 2. continuity / surface ----------------
r,th,ph,nn=symbols('r theta phi n', positive=True)
j=r**nn  # radial entropy current magnitude
flux = 4*pi*Rr**2*j.subs(r,Rr)
divJ = sp.diff(r**2*j, r)/r**2
vol = integrate(divJ*4*pi*r**2, (r,0,Rr))
chk("S1_divergence_theorem_radial", simplify(vol - flux) == 0, "int div J dV = closed-surface flux")
# Poynting: dipole far field <S> ~ sin^2(theta)/r^2 ; total power through sphere independent of R
S0=symbols('S0',positive=True)
Ptot = integrate(integrate(S0*sin(th)**2/Rr**2*Rr**2*sin(th),(th,0,pi)),(ph,0,2*pi))
chk("S2_poynting_sphere_R_independent", sp.diff(Ptot,Rr)==0 and simplify(Ptot-8*pi*S0/3)==0, str(Ptot))
# steady state: dS/dt = -outflow + production = 0 -> outflow = production
Sout,Sprod=symbols('Sout Sprod')
chk("S3_steady_state_all_produced_entropy_crosses", solve(Eq(-Sout+Sprod,0),Sout)[0]==Sprod)
# Clausius entropy flux of heat q crossing at boundary temperature: q/T_b ; blackbody s = 4u/3T
ratio = 5772/255
chk("S4_sun_earth_entropy_flux_ratio", 20 < ratio < 25, f"same power re-emitted at 255 K carries {ratio:.1f}x the entropy it arrived with at 5772 K (sigma>0 on absorption)")
# ordinary matter: entropy extensive (R^3) vs area bound (R^2); gravity intervenes first
s_per_m3 = 8.4*1000/0.018*6.022e23  # ~8.4 k per water molecule (order-of-magnitude assumption)
Rcross = 3/(4*lp2)/s_per_m3  # (4/3)pi R^3 s = 4 pi R^2/(4 lp2)
Rbh = math.sqrt(3*cc**2/(8*math.pi*GG*1000))
chk("S5_matter_collapses_before_area_bound", Rbh < Rcross, f"water: area-bound crossover R ~ {Rcross:.1e} m, but a water ball is its own Schwarzschild radius at R ~ {Rbh:.1e} m")

# ---------------- 3. halves and departures ----------------
k_=symbols('k'); K_,V_=symbols('K V')
chk("H1_virial_general_k", solve(Eq(2*K_, k_*V_),K_)[0].subs({k_:-1}) == -V_/2)
# Kerr / Reissner-Nordstrom share
chi=symbols('chi',positive=True)
rp=1+sqrt(1-chi**2); rm=1-sqrt(1-chi**2)   # units M=1, G=c=1
TS_RN=(rp-rm)/4; MRN=(rp+rm)/2
chk("H2_RN_share", simplify(TS_RN/MRN - sqrt(1-chi**2)/2)==0)
chi52 = sp.nsolve(sqrt(1-chi**2)/2-0.48, chi, 0.3)
chk("H2b_spin_or_charge_for_48pct", abs(float(chi52)-0.28)<0.001, f"T S/Mc^2 = 0.48 at chi (or Q/M) = {float(chi52):.4f}")
D=symbols('D',positive=True)
chk("H3_smarr_D_dim", simplify(((D-3)/(D-2)).subs(D,4))==R(1,2) and ((D-3)/(D-2)).subs(D,5)==R(2,3))
# FRW shell: G M/(R c^2) = Omega/2 with R = c/H, M = rho 4/3 pi R^3, Omega = 8 pi G rho/(3H^2)
rhoM=symbols('rho_m',positive=True)
Om = 8*pi*G*rhoM/(3*H**2)
chk("H4_FRW_shell_half", simplify(G*(rhoM*4*pi/3*(c/H)**3)/((c/H)*c**2) - Om/2)==0)
# cosmic horizon: E_MS = c^4 R/2G, S = kB pi R^2 c^3/(G hbar), T_GH = hbar c/(2 pi kB R)
EMS=c**4*Rr/(2*G); Sh=kB*pi*Rr**2*c**3/(G*hbar); TGH=hbar*c/(2*pi*kB*Rr)
chk("H5_cosmic_TS_equals_E", simplify(TGH*Sh - EMS)==0, "T_GH S = E (ratio 1)")
chk("H5b_cosmic_dEdS_half", simplify(sp.diff(EMS,Rr)/sp.diff(Sh,Rr) - TGH/2)==0 and simplify(sp.diff(EMS,Rr)/sp.diff(Sh,Rr)*Sh - EMS/2)==0, "first-law temperature dE/dS = T_GH/2 gives the half")
# capacitor step: dissipated = 1/2 C V^2 for any R
Rs_,C_,V0,t=symbols('R_s C V_0 t',positive=True)
i_t = V0/Rs_*exp(-t/(Rs_*C_))
Ediss = integrate(i_t**2*Rs_,(t,0,oo))
chk("H6_capacitor_step_half", simplify(Ediss - C_*V0**2/2)==0)
# ramp charging over time Tr: v_s = V0 t/Tr; exact dissipation, then hold at V0
Tr=symbols('T_r',positive=True); tau=Rs_*C_
q=sp.Function('q')
# current for ramp: i(t) = C V0/Tr (1 - exp(-t/tau)) for t<Tr; after Tr residual decays
i1 = C_*V0/Tr*(1-exp(-t/tau))
E1 = integrate(i1**2*Rs_,(t,0,Tr))
vc_Tr = V0 - V0*tau/Tr*(1-exp(-Tr/tau))  # capacitor voltage at end of ramp
E2 = C_*(V0-vc_Tr)**2/2
Ed = simplify(E1+E2)
frac = sp.lambdify((Tr,), (Ed/(C_*V0**2)).subs({Rs_:1,C_:1,V0:1}))
f10=frac(10.0); f1e=frac(1e-6)
chk("H7_ramp_charging_departs", abs(f1e-0.5)<1e-3 and f10 < 0.1, f"dissipated/CV^2: step-limit {f1e:.4f}; ramp over 10 RC {f10:.4f} -> stored share of drawn {0.5/(0.5+f10):.3f}")
# polytropes / adiabatic index: retained heat share of released gravity 1/(3(g-1)); radiated (3g-4)/(3(g-1))
g=symbols('gamma',positive=True)
ret=1/(3*(g-1)); rad=(3*g-4)/(3*(g-1))
chk("H8_gamma_shares", simplify(ret+rad-1)==0 and ret.subs(g,R(5,3))==R(1,2) and ret.subs(g,R(4,3))==1)
g52=solve(Eq(ret,R(52,100)),g)[0]
chk("H8b_gamma_for_52_48", abs(float(g52)-1.641)<0.001, f"retained 52 / radiated 48 at gamma = {float(g52):.4f}; gamma=1.4 gives {float(ret.subs(g,R(7,5))):.3f}")
# Eddington standard model Gamma_1(beta), gas fraction beta, gas gamma = 5/3
beta=symbols('beta',positive=True)
G1 = beta + (4-3*beta)**2*(R(2,3))/(beta+12*R(2,3)*(1-beta))
chk("H8c_Gamma1_limits", simplify(G1.subs(beta,1)-R(5,3))==0 and simplify(G1.subs(beta,sp.Symbol('e',positive=True)).limit(sp.Symbol('e',positive=True),0)-R(4,3))==0)
# gas + radiation star, constant gas-pressure fraction beta (Eddington): u = P(beta/(gamma-1) + 3(1-beta)), 3 int P dV = -W
Pint=symbols('Pint'); Wg=symbols('W',negative=True)
Uint=(beta/(R(5,3)-1)+3*(1-beta))*(-Wg/3)
Etot=simplify(Uint+Wg)
chk("H8d_gas_radiation_star", simplify(Etot - beta*Wg/2)==0, "E = beta W/2: retained 1-beta/2, radiated beta/2 of |W|")
b52=solve(Eq(1-beta/2,R(52,100)),beta)[0]
chk("H8e_beta_for_52_48", b52==R(24,25), f"retained 52 / radiated 48 at gas fraction beta = {b52}; beta=0.9 gives retained {1-0.9/2:.2f}")
# Yukawa circular orbit: K/|V| = (1 + r/lambda)/2
lam,gY,mY=symbols('lambda g m',positive=True)
VY=-gY*exp(-r/lam)/r
Kc = r*sp.diff(VY,r)/2   # circular orbit m v^2 = r dV/dr
chk("H9_yukawa_share", simplify(Kc/(-VY) - (1+r/lam)/2)==0)
rl=solve(Eq((1+x)/2,R(52,100)),x)[0]
chk("H9b_yukawa_52", rl==R(1,25), f"K/|V| = 0.52 at r/lambda = {rl}")
# finite averaging window for a harmonic mode: <cos^2> over [0,tau] = 1/2 + sin(2 w tau)/(4 w tau)
w,tt=symbols('omega tau',positive=True)
avg=integrate(cos(w*t)**2,(t,0,tt))/tt
chk("H10_window_average", simplify(avg - (R(1,2)+sin(2*w*tt)/(4*w*tt)))==0, "bound |dev| <= 1/(4 omega tau): 0.02 needs omega tau >= 12.5")
# quantum oscillator: <K> = <V> in every number state (virial k=2, quantum)
N=8; a=sp.zeros(N); 
for n_ in range(1,N): a[n_-1,n_]=sqrt(n_)
ad=a.T; X_=(a+ad)/sqrt(2); Pq=sp.I*(ad-a)/sqrt(2)
Kop=(Pq*Pq)/2; Vop=(X_*X_)/2
okq=all(simplify(Kop[n_,n_]-Vop[n_,n_])==0 for n_ in range(N-1))
chk("H11_quantum_oscillator_K_eq_V", okq, "hbar omega (n+1/2)/2 each, n=0..6")
# Koide as share: Q = (1 + rho)/3 with rho = w_harm/w_DC
xK,yK,dK=symbols('x y delta',positive=True)
ph3=[dK+2*pi*kk_/3 for kk_ in range(3)]
s=[xK+yK*cos(p_) for p_ in ph3]
Qk=sp.simplify(sp.trigsimp(sp.expand_trig(sum(si**2 for si in s)/(sum(s))**2)))
rhoK=(yK**2/2)/xK**2
chk("H12_koide_share_form", simplify(Qk - (1+rhoK)/3)==0, str(Qk))
Qobs=0.66666446; dQ=0.00000508
chk("H12b_koide_no_room_for_52_48", abs((3*Qobs-1)-1) < 3*3*dQ and abs(52/48-1)>0.08, f"rho_obs = 3Q-1 = {3*Qobs-1:.6f} +/- {3*dQ:.1e}; 52/48 would need Q = {(1+52/48)/3:.4f}")
# MaxEnt over K channels -> 1/K
pk=symbols('p1:4',positive=True); lmb=symbols('lam')
L_=-sum(p_*sp.log(p_) for p_ in pk) - lmb*(sum(pk)-1)
solm=solve([sp.diff(L_,p_) for p_ in pk]+[sum(pk)-1], list(pk)+[lmb], dict=True)
chk("H13_maxent_uniform", all(simplify(solm[0][p_]-R(1,3))==0 for p_ in pk))
# Equipartition statistics: x ~ N(0,s), y/sqrt2 ~ N(0,s): rho ~ F(1,1); P(|rho-1|<eps)
uu=symbols('u',positive=True); eps=3*3*dQ
pdfF11=1/(pi*sqrt(uu)*(1+uu))
chk("H14a_F11_normalised", simplify(integrate(pdfF11,(uu,0,oo))-1)==0)
Pwin=float(sp.Integral(pdfF11,(uu,1-eps,1+eps)).evalf())
chk("H14b_equipartition_cannot_give_exact_split", Pwin < 1e-4, f"P(|rho-1| < {eps:.2e}) for one thermal draw per channel = {Pwin:.2e}")
# DOF-counted equipartition: zero axis 1 DOF, alpha-beta plane 2 DOF -> <|v_ab|^2> = 2 <v0^2> -> Q = 1 (K=3 case)
v0s,vab=symbols('v0sq vabsq',positive=True)
Qv = (v0s+vab)/(3*v0s)
chk("H15_dof_equipartition_gives_Q1", Qv.subs(vab,2*v0s)==1 and Qv.subs(vab,v0s)==R(2,3))
# one oscillator: q = x, p = y/sqrt2 rotating at omega; time averages <q^2> = <p^2> (virial k=2)
Aamp,ww=symbols('A_amp w',positive=True)
qq=Aamp*cos(ww*t); pp=Aamp*sin(ww*t)
Tper=2*pi/ww
chk("H16_one_oscillator_equal", simplify(integrate(qq**2,(t,0,Tper))-integrate(pp**2,(t,0,Tper)))==0, "equal time-averages; needs x and y/sqrt2 to be conjugate (the gap)")
# EM plane wave u_E = u_B ; SIL: 1/2 C V^2 = 1/2 L I^2 at V = Z0 I
Lq,Cq,Iq=symbols('L C_l I',positive=True)
Vq=sqrt(Lq/Cq)*Iq
chk("H17_SIL_equal_energies", simplify(Cq*Vq**2/2 - Lq*Iq**2/2)==0)
e0,mu0_,E0=symbols('epsilon0 mu0 E0',positive=True)
B0=E0*sqrt(e0*mu0_)
chk("H18_plane_wave_uE_eq_uB", simplify(e0*E0**2/2 - B0**2/(2*mu0_))==0)

# ---------------- 4. Koide as Clarke transform ----------------
Cl = sqrt(R(2,3))*Matrix([[1,-R(1,2),-R(1,2)],[0,sqrt(3)/2,-sqrt(3)/2],[1/sqrt(2),1/sqrt(2),1/sqrt(2)]])
chk("K1_clarke_power_invariant_orthogonal", simplify(Cl*Cl.T - sp.eye(3))==sp.zeros(3))
s1,s2,s3=symbols('s1 s2 s3',positive=True); sv=Matrix([s1,s2,s3]); v=Cl*sv
chk("K2_v0", simplify(v[2]-(s1+s2+s3)/sqrt(3))==0)
Qs=(s1**2+s2**2+s3**2)/(s1+s2+s3)**2
chk("K3_Q_from_clarke", simplify(Qs - (v[2]**2+v[0]**2+v[1]**2)/(3*v[2]**2))==0)
chk("K3b_Q23_iff_equal_power", simplify((Qs-R(2,3))*3*v[2]**2 - (v[0]**2+v[1]**2-v[2]**2))==0, "Q=2/3 <=> |v_ab|^2 = |v_0|^2 (45 deg)")
chk("K3c_cos2theta", simplify(v[2]**2/(sv.T*sv)[0] - 1/(3*Qs))==0)
me,mmu,mtau=0.51099895000,105.6583755,1776.93
import math as m_
sq=[m_.sqrt(me),m_.sqrt(mmu),m_.sqrt(mtau)]
v0n=sum(sq)/m_.sqrt(3); tot=sum(z*z for z in sq); vabn=tot-v0n**2
ang=m_.degrees(m_.acos(v0n/m_.sqrt(tot)))
chk("K4_leptons_45deg", abs(ang-45)<1e-3, f"angle {ang:.5f} deg; |v_ab|^2/|v_0|^2 = {vabn/v0n**2:.6f}; Q = {tot/sum(sq)**2:.8f}")
# Brannen coordinates (x, r, delta) -> (s1,s2,s3) invertible: Jacobian
xx,rr,dd=symbols('x r delta',positive=True)
F=Matrix([xx*(1+rr*cos(dd+2*pi*kk_/3)) for kk_ in range(3)])
J=simplify(F.jacobian([xx,rr,dd]).det())
chk("K5_brannen_is_coordinate_change", simplify(J - R(3,2)*sqrt(3)*xx**2*rr)==0 or simplify(J + R(3,2)*sqrt(3)*xx**2*rr)==0, f"det J = {J} (nonzero for x,r>0): 3 numbers <-> 3 numbers")
gradQ=Matrix([sp.diff(Qs,z) for z in (s1,s2,s3)])
chk("K6_koide_set_is_2D_cone", simplify(gradQ.dot(sv))==0 and simplify(gradQ.subs({s1:1,s2:2,s3:3}).norm())!=0, "Q homogeneous deg 0: level set Q=2/3 is a cone (2D) through origin")
# form factor / ripple factor
mean=(s1+s2+s3)/3; rms2=(s1**2+s2**2+s3**2)/3
chk("K7_Q_is_formfactor_sq_over_3", simplify(Qs - (rms2/mean**2)/3)==0, "Koide <=> FF = sqrt2 <=> ripple factor 1")
# bus magnitudes with ideal 120-degree angles: V1 = mean, V2 = S1 (DFT) -> Q = (1 + 2 VUF^2)/3
Va,Vb,Vc=symbols('V_a V_b V_c',positive=True)
aop=sp.exp(2*pi*sp.I/3)
pha=[Va, Vb*sp.exp(-2*pi*sp.I/3), Vc*sp.exp(2*pi*sp.I/3)]
V1=(pha[0]+aop*pha[1]+aop**2*pha[2])/3; V2=(pha[0]+aop**2*pha[1]+aop*pha[2])/3
V1s=sp.nsimplify(sp.simplify(sp.expand_complex(V1)))
VUF2=sp.simplify(sp.expand_complex(V2*sp.conjugate(V2))/sp.expand_complex(V1*sp.conjugate(V1)))
QV=(Va**2+Vb**2+Vc**2)/(Va+Vb+Vc)**2
chk("K8_Q_equals_1p2VUF2_over3", simplify(QV - (1+2*VUF2)/3)==0, "magnitude-only unbalance, ideal angles")
import cmath
def vuf(mags,angs):
    ph=[mm*cmath.exp(1j*m_.radians(aa)) for mm,aa in zip(mags,angs)]
    A_=cmath.exp(2j*m_.pi/3)
    v1=(ph[0]+A_*ph[1]+A_**2*ph[2])/3; v2=(ph[0]+A_**2*ph[1]+A_*ph[2])/3
    return abs(v2)/abs(v1)
vu=vuf([1,1,1],[0,-118,120]); Qa=3/9
chk("K9_magnitude_Q_blind_to_angle_unbalance", Qa==1/3 and vu>0.005, f"angles 0,-118,120 deg, equal magnitudes: Q-1/3 = 0, VUF = {100*vu:.3f} %")
Q2=(1+2*0.02**2)/3
chk("K10_Q_at_2pct_VUF", abs(Q2-1/3-2.667e-4)<1e-6, f"Q - 1/3 = {Q2-1/3:.3e} at VUF 2 %; Koide 2/3 needs VUF = {m_.sqrt(0.5):.4f}")
# PVUR (max dev/mean) vs RMS dev/mean: ratio in [sqrt(3/2), sqrt(2)] for zero-sum deviations
tt_=symbols('t_',real=True)
dvec=[cos(tt_), cos(tt_-2*pi/3), cos(tt_+2*pi/3)]  # all zero-sum triples up to scale
rmsd=sqrt(sum(z**2 for z in dvec)/3)
import numpy as np
ts=np.linspace(0,2*np.pi,20001); rr_=[max(abs(m_.cos(t0)),abs(m_.cos(t0-2*m_.pi/3)),abs(m_.cos(t0+2*m_.pi/3)))/m_.sqrt(0.5) for t0 in ts]
chk("K11_PVUR_over_CV_bounds", abs(min(rr_)-m_.sqrt(1.5))<1e-4 and abs(max(rr_)-m_.sqrt(2))<1e-6, f"PVUR/CV in [{min(rr_):.4f}, {max(rr_):.4f}]")
# instantaneous balanced set: zero-sequence 0 -> cos^2 theta = 0, Q undefined
chk("K12_balanced_instantaneous_v0_zero", simplify(sum(cos(w*t+2*pi*kk_/3) for kk_ in range(3)))==0)

# ---------------- summary ----------------
npass=sum(1 for _,o,_ in res if o); nfail=len(res)-npass
print(f"\nCHECKS: {npass} passed, {nfail} failed of {len(res)}")
import json; json.dump({"passed":npass,"failed":nfail,"checks":[{"id":a,"ok":b,"note":c} for a,b,c in res]}, open("spider_encoding_checks_output.json","w"), indent=1)
