<!-- Generated from IAM-Web's one master of the long form (researcher edition) by code; never edited by hand. Each commit of this file is a dated public record of the derivation. -->

# IAM's Law, derived long form: from the moment of actualization to the horizon ledger and the Friedmann equation

**How each step is tagged.**
- **ESTABLISHED**: a result of the literature, with its citation (see Citations).
- **DERIVED**: algebra, calculus or numerics; every algebraic and numerical check the steps rely on is computed by one checking script in the repository.
- **IAM IDENTIFICATION**: IAM's reading of an established quantity, or a premise of the theory. Stated as such; not derived.
- **GAP**: a missing step, stated with the calculation or data that would close it.

**How to read the numbering.** Section *n* is Step *n* of the law. Inside it, (*n.k*) numbers each line of the derivation. Equations are written out. Each line ends with its tag. Koide's relation is Step 4.

## 0. The law and the chain in one page

**The law.** Every irreversible transition from a quantum superposition to a classical record, that is, every decoherence event on a timelike worldline, pays
$$\Delta E_{\rm rec}=k_BT\ln2\quad\text{per bit},$$
on the nearest available encoding surface, at that surface's local temperature $T$. That temperature is denoted $T_H$. **IAM IDENTIFICATION** (the root statement; it rests on Landauer, ESTABLISHED, with the receiving surface named).

**The chain, link by link** (each link is a section below):

| link | what is derived | what is not |
|---|---|---|
| 1 Actualization | sudden write of an einselected pointer into a blank Gaussian surface: held $=$ spent $=U$, equal to $k_BT$ times the two relative entropies | which real surfaces meet the four conditions |
| 2 The price | $\chi\le U/k_BT$: each nat costs $\ge k_BT$ held and $\ge k_BT$ spent; floor reached only by faint records | the finite-time price |
| 2b The funding | the payment is free energy degraded to heat; Rovelli's trace bound reduces to Step 2 | the Past Hypothesis (a postulate) |
| 3 The virial half | four halves are one Euler identity (degree ratio 1:2); exact half only at $d=3$ | the radiated half of a $1/r$ capture as dissipated work |
| 4 Koide | the record keeps (scale, action), drops δ; one bit iff $Q=2/3$; a sudden write gives $Q=2/3$ by geometry | that the lepton triple *is* such a written vector (X11) |
| 5 The arrow | per record, $\beta W_{\rm diss}=U/k_BT\ge\chi$; the arrow lives in the dropped coordinate | the low-entropy start |
| 6 The entropy functional | $\dot H=-4\pi G(\rho+P)+(G/2\pi)H^3\dot S_{\rm info}$; with the record constraint, $E(a)=e^{1-1/a}$, $w_{\rm info}=-1-1/3a$ | the rate $\dot S_{\rm info}$ from local writing; the record constraint itself |
| 7 The ledger | heat $k_BT\ln2$ at a horizon adds exactly $4\ln2\,\ell_P^2$; $\beta_mS_{\rm geo}=\tfrac12M_mc^2/k_BT_{GH}$ per e-fold iff $\beta_m=\Omega_m/2$ | which booking carries a local record onto the ledger |
| → Friedmann | $H_m^2=H_{\Lambda\rm CDM}^2+\beta_mE(a)H_0^2$, $\beta_m=\Omega_m/2$ (Section 10) | $\beta_m$: derived top-down (Section 16.1), one identification left ('mass is the writing capacity', 7c); not derived bottom-up |

**In grid language.** A write is a breaker closed from a stiff bus onto an uncharged cable: half the energy stays as the cable's charge (the record kept), half is burned or radiated (the price paid). The meter (the horizon) books the transaction. The open question is the meter's rule: it books exactly half the rest energy of matter inside it per e-fold, and no local process yet found sends that much.

## 1. Step 1: actualization, the moment a bit becomes part of the record

**(1.1) The system and its local surface.** A system $S$ is in a superposition $|\psi_S\rangle=\sum_\sigma c_\sigma|\sigma\rangle$ of eigenstates of $\hat\sigma$ ($\sigma=\pm1$ for one bit). Its local surface is modelled as harmonic modes $(x_k,p_k)$ at temperature $T$, coupled linearly:
$$H=H_S+\sum_k\Big(\frac{p_k^2}{2m_k}+\tfrac12m_k\omega_k^2x_k^2\Big)+\hat\sigma\otimes B,\qquad B=\sum_kc_kx_k,\qquad[H_S,\hat\sigma]=0.$$
**ESTABLISHED** (Caldeira & Leggett 1983; pure-dephasing independent-boson model; Zurek 2003).

**(1.2) Einselection.** The observable that commutes with the interaction is the one the environment records:
$$[\hat\sigma\otimes\mathbb 1,\ \hat\sigma\otimes B]=0,\qquad[\hat\sigma_x\otimes\mathbb 1,\ \hat\sigma\otimes B]\neq0 .$$
Its eigenstates are the pointer states. **ESTABLISHED** (Zurek 1981, 1982, 2003) / **DERIVED** for this coupling.

**(1.3) The record.** Unitary evolution entangles system and surface,
$$|\psi_S\rangle|E_0\rangle\to\sum_\sigma c_\sigma|\sigma\rangle|E_\sigma\rangle,\qquad\rho_S=\sum_{\sigma\sigma'}c_\sigma c^*_{\sigma'}\langle E_{\sigma'}|E_\sigma\rangle|\sigma\rangle\langle\sigma'|\ \xrightarrow{\ \langle E_+|E_-\rangle\to0\ }\ \sum_\sigma|c_\sigma|^2|\sigma\rangle\langle\sigma| .$$
$|E_\sigma\rangle$ is the surface displaced by $-\sigma c_k/m_k\omega_k^2$ in each mode. The record is the which-σ information in the conditional surface states $\rho_\sigma$. **ESTABLISHED** (Zurek 2003).

**(1.4) A sudden write into a blank surface.** The surface starts blank, $\langle x_k\rangle=0$. The coupling switches on faster than every mode period, so for $\sigma=+1$
$$x_k(t)=x_k^{\rm eq}(1-\cos\omega_kt),\qquad x_k^{\rm eq}=-\frac{c_k}{m_k\omega_k^2}.$$
**DERIVED** (solution of $m\ddot x=-m\omega^2x-c$ from rest).

**(1.5) The energies of one mode.** With $U_k\equiv c_k^2/2m_k\omega_k^2$,
$$H_B(t)=\tfrac12m\dot x^2+\tfrac12m\omega^2x^2=2U_k(1-\cos\omega t),\qquad H_{\rm int}(t)=c\,x(t)=-2U_k(1-\cos\omega t),\qquad H_B+H_{\rm int}=0,$$
$$\langle H_B\rangle_t=2U_k,\qquad\langle H_{\rm int}\rangle_t=-2U_k .$$
**DERIVED.**

**(1.6) The split.** Write $x=x^{\rm eq}+y$.
- The static displacement carries $\tfrac12m\omega^2(x^{\rm eq})^2=U_k$: the **held** part, the polaron cloud that now dresses the pointer.
- The ringing $y=-x^{\rm eq}\cos\omega t$ carries $\tfrac12m\dot y^2+\tfrac12m\omega^2y^2=U_k$: the **spent** part. In a continuum of modes it dephases, a wave packet radiated away from the pointer.
- The new equilibrium $W(x)=\tfrac12m\omega^2x^2+cx$ has $W(x^{\rm eq})=-U_k$. Of the $2U_k$ the source puts in, $U_k$ stays and $U_k$ leaves.

Summed over modes, for any spectral density and any coupling strength,
$$\boxed{E_{\rm held}=E_{\rm spent}=U=\sum_k\frac{c_k^2}{2m_k\omega_k^2}} .$$
**DERIVED** This was also verified for a 2,000-mode ohmic, sub-ohmic, super-ohmic and random bath.

**(1.7) The same split as information.** With $\rho_0$ the blank thermal state and $\rho_1$ the thermal state about the new equilibrium:
- spent $=$ dissipated work of the sudden write, $W-\Delta F=k_BT\,D(\rho_0\Vert\rho_1)$ **ESTABLISHED** (Kawai, Parrondo & Van den Broeck 2007);
- held $=$ the record's non-equilibrium free energy, $k_BT\,D(\rho_1\Vert\rho_0)$ **ESTABLISHED** (Esposito & Van den Broeck 2011).

For one classical mode, with $s^2=k_BT/m\omega^2$,
$$k_BT\,D(\rho_1\Vert\rho_0)=k_BT\,D(\rho_0\Vert\rho_1)=k_BT\frac{(x^{\rm eq})^2}{2s^2}=U_k$$
at every temperature. **DERIVED.** For a quantum mode (displaced thermal state) both relative entropies equal $\beta\hbar\omega|\alpha|^2$ (0.1750000000 at $\beta\hbar\omega=0.7$, $\alpha=0.5$, 140 levels). **DERIVED** (numerical, truncated Fock space).

**(1.8) The four conditions, each with its failure mode.**

| condition | if it fails | check |
|---|---|---|
| sudden write, $\omega_{\max}\tau\ll1$ | a ramp of duration τ spends $\mathrm{sinc}^2(\omega\tau/2)$ of $U$ per mode: 0.999167 at $\omega\tau=0.1$, 0.708073 at 2, 0.002213 at 6 | **DERIVED** |
| Gaussian (quadratic) surface | held share $\tfrac12+\kappa_3s/12\kappa_2+\dots$ (skewness); spin bath 0.29-0.98 | |
| no opposing record (was: blank start) | overwriting $+\to-$ spends $4U$ (Marcus reorganization energy); a same-sign write onto an already-written field still splits half (preheating second burst 0.4995, 17.1(3)) | |
| a recorder with its own energy | Zurek's 1982 spin bath has $H_E=0$ and writes records at zero energy | |

**(1.9) Reading.** *Actualization* is this moment: the local surface is the recorder at its own temperature; the held half is the record kept; the spent half is the price, radiated outward. **IAM IDENTIFICATION** ("written LOCALLY by matter on its own nearest 2D encoding surface").

**(1.10) GAP: which real surfaces meet the four conditions.** The free Maxwell field is exactly linear and Gaussian. *Calculation:* switch a classical dipole on in the free Maxwell field over a time τ; compute the static near-field (held) and radiated (spent) energies mode by mode; show they are equal as $\tau\to0$ and follow $\mathrm{sinc}^2$ otherwise; give the cutoff dependence of $U$.

---

**What 'sudden' means, derived.** For a dipole switched on in the free field, radiated/held = 2 I_f (a/c tau)^3 exactly, so the equal split needs the write to be as fast as the surface's own light-crossing time (tau* = 2.90 a/c for a smooth switch); slower writes keep the record and spend almost nothing. A stiff source charging a store (the capacitor) spends exactly the held energy for ANY resistance, and almost nothing if ramped slowly. Applied: black-hole formation (collapse from rest at 2-10 r_s takes 4.4-50 r_s/c) is 1e8 times nearer sudden than the crossover, but reaches the exact half only for collapse starting within about 1.50 r_s (split 0.28 from 2 r_s, 2e-4 from 10 r_s; whether the dipole rule applies to a trapped surface is open); the electroweak crossover (7e13 write times long) is adiabatic, so the first write sits on the half only if the Higgs potential acts as a stiff source (open). **[DERIVED (classical); applications CALCULATED; the stiff-source question OPEN]** The quantum twin: a mode whose mass switches on from vacuum creates (w2 - w1)^2/(4 w1 w2) quanta when sudden and an exponentially small number when slow against its own period **[DERIVED]**; the crossover bars were missed as written (1.4e-14, not < 1e-15).

**Established physics that measures or extends link 1's sudden write.**
- **Measured in the laboratory: the dynamical Casimir effect** (Wilson et al., Nature 479, 376, 2011). Changing a mode's boundary faster than its own period creates photon pairs from vacuum; a sudden frequency jump w1 -> w2 squeezes vacuum by r = (1/2) ln(w2/w1), and sinh^2 r = (w2 - w1)^2/(4 w1 w2), the quantum result above. **[ESTABLISHED, measured]**
- **The same rule for an expanding universe: cosmological particle creation** (Parker, PRL 21, 562, 1968). Expansion creates particles only where it is non-adiabatic for the mode; particle number is an adiabatic invariant, so slow expansion creates almost none (the tanh model gives |beta|^2 = sinh^2(pi w_-/rho)/(sinh(pi w_in/rho) sinh(pi w_out/rho)), whose sudden limit is again (w2 - w1)^2/(4 w1 w2); Birrell and Davies 1982). **[ESTABLISHED]**
- **How many records a transition leaves: the Kibble-Zurek mechanism** (Kibble 1976; Zurek 1985). Crossing a continuous transition in a time tau_Q leaves defects n ~ tau_Q^(-d nu/(1 + z nu)) (exponent 1/2 for the 1D quantum Ising chain, measured on 2,000 coupled two-level systems, King et al., Nature Physics 18, 1324, 2022); a gapped crossing leaves exponentially few. It counts what link 1 calls the records a transition writes, against how sudden it is. **[ESTABLISHED, measured]** The Standard Model crossover has no critical point and no first-order jump, so it leaves no domains: consistent with the principle that a record needs distinct outcomes. **[CALCULATED reading]**

**The cell's instance of link 1.** Restores are single transfers, one SAM per restore (strand coupling 5.08-8.21, GSE116482 hairpins): each restore is one discrete write (measured; reading is interpretation). One selection step: the eraser is not proofreading the unmarked side (knockout/wild type 0.985-1.292). The methylation mark as an einselected record (stable, copied, enzyme-maintained) is an IAM identification, not yet a derivation.

### 1c. Who can write: rest mass, proper time, and the electroweak split

- A massive particle's worldline is timelike: it carries proper time. A photon's is null: zero proper time along the path. (ESTABLISHED)
- Electroweak symmetry breaking gives the fermions and the W and Z their mass (Yukawa couplings to the Higgs); the photon stays massless, protected by the unbroken electromagnetic gauge symmetry. (ESTABLISHED)
- Writing needs proper time, so only massive matter writes; light writes nothing in flight (the photon exemption) and carries a source's record unchanged. The electroweak split is therefore what sorts the world into the sector that writes (matter) and the sector that does not (light): the same split the cosmology chapters use as mu < 1 for matter and Sigma = 1 for light. (IAM IDENTIFICATION)
- Parker's theorem (PRL 21, 562, 1968): an isotropically expanding universe creates no massless, conformally coupled quanta (photons in a flat FRW universe). Expansion itself cannot write light; this is the established physics beside the photon exemption. (ESTABLISHED)
- No gauge-invariant quantity records a choice at the electroweak epoch (Elitzur), so 'the electroweak transition is a write' is an identification to be stated with care, not a theorem.

### 1d. The maintained hold is the same rule, slow
One function of how slow a write is, $x$ in kicks: held/lost $=(x/2)/\big(1-(1-e^{-x})/x\big)$. The surfaces check derived it for a ramp write ($x$ = ramp time over the system's own clock); the IAM-A derivation derived the same function for the cell's restore, $x=w=2M$. Sudden ($x\to0$): kept = lost. The cell ($x=2M=41.9$): odds 21.45, $E_{\rm hold}=3.066\,kT$, $\varphi=0.1464\approx\ln M/M$. The cell's position $x=2M$ gives about 21 restores per loss, and $\varphi=\ln({\rm odds})/M$ follows; $\varphi$ is a share of the coin, not the slow-end version of the energy half. Status: identity confirmed symbolically; sudden limit odds = 1. Open: the horizon's held share 1 as the adiabatic limit ($x\to\infty$) needs its own mapping of $x$; a direct mapping attempt failed.

I don't see any derivation content to edit in Block 2 — only the heading "## 2. Step 2: the price" was provided, with no equations, status words, or working-process notes beneath it. Please supply the full text of the block (including its equations, citations, and any bracketed check codes or notes) so I can produce the edited version.

**(2.1) Landauer.** Erasing one bit into a bath at $T$ dissipates at least $k_BT\ln2$. In the one-molecule gas, $\int_{V/2}^{V}k_BT\,dV'/V'=k_BT\ln2$. **ESTABLISHED** (Landauer 1961; Bennett 1982; measured near the bound by Bérut et al. 2012) / **DERIVED.**

**(2.2) What a write buys.** Let χ be the record's information about σ (mutual or Holevo information). For any reference $Q$, $\chi\le\sum_\sigma p_\sigma D(\rho_\sigma\Vert Q)$ (the golden formula). With $Q=\rho_0$,
$$\chi\le\sum_\sigma p_\sigma D(\rho_\sigma\Vert\rho_0)=\frac{U}{k_BT}\qquad\Longrightarrow\qquad E_{\rm held}=E_{\rm spent}=U\ge k_BT\,\chi .$$
**ESTABLISHED** (Holevo 1973) / **DERIVED.** Binary Gaussian record:

| $U/k_BT$ | χ (nats) | χ/(U/k_BT) |
|---|---|---|
| 0.001 | 0.000999 | 0.999 |
| 0.1 | 0.091091 | 0.911 |
| 1 | 0.500072 | 0.500 |
| 10 | 0.693136 | 0.069 |

**(2.3) A crisp bit costs more than the floor.** A 0.99-bit record needs $U=3.9565\,k_BT$ held and $3.9565\,k_BT$ spent, 5.7 times $k_BT\ln2$ in each half. **DERIVED.** The $k_BT$-per-nat floor is reached only by faint records.

**(2.4) The book's local chain.**
$$E_{\rm Landauer}=Q=T\Delta S_{\min}=-E_f=\langle K\rangle=\tfrac12|\langle V\rangle| .$$
Its third equality is the identity $T(Q/T)=Q$ (**DERIVED**). The reading: "one half paying the cost of the record, one half sustaining the bound state."

**(2.5) The established pair behind that reading.** Paid half $=$ spent $=k_BT\,D(\rho_0\Vert\rho_1)$, the receipt. Sustaining half $=$ held $=k_BT\,D(\rho_1\Vert\rho_0)$, the deposit that keeps the record readable. The law's $k_BT\ln2$ is the floor under each half per bit of χ. **IAM IDENTIFICATION** (on DERIVED).

**(2.6) Bound locally, equality at a horizon.** At a local surface Landauer is a bound met only by faint records (2.3). At a horizon, heat $\delta Q$ at $T$ raises $S$ by exactly $\delta Q/T$, so $k_BT\ln2$ buys exactly one bit (Section 7.1). **DERIVED.**

**(2.7) GAP: the finite-time price.** The sudden write is the most dissipative protocol. *Calculation:* for the Step-1 Gaussian surface, find the protocol $c_k(t)$ of duration τ that minimizes the spent energy at fixed χ, giving $E_{\rm spent}(\tau,\chi)$ between $U$ and the quasi-static floor. Read the finite-time Landauer literature first; no citation is used here.

---

**The cell's instance of link 2.** (DERIVED): the floor, held error where thermal kicks win against one full coin, 1/(1+e^M), M = 20.94 (the price at the cell's own temperature, 310.15 K, and coin, 54 kJ/mol). (DERIVED, given adenylate kinase at equilibrium): the price of one restore is one SAM = 2 ATP. (calculated from sourced data): one erasure and restore costs 14-63 coins, 420-1,900 times Landauer's floor: the cell's finite-time price in practice, far above the floor; this gap now has a measured cell case.

Link 1d derives one protocol in closed form at every speed, the linear ramp (held/lost $=(x/2)/(1-(1-e^{-x})/x)$, sudden limit 1): its spent energy at duration τ is an upper bound on the minimum this gap asks for, not the minimum. The optimal protocol is still open.

The supplied block contains only a heading, with no derivation content, equations, or status-labeled material beneath it:

## 2b. Step 2b: where the free energy that pays each write comes from

There is nothing else in the submitted text to edit — no equations, citations, numbers, or status words to preserve or clean up. Please provide the full content of block 2b so it can be processed.

**(2b.1) The payment is degraded free energy, not new energy.** Energy is conserved at every write (exactly so in Step 1). The write consumes free energy $F=E-TS$: energy that arrives ordered and leaves as heat at the local temperature. The surroundings gain $\Delta S=Q/T_{\rm loc}\ge k\chi$. **ESTABLISHED** (Clausius; Landauer 1961; Bennett 1982).

**(2b.2) In the Step-1 model.** The sudden switch from blank costs the agent no work, $W=\langle-Fx\rangle_0=0$; the free energy falls by $\Delta F=-U$; the heat to the surface is $W-\Delta F=U$. The write is paid from the pointer-surface binding free energy. **DERIVED.**

**(2b.3) Rovelli's conditions for a trace.** Hot environment $T_e$, cold slowly damped memory $T_m$, rare couplings. With separation, past imbalance $T_e>T_m$ and long memory,
$$\Delta S=\frac{\Delta E}{T_m}-\frac{\Delta E}{T_e},\qquad I<\frac{\Delta S}{k}=\frac{C_m(T_e-T_m)^2}{kT_eT_m}\big(1-e^{-\tau_m/\tau_{em}}\big).$$
**ESTABLISHED** (Rovelli 2022, eqs. 1, 4-6) / **DERIVED.**

**(2b.4) The join.** For a work source ($T_e\to\infty$), Rovelli's bound becomes $I<\Delta E/kT_m$, which is Step 2's $\chi\le U/k_BT$. **DERIVED.**

**(2b.5) The cell's chain.** Sunlight (photosphere 5772 K) → chlorophyll → sugar → ATP → mark on DNA (heat at 310 K) → infrared to space (~255 K).
- One ATP in the cytosol, $\Delta G_{\rm ATP}\approx54$ kJ/mol: $\mathcal M=\Delta G_{\rm ATP}/RT=20.94$, i.e. 30.21 Landauer bits. **DERIVED**
- In steady state Earth exports $T_{\rm sun}/T_{\rm earth}=22.67$ times the photon entropy it imports. **ESTABLISHED** (blackbody $s=\tfrac43u/T$) / **DERIVED** (albedo 0.30 assumed).
- Upper bound on records sunlight can fund on Earth: $\dot S=\tfrac43P_{\rm abs}(1/T_{\rm earth}-1/T_{\rm sun})=6.08\times10^{14}$ W/K, at most $6.4\times10^{37}$ bits/s. **DERIVED** (an upper bound; albedo assumed).

**(2b.6) The star's chain.** $4p+2e\to{}^4{\rm He}$ releases 26.73 MeV, 0.712 % of the rest energy; contraction at $\gamma=5/3$ radiates half of $|\Delta W|$, since $E=\frac{3\gamma-4}{3(\gamma-1)}W$. **DERIVED** (CODATA) / **ESTABLISHED** (stellar virial; Chandrasekhar 1939).

**(2b.7) Both reservoirs trace to the low-entropy start.** Nuclear free energy exists because H and He froze out of equilibrium at nucleosynthesis; gravitational free energy exists because the early universe was smooth (Penrose's Weyl curvature hypothesis, 1979). The Past Hypothesis (name: Albert 2000) is a postulate. **ESTABLISHED as a postulate.** This initial condition is not derived: IAM does not remove that initial condition.

**(2b.8) Where the book books the spent heat.** It is ordinary heat crossing the apparent horizon, part of $-dE$ and so of $S_{\rm geo}$, not of $S_{\rm info}$. Radiated baryon virial heat is $3.15\times10^{-8}\rho_\Lambda c^2$. By sign it cannot be the record term: heat has $\rho+P=\tfrac43\rho>0$; the record component has $\rho_x+P_x=-H^3\dot S_{\rm info}/8\pi^2<0$. **DERIVED**. This number implies $T_{\rm gas}=3.2\times10^6$ K, matching a 213 km/s virial gas. **DERIVED**.

**(2b.9) The horizon half in equipartition form.** Padmanabhan's $E=\tfrac12nk_BT$, $n=A/L_P^2$:
- Schwarzschild: $\tfrac12(A/L_P^2)k_BT_H=Mc^2=2T_HS$, so equipartition is Smarr. **DERIVED.**
- Newton: equipartition on a sphere with the Unruh temperature gives $\kappa=GM/r^2$. **DERIVED.**
- de Sitter: $\tfrac12nk_BT_{GH}=2T_{GH}S=2E_{\rm MS}=|(\rho+3p)V|$. **DERIVED** (identity for which temperature prices a horizon bit; no verdict).

**ESTABLISHED** (Padmanabhan 2010, Mod. Phys. Lett. A 25, 1129).

**(2b.10) GAPs.** The Past Hypothesis (not closable inside IAM or established physics). The biosphere's record efficiency: *calculation* — set a bottom-up count of records written by life per second (methylation writes and maintenance per cell × cells, using the book's measured cell numbers) against $6.4\times10^{37}$ bits/s. Also missing on the web: a published free-energy cost per maintained CpG per division including proofreading, with the closest physics being Ouldridge, Govern & ten Wolde (2017).

## 3. Step 3: the virial half

**(3.1) Lagrange's identity.** Newton, $m_i\ddot{\mathbf r}_i=\mathbf F_i$; moment of inertia $I=\sum_im_i\mathbf r_i^2$; Clausius' virial $G=\sum_i\mathbf p_i\cdot\mathbf r_i=\tfrac12\dot I$:
$$\tfrac12\ddot I=2T+\sum_i\mathbf r_i\cdot\mathbf F_i .$$
**ESTABLISHED** (Clausius 1870; Collins 1978 §1.2; Goldstein §3.4).

**(3.2) Euler's theorem.** For pair potentials $V=\sum_{i<j}C|\mathbf r_i-\mathbf r_j|^k$, $\sum_i\mathbf r_i\cdot\mathbf F_i=-kV$, so on Newtonian trajectories
$$\boxed{\tfrac12\ddot I=2T-kV}\qquad(k=-1:\ \tfrac12\ddot I=2T+V).$$
**DERIVED.**

**(3.3) The time average.** For bounded motion $(G(\tau)-G(0))/\tau\to0$, so $2\langle T\rangle=k\langle V\rangle$. On a Kepler ellipse $\langle1/r\rangle_t=1/a$: $\langle V\rangle=2E$, $\langle T\rangle=-E$. **DERIVED.**

**(3.4) Binding from rest.** With $E_i=0$: $Q_{\rm rad}=-E_f=-(1+\tfrac k2)\langle V\rangle$ and $\langle K\rangle=\tfrac k2\langle V\rangle$, equal only for $k=-1$. **DERIVED**

**(3.5) The static form.** A store $E_s=Ax^n$ charged by a constant force $F$ (source $-Fx$, degree 1). At $nAx_*^{n-1}=F$:
$$\frac{E_{\rm held}}{W_{\rm source}}=\frac{Ax_*^n}{Fx_*}=\frac1n,\qquad E_{\rm spent}=\frac{n-1}{n}W_{\rm source}.$$
**DERIVED.** For $n=2$ this is the Step-1 write and the capacitor charged from a stiff bus.

**(3.6) The generalized virial.** Kinetic energy of degree $m$ in $p$, potential of degree $k$ in $r$: $m\langle T\rangle=k\langle V\rangle$, held fraction $-k/m$: ½ for $(2,-1)$; 1 for ultra-relativistic $(1,-1)$. **DERIVED.**

**(3.7) One identity, four halves.** An Euler-weighted balance at a stationary point or on a long average,
$$n_{\rm store}\,E_{\rm held}=|n_{\rm source}|\,|E_{\rm source}|\quad\Longrightarrow\quad\frac{E_{\rm held}}{|E_{\rm source}|}=\frac{|n_{\rm source}|}{n_{\rm store}} .$$

| half | source (degree) | store (degree) | held/source |
|---|---|---|---|
| sudden write into a quadratic surface | force × x (1) | $\tfrac12m\omega^2x^2$ (2) | ½ |
| $1/r$ binding from rest | $V$ ($\lvert k\rvert=1$) | $p^2/2m$ (2) | ½ |
| capacitor from a stiff bus | $QV$ (1) | $Q^2/2C$ (2) | ½ |
| Smarr, Schwarzschild | $Mc^2$ (1) | $S\propto M^2$ (2) | $T_HS=\tfrac12Mc^2$ |

**DERIVED**; Euler's theorem ESTABLISHED. The record is always paid from a degree-1 term into a degree-2 store.

**(3.8) The thermodynamic pair is the finite-temperature completion.** For a store of degree $n$ driven from its blank thermal state,
$$\big(k_BT\,D(\rho_1\Vert\rho_0),\ k_BT\,D(\rho_0\Vert\rho_1)\big)\xrightarrow{T\to0}\Big(\frac1n,\frac{n-1}{n}\Big)W_{\rm source};$$
quartic store at β = 3000: 1.0045 and 0.9983 of (1/4, 3/4); exact at every $T$ only for $n=2$. **DERIVED**, as a numerical low-T limit, not a proof for all $n$.

**(3.9) What only coincides.** The harmonic mode's $\langle K\rangle=\langle V\rangle$ keeps both halves in the mode (nothing leaves), so it is not a held/spent pair. **DERIVED** (negative). Koide's 2/3 is an equal split of a quadratic norm, not a source/store balance (Section 4.6 shows a sudden write can make it one).

**(3.10) The same identity in $d$ space dimensions**. Gravity in $d$ dimensions has degree $-(d-2)$:
$$K_{\rm kept}=\frac{d-2}{2}|V|,\qquad Q_{\rm rad}=\frac{4-d}{2}|V| ,$$
equal only at $d=3$ **DERIVED**; circular orbits stable only for $d<4$ (Ehrenfest 1917, ESTABLISHED). Tangherlini's Smarr: $T S/M=(d-2)/(d-1)$, ½ only at $d=3$ **DERIVED**. The virial share $(d-2)/2$ and the Smarr share $(d-2)/(d-1)$ agree only at $d=2$ and $d=3$ **DERIVED**. The sudden-write and capacitor ½, the Bekenstein-Hawking ¼ and the horizon Landauer $4\ln2\,\ell_P^2$ do not depend on $d$. IAM has no principle of its own that selects $d=3$; it inherits the Ehrenfest-Tangherlini window through the virial half (**GAP**, a restatement).

**(3.11) Reading.** The virial $K$ and $Q$ are the record's held and spent halves: one identity from the atom (photon radiated, motion retained) to the horizon (Smarr). **IAM IDENTIFICATION.**

**(3.12) GAP: the radiated half as dissipated work.** *Calculation:* a classical charge captured from rest into a bound orbit, coupled to the Maxwell field as the Step-1 Gaussian bath at $T$; compute the field's $D(\rho_0\Vert\rho_1)$ for the radiated packet and compare $k_BT\,D$ with $|E_f|=\langle K\rangle$. This joins the information form (Step 1) to the mechanical form (Step 3) for $1/r$ binding.

## 4. Step 4: Koide as the record's compression from 3D to 2D

**(4.1) The state and its two pieces.** $\vec s=(\sqrt{m_e},\sqrt{m_\mu},\sqrt{m_\tau})$, $\hat n=(1,1,1)/\sqrt3$, $\vec s=\vec s_\parallel+\vec s_\perp$. With $Q=\sum m/(\sum\sqrt m)^2$,
$$\cos^2\theta=\frac{|\vec s_\parallel|^2}{|\vec s|^2}=\frac1{3Q}.$$
**DERIVED.** $Q=2/3$ is θ = 45° (Foot 1994, ESTABLISHED).

**(4.2) Brannen's form.** $s_k=x[1+r\cos(\delta+2\pi k/3)]$. The $Z_3$ sums give
$$|\vec s_\parallel|^2=3x^2,\qquad|\vec s_\perp|^2=\tfrac32x^2r^2,\qquad Q=\tfrac13\big(1+\tfrac{r^2}2\big),$$
independent of δ. Koide holds iff $|\vec s_\perp|=|\vec s_\parallel|$ ($r=\sqrt2$). **DERIVED**

**(4.3) The plane is one canonical pair.** In Clarke coordinates $(\alpha,\beta)$ on the plane, the rotation about (1,1,1) that advances δ is generated by $H=\omega(\alpha^2+\beta^2)/2$: action $I=|\vec s_\perp|^2/2$, angle δ, $\{I,\delta\}=1$; the axis is conserved; $Q$ and $I$ do not depend on δ. **DERIVED.** In grid language: the axis is the zero sequence; the plane is the balanced set, one phasor.

**(4.4) The record map.**
$$R:(x,r,\delta)\longmapsto(x,Q)\quad(\text{equivalently }(|\vec s_\parallel|,I)).$$
It keeps the scale and the plane's action and drops the angle: three numbers become two. **DERIVED** (definition).
- It is many-to-one: δ = 2/9 and 2/9 + 0.1 give the same $(x,Q)$, $Q=0.666666666667$, with different mass triples. Solving $R(\vec s)=(1,2/3)$ returns $x=1$, $r=\sqrt2$ with δ free.
- On the Koide circle the all-positive arcs are exactly 1/4 of the circle.
- Keeping δ, the map is invertible: $\det\partial(s_0,s_1,s_2)/\partial(x,r,\delta)=-\tfrac{3\sqrt3}{2}x^2r$.

All **DERIVED**. The full Koide parametrization (scale, $Q$, δ) is an invertible change of coordinates; the compression is the record that drops δ.

**(4.5) The entropy form: one bit.** With $p_i=m_i/\sum m$ and $q_i=\sqrt{m_i}/\sum\sqrt m$,
$$Q=e^{-H_{1/2}(p)}=\sum_iq_i^2=\frac{1+\mathrm{CV}^2}{3},$$
$$D_{1/2}(p\Vert u)=\ln3-H_{1/2}(p)=-\ln F=D_2(q\Vert u)=\ln(3Q),\qquad F=\Big(\sum_k\sqrt{p_k/3}\Big)^2=\frac1{3Q},$$
and all equal $\ln2$ (one bit) iff $Q=2/3$. **DERIVED.**

Measured (PDG 2024): $Q=0.666664463$, $H_{1/2}=0.40546841$ nats against $\ln(3/2)=0.40546511$, $D_{1/2}=0.99999523$ bits, $3Q-1=|\vec s_\perp|^2/|\vec s_\parallel|^2=0.99999339$. **DERIVED.**

For $N$ generations $Q=(1+\mathrm{CV}^2)/N$ and $D_{1/2}=\ln(1+\mathrm{CV}^2)$, one bit at CV = 1 for any $N$. Brannen's form on $N$ phases at $r=\sqrt2$ gives $Q=2/N$. So the 3 of 2/3 is $N$, and the 2 is one canonical pair. (Prior art for the $\sum q^2$ form: Zenodo 19271888, not peer reviewed; Hübner 2026 preprint names $1/Q$ an effective participant number.)

**(4.6) Why 2/3 is one canonical pair's energy.** For ν quadratic degrees of freedom ($\chi^2_\nu$), $\mathrm{CV}^2=2/\nu$ and
$$Q_\nu=\frac{\nu+2}{3\nu}:\quad\nu=1\to1,\quad\nu=2\to\tfrac23,\quad\nu\to\infty\to\tfrac13 .$$
One canonical pair $(q,p)\sim N(0,1)$ has $E=(q^2+p^2)/2$ exponential, CV = 1. **DERIVED.**

**(4.7) Why the plane carries the axis's norm: four routes tested**

1. *Classical three-component pure-dephasing pointer.* Invariance about (1,1,1) forces $\Lambda=aP_\parallel+bP_\perp$ (a 2-dimensional solution space); held = spent holds sector by sector for any split, and extremizing over directions gives 0° or 90°, never 45°. The write records the input split and makes none. **DERIVED (negative): this route is retired.**
2. *The Step-3 degree ratio between axis and plane.* Euler's half gives Koide only for the role (axis, plane) = (held, spent); (source, held) gives $Q=1/2$; (held, source) gives $Q=1$. **DERIVED.**
3. *The sudden write (candidate ii).* A constant force switched on at $t=0$ on a mode at rest: $q=q_*(1-\cos\omega t)$, $p=m\omega q_*\sin\omega t$. In the energy metric $(\alpha,\beta)=(q-q_*,\,p/m\omega)$ the orbit is $\alpha^2+\beta^2=q_*^2$, a circle about the new equilibrium whose radius equals its distance from the blank point. Take axis = the held offset and plane = the ringing pair:
$$\vec s=q_*\hat n+\alpha\hat e_1+\beta\hat e_2\ \Longrightarrow\ |\vec s_\perp|=|\vec s_\parallel|,\qquad Q=\tfrac23\ \text{at every }t,$$
with nothing fitted. Read on three phases the vector is exactly Brannen's form with $r=\sqrt2$ and $\delta=\omega t+\pi$: δ is the ringing phase. $F=\cos^2\theta=$ held/(held + spent) $=1/2$, so the Step-1 pair is the Step-4 bit. **DERIVED** (construction) / **IAM IDENTIFICATION** (that the lepton triple is such a written vector, zero sequence = held offset, balanced set = ringing pair, in one energy metric).
   - *An instance in established cosmology.* Preheating at the end of inflation (Kofman, Linde & Starobinsky, PRL 73, 3195, 1994) is a sudden write by link 1's rule inside KLS's band $k<k_*$, and its created quanta hold 0.500 of their energy in the held (interaction) term against the ringing pair for $k\ll k_*$ (0.4998 at $0.05k_*$, 0.495 at $k_*$): the equal split of route 3, in one canonical pair. A second burst onto the already-written field splits its NEW energy 0.4995 (0.4984 to 0.4996 across modes): on a linear field the split of each write's own energy is set by the field's linearity, not by a clean slate; the warm-start departure is in the total, from energy already present. **DERIVED** (numeric) / it is an instance of the construction, not evidence that the lepton triple is written this way.
   - Failure modes, each signed: a ramp gives $Q=(1+\mathrm{sinc}^2(\omega\tau/2))/3<2/3$ (0.569358 at $\omega\tau=2$); a warm start gives $Q_T=\tfrac23+\dfrac{2k_BT}{3kq_*^2}>\tfrac23$; one quadrature only gives $r=1$, $Q=1/2$.
4. *Quantum: the three-component pointer is spin 1.* $(J_k)_{ij}=-i\varepsilon_{kij}$, $J^2=2$, $J_n$ eigenvalues $-1,0,1$, the $m=0$ eigenvector is the (1,1,1) axis; in the stretched state $m=1$, $\langle J_n\rangle^2=\langle J_\perp^2\rangle=1$. For spin $j$, axis $j^2$ and plane $j$, so $Q(j)=(j+1)/(3j)=Q_\nu(\nu=2j)$, equal only at $j=1$. The Euler balance on the Casimir would pick $j=1/2$ ($Q=1$): the dimension, not the degree ratio, fixes $j=1$. **DERIVED** / **ESTABLISHED** (spin-coherent states: Radcliffe 1971; Arecchi et al. 1972; vector model: Gomez 2008) / **IAM IDENTIFICATION** (lepton triple ↔ stretched spin-1 pointer).

(MaxEnt and equipartition per quadrature were also tested: MaxEnt gives 2/3 only with two-sector counting put in; equipartition per quadrature gives $Q=1$.)

**(4.8) The caveat, plainly.** In nature δ is measured (0.222225 rad). "Dropped" refers to the invariant $(x,Q)$, not to the masses. That a physical process forgets δ is not shown. The measured $Q$ sits 0.43σ below 2/3 (PDG 2024; unchanged in the PDG 2025 update, $m_\tau=1776.93\pm0.09$ MeV), so neither signed departure of (4.7.3) is measured yet.

**(4.8b) Koide's half is the ledger's half (exact identity).** Let $v=(\sqrt{m_1},\sqrt{m_2},\sqrt{m_3})$. Then
$$|v|^2=\textstyle\sum m_i\ \ (\text{the rest energy}),\qquad |v\cdot\hat n|^2=\frac{(\sum\sqrt{m_i})^2}{3}\ \ (\hat n=(1,1,1)/\sqrt3),\qquad
\frac{|v\cdot\hat n|^2}{|v|^2}=\frac1{3Q}.$$
At Koide's $Q=2/3$ the axis (the scale the record keeps, Step 4) carries **exactly half the rest energy**; the plane (the dropped phase $\delta$) carries
the other half. Reading (IAM IDENTIFICATION): matter's rest energy written as a Koide record from 3D onto the 2D surface holds half as record and
leaves half as working energy; the ledger books the held half, which is $\tfrac12M_mc^2/k_BT_{GH}$, i.e. $\beta_m=\Omega_m/2$. The $1/3$ in the record
term's $1/6$ of matter's horizon-flux entropy is the shell geometry $1/d$. Charged leptons: $Q=0.666661$ (PDG 2024), axis share 0.500004 (both unchanged with PDG 2025).
GAP: that every record matter writes is Koide-shaped (three components, $Q=2/3$), not only the lepton masses; the light quarks do not satisfy Koide.
Step 4 already shows a sudden write gives $Q=2/3$ by geometry; the step to close is that matter's records on the local surface are such writes.

**(4.9) GAPs.** (i) X11 dynamics: show the lepton triple is read out as (held offset, ringing pair). *Calculation:* the Step-1 multi-mode Gaussian surface written with a $Z_3$-symmetric three-phase readout; compute $Q(t;\tau,T)=\tfrac23-(\text{rise-time term})+(\text{thermal term})$, with each mode weighted by $\mathrm{sinc}^2(\omega_k\tau/2)$. Data that decide: the τ mass as it settles. (ii) $N\le3$ inside IAM rests on a single-harmonic encoding (Step 6, conjecture); CV = 1 and positivity alone allow $N\ge4$ (Monte Carlo positive fractions 0.156, 0.103, 0.066 for $N=4,5,6$, provisional). *Calculation:* the Step-4 pointer with $N$ components.

## 5. Step 5: the arrow

**(5.1) What is established.**
1. Fine-grained entropy is constant under Hamiltonian/unitary evolution; coarse-grained entropy can grow; the thermodynamic arrow needs a coarse-graining and a low-entropy start. **ESTABLISHED** (Jaynes 1965; Lebowitz 1993) / **DERIVED.**
2. No channel increases distinguishability: $D$, $D_{1/2}$, $D_2$ are monotone. **ESTABLISHED** (Lindblad 1975; Uhlmann 1977; van Erven & Harremoës 2014) / **DERIVED** (2,000 random pairs, a test, not a proof).
3. Logically irreversible steps must dissipate; bijective steps need not. Forgetting a coordinate held at $N$ equiprobable values costs at least $k_BT\ln N$. **ESTABLISHED** (Landauer 1961; Bennett 1973, 1982, 2003) / **DERIVED.**
4. $\langle W_{\rm diss}\rangle=k_BT\,D(\rho_F\Vert\rho_R)$; Crooks $P_F(W)/P_R(-W)=e^{\beta(W-\Delta F)}$. **ESTABLISHED** (Kawai, Parrondo & Van den Broeck 2007; Parrondo, Van den Broeck & Kawai 2009; Crooks 1999).
5. $W_{\rm meas}+W_{\rm eras}\ge k_BT\,I$. **ESTABLISHED** (Sagawa & Ueda 2009).

**(5.2) The fluctuation theorems hold exactly for the Step-1 write.** Forward work $W=-Fx$, $x\sim\rho_0$: $\langle e^{-\beta W}\rangle=e^{\beta U}=e^{-\beta\Delta F}$ with $\Delta F=-U$. With $u=U/k_BT$, $\beta W_F\sim N(0,2u)$ and $\beta W_R\sim N(2u,2u)$ satisfy Crooks exactly, with mean $\beta W_{\rm diss}=u=D(\rho_0\Vert\rho_1)$. **DERIVED.**

**(5.3) The arrow per record.**
$$\beta\langle W_{\rm diss}\rangle=\frac{U}{k_BT}\ \ge\ \chi ,$$
margins $10^{-6}$, 0.0089, 0.50 and 9.31 at $U/k_BT=0.001,0.1,1,10$. Every nat written by a sudden write makes the reversed history at least one nat less likely. **DERIVED.**

**(5.4) Write and un-write.** Sudden write costs the writer $U$; sudden erase back to blank costs $k_BT\,D(\rho_1\Vert\rho_0)=U$; the round trip $2U\ge k_BT\,\chi$. "Un-writing costs at least what writing cost" is not a general theorem (Bennett: a write into a blank memory can be reversible); what holds is the sum bound. **DERIVED** / **ESTABLISHED.**

**(5.5) A many-to-one record cannot be inverted from itself.** The Koide record $R$ has a circle of preimages. If δ was held at $N$ values and forgotten, resetting it cost at least $k_BT\ln N$; if δ is kept the record is a bijection. The arrow lives in the dropped coordinate; physically, restoring it means un-dephasing the spent half, the ringing already radiated. **DERIVED** (with Landauer-Bennett ESTABLISHED).

**(5.6) The KR1 simulation, first result**: (b) MET — rewinding from the Koide record spreads $Q$ by 0.095 against 0.0067 with the phase kept (classical rotation); (d) MET — a one-tile overwrite loses 1.31 bits per tick (classical) and 0.69 (qutrit). Items (a) and (c) were invalid due to a script error and are queued for rerun. **Measured on the simulator; provisional until the rerun.**

**(5.7) Where IAM's claim begins.** Einselection supplies an objective coarse-graining (Zurek 1981, 2003; ESTABLISHED). IAM's claim: every decoherence event on a timelike worldline is such a write, paid at the nearest surface at its local temperature; the accumulated ledger $E(a)$ is the arrow; and the record matter writes keeps pointer values and drops phases. **IAM IDENTIFICATION.** The low-entropy start is not removed.

**(5.8) GAP: the arrow as the spent half.** *Calculation:* run the Step-1 write in a finite bath of $N$ modes, reverse every bath momentum at time $t$ (Loschmidt echo), and measure the un-writing fidelity against $t$, $N$ and $U/k_BT$. Prediction: the echo survives only while the spent half has not dephased; its loss rate tracks $U/k_BT$.

---

**The cell's instance of link 5.** None derived yet. Candidates: OPEN: the mitotic clock tick as writer error made heritable, a cell record that is copied forward and not restored; calculated: an erased mark stays unmarked 1-6 h; proposed: UV damage, caused by light, written by the cell, partly unrepairable.

## 6. Step 6: the entropy functional

Units $\hbar=c=k_B=1$ in this section unless shown.

**(6.1) Jacobson.** $\delta Q=T\,\delta S$ on every local Rindler horizon, with $S=\eta A$, gives
$$R_{ab}-\tfrac12Rg_{ab}+\Lambda g_{ab}=\frac{2\pi}{\hbar c\,\eta}T_{ab}\equiv\frac{8\pi G}{c^4}T_{ab},\qquad G=\frac{c^3}{4\hbar\eta},\qquad\eta=\frac1{4\ell_P^2}.$$
**ESTABLISHED** (Jacobson 1995). This rests on the hypothesis that gravity is thermodynamic in origin.

**(6.2) Cai-Kim on the apparent horizon.** Flat FRW: $\tilde r_A=1/H$, $A_H=4\pi/H^2$, $S_{\rm geo}=\pi/GH^2$, $T_H=H/2\pi$, Misner-Sharp energy $E=\tilde r_A/2G$.
1. $T_H\,dS_{\rm geo}=\dfrac{H}{2\pi}\Big(-\dfrac{2\pi}{GH^3}\Big)\dot H\,dt=-\dfrac{\dot H\,dt}{GH^2}$.
2. The energy crossing in $dt$: $-dE=A_H(\rho+P)\tilde r_AH\,dt=\dfrac{4\pi(\rho+P)}{H^2}dt$.
3. $-dE=T_H\,dS_{\rm geo}$ gives $\boxed{\dot H=-4\pi G(\rho+P)}$.
4. With continuity $\dot\rho=-3H(\rho+P)$: $\frac d{dt}\big(H^2-\tfrac{8\pi G}3\rho\big)=2H\dot H+8\pi GH(\rho+P)=0$, so $H^2=\tfrac{8\pi G}3\rho+\tfrac\Lambda3$, Λ an integration constant.

**ESTABLISHED** (Cai & Kim 2005; Hayward 1998; Akbar & Cai 2007) / **DERIVED**.

**(6.3) The record term.** IAM adds the informational entropy:
$$S_{\rm total}=S_{\rm geo}+S_{\rm info},\qquad -dE=T_H\,d(S_{\rm geo}+S_{\rm info}).$$
**IAM IDENTIFICATION**. This is the single new physical identification introduced by IAM's Law.

**(6.4) The modified second Friedmann equation.**
$$\frac{4\pi(\rho+P)}{H^2}=-\frac{\dot H}{GH^2}+\frac{H}{2\pi}\dot S_{\rm info}\quad\Longrightarrow\quad\boxed{\dot H=-4\pi G(\rho+P)+\frac{G}{2\pi}H^3\dot S_{\rm info}}$$
**DERIVED.** The record acts as a component with
$$\rho_x+P_x=-\frac{H^3\dot S_{\rm info}}{8\pi^2}<0\qquad(\dot S_{\rm info}>0),$$
so $w<-1$ whatever the history of $S_{\rm info}$. **DERIVED**, given Assumptions 1-2.

**(6.5) The record constraint and the activation function.** The premise (conjecture):
$$\dot\rho_{\rm info}=\rho_{\rm info}\frac Ha\quad\Longleftrightarrow\quad\frac{d\ln\rho_{\rm info}}{da}=\frac1{a^2}.$$
**IAM IDENTIFICATION** (premise). It integrates, with $E(1)=1$ set by definition, to
$$E(a)\equiv\frac{\rho_{\rm info}(a)}{\rho_{\rm info}(1)}=\exp\Big(1-\frac1a\Big)=e^{-z}.$$
**DERIVED.** Properties: $E\to0$ as $a\to0$; $E(\infty)=e$; inflection at $a=1/2$ ($z=1$); $dE/d\ln a=E/a$, largest at $a=1$. From continuity with $d\ln\rho_{\rm info}/d\ln a=1/a$,
$$w_{\rm info}(a)=-1-\frac1{3a}:\quad-1.67\ (a=0.5),\ -1.33\ (a=1),\ -1.17\ (a=2)$$
. The constraint is the balance law
$$\frac{d\ln E}{d\ln a}=1-\ln E$$
(a writer at one per e-fold with no eraser). **DERIVED.**

**(6.6) What the first law then requires.** Setting the record component of (6.4) to $\rho_{\rm info}=(3H_0^2/8\pi G)\beta_mE(a)$ with $w_{\rm info}$:
$$\dot S_{\rm info}=\frac{8\pi^2\rho_{\rm info}}{3aH^3},\qquad\frac{dS_{\rm info}}{d\ln a}\Big|_{a=1}=\beta_m\frac{\pi}{GH_0^2}=\beta_mS_{\rm geo}(1).$$
**DERIVED.** At $H_0=67.16$: $S_{\rm geo}=2.2817\times10^{122}$ nats and $\beta_mS_{\rm geo}=3.597\times10^{121}$ nats $=5.19\times10^{121}$ bits per e-fold. **DERIVED.**

**(6.7) Tension in the rate.** With $\dot I\propto\rho_mD^nfH$ and $dS_{\rm info}/dt\propto\dot I/T_HA_H$, $n=7/2$ makes the writing per e-fold fall as $1/a$ in the matter era, while (6.6) needs it to rise as $E(a)a^5$; matching needs $n_{\rm eff}=19/2+1/a$. **DERIVED.** This was originally presented as an open problem. So the shape of $E(a)$ rests on the record constraint alone.

**(6.8) Where the term sits in established non-equilibrium spacetime thermodynamics.** Eling, Guedens & Jacobson derive field equations from $dS=\delta Q/T+d_iS$. The book's first law $T(dS_{\rm geo}+dS_{\rm info})=\delta Q$ is that balance with
$$d_iS=-dS_{\rm info}.$$
**ESTABLISHED** (EGJ 2006) / **DERIVED** (the placement) / **IAM IDENTIFICATION** (that this slot is the record). Two cautions carried: a 2026 analysis warns that a term put into the apparent-horizon Clausius relation ad hoc can make the derivation circular unless it is prescribed (Pai, Namboothiri & Mathew, arXiv:2604.04731); and in FRW, being shear-free, EGJ's own $d_iS$ is second order and carries no heat. Steps 1-5 are offered as the prescription.

**(6.9) How Steps 1-5 supply $S_{\rm info}$.** Record $i$, written at a local surface at $T_i$, carries $\chi_i$ nats and spends $U_i\ge k_BT_i\chi_i$ there (Steps 1-2); its entry is non-negative because the write is many-to-one (Step 5). Two candidate ledger entries:
- **count:** $\Delta S_{\rm info}=k\chi_i$;
- **heat:** $\Delta S=U_i/T_H$, if the spent heat reached the horizon.

They differ by at least $T_i/T_H$: $1.172\times10^{32}$ for a 310 K cell against $T_{GH}=2.6459\times10^{-30}$ K. Heat moving outward only gains entropy, $Q(1/T_2-1/T_1)>0$. **DERIVED** (numbers). The ATM picture — the transaction is local, the bank shows the balance — gives the count booking, with the fee travelling outward as ordinary heat that Cai-Kim already books in $-dE$:
$$S_{\rm info}(t)=k\sum_{i\ \text{written by }t}\chi_i,\qquad\dot S_{\rm info}\ge0 .$$
**IAM IDENTIFICATION.** Count pricing uses bits $=Q/k_BT_{\rm gas}\ln2$ each at $k_BT_{GH}\ln2$.

**(6.10) The variational form.** The action with the constraint as a Lagrange multiplier returns eq:Hm in the FRW minisuperspace; it assumes the activation function it returns, and $H/a$ is not a spacetime scalar. **IAM IDENTIFICATION** (CONJECTURE; covariant form OPEN).

**(6.11) GAPs.** (i) The rate: no local process is yet shown to write at (6.6) (the bottom-up writers). (ii) The record constraint is a premise. (iii) A covariant action.

Nothing yet bears on (i)-(iii). Link 1d prices each local write by its speed; it does not say how many writes happen per e-fold. The preheating runs bear on the first write (17.1, item 3), not on today's rate.

The message you sent contains only a section heading — "## 7. Step 7: the horizon as the ledger, followed through temperature and distance" — with no accompanying derivation text, equations, or notes to edit. There is no content here for me to process under the editing rules.

Please paste the full text of Block 7, including its equations, citations, and working notes, and I'll produce the edited version.

**(7.1) The entry price at the bank.** For a Killing horizon, $T=\hbar\kappa/2\pi k_B$ and $\delta E=\kappa c^3\delta A/8\pi G$ (eq:th:dAmin). Heat $k_BT\ln2$ adds
$$\delta A=4\ln2\,\frac{\hbar G}{c^3}=4\ln2\,\ell_P^2\qquad\text{for every }\kappa:$$
Landauer as an equality. One bit is $7.2428\times10^{-70}$ m². For a Schwarzschild hole $N_{\rm bits}k_BT_H\ln2=\tfrac12Mc^2$: the ln 2 cancels and the bill is the Smarr half. Energy $k_BT_H\ln2$ on a carrier no larger than $r_s$ holds at most half a bit, so a tile is a unit of the count, not an address. **DERIVED** (Bekenstein 1973, Hawking 1975, ESTABLISHED).

**(7.2) The local records, summed.** With the count booking,
$$A_{\rm info}=4\ell_P^2\sum_i\chi_i,\qquad N_{\rm bits}=\frac{\sum_i\chi_i}{\ln2},\qquad\frac{dA_{\rm info}/d\ln a}{A_H}\Big|_{a=1}=\beta_m\ \ (\text{when the book's rate holds}).$$
**DERIVED.**

**(7.3) The temperature ladder.** From the writer outward:

| surface | temperature | formula | source |
|---|---|---|---|
| cell chromatin | 310.15 K | measured | measured |
| black hole | $T_{\rm BH}=\hbar c^3/8\pi GMk_B$ | Hawking 1975 | ESTABLISHED |
| cosmic apparent horizon | $T_{GH}=\hbar H/2\pi k_B=2.6459\times10^{-30}$ K today | Gibbons-Hawking 1977 | ESTABLISHED |
| de Sitter floor | $T_{dS}=\sqrt{\Omega_\Lambda}\,T_H=0.8275\,T_H$ | — | ESTABLISHED form |

- At the ATM (local surface) Landauer is a bound paid twice (held and spent); at the bank it is an equality. Fee/entry $=T_{\rm loc}/T_H\approx10^{32}$ for a warm cell. **DERIVED.**
- Only the ratio $T_{\rm ledger}/T_{\rm writer}$ enters a heat booking; pricing writer and ledger both at $T_{dS}$ is identical to pricing both at $T_H$ (DERIVED). The one booking that moves the shape is "written at $T_H$, booked at $T_{dS}$" (the 1.0079 booking); same-surface bookings give 1.218 or 0.690 (sixteen-thirds, dark sector). That assignment is fitted. **IAM IDENTIFICATION.**

**(7.4) The distance to the horizon: the Hubble sphere is a saturated Bekenstein region.** Bekenstein's bound is $S\le E/T_R$ with $T_R=\hbar c/2\pi k_BR$. For the matter-plus-everything content of the Hubble sphere, $M_H=\rho_{\rm crit}V_H$:
$$\frac{GM_H}{R_Hc^2}=\frac12,\qquad r_s(M_H)=R_H,\qquad T_{\rm BH}(M_H)=\tfrac12T_{GH}.$$
**DERIVED.** The horizon's temperature identity: $T_HS_{\rm geo}/V_H=3H^2/8\pi G=\rho_{\rm crit}$ exactly. **ESTABLISHED** (Bekenstein 1981, PRD 23, 287).

**(7.5) Rate along the distance.** Pricing at $T_H$ is the Bekenstein bound of the Hubble radius, $\pi E/\hbar H=E/2k_BT_H$. Per writing particle the book's requirement is
$$\Gamma_{\rm req}=\pi E(a)a^2\frac{mc^2}{\hbar}\ \ \text{nats/s},\qquad\frac{\Gamma_{\rm req}}{\pi mc^2/\hbar}=E(a)a^2 ,$$
with $H$ cancelling: exactly Bekenstein's information-transfer rate today, 0.0920, 0.0150, 0.0031 of it at $z=1,2,3$, and 4.478e24 nats/s for a proton. Bekenstein's rate over Margolus-Levitin's $2E/\pi\hbar$ is $\pi^2/2$. **DERIVED.** Bekenstein's 1981 rate bound is heuristic (Deffner 2020): **GAP** (the right scale, not a theorem).

**(7.6) Light-travel distance.** A record's consequences propagate at most at $c$. GW170817 puts gravitational waves on the light cone, so by the proper-time criterion they write nothing in flight (time-and-rate). If records must reach the Hubble sphere at $\le c$, booking lags writing by 0.73 e-folds (written today) to 1.07 (written at $z=3$); records reaching it today were written at $z=1.48$, $E=0.227$ (time-and-rate, numeric, provisional). Cai-Kim books at the same instant by homogeneity. **IAM IDENTIFICATION** (which reading IAM intends is Question 13.6).

**(7.7) Clocks.** "Clocks run faster where there is more actualization" is opposite to GR proper time (Pound-Rebka; GPS) and to timescape (void clocks faster); it holds only for matter's expansion bookkeeping, $H_m/H=1.0759$ today, 1.0090 at $z=1$ (time-and-rate). **ESTABLISHED** (GR clock tests) / **IAM IDENTIFICATION** (the matter-sector clock is a construct the book does not define).

**(7.8) Reading.** One nat written locally is one nat on the ledger. **IAM IDENTIFICATION.**

**(7.9) GAP: the booking rule.** The established entropy balance gives the cost of a record made inside a region as a flux through the region's boundary at the boundary's temperature (de Groot-Mazur), not as storage on it; flux-to-storage is established only for horizons. *Calculation:* in the book's Cai-Kim form follow one radiating local record; show its spent heat $U$ enters through $-dE$ (hence $S_{\rm geo}$), so that $S_{\rm info}$ has only the count $k\chi$ to book. It uses Cai-Kim as the book does and touches neither $\beta_m$ nor $E(a)$.

### 7c. Why the ledger books rest energy, all of it, at the half
- [derived identity, not fitted] The rate the ledger needs, per unit of matter's rest energy, is pi E(a) a^2 m c^2/hbar nats per second: today (E a^2 = 1) every proton writes exactly at Bekenstein's maximum rate for its rest energy (Bekenstein 1981). Rest energy is the writing capacity.
- [derived + established] All of it: both ledgers book gravitating mass (beta_m T_GH S_geo = M_m c^2/2 iff beta_m = Omega_m/2; Schwarzschild TS = Mc^2/2 for any composition), and binding of every kind gravitates like rest mass (MICROSCOPE to 2.7e-15; lunar laser ranging to about 1e-4).
- [derived] At the half: Koide's axis share is 1/(NQ), and one write gives Q = 2/N, so the held part is exactly 1/2 for any number of components and in any dimension; the virial and Smarr halves equal it only because space has three dimensions.
- [open] Derive 'rest mass is the record of a write and its writing capacity' instead of reading it from the Bekenstein match; and the record constraint E(a) (Section 10 status block).

## 8. What the requirement $\beta_mS_{\rm geo}$ is, exactly

These identities hold in units $\hbar=c=k_B=1$, flat matter plus Λ, at $a=1$ unless stated. They restate the book's numbers.

**(8.1) Half of matter's rest energy at the bank's temperature.**
$$\beta_mS_{\rm geo}=\frac12\,\frac{M_mc^2}{k_BT_{GH}}\quad\Longleftrightarrow\quad\beta_m=\frac{\Omega_m}2,$$
$M_m=\Omega_m\rho_{\rm crit}V_H$ (matter inside the apparent horizon, $1.47\times10^{22}M_\odot$). **DERIVED**

**(8.2) One sixth of matter's horizon flux.** The matter flux per e-fold, $-dE=A_H\rho_m\tilde r_AH(1/H)=4\pi\rho_m/H^3=3M_m$, carries $8\pi^2\rho_m/H^4=3\Omega_mS_{\rm geo}$ at $T_H$; $\beta_mS_{\rm geo}$ is exactly 1/6 of it:
$$\frac16=\underbrace{\frac12}_{\text{share of the content}}\times\underbrace{\frac13}_{V_H/(A_H\tilde r_A)=1/d}.$$
**DERIVED.** The 1/3 is geometry (the same 3 as continuity's $3H$), not a new physical factor.

**(8.3) At any $a$.** The required record is $E(a)a^2/6$ of the matter-flux entropy, with $H$ and radiation dropping out: 1/6 today, 0.0153 at $z=1$, 0.0025 at $z=2$. **DERIVED**

**(8.4) The ledger entry rule.** $k_BT_{GH}\,dS_{\rm info}=V_H\,d\rho_{\rm info}$ follows from eq:law_F2info plus continuity; today it is (balance sheet $\tfrac12\rho_m$) × (turnover $d\ln E/d\ln a=1$) $=\tfrac12M_mc^2$ per e-fold. **DERIVED**

In Padmanabhan's counting the record per e-fold today is matter's $|N_{\rm bulk,m}|$ and $\beta_m=|N_{\rm bulk,m}|/N_{\rm sur}$ (Padmanabhan 2012).

**(8.5) The Bekenstein ceiling after today (proposal).** With the book's $E$, $a^3\,dE/d\ln a=E\,a^2$: 1 at $a=1$, 6.59 at $a=2$. A record held to the ceiling $a^3\,dE/d\ln a\le1$ for $a>1$ would follow
$$E_{\rm sat}(a)=\frac{4-a^{-3}}{3},\qquad E_{\rm sat}(1)=1,\qquad E_{\rm sat}\to\tfrac43 .$$
**DERIVED.** Nothing at $a\le1$ changes, so DESI DR2 cannot see it; it would change the book's far future ($\rho_{\rm info}/\rho_\Lambda\to0.307$ not 0.626; $H_{m,\infty}=63.53$ not 70.86 at $H_0=67.16$).

**(8.6) Which half.** At $d=3$ the virial share $(d-2)/2$ and the Smarr/Komar share $(d-2)/(d-1)$ are both ½. For general $d$ they differ: $\beta_m(d)=(d-2)(d-1)\Omega_m/4$ (virial ratio on the Smarr source) against $(d-2)\Omega_m/2$ (Smarr/Komar and Padmanabhan tiles); only Smarr/Komar keeps one rule at both horizons. The Misner-Sharp calculation, booked at $T_{GH}/2$, also gives the Smarr half, numerically at one epoch only. **GAP** (which half is the record's half).

**(8.7) Status: $\beta_m=\Omega_m/2$ is derived top-down.** (8.1)-(8.4) are exact and say what $\beta_m=\Omega_m/2$ *is*: the bank books, per e-fold today, half of matter's rest energy at its own temperature. The value itself follows from general relativity, top-down:
Einstein's equations $\to$ Tolman mass $=$ ADM mass and the GR virial identity ($2K+U=0$ in the weak field) $\to$ Komar at a horizon $=2TS$,
Smarr as Euler's theorem for $M(A,J)$ of degree ½ $\to$ matter's own Smarr half priced at the horizon temperature:
$\beta_mS_{\rm geo}=\tfrac12M_mc^2/k_BT_{GH}$ per e-fold, exact iff $\beta_m=\Omega_m/2$ (8.1). It is not derived bottom-up from counted local writes; the one open step is "why matter's own capacity": why the ledger books the enclosed matter's whole rest energy at the half, when locally counted writes fall short. The free-coupling fit gives $\beta=0.155\pm0.029$, $-0.09\sigma$ from $\Omega_m/2$, and that fit is set by the three $H_0$ entries. **DERIVED** (top-down) / **GAP** (bottom-up).

## 10. (a) The chain from actualization to the Friedmann equation with $\beta_m=\Omega_m/2$, as far as it is derived

Each link names its tag. "Given" marks the premise a link stands on.

1. **Einselection.** The pointer basis is fixed by $[\hat\sigma,H_{\rm int}]=0$. ESTABLISHED (Zurek) / DERIVED.
2. **Actualization.** A sudden write into a blank linear-Gaussian local surface splits the source's $2U$ into held $U$ and spent $U$ exactly, $=k_BT\,D(\rho_1\Vert\rho_0)$ and $k_BT\,D(\rho_0\Vert\rho_1)$. DERIVED, given the four conditions of (1.8).
3. **The price.** $\chi\le U/k_BT$: every nat costs $\ge k_BT$ held and $\ge k_BT$ spent at the local temperature; $k_BT\ln2$ per bit is the floor. ESTABLISHED (Landauer, Holevo) / DERIVED.
4. **The funding.** The payment is free energy degraded to heat (here the pointer-surface binding); Rovelli's trace bound is the same bound. DERIVED; the reservoirs trace to the Past Hypothesis, ESTABLISHED as a postulate.
5. **The half.** The write, $1/r$ binding, the capacitor and Smarr are one Euler balance with degree ratio 1:2; the thermodynamic pair is its finite-$T$ completion; the half is exact only at $d=3$. DERIVED.
6. **The record and the arrow.** The record is many-to-one (Koide: keeps scale and action, drops δ); per record $\beta W_{\rm diss}=U/k_BT\ge\chi$; un-doing needs the dropped coordinate. DERIVED; Koide's equal split follows from a sudden write by geometry, with the lepton identification IAM IDENTIFICATION.
7. **The entry.** One nat written locally is one nat on the ledger: $S_{\rm info}=k\sum_i\chi_i$, $\dot S_{\rm info}\ge0$. IAM IDENTIFICATION (count booking; Assumption 2).
8. **The geometric first law.** $-dE=T_H\,dS_{\rm geo}$ on the apparent horizon gives $\dot H=-4\pi G(\rho+P)$ and, with continuity, $H^2=\tfrac{8\pi G}3\rho+\tfrac\Lambda3$. ESTABLISHED (Jacobson; Cai-Kim) / DERIVED, given Assumption 1.
9. **The completed first law.** $-dE=T_H\,d(S_{\rm geo}+S_{\rm info})$ gives $\dot H=-4\pi G(\rho+P)+\frac G{2\pi}H^3\dot S_{\rm info}$, a component with $\rho+P<0$. DERIVED, given link 7 / Assumption 2. In EGJ's form it is the slot $d_iS=-dS_{\rm info}$.
10. **The history.** Given the record constraint $\dot\rho_{\rm info}=\rho_{\rm info}H/a$ (premise): $E(a)=e^{1-1/a}$, $w_{\rm info}=-1-1/3a$, balance law $d\ln E/d\ln a=1-\ln E$. IAM IDENTIFICATION (premise) / DERIVED.
11. **The requirement.** $\dot S_{\rm info}=8\pi^2\rho_{\rm info}/3aH^3$; today $dS_{\rm info}/d\ln a=\beta_mS_{\rm geo}=3.597\times10^{121}$ nats. DERIVED.
12. **The coupling.** $\beta_mS_{\rm geo}=\tfrac12M_mc^2/k_BT_{GH}$ per e-fold if and only if $\beta_m=\Omega_m/2$; equivalently 1/6 of matter's horizon-flux entropy (½ of the content × 1/$d$); at any $a$, $E(a)a^2/6$. DERIVED identities. The value $\beta_m=\Omega_m/2=0.15765$ itself follows from Assumption 3. IAM IDENTIFICATION; derived top-down since (Section 16.1).
13. **The matter-sector Friedmann equation.** Placing the term in the matter perturbations only (the sector rule; CONJECTURE):
$$\boxed{H_m^2=H_{\Lambda{\rm CDM}}^2+\beta_mE(a)H_0^2=\frac{8\pi G}3\rho_m+\frac\Lambda3+\frac{\Omega_m}2e^{1-1/a}H_0^2}$$
with $\mu(a)=H^2_{\Lambda\rm CDM}/(H^2_{\Lambda\rm CDM}+\beta_mE(a)H_0^2)$, $\mu(1)=0.8638$, $\mu_0=-0.13618$, $\Sigma=1$, and matter-sector $H_0=67.16\sqrt{1+\beta_m}=72.26$. The background Friedmann equation is unchanged. IAM IDENTIFICATION (sector rule) / DERIVED (numbers).

**Where the chain is closed and where it is open.**
- *Closed:* links 1-6 (local physics, under the stated conditions), and links 8-13 algebraically, given Assumptions 1-3, the count booking and the record constraint.
- *Open between 7 and 11 (the rate).* Nothing yet shown writes $\tfrac12M_mc^2/k_BT_{GH}$ per e-fold today with history $E(a)a^2$: the count booking of known writers gives $\sim10^{-19}$ of it, the book's own virial rule $\sim10^{-6}$, starlight priced at the bank $\sim10^{-5}$ (the bottom-up writers).
- *Open at 10:* the record constraint is a premise; the book's own rate law disagrees with it in the matter era ($n_{\rm eff}=19/2+1/a$).
- *Open at 12:* the identities say exactly what $\beta_m=\Omega_m/2$ books; they do not supply the writer, and "which half" (virial or Smarr/Komar) is not fixed beyond $d=3$.
- *Open at 13:* the sector rule is an argument from symmetry.
- *Which half (open at 12): ANSWERED IN PRINCIPLE.* N components: Koide's axis share is 1/(NQ); one write gives Q = 2/N, so the held part is exactly 1/2 in every dimension and for every N, the sudden-write half; the virial half (d-2)/2 and Smarr's (d-2)/(d-1) equal it only at d = 3. At a black hole Smarr's heat share is 1/2 at zero spin and Christodoulou-Ruffini's equal split holds only at extremal spin, so 'every record is Koide-shaped' is not shown there.
- *Why matter's own capacity: beta_m = Omega_m/2 DERIVED TOP-DOWN* (Einstein -> Tolman/Komar half -> matter's own Smarr half at T_GH). Candidate for the last step, IAM IDENTIFICATION: matter's rest energy written as a Koide record keeps exactly half on the axis (Q = 2/3 gives axis share 1/2); charged-lepton masses are bare Higgs writes, which is why Koide holds for pole masses and fails for running ones. Open: binding energy (the proton is ~99 % QCD) and dark matter booked the same way.
- *Booking (gap-list item 1): SETTLED for the book's booking.* Heat crossing at T_GH (eq:Sdot) is self-consistent with weight 1, and identical with the Kodama-Hayward temperature inside the unified first law; the long form's 0.76 was an inconsistent hybrid. The tally booking is real but carries Smarr's half; the two cannot both hold for a growing record. The book sets the booking; the math found it consistent.
- *The writer (open between 7 and 11): REFRAMED.* No counted local writer reaches the rate (8e-20 to 3e-19; black holes 3-29 orders short). The Weyl energy of collapsing halos (Clifton-Ellis-Tavakol, no free parameter) supplies 0.51-0.77 of today's amplitude, but every matter-sourced history is front-loaded; no established galaxy-formation scale sets a threshold that fixes the shape. What the rate IS: exactly half of all matter's rest energy at T_GH per e-fold, i.e. today every proton writes exactly at Bekenstein's maximum information rate for its rest energy, a property of matter's whole rest energy on the ledger, consistent with the Koide-half reading above. 'Why now' reads as the rate reaching that ceiling today; prediction if it saturates: E(a) = (4 - a^-3)/3 -> 4/3, w(a = 2) = -1.032 against the book's -1.167.
- *The record constraint (open at 10): STILL A PREMISE.* The saturation history is the first alternative computed. Rate accounting and the constraint disagree.
- *Binding at the half (open under beta_m): ANSWERED from established physics*. Both ledgers book GRAVITATING mass: beta_m T_GH S_geo = (1/2) M_m c^2 holds iff beta_m = Omega_m/2 with M_m from Omega_m, and Schwarzschild TS = Mc^2/2 for any composition. Every form of binding gravitates like rest mass: MICROSCOPE (Ti vs Pt) to 2.7e-15 (Touboul et al., PRL 129, 121102, 2022); gravitational self-energy to ~1e-4 by lunar laser ranging (Hofmann and Mueller, CQG 35, 035015, 2018). So the proton's ~99 % strong binding is booked at the same half. Consequence: the ledger's half is the write's own half (sudden write, any N, any d), which Koide displays only where a mass is one bare write (the leptons). beta_m's last step is an IAM identification WITH ITS REASONS IN HAND: (i) WHY REST ENERGY: rest mass is what makes a worldline timelike, so it carries proper time and can write, while light (null, no proper time) writes nothing (dual-sector reason); and the rate the ledger needs is exactly every proton writing at Bekenstein's maximum for its own rest energy today, pi E(a) a^2 mc^2/hbar (not fitted: from beta_m = Omega_m/2 and Cai-Kim), i.e. mass is the writing capacity; (ii) WHY ALL OF IT: Smarr's half does not depend on composition and every form of binding gravitates like rest mass (above); (iii) WHY HALF: the write's own half, any N and any d; Koide shows it plainly only where a mass is one bare Higgs write, the charged-lepton pole masses, and strong binding laid over the quark masses masks it (heavy quarks, mostly Higgs mass, Q ~ 0.669). What is left is to DERIVE 'rest mass is the record of a write and its writing capacity' rather than read it from the Bekenstein match. Caveat: the strong-binding share common to every nucleon is tested by Omega_b from nuclear counting (BBN) agreeing with Omega_b from the CMB, not by MICROSCOP
- *E(a), the one bare premise, and its test:* the book's record constraint, E(a) = exp(1 - 1/a), against the saturation history after today, E(a) = (4 - a^-3)/3 -> 4/3 (SATURATION). They differ in the future dark-energy equation of state: w(a = 2) = -1.032 (saturation) against -1.167 (book). Today's growth data do not separate the two record histories, with no preference on the 7 growth points; a full chain with DESI BAO and full-shape growth is the deciding run.

## Citations

**Decoherence and records.** Zurek, Phys. Rev. D 24, 1516 (1981), doi:10.1103/PhysRevD.24.1516. Zurek, Phys. Rev. D 26, 1862 (1982), doi:10.1103/PhysRevD.26.1862. Zurek, Rev. Mod. Phys. 75, 715 (2003), doi:10.1103/RevModPhys.75.715. Zurek, Habib & Paz, Phys. Rev. Lett. 70, 1187 (1993), doi:10.1103/PhysRevLett.70.1187. Caldeira & Leggett, Ann. Phys. 149, 374 (1983), doi:10.1016/0003-4916(83)90202-6. Riedel & Zurek, Phys. Rev. Lett. 105, 020404 (2010), doi:10.1103/PhysRevLett.105.020404.

**Information thermodynamics.** Landauer, IBM J. Res. Dev. 5, 183 (1961), doi:10.1147/rd.53.0183. Bennett, IBM J. Res. Dev. 17, 525 (1973), doi:10.1147/rd.176.0525. Bennett, Int. J. Theor. Phys. 21, 905 (1982), doi:10.1007/BF02084158. Bennett, Stud. Hist. Phil. Mod. Phys. 34, 501 (2003), doi:10.1016/S1355-2198(03)00039-X. Bérut et al., Nature 483, 187 (2012), doi:10.1038/nature10872. Sagawa & Ueda, Phys. Rev. Lett. 102, 250602 (2009), doi:10.1103/PhysRevLett.102.250602. Holevo, Probl. Peredachi Inf. 9(3), 3 (1973). Rovelli, "Memory and Entropy", Entropy 24, 1022 (2022), doi:10.3390/e24081022. Guryanova, Friis & Huber, Quantum 4, 222 (2020), doi:10.22331/q-2020-01-13-222. Ouldridge, Govern & ten Wolde, Phys. Rev. X 7, 021004 (2017), doi:10.1103/PhysRevX.7.021004.

**Dissipation and the arrow.** Kawai, Parrondo & Van den Broeck, Phys. Rev. Lett. 98, 080602 (2007), doi:10.1103/PhysRevLett.98.080602. Parrondo, Van den Broeck & Kawai, New J. Phys. 11, 073008 (2009), doi:10.1088/1367-2630/11/7/073008. Crooks, Phys. Rev. E 60, 2721 (1999), doi:10.1103/PhysRevE.60.2721. Esposito & Van den Broeck, EPL 95, 40004 (2011), doi:10.1209/0295-5075/95/40004. Lindblad, Commun. Math. Phys. 40, 147 (1975), doi:10.1007/BF01609396. Uhlmann, Commun. Math. Phys. 54, 21 (1977), doi:10.1007/BF01609834. van Erven & Harremoës, IEEE Trans. Inf. Theory 60, 3797 (2014), doi:10.1109/TIT.2014.2320500. Jaynes, Am. J. Phys. 33, 391 (1965), doi:10.1119/1.1971557. Lebowitz, Physics Today 46(9), 32 (1993), doi:10.1063/1.881363. Penrose, "Singularities and time-asymmetry", in Hawking & Israel (eds.), *General Relativity: An Einstein Centenary Survey*, CUP (1979) (through secondary sources). Albert, *Time and Chance*, Harvard University Press (2000) (through secondary sources).

**Virial theorem and halves.** Clausius (1870). Collins, *The Virial Theorem in Stellar Astrophysics*, Pachart (1978), §1.2. Chandrasekhar (1939), stellar structure. Smarr, Phys. Rev. Lett. 30, 71 (1973). Tangherlini, Nuovo Cim. 27, 636 (1963), doi:10.1007/BF02784569. Myers & Perry, Ann. Phys. 172, 304 (1986), doi:10.1016/0003-4916(86)90186-7. Kastor, Ray & Traschen, Class. Quantum Grav. 26, 195011 (2009), doi:10.1088/0264-9381/26/19/195011. Ehrenfest, Proc. KNAW 20, 200 (1917). Athas et al. (1994), adiabatic switching, doi:10.1109/92.335009.

**Horizon thermodynamics.** Bekenstein, Phys. Rev. D 7, 2333 (1973). Bekenstein, Phys. Rev. D 23, 287 (1981). Bekenstein, Phys. Rev. Lett. 46, 623 (1981), doi:10.1103/PhysRevLett.46.623, rate form as restated by Deffner, Phys. Rev. Research 2, 013161 (2020). Hawking, Commun. Math. Phys. 43, 199 (1975), doi:10.1007/BF02345020. Gibbons & Hawking, Phys. Rev. D 15, 2738 (1977). Jacobson, Phys. Rev. Lett. 75, 1260 (1995), doi:10.1103/PhysRevLett.75.1260. Eling, Guedens & Jacobson, Phys. Rev. Lett. 96, 121301 (2006), doi:10.1103/PhysRevLett.96.121301. Pai, Namboothiri & Mathew, arXiv:2604.04731 (2026, preprint). Chirco & Liberati, Phys. Rev. D 81, 024016 (2010), doi:10.1103/PhysRevD.81.024016. Cai & Kim, JHEP 2005(02), 050, doi:10.1088/1126-6708/2005/02/050. Hayward, Class. Quantum Grav. 15, 3147 (1998), doi:10.1088/0264-9381/15/10/017. Padmanabhan, Mod. Phys. Lett. A 25, 1129 (2010), doi:10.1142/S021773231003313X. Padmanabhan, arXiv:1206.4916 (2012). Cai, JHEP 11 (2012) 016, arXiv:1207.0622. Wang, Gong & Abdalla, Phys. Rev. D 74, 083520 (2006). Deser & Levin, Class. Quantum Grav. 14, L163 (1997), arXiv:gr-qc/9706018. Bousso, Rev. Mod. Phys. 74, 825 (2002), doi:10.1103/RevModPhys.74.825. Margolus & Levitin, Physica D 120, 188 (1998), doi:10.1016/S0167-2789(98)00054-2. Lloyd, Phys. Rev. Lett. 88, 237901 (2002), doi:10.1103/PhysRevLett.88.237901. Egan & Lineweaver, ApJ 710, 1825 (2010), doi:10.1088/0004-637X/710/2/1825.

**Cosmology and structure.** Clifton, Ellis & Tavakol, Class. Quantum Grav. 30, 125009 (2013), arXiv:1303.5612. Dekel & Birnboim, MNRAS 368, 2 (2006). Moster, Naab & White, MNRAS 428, 3121 (2013). Behroozi, Wechsler & Conroy, ApJ 770, 57 (2013); Behroozi et al., MNRAS 488, 3143 (2019). Fukugita & Peebles, ApJ 616, 643 (2004). Madau & Dickinson, ARA&A 52, 415 (2014), doi:10.1146/annurev-astro-081811-125615. Tinker et al., ApJ 688, 709 (2008). Sheth & Tormen, MNRAS 308, 119 (1999). DESI Collaboration, DESI DR2 Results II, arXiv:2503.14738 (2025). Abbott et al., ApJL 848, L13 (2017), doi:10.3847/2041-8213/aa920c. Pound & Rebka, Phys. Rev. Lett. 4, 337 (1960), doi:10.1103/PhysRevLett.4.337. Ashby, Living Rev. Relativ. 6, 1 (2003), doi:10.12942/lrr-2003-1. Wiltshire, New J. Phys. 9, 377 (2007).

**Koide.** Koide, Phys. Rev. D 28, 252 (1983), doi:10.1103/PhysRevD.28.252. Koide, Mod. Phys. Lett. A 5, 2319 (1990), doi:10.1142/S0217732390002663 (through hep-ph/0506247). Koide, arXiv:hep-ph/0506247 (2005). Foot, arXiv:hep-ph/9402242 (1994). Brannen, Found. Phys. 40, 1681 (2010), doi:10.1007/s10701-010-9465-8. Hübner, arXiv:2605.09651 (2026, preprint). Zenodo record 19271888 (2026, not peer reviewed). Gomez, arXiv:0803.3834 (2008). Radcliffe, J. Phys. A 4, 313 (1971), doi:10.1088/0305-4470/4/3/009. Arecchi, Courtens, Gilmore & Thomas, Phys. Rev. A 6, 2211 (1972), doi:10.1103/PhysRevA.6.2211. ALEPH/DELPHI/L3/OPAL/SLD, Phys. Rep. 427, 257 (2006), doi:10.1016/j.physrep.2005.12.006.

**Cell.** Ming et al., Cell Res. 30, 980 (2020), doi:10.1038/s41422-020-0359-9. Busto-Moner et al., PLoS Comput. Biol. (2020). Genereux et al. (2005), as cited in book p6_02. Loyfer et al. (2023), as cited in book p6_02. Pradhan et al., J. Biol. Chem. (1999). Hopfield, PNAS (1974), as cited in book p6_02.

---
