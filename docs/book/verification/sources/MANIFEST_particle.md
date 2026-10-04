# MANIFEST: particle physics of Part 2 rebuilt as three chapters (2026-10-02)

Deliverable of the particle-physics rebuild. Only new files are delivered (zip, repo-relative paths). Existing files change only through the
insertion blocks below. Every anchor below was checked to occur exactly once in the file at HEAD 30dc9e7 (clone of 2026-10-02).

## New files

| file | content |
|---|---|
| `docs/book/part2/p2_15a_lepton_koide.tex` | Chapter "Three charged leptons and the Koide relation", `\label{ch:koide}`; 3 figures, 2 tables + status table |
| `docs/book/part2/p2_15b_electron_mass.tex` | Chapter "The electron rest mass as a fixed point", `\label{ch:electronmass}`; 1 figure, 2 tables + status table |
| `docs/book/part2/p2_22b_higgs_record.tex` | Chapter "The Higgs field and the onset of proper time", `\label{ch:higgsrecord}`; 1 figure, 1 table + status table |
| `docs/book/bib_particle.bib` | 10 new entries (DOIs checked on CrossRef; 2 arXiv ids on the arXiv API). Add to the `\bibliography{}` line. |
| `docs/book/figscripts/fig_p2_particle.py` | draws the five figures; imports `_bookstyle.py` |
| `docs/book/figures/part2/fig_koide_flavour.{pdf,png}` | flavour-space vector, (1,1,1) and the 45° circle (3D), and the plane Σ√m = 3x |
| `docs/book/figures/part2/fig_koide_orbit.{pdf,png}` | the charge orbit with the three phases; √m(φ)/x |
| `docs/book/figures/part2/fig_koide_sweep.{pdf,png}` | the δ sweep (static version of the animation) |
| `docs/book/figures/part2/fig_electron_fp.{pdf,png}` | the two sides of the fixed point; H0 dependence; α exponent |
| `docs/book/figures/part2/fig_higgs_proper.{pdf,png}` | masses and Compton times at the crossover; −ln E = z |
| `docs/verification/scripts/verify_particle_book.py` + `_output.txt` | every number of the three chapters (K1-K18, E1-E12, H1-H10) |

## main.tex and bibliography

- Retire `part2/p2_15_particle_masses.tex` and its figures `fig_koide`, `fig_electron_h0` (no other file references `fig:koide`, `fig:electron_h0`).
  The figures-pass table `tab:particle_numbers` (figures_insertions_p2.json, p2_15 entry) is not needed: its numbers are in `tab:ko:fit`,
  `tab:em:factors` and the H0 table of p2_15b (checked: 1.2018, 0.832107/0.832112, −231 ppm, +6.6 ppm, +3.27 %, ±0.32 %, 0.66666051,
  313.84 MeV, 0.22227, 26.9 MeV all reproduced; PDG 2024 values added).
- Add `bib_particle` to the `\bibliography{...}` list in main.tex. Drop `EuclidMG2025` / `EuclidReview2025` from it if the lead already added them.
- Option (not done): the electroweak chapters could move ahead of p2_15a/p2_15b so that mass generation precedes the mass relations; the
  new chapters reference each other by label and work in either order.

## Sources read in full (line count first; every line read in chunks of at most 60 lines)

| source | lines | role |
|---|---|---|
| `docs/papers/Koide_Mahaffey.pdf` (PDF text) | 839 | the Koide chain carried in p2_15a (lead: this is the authoritative paper) |
| `book_sources_wave3.zip: particle/Koide_Paper.tex` | 359 | earlier conditional version; equation forms; identical to `Koide_Paper_revised.tex` (diff empty) |
| `particle/Koide_Paper_revised.tex` | 359 | identical to the above (checked by diff; not re-read line by line) |
| `particle/Koide_ratio_from_holographic_boundary.pdf` (PDF text) | 366 | earlier version; the virial route 2x² = y² (carried as \analogy) |
| `particle/generate_koide_figures.py` (RTF-wrapped) | 101 | figure logic; see "Figures" below |
| `particle/3D_flavor_space.png`, `2D_holographic_screen.png` | images | inspected; replaced |
| `docs/papers/Electron_Rest_Mass_from__IAM.pdf` (PDF text) | 475 | carried in p2_15b |
| `particle/electron_mass_referee_safe.tex` | 486 | source of the same paper |
| `docs/papers/Electroweak_Symmetry_Breaking_and_the_Matter_Sector.pdf` (PDF text) | 200 | already in p2_22; β_m composition moved there from p2_15 |
| `particle/iam_higgs_duration.tex` | 746 | carried in p2_22b (first reading) |
| `docs/verification/particle/KOIDE_CHECK.md` | 13 | |
| `docs/verification/particle/ELECTRON_MASS_CHECK.md` | 13 | |
| `docs/verification/particle/ELECTROWEAK_CHECK.md` | 22 | |
| `docs/verification/PAPER_ERRATA.md` rows EM1-EM4, KO1-KO4, EW1-EW5, TR1/TR7/TR8 (lines 260-289 of 341) | 30 | rows read; file not read in full |
| `docs/book/part2/p2_15_particle_masses.tex` | 89 | retired by this deliverable |
| `docs/book/part2/p2_22_electroweak.tex` | 109 | kept; blocks below |
| `docs/book/figscripts/_bookstyle.py` | 98 | house style |
| `docs/verification/scripts/verify_koide.py` / `_output.txt` | 13 / 10 | reproduced |
| `docs/verification/scripts/verify_electron_mass.py` / `_output.txt` | 11 / 10 | reproduced |
| `docs/verification/scripts/verify_entanglement_electroweak.py` / `_output.txt` | 63 / 33 | W1-W12 used |
| figures-pass zip `v4e8319c1_p2_figures_tables.zip`: `figures_insertions_p2.json` p2_15 and p2_22 entries | — | numbers checked; not reused |

PDF line counts are of the text extracted here with pypdfium2 (839/475/200); the check files quote 846/485/207 from a different extractor.

## Numbers that changed against the old p2_15

- Masses: PDG 2024 (m_τ = 1776.93 ± 0.09 MeV) replaces PDG 2022 as the main input. Q = 0.66666446 ± 0.00000508 (0.43σ); the 2022 value
  0.66666051 is quoted alongside. x² = 313.851 MeV, δ = 0.222225 rad (2022: 313.84, 0.22227).
- Electron: the fixed point as derived is 0.5762 m_e; the identified (2π)^(3/10) is labelled \fitted. Sector H0 values: 67.16 → −0.14 %,
  72.26 → +2.82 %.
- Koide count: "at most three" over all offsets, and "exactly three" at δ = 0 and at the measured offset (lead, 2026-10-02).
- E(a) = e^(−z) is stated explicitly (1 − 1/a = −z).

## Figures

The archive's 3D figure plotted a "toy symmetric" point at (171.6, 20.8, 20.8) √MeV: the largest root on the electron axis (φ = 0 given to the
electron) with a scale built from √(m_μ m_τ); the 2D figure also put the electron at φ = 0. The new `fig_koide_flavour` plots (√m_e, √m_μ, √m_τ) = (0.7148, 10.2790, 42.1536) √MeV on
axes labelled in that order (checked in the script output), the (1,1,1) direction, and the 45° circle at the measured x. The τ sits at k = 0
(the maximum), the electron at k = 1, the muon at k = 2. The animation (linear rescaling of the masses, not the δ of the encoding) is replaced
by `fig_koide_sweep`, which varies δ in the encoding itself. Render-then-verify overlap counts: flavour 11 (3D tick labels only), orbit 2,
sweep 1, electron 2, Higgs 3 (axis-adjacent labels); all inspected visually.

## Insertion and replacement blocks

Format: target file | anchor line copied exactly | action and LaTeX. Blocks for one file are listed in increasing line order; apply
from the bottom up, or match by anchor text.

### 1. `docs/book/main.tex` (line 31) — REPLACE

Anchor:
```latex
\input{part2/p2_15_particle_masses}
```
Replace with:
```latex
\input{part2/p2_15a_lepton_koide}
\input{part2/p2_15b_electron_mass}
```
Note: p2_15 is retired; the two new chapters take its slot (the new line holds two \input lines).

### 2. `docs/book/main.tex` (line 38) — INSERT AFTER

Anchor:
```latex
\input{part2/p2_22_electroweak}
```
Insert after it:
```latex
\input{part2/p2_22b_higgs_record}
```
Note: new chapter directly after the electroweak chapter.

### 3. `docs/book/part2/p2_22_electroweak.tex` (line 5) — REPLACE

Anchor:
```latex
% (Chapter ch:particle); eta and the Sakharov conditions (ch:baryon); before/after the transition and the arrow of time (ch:time, Part 5).
```
Replace with:
```latex
% (Chapter ch:higgsrecord, Section sec:ew:betam); eta and the Sakharov conditions (ch:baryon); before/after the transition and the arrow of time (ch:time, Part 5).
```
Note: header comment only.

### 4. `docs/book/part2/p2_22_electroweak.tex` (line 9) — REPLACE

Anchor:
```latex
Chapter~\ref{ch:particle} stated the result: a matter sector able to decohere gravitationally exists from electroweak symmetry breaking, and the
```
Replace with:
```latex
The result of this chapter is that a matter sector able to decohere gravitationally exists from electroweak symmetry breaking, and the
```

### 5. `docs/book/part2/p2_22_electroweak.tex` (line 44) — REPLACE

Anchor:
```latex
with it the baryonic part of $\beta_m$ (Chapters~\ref{ch:particle} and~\ref{ch:baryon}). The phase of the quark mixing matrix is too small to produce
```
Replace with:
```latex
with it the baryonic part of $\beta_m$ (Section~\ref{sec:ew:betam} and Chapter~\ref{ch:baryon}). The phase of the quark mixing matrix is too small to produce
```

### 6. `docs/book/part2/p2_22_electroweak.tex` (line 64) — REPLACE

Anchor:
```latex
Chapter~\ref{ch:particle}, the matter sector exists from here on. Dark matter, whose particle nature is unknown, joins it by the same criterion if it
```
Replace with:
```latex
Section~\ref{sec:hr:proper} (a degree of freedom decoheres gravitationally only if it accumulates proper time), the matter sector exists from here on. Dark matter, whose particle nature is unknown, joins it by the same criterion if it
```

### 7. `docs/book/part2/p2_22_electroweak.tex` (line 69) — REPLACE

Anchor:
```latex
(Chapter~\ref{ch:particle} gives the bound). Its exemption from writing follows from an established symmetry, not from an added assumption. \derived{}
```
Replace with:
```latex
($m_\gamma<10^{-18}$\,eV~\cite{PDG2024}). Its exemption from writing follows from an established symmetry, not from an added assumption. \derived{}
```

### 8. `docs/book/part2/p2_22_electroweak.tex` (line 88) — REPLACE

Anchor:
```latex
Chapter~\ref{ch:particle} and Chapter~\ref{ch:time} leave open whether the settling of the Higgs field into its vacuum is itself a record written at
```
Replace with:
```latex
Chapter~\ref{ch:time} and Chapter~\ref{ch:higgsrecord} take up whether the settling of the Higgs field into its vacuum is itself a record written at
```

### 9. `docs/book/part2/p2_22_electroweak.tex` (line 85) — INSERT AFTER

Anchor:
```latex
\calc{} The matter sector exists from the first row. It writes in quantity only from the fifth, once structure forms.
```
Insert after it:
```latex

\section{The composition of $\beta_m$}\label{sec:ew:betam}
The coupling $\beta_m=\Omega_m/2$ (Chapter~\ref{ch:virial}) divides between the two kinds of matter. With the Planck 2018 values $\Omega_m=0.3153$
and $\Omega_b=0.0493$~\cite{Planck2018VI},
\begin{equation}
\beta_m=\frac{\Omega_b}{2}+\frac{\Omega_{\rm dm}}{2}=0.0247+0.1330=0.1577.\label{eq:ew:betam}
\end{equation}
\calc{} The baryonic part, $15.6\,\%$, is set by the baryon asymmetry (Chapter~\ref{ch:baryon}); the dark part, $84.4\,\%$, by the dark-matter
density, which the microwave background fixes at $z\approx1090$ (Chapter~\ref{ch:higgsrecord}). Both are measured inputs; $\beta_m$ adds no
parameter to them. That $\beta_m/\Omega_m=1/2$ for any revision of $\Omega_m$ is the definition, not a test. \derived
```
Note: insert after this line, and after any figure block already inserted at the same anchor (fig_ew_timeline from the figures pass); before \section{The vacuum question}.

### 10. `docs/book/part2/p2_22_electroweak.tex` (line 99) — REPLACE

Anchor:
```latex
\item $\mu_0=-0.136$, that is $\mu(z{=}0)=0.864$: Euclid clustering. The complete first data release in mid-2027 reaches $\sigma(\mu_0)\approx0.08$, a
```
Replace with:
```latex
\item $\mu_0=-0.136$, that is $\mu(z{=}0)=0.864$: Euclid clustering and lensing. Euclid's published forecasts use $\mu(z)=1+\mu_0\,\Omega_{\rm DE}(z)/\Omega_{\rm DE}(0)$: the 68\,\% error on $1+\mu_0$ is $23.3\,\%$ with conservative scale cuts (all primary probes), $4\,\%$ for weak lensing with photometric clustering taken to $k\sim4\,{\rm Mpc}^{-1}$, and of order $1\,\%$ for all probes with optimistic cuts~\cite{EuclidMG2025,EuclidReview2025}. \observed{} IAM's $\mu(z)$ falls with redshift faster than that template ($\mu(z{=}1)=0.982$ against $0.958$ for the same $\mu_0$), so the significance is not $|\mu_0|/\sigma$; matching the linear-growth deficit gives a template-equivalent $\mu_0$ of $-0.03$ to $-0.07$, about $0.3\sigma$ with conservative cuts, $1.5$--$1.8\sigma$ at $4\,\%$ and $6$--$7\sigma$ at $1\,\%$. \calc{} A forecast with IAM's own $\mu(z)$ is open (Chapter~\ref{ch:predictions}). \openprob
```
Note: lead directive 2026-10-02 (Euclid). Line 100 is deleted (next block).

### 11. `docs/book/part2/p2_22_electroweak.tex` (line 100) — REPLACE

Anchor:
```latex
$1.7\sigma$ test; the final survey reaches $\approx0.04$, $3.4\sigma$ (Chapter~\ref{ch:predictions}). \prediction
```
Replace with:
```latex
(delete this line; its content is replaced by the new line 99)
```

### 12. `docs/book/part5/p5_06_nonlocality.tex` (line 61) — REPLACE

Anchor:
```latex
appears in the electron mass of Chapter~\ref{ch:particle} and in the decoherence argument, and not in the cosmology, which takes $7/2$. Whether the
```
Replace with:
```latex
appears in the electron mass of Chapter~\ref{ch:electronmass} and in the decoherence argument, and not in the cosmology, which takes $7/2$. Whether the
```

### 13. `docs/book/part5/p5_09_open.tex` (line 20) — REPLACE

Anchor:
```latex
\textbf{D3. The electron-mass prefactor.} $(2\pi)^{3/10}$ was found by search (Chapter~\ref{ch:particle}).
```
Replace with:
```latex
\textbf{D3. The electron-mass prefactor.} $(2\pi)^{3/10}$ was found by search (Chapter~\ref{ch:electronmass}).
```

### 14. `docs/book/part5/p5_09_open.tex` (line 26) — REPLACE

Anchor:
```latex
 measured, not derived (Chapter~\ref{ch:particle}). & A principle that excludes $n=2$ and fixes $\delta$. & we \\
```
Replace with:
```latex
 measured, not derived (Chapter~\ref{ch:koide}). & A principle that excludes $n=2$ and fixes $\delta$. & we \\
```

### 15. `docs/book/appendices/app_F_glossary.tex` (line 24) — REPLACE

Anchor:
```latex
\textbf{AdS/CFT} & The correspondence between gravity in anti-de Sitter space and a conformal field theory on its boundary. Cited as the setting in which a fixed-charge sector reduces to an orbit; in de Sitter space the reduction is open. {\footnotesize(Ch.~\ref{ch:particle})}\\[1pt]
```
Replace with:
```latex
\textbf{AdS/CFT} & The correspondence between gravity in anti-de Sitter space and a conformal field theory on its boundary. Cited as the setting in which a fixed-charge sector reduces to an orbit; in de Sitter space the reduction is open. {\footnotesize(Ch.~\ref{ch:koide})}\\[1pt]
```

### 16. `docs/book/appendices/app_F_glossary.tex` (line 97) — REPLACE

Anchor:
```latex
\textbf{Charge orbit} & An $S^1$ on which the square roots of the charged-lepton masses are encoded as $x+y\cos(\phi+\delta)$; equipartition between the two modes gives the Koide ratio 2/3. {\footnotesize(Ch.~\ref{ch:particle})}\\[1pt]
```
Replace with:
```latex
\textbf{Charge orbit} & An $S^1$ on which the square roots of the charged-lepton masses are encoded as $x+y\cos(\phi+\delta)$; equipartition between the two modes gives the Koide ratio 2/3. {\footnotesize(Ch.~\ref{ch:koide})}\\[1pt]
```

### 17. `docs/book/appendices/app_F_glossary.tex` (line 137) — REPLACE

Anchor:
```latex
\textbf{Dark matter} & Non-luminous matter that clusters; it decoheres and writes records like baryons and carries 84.4\,\% of $\beta_m$. The framework's identification of dark matter with the geometric half is a conjecture. {\footnotesize(Chs.~\ref{ch:particle}, \ref{ch:time})}\\[1pt]
```
Replace with:
```latex
\textbf{Dark matter} & Non-luminous matter that clusters; it decoheres and writes records like baryons and carries 84.4\,\% of $\beta_m$. The framework's identification of dark matter with the geometric half is a conjecture; any version of it must reproduce a dark-matter density already fixed at $z\approx1090$. {\footnotesize(Chs.~\ref{ch:electroweak}, \ref{ch:higgsrecord}, \ref{ch:time})}\\[1pt]
```

### 18. `docs/book/appendices/app_F_glossary.tex` (line 181) — REPLACE

Anchor:
```latex
\textbf{Electron fixed point} & The condition that the electron's rest energy equals the cost of maintaining its own record at the cosmic-horizon temperature; it reproduces $m_e$ within the 0.3\,\% set by $H_0$, with one prefactor found numerically. {\footnotesize(Ch.~\ref{ch:particle})}\\[1pt]
```
Replace with:
```latex
\textbf{Electron fixed point} & The condition that the electron's rest energy equals the cost of maintaining its own record at the cosmic-horizon temperature; it reproduces $m_e$ within the 0.3\,\% set by $H_0$, with one prefactor, $(2\pi)^{3/10}$, found numerically; without it the fixed point is $0.576\,m_e$. {\footnotesize(Ch.~\ref{ch:electronmass})}\\[1pt]
```

### 19. `docs/book/appendices/app_F_glossary.tex` (line 182) — REPLACE

Anchor:
```latex
\textbf{Electroweak symmetry breaking} & The epoch at which the Higgs field takes its vacuum value $v=246.22$~GeV and the $W$, $Z$ and charged fermions acquire mass; in the Standard Model a smooth crossover centred at $T_c=159.5$~GeV, $t\approx9\times10^{-12}$~s. The matter sector exists from this epoch; the photon stays massless under the unbroken $U(1)$. {\footnotesize(Chs.~\ref{ch:particle}, \ref{ch:time}, \ref{ch:electroweak})}\\[1pt]
```
Replace with:
```latex
\textbf{Electroweak symmetry breaking} & The epoch at which the Higgs field takes its vacuum value $v=246.22$~GeV and the $W$, $Z$ and charged fermions acquire mass; in the Standard Model a smooth crossover centred at $T_c=159.5$~GeV, $t\approx9\times10^{-12}$~s. The matter sector exists from this epoch; the photon stays massless under the unbroken $U(1)$. {\footnotesize(Chs.~\ref{ch:time}, \ref{ch:electroweak})}\\[1pt]
```

### 20. `docs/book/appendices/app_F_glossary.tex` (line 311) — REPLACE

Anchor:
```latex
\textbf{Koide relation} & $Q=(m_e+m_\mu+m_\tau)/(\sqrt{m_e}+\sqrt{m_\mu}+\sqrt{m_\tau})^2=2/3$ to $10^{-5}$ for the charged-lepton pole masses; reproduced by equipartition on a charge orbit, which also allows at most three generations. {\footnotesize(Ch.~\ref{ch:particle})}\\[1pt]
```
Replace with:
```latex
\textbf{Koide relation} & $Q=(m_e+m_\mu+m_\tau)/(\sqrt{m_e}+\sqrt{m_\mu}+\sqrt{m_\tau})^2=2/3$ within $2.2\times10^{-6}$ ($0.43\sigma$, PDG 2024) for the charged-lepton pole masses; reproduced by equipartition on a charge orbit, which allows at most three generations and exactly three at the measured offset; the offset $\delta=0.2222$ is not derived. {\footnotesize(Ch.~\ref{ch:koide})}\\[1pt]
```

### 21. `docs/book/appendices/app_F_glossary.tex` (line 347) — REPLACE

Anchor:
```latex
\textbf{Matter sector} & Degrees of freedom on timelike worldlines, which decohere and write records; it exists from electroweak symmetry breaking. {\footnotesize(Chs.~\ref{ch:dual}, \ref{ch:particle})}\\[1pt]
```
Replace with:
```latex
\textbf{Matter sector} & Degrees of freedom on timelike worldlines, which decohere and write records; it exists from electroweak symmetry breaking. {\footnotesize(Chs.~\ref{ch:dual}, \ref{ch:electroweak})}\\[1pt]
```

### 22. `docs/book/appendices/app_F_glossary.tex` (line 407) — REPLACE

Anchor:
```latex
\textbf{Photon sector} & Degrees of freedom on null worldlines, which write no records in flight; they see the $\Lambda$CDM background. {\footnotesize(Chs.~\ref{ch:dual}, \ref{ch:particle})}\\[1pt]
```
Replace with:
```latex
\textbf{Photon sector} & Degrees of freedom on null worldlines, which write no records in flight; they see the $\Lambda$CDM background. {\footnotesize(Chs.~\ref{ch:dual}, \ref{ch:electroweak})}\\[1pt]
```

### 23. `docs/book/appendices/app_F_glossary.tex` (line 412) — REPLACE

Anchor:
```latex
\textbf{Planck length, Planck mass} & $\lP=\sqrt{\hbar G/c^3}=1.616\times10^{-35}$~m and $\mP=\sqrt{\hbar c/G}=2.176\times10^{-8}$~kg. {\footnotesize(Chs.~\ref{ch:surfaces}, \ref{ch:particle})}\\[1pt]
```
Replace with:
```latex
\textbf{Planck length, Planck mass} & $\lP=\sqrt{\hbar G/c^3}=1.616\times10^{-35}$~m and $\mP=\sqrt{\hbar c/G}=2.176\times10^{-8}$~kg. {\footnotesize(Chs.~\ref{ch:surfaces}, \ref{ch:electronmass})}\\[1pt]
```

### 24. `docs/book/appendices/app_F_glossary.tex` (line 417) — REPLACE

Anchor:
```latex
\textbf{Pole mass} & The mass of a particle defined by the pole of its propagator; Koide's relation holds for pole masses. {\footnotesize(Ch.~\ref{ch:particle})}\\[1pt]
```
Replace with:
```latex
\textbf{Pole mass} & The mass of a particle defined by the pole of its propagator; Koide's relation holds for pole masses. {\footnotesize(Ch.~\ref{ch:koide})}\\[1pt]
```

### 25. `docs/book/appendices/app_E_formulas.tex` (line 138) — REPLACE

Anchor:
```latex
\subsection*{Chapter~\ref{ch:particle}: The particle scale}
```
Replace with:
```latex
\subsection*{Chapter~\ref{ch:electronmass}: The electron rest mass as a fixed point}
```

### 26. `docs/book/appendices/app_E_formulas.tex` (line 139) — REPLACE

Anchor:
```latex
\iamfsentry{93}{mc^2=E_{\rm bit}\,\frac{N(m)}{f(\alpha)},\qquad E_{\rm bit}=k_BT_{GH}\ln2=\frac{\hbar H_0\ln2}{2\pi}}{The electron's rest energy as the cost of maintaining its own record at the cosmic-horizon temperature.}{\conjecture{} (three stated assumptions)}{Chapter~\ref{ch:particle}, section ``The electron as a fixed point''}
```
Replace with:
```latex
\iamfsentry{93}{mc^2=E_{\rm bit}\,\frac{N(m)}{f(\alpha)},\qquad E_{\rm bit}=k_BT_{GH}\ln2=\frac{\hbar H_0\ln2}{2\pi}}{The electron's rest energy as the cost of maintaining its own record at the cosmic-horizon temperature.}{\conjecture{} (three stated assumptions)}{Chapter~\ref{ch:electronmass}, section ``The fixed point and its solution''}
```

### 27. `docs/book/appendices/app_E_formulas.tex` (line 140) — REPLACE

Anchor:
```latex
\iamfsentry{94}{m_e=(2\pi)^{-1/10}\left[\frac{\hbar H_0\ln2\;m_P^{3/2}}{\alpha^{5/2}c^2}\right]^{2/5}}{The electron-mass fixed point; the factor $(2\pi)^{3/10}$ inside the prefactor $(2\pi)^{-1/10}$ was found by numerical search.}{\fitted{} (prefactor $(2\pi)^{3/10}$ selected numerically by search); precision limited by $H_0$ to $\pm0.3\,\%$}{Chapter~\ref{ch:particle}, section ``The electron as a fixed point''}
```
Replace with:
```latex
\iamfsentry{94}{m_e=(2\pi)^{-1/10}\left[\frac{\hbar H_0\ln2\;m_P^{3/2}}{\alpha^{5/2}c^2}\right]^{2/5}}{The electron-mass fixed point; the factor $(2\pi)^{3/10}$ inside the prefactor $(2\pi)^{-1/10}$ was found by numerical search.}{\fitted{} (prefactor $(2\pi)^{3/10}$ selected numerically by search); precision limited by $H_0$ to $\pm0.3\,\%$}{Chapter~\ref{ch:electronmass}, section ``The fixed point and its solution''}
```

### 28. `docs/book/appendices/app_E_formulas.tex` (line 140) — INSERT AFTER

Anchor:
```latex
\iamfsentry{94}{m_e=(2\pi)^{-1/10}\left[\frac{\hbar H_0\ln2\;m_P^{3/2}}{\alpha^{5/2}c^2}\right]^{2/5}}{The electron-mass fixed point; the factor $(2\pi)^{3/10}$ inside the prefactor $(2\pi)^{-1/10}$ was found by numerical search.}{\fitted{} (prefactor $(2\pi)^{3/10}$ selected numerically by search); precision limited by $H_0$ to $\pm0.3\,\%$}{Chapter~\ref{ch:particle}, section ``The electron as a fixed point''}
```
Insert after it:
```latex
\subsection*{Chapter~\ref{ch:koide}: Three charged leptons and the Koide relation}
```
Note: heading for entries 95 and 96.

### 29. `docs/book/appendices/app_E_formulas.tex` (line 141) — REPLACE

Anchor:
```latex
\iamfsentry{95}{Q=\frac{m_e+m_\mu+m_\tau}{(\sqrt{m_e}+\sqrt{m_\mu}+\sqrt{m_\tau})^2}=0.66666051}{Koide's relation for the charged-lepton pole masses.}{\observed}{Chapter~\ref{ch:particle}, section ``The charged-lepton pattern''}
```
Replace with:
```latex
\iamfsentry{95}{Q=\frac{m_e+m_\mu+m_\tau}{(\sqrt{m_e}+\sqrt{m_\mu}+\sqrt{m_\tau})^2}=0.66666446}{Koide's relation for the charged-lepton pole masses (PDG 2024; $0.66666051$ with the 2022 $m_\tau$).}{\observed}{Chapter~\ref{ch:koide}, section ``Koide's relation''}
```

### 30. `docs/book/appendices/app_E_formulas.tex` (line 142) — REPLACE

Anchor:
```latex
\iamfsentry{96}{\sum_k\sqrt{m_k}=3x,\qquad\sum_km_k=3x^2+\tfrac32y^2,\qquad Q=\tfrac13\Big(1+\frac{y^2}{2x^2}\Big)=\tfrac23}{Equal partition between the constant mode and the first harmonic on a charge orbit gives $Q=2/3$ for three phases.}{\derived{} (given equal partition between the two modes and the $S^1$ reduction, both \conjecture); $\delta=0.22227$ is \measured}{Chapter~\ref{ch:particle}, section ``The charged-lepton pattern''}
```
Replace with:
```latex
\iamfsentry{96}{\sum_k\sqrt{m_k}=3x,\qquad\sum_km_k=3x^2+\tfrac32y^2,\qquad Q=\tfrac13\Big(1+\frac{y^2}{2x^2}\Big)=\tfrac23}{Equal partition between the constant mode and the first harmonic on a charge orbit gives $Q=2/3$ for three phases.}{\derived{} (given equal partition between the two modes and the $S^1$ reduction, both \conjecture); $\delta=0.2222$ is \calc{} from the measured masses and not derived}{Chapter~\ref{ch:koide}, section ``The Koide value''}
```

### 31. `docs/book/appendices/app_N_notation.tex` (line 230) — REPLACE

Anchor:
```latex
$m_e$, $\alpha$, $\mP$ & electron mass, fine-structure constant, Planck mass & kg; ---; kg & Ch.~\ref{ch:particle}\\
```
Replace with:
```latex
$m_e$, $\alpha$, $\mP$ & electron mass, fine-structure constant, Planck mass & kg; ---; kg & Ch.~\ref{ch:electronmass}\\
```

### 32. `docs/book/appendices/app_N_notation.tex` (line 231) — REPLACE

Anchor:
```latex
$N(m)$, $f(\alpha)$ & encoding-cell count and electromagnetic suppression factor of the electron fixed point & --- & Ch.~\ref{ch:particle}\\
```
Replace with:
```latex
$N(m)$, $f(\alpha)$ & encoding-cell count and electromagnetic suppression factor of the electron fixed point & --- & Ch.~\ref{ch:electronmass}\\
```

### 33. `docs/book/appendices/app_N_notation.tex` (line 232) — REPLACE

Anchor:
```latex
$Q$ (Koide) & Koide ratio of the charged-lepton masses & --- & Ch.~\ref{ch:particle}\\
```
Replace with:
```latex
$Q$ (Koide) & Koide ratio of the charged-lepton masses & --- & Ch.~\ref{ch:koide}\\
```

### 34. `docs/book/appendices/app_N_notation.tex` (line 233) — REPLACE

Anchor:
```latex
$x$, $y$, $\delta$, $\phi_k$ & constant mode, first harmonic, offset and phases of the charge-orbit encoding & MeV$^{1/2}$; rad & Ch.~\ref{ch:particle}\\
```
Replace with:
```latex
$x$, $y$, $\delta$, $\phi_k$ & constant mode, first harmonic, offset and phases of the charge-orbit encoding & MeV$^{1/2}$; rad & Ch.~\ref{ch:koide}\\
```

### 35. `docs/book/appendices/app_N_notation.tex` (line 234) — REPLACE

Anchor:
```latex
$v$, $y_f$ & Higgs vacuum expectation value $(\sqrt2G_F)^{-1/2}=246.22$~GeV; Yukawa coupling, $m_f=y_fv/\sqrt2$ & GeV; --- & Chs.~\ref{ch:particle}, \ref{ch:electroweak}\\
```
Replace with:
```latex
$v$, $y_f$ & Higgs vacuum expectation value $(\sqrt2G_F)^{-1/2}=246.22$~GeV; Yukawa coupling, $m_f=y_fv/\sqrt2$ & GeV; --- & Chs.~\ref{ch:electroweak}, \ref{ch:higgsrecord}\\
```

### 36. `docs/verification/PAPER_ERRATA.md` (line 275) — INSERT AFTER

Anchor:
```
| EM4 | | §9 | n = 3 generations from companion | at most three (KO2) | confirmed | `particle/ELECTRON_MASS_CHECK.md` |
```
Insert after it:
```
| EM5 | | §3.5 | T_C/T_GH separated by 47 orders of magnitude | m_e c²/ħH0 = 3.55 × 10³⁸, 38.6 orders (E7) | found 2026-10-02 | `book/part2/p2_15b_electron_mass.tex`; `scripts/verify_particle_book.py` |
| EM6 | | §2, §3.1 | Compton sphere saturates the Bekenstein bound | Bekenstein bound for m_e c² in λ̄_C is 2π; the area count π(m_P/m)² = 1.79 × 10⁴⁵ exceeds it by 2.9 × 10⁴⁴; it is the area law applied to the Compton sphere (E11) | found 2026-10-02 | `book/part2/p2_15b_electron_mass.tex`; `scripts/verify_particle_book.py` |
| EM7 | | §4, Eqs 13-14 | fixed point gives m_e | as derived (Eq. 13) it gives 0.5762 m_e; the identified (2π)^(3/10) = 1.7356 (equivalently a coefficient (2π)^(3/4) = 3.969 in N) is a fitted factor (E2, E3, E9) | found 2026-10-02 | `book/part2/p2_15b_electron_mass.tex`; `scripts/verify_particle_book.py` |
| EM8 | | §3.3, §3.4 | dimensional consistency uniquely produces (m_P/m)^(3/2); (r_e/λ̄_C)^(3/2) a phase-space volume | every power of the dimensionless m_P/m is homogeneous; the exponent is assumed; a volume ratio of lengths would be cubed | found 2026-10-02 | `book/part2/p2_15b_electron_mass.tex`; `scripts/verify_particle_book.py` |
| EM9 | | §5 | H0 = 67.4 | book sector values: photon sector 67.16 gives −0.14 %, matter sector 72.26 gives +2.82 % (8.8 × the H0-propagated spread); which H prices the bit is open (E4) | found 2026-10-02 | `book/part2/p2_15b_electron_mass.tex`; `scripts/verify_particle_book.py` |
```
Note: new rows after EM4.

### 37. `docs/verification/PAPER_ERRATA.md` (line 280) — INSERT AFTER

Anchor:
```
| KO4 | | Acknowledgements | named correspondent | remove | confirmed | `particle/KOIDE_CHECK.md` |
```
Insert after it:
```
| KO5 | | Theorem 1 | n = 3 unique | refines KO2: n ≥ 4 excluded for every δ; n = 2 admissible for π/4 < δ < 3π/4 (mod π), half of all offsets; n = 3 for a quarter; at δ = 0 and at the measured δ = 0.2222, exactly three (K10, K18). The condition is on the sign of the encoded amplitude, not on the mass; with signed amplitudes Q_n = 2/n (K11) | found 2026-10-02 | `book/part2/p2_15a_lepton_koide.tex`; `scripts/verify_particle_book.py` |
| KO6 | | §III D, Eqs 8-10 | δA = (8πG/c⁴)E; κ_min = c²/ℓ_P gives δA_min = 4ℓ_P² | Eq. 8 is not an area dimensionally; δA_min = 4ℓ_P² follows from the first law at any κ for δS = 1 nat; Eq. 10 restates the area law (extends KO3) | found 2026-10-02 | `book/part2/p2_15a_lepton_koide.tex`; `scripts/verify_particle_book.py` |
| KO7 | | §V C, §IX (ii) | w₂/w₁ ≲ 10⁻⁵, k_BT_enc ≲ ω₀²/8 | on Z₃ the second harmonic aliases onto the first; Q shifts by up to 0.67 a₂/y, so the data need a₂/y ≲ 10⁻⁵, w₂/w₁ ≲ 10⁻¹⁰, k_BT_enc ≲ ω₀²/15; three masses cannot test the two-mode truncation (K15-K17) | found 2026-10-02 | `book/part2/p2_15a_lepton_koide.tex`; `scripts/verify_particle_book.py` |
| KO8 | | §VIII | PDG 2022 masses | PDG 2024 (m_τ = 1776.93 ± 0.09): Q = 0.66666446 ± 0.00000508 (0.43σ); m_τ from Q = 2/3 is 1776.969 MeV (−0.43σ); Q = 2/3 and δ = 2/9 cannot both be exact (2/9 − δ = 1.75 × 10⁻⁷ rad at Q = 2/3, σ = 4 × 10⁻¹⁰) (K1, K8, K13) | found 2026-10-02 | `book/part2/p2_15a_lepton_koide.tex`; `scripts/verify_particle_book.py` |
```
Note: new rows after KO4.

### 38. `docs/verification/PAPER_ERRATA.md` (line 286) — INSERT AFTER

Anchor:
```
| EW5 | | §8 test 2 | 5.4σ with DESI Y5 | unsourced (as SP5) | confirmed | `particle/ELECTROWEAK_CHECK.md` #7 |
```
Insert after it:
```
| **The Higgs Boson and the Origin of Duration (`particle/iam_higgs_duration.tex`, Mar 2026)**, read in full 2026-10-02 (746 lines) |||||||
| HD1 | | Abstract, §2 | gravity, electromagnetism and the strong force all satisfy 2⟨K⟩+⟨V⟩ = 0 | the Cornell potential is linear at confinement, 2⟨K⟩ = +⟨V_lin⟩; only 1/r interactions have the virial half | found 2026-10-02 | `book/part2/p2_22b_higgs_record.tex`; `scripts/verify_particle_book.py` |
| HD2 | | Abstract, §1, §3 | without weak CP and P violation no arrow of time, no irreversible decoherence | the thermodynamic arrow, decoherence and Landauer's bound hold for every interaction (as p2_22) | found 2026-10-02 | `book/part2/p2_22b_higgs_record.tex`; `scripts/verify_particle_book.py` |
| HD3 | | §4.1, §4.2 | before EWSB no handedness, no CP violation; CP violation established at the transition | SU(2)_L × U(1)_Y is chiral at every temperature; the CP phase lies in the Yukawa couplings on both sides | found 2026-10-02 | `book/part2/p2_22b_higgs_record.tex`; `scripts/verify_particle_book.py` |
| HD4 | | §4.1, §6 | E(a) ≡ 0 exactly before EWSB; E steps away from zero at the Higgs moment | E = exp(1 − 1/a) = e^(−z); ln E(a_EW) = −2.04 × 10¹⁵; no step (H4) | found 2026-10-02 | `book/part2/p2_22b_higgs_record.tex`; `scripts/verify_particle_book.py` |
| HD5 | | Abstract, §4.2, Table 2 | t ≈ 10⁻¹² s at ~100 GeV; symmetry breaking by a fluctuation into one minimum | crossover at 159.5 ± 1.5 GeV, t = 9.2 × 10⁻¹² s (as EW3); no order parameter (as EW4) | found 2026-10-02 | `book/part2/p2_22b_higgs_record.tex`; `scripts/verify_particle_book.py` |
| HD6 | | §7 | the Higgs field decohered into one direction: the first decoherence event, the first bit on the horizon | vacuum points are gauge-related (Elitzur 1975); no gauge-invariant record; irreversible processes occur before the crossover; not carried as a claim (as EW4) | found 2026-10-02 | `book/part2/p2_22b_higgs_record.tex`; `scripts/verify_particle_book.py` |
| HD7 | | Abstract, §5, §6 | Ω_b (15.6 % of β_m) set at 10⁻¹² s by weak CP violation; η determined at EWSB | the SM crossover cannot produce η; when η was set is unknown (leptogenesis is one earlier route) | found 2026-10-02 | `book/part2/p2_22b_higgs_record.tex`; `scripts/verify_particle_book.py` |
| HD8 | | Abstract, §5 | Ω_dm (84.4 %) accumulated over 13.8 Gyr of decoherence | the CMB fixes Ω_c h² = 0.120 at z ≈ 1090; the comoving dark-matter density was in place at recombination | found 2026-10-02 | `book/part2/p2_22b_higgs_record.tex`; `scripts/verify_particle_book.py` |
| HD9 | | Abstract, §4 Table 2 | cosmological virialisation Ω_m/[β_m E(a)] = 2 exactly today | an identity of β_m = Ω_m/2 and E(1) = 1 (H7) | found 2026-10-02 | `book/part2/p2_22b_higgs_record.tex`; `scripts/verify_particle_book.py` |
| HD10 | | Table 2 | ~9 Gyr: E ≈ 17 %; QCD 10⁻⁶ s; recombination 0.3 eV; GUT and gravity rows | E = 0.64 at age 9 Gyr (z = 0.44); E = 0.17 at z = 1.77 (3.67 Gyr); QCD 1.4-2.6 × 10⁻⁵ s; 0.256 eV; hypothetical rows not carried (H5, H6, W2, W5) | found 2026-10-02 | `book/part2/p2_22b_higgs_record.tex`; `scripts/verify_particle_book.py` |
| HD11 | | §4.3, §8 | Σ = 1 a retrodiction, confirmed in 1983 | photon masslessness does not establish Σ = 1; weak lensing tests it (as p2_22) | found 2026-10-02 | `book/part2/p2_22b_higgs_record.tex`; `scripts/verify_particle_book.py` |
| HD12 | | §6, §9, §10 | β_m recovered to 0.2σ without fitting; Euclid DR1 October 2026 decisive; σ(Σ0) ≈ 0.02 | untraced (as TR1); DR1 mid-2027 (as TR8); Euclid sensitivity stated from EuclidMG2025/EuclidReview2025 with IAM-template caveat | found 2026-10-02 | `book/part2/p2_22b_higgs_record.tex`; `scripts/verify_particle_book.py` |
| HD13 | | §1-§3, §5 | "being/becoming", efficient cause, "Davar", "God particle", "origin of duration" | not carried; physics content stated as the onset of rest-mass proper time | found 2026-10-02 | `book/part2/p2_22b_higgs_record.tex`; `scripts/verify_particle_book.py` |
```
Note: new rows after EW5 (a new paper block).

## Unresolved / for the lead

1. Errata rows EM5-EM9, KO5-KO8 and HD1-HD13 (blocks above) are new findings of this rebuild; `docs/verification/particle/` has no check file
   for the Higgs paper yet (a `HIGGS_DURATION_CHECK.md` should be written from the HD rows). `appendices/app_B_errata_physics.tex` was not
   examined for these rows.
2. The lead asked for the verified Koide numbers to go into `verify_koide.py`; that file is existing and was not edited. All numbers are in the
   new `verify_particle_book.py` (K1-K18).
3. Which expansion rate prices the electron's bit (photon-sector 67.16 or matter-sector 72.26) is not fixed by the ingredients; the chapter
   states the numbers and labels the choice \openprob.
4. The Euclid statements follow the lead's directive (EuclidMG2025, EuclidReview2025; `verify_euclid_template.py`, which was not read here).
   The −0.03 to −0.07, 0.3σ, 1.5-1.8σ and 6-7σ figures are the lead's, not recomputed in this deliverable.
5. The book was not compiled (no TeX in the sandbox). Static checks: braces and environments balance in all three chapters and in every
   block; every \ref resolves to a label in the book or the new files; every \cite resolves to iam.bib or bib_particle.bib; no new label
   collides with an existing one; no `[h]` floats; no non-ASCII characters in the .tex files.
6. `p5_03_time.tex` line 24 (t ≈ 10⁻¹² s, flagged in ELECTROWEAK_CHECK #4) was not touched here.
7. Part 5 and the appendices may contain other statements that depend on the old p2_15 wording ("at most three", "6.6 ppm"); only the
   lines that reference `ch:particle` were changed.
