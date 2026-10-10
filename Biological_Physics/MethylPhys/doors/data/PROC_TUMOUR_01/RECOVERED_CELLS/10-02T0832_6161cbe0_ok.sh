set -e; cd cbook2/chapters && python3 - <<'PY'
p="07_meta.tex"; s=open(p).read()
a="""A reference measured on one platform does not transfer to another. On 450K arrays, purified neutrophils read 0.932 against
the EPIC reference, monocytes 0.916, NK cells 0.904 (DIAG-450K-01, eight donors). \\measured{} The array chemistry differs,
so the entropy the same cell shows differs. Every cell type needs its own reference on every platform. The 450K neutrophil
reference is pending, and chain v3 reads EPIC v1 arrays only."""
b="""A reference measured on one platform does not transfer to another. On 450K arrays from eight donors, purified neutrophils
read 0.932 on references measured on EPIC arrays, monocytes 0.916, NK cells 0.904; none of the eight in Normal. A development
reference built from purified 450K arrays, read leave-one-donor-out, put all eight in Normal (DIAG-450K-01). \\measured{} The
array chemistry differs, so the entropy the same cell shows differs, and every cell type needs its own reference on every
platform. The 450K neutrophil reference is not yet frozen into the chain; chain v3 reads EPIC v1 arrays only."""
assert s.count(a)==1; open(p,"w").write(s.replace(a,b)); print("ok")
PY
cat > 08_iama.tex <<'EOF'
\chapter{IAM-A: copy error on single molecules}\label{ch:iama}

\section{What a molecule shows that an array cannot}

An array reports, at each site, the fraction of copies that are methylated. A sequencing read reports one molecule: a run of
CpGs, each methylated or not, on one strand of one cell's DNA. On a molecule that is otherwise methylated, an unmethylated CpG
between two methylated neighbours is a site the maintenance machinery failed to copy at the last division. That is the
\emph{copy error}, and it is the most direct reading of the writing process the book's physics is about.

\section{The statistic}

A molecule qualifies when it carries at least six CpG calls and at least 80~\% of them are methylated. Each interior call is
an opportunity; an unmethylated interior call with both neighbours methylated is an isolated error. The copy error is
\begin{equation}
  \varepsilon=\frac{\sum\text{isolated errors}}{\sum\text{opportunities}},
  \qquad E_{\rm hold}=\kB T\ln\frac{1-\varepsilon}{\varepsilon},
  \label{eq:eps}
\end{equation}
computed on the methylated channel only. The unmethylated channel is not used: there, bisulfite conversion failure dominates
the isolated calls. The chain requires at least 100,000 opportunities and reports the reading separately on odd and even
molecules as a repeat check. \derived{} for the form; the selection rules are \calibrated.

\section{The floor}

If the cell holds each site with an energy $E$ against thermal kicks, a two-state Boltzmann factor gives the least error that
energy allows:
\begin{equation}
  \varepsilon_0=\frac{1}{1+e^{\phi M}},\qquad \Hmin=H(\varepsilon_0).
  \label{eq:eps0}
\end{equation}
\derived{} for the form. With $M=20.94$ and the holding energy measured across 56 healthy cell types, $\phi=0.1628$
(Chapter~\ref{ch:landauer}), $\varepsilon_0=0.032$ and $\Hmin=0.2044$ bits. \measured{} for the height. The form is IAM's Law at
the chromatin surface; the height is still taken from healthy molecules, and deriving $\phi$ from the chemistry of
maintenance is the central open problem of the cellular rung (Chapter~\ref{ch:open}).

\section{The healthy position, and the reading}

A healthy cell does not sit exactly on the floor. Its position above the floor, $P$, is measured once per cell type and per
read pipeline, and frozen (\texttt{iama\_positions\_v1.json}):
\begin{equation}
  \text{IAM-A}=\frac{H(\varepsilon)}{P_{\rm cell}\,H(\varepsilon_0)},\qquad P_{\rm neutrophil}=1.099 .
  \label{eq:iama}
\end{equation}
\calibrated{} $P$ was measured on three healthy granulocyte donors (range 1.084--1.108, coefficient of variation 1.2~\%), on
the read-level files of one public atlas~\cite{Loyfer2023} with the pipeline named \texttt{loyfer\_pat\_v1}. The floor then
sits at $A=1/P=0.910$, below Normal. A reading below it is not a cell doing better than physics allows; it is an instrument
fault, and the chain treats it as one.

\begin{center}\small
\begin{tabular}{@{}lll@{}}\toprule
test & result & \\\midrule
healthy donors on $\varepsilon_0$ alone & 1.084--1.127 & \measured \\
healthy donors with $P$, leave-one-donor-out & 0.978--1.040 & \measured \\
odd/even molecule halves & differ by $\le0.002$ & \measured \\
simulated 2~\% rise in copy error & 1.285--1.346 & \measured \\
same cells on a second read pipeline, bare floor & 0.70--0.79 & \measured \\\bottomrule
\end{tabular}
\end{center}
The last line is why $P$ is locked to a pipeline: the same cells read through a different alignment and extraction show a
different error. A new pipeline needs its own $P$, measured on healthy cells, before it can read anything.

\begin{rulebox}
The floor and the reading must be defined by the same statistic on the same selection of molecules. An earlier floor of
$\varepsilon_0=0.0227$, taken from a different statistic, made healthy molecules look 3.4--4.8 times too error-prone
(PROC-MOLECULE-01). The lesson is now a rule of the chain.
\end{rulebox}

\section{One energy, many cells}

Read on the same 399 genomic windows for 153 samples of 56 healthy cell types, with no labels and no floor
(PROC-CHANNEL-01):
\begin{itemize}
\item the copy error ranges 0.024--0.042 and the holding energy $3.41\pm0.12\,\kB T$: close to one energy in every cell type;
\item on one physics floor (Eq.~\ref{eq:eps0}) healthy cell types read 0.79--1.23, median 1.01; 27 of 56 inside Normal. One
      floor is not yet precise enough for an individual reading, which is why each cell type carries its own $P$;
\item on the 26,800 CpGs that all 56 cell types keep methylated, the same bits in every cell, the copy error still differs
      1.66-fold from the lowest (naive CD8 T cells, lung macrophages, B cells) to the highest (smooth muscle, heart
      fibroblasts, erythroid progenitors), and the difference is stable across donors (intraclass correlation 0.80);
\item cells do not sort by turnover: colon epithelium (a 3.4-day lifespan) and cardiomyocytes (56,000 days) carry the same
      copy error, 0.031.
\end{itemize}
\measured{} for all four. The architecture of each cell's maintenance, not how often it is replaced, sets its position.

\section{Tumours}

The same statistic reads tumours against the same person's normal tissue with no reference population: in six early-onset
colorectal pairs the tumour's copy error is higher in all six, by 7--33~\%, and in oral squamous carcinoma in four of four,
most of it surviving when 5-hydroxymethylcytosine is removed (PROC-TUMOUR-01; Chapter~\ref{ch:reach}). \measured
EOF
cat > 09_cscore.tex <<'EOF'
\chapter{The C-score: where the departures sit}\label{ch:cscore}

\section{Two people with the same $A$}

$A$ is an average over a cell's identity sites. Two specimens can read the same $A$ with very different patterns underneath:
a small departure spread evenly over every site, or a large departure concentrated in a few regions of the genome. The second
is what a clone, a deletion or a regional failure of maintenance would leave. The C-score reads that difference, on the same
gauge: healthy is 1.

\section{The residual map}

For each identity site, in genome order, the chain computes how far this specimen's entropy sits from the healthy cell's,
in units of the healthy spread at that site:
\begin{equation}
  z_i=\frac{H(\beta_i)-H(\mathrm{ref}_i)}{s_i},
  \label{eq:z}
\end{equation}
with $\mathrm{ref}_i$ the healthy neutrophil mean entropy at that site (isolated cells) or the person's own expectation
$H(e_i)$ (whole blood), and $s_i$ the shrunken standard deviation of $H$ across the six reference arrays. The map is the cell's
own sky: only measured sites, placed by genome position, nothing interpolated. \derived{} (the construction)

\section{The score}

Consecutive runs of 50 sites are averaged. If the departures are scattered at random, the block means vary as $1/\sqrt{50}$
of the site values; if they cluster, the block means vary more. The clustering statistic is
\begin{equation}
  c=\frac{\mathrm{var}\left(\sqrt{50}\;\overline{z}_{\rm block}\right)}{\mathrm{var}(z)},\qquad
  C=\frac{c}{c_{\rm healthy}},\qquad c_{\rm healthy}=1.1104,
  \label{eq:C}
\end{equation}
with $c_{\rm healthy}$ the median over the six reference arrays, each read against the other five
(\texttt{neutrophil\_reference\_v1\_1.json}). \calibrated{} The healthy arrays read $C=0.70$--$1.23$. A score needs at least
ten blocks.

\begin{keybox}[title=Status]
The Met-A C-score is built into chain v3 and printed with every reading, with no band: six healthy arrays are too few to set
one. On the acceptance run, isolated neutrophils read $C=0.69$--$1.21$ and whole bloods $0.78$--$1.49$; in whole blood the
residual also carries composition error. \measured{} The IAM-A C-score, the same map built from per-region copy error on
molecules, is defined and not yet built. \openprob
\end{keybox}

\section{Why it matters}

A loss spread thinly across the genome and a loss concentrated in one region can mean different biology, and only the
second is visible as structure. Serial readings of one person (Chapter~\ref{ch:serial}) make the map sharper with every draw,
because a person's own stable regions become their own reference.
EOF
cat > 10_temperature.tex <<'EOF'
\chapter{Temperature, and other species}\label{ch:temperature}

\section{The floor is a function of temperature}

The Mahaffey number falls as body temperature rises, if the free energy of ATP hydrolysis is held fixed: 22.94 for a salmonid
at 10~\textdegree C, 20.94 for a human at 37~\textdegree C, 20.84 for a dog at 38.5~\textdegree C, 20.74 for a bird at
40~\textdegree C. \calc{} Holding $\dGATP$ fixed is an assumption; the free energy depends on temperature and on the cell's
ATP, ADP and phosphate concentrations.

The floor of Eq.~\eqref{eq:eps0} makes a sharper statement. If a cell holds each site with a fixed energy $E_{\rm hold}$,
measured at 37~\textdegree C as $3.41\,\kB T$, then at another temperature
\begin{equation}
  \varepsilon_0(T)=\frac{1}{1+\exp\!\left(E_{\rm hold}/\kB T\right)} ,
  \label{eq:eps0T}
\end{equation}
with no exponent to choose. \derived{} given a fixed holding energy. At 10~\textdegree C this gives $\varepsilon_0=0.0233$ and a
floor $0.78$ times the human one; at 38.5~\textdegree C, $1.012$ times (Figure~\ref{fig:eps0T}). \calc{} The alternative is a
cell that holds its error fixed in units of $\kB T$, adjusting its holding energy with temperature. The two are distinguished by
one measurement.

\begin{figure}[t]\centering
\includegraphics[width=0.8\textwidth]{fig_eps0_T.pdf}
\caption{The floor against body temperature. Solid: fixed holding energy ($3.41\,\kB T$ at 37~\textdegree C), Eq.~\eqref{eq:eps0T}.
Dashed: error held fixed in units of $\kB T$. Marks: dog (38.5~\textdegree C), human (37~\textdegree C), salmonid
(10~\textdegree C). \calc}
\label{fig:eps0T}
\end{figure}

\section{What the fish say so far}

Three salmonid sets have been read with the same single-molecule statistic, each pre-registered:
\begin{center}\small
\begin{tabular}{@{}llll@{}}\toprule
set & tissue, method & holding energy & instrument \\\midrule
Methow River steelhead, 20 fish & red blood cells, RRBS & 3.31 $\kB T$ (median) & halves agree, ICC 0.998 \\
 & sperm, RRBS & 4.02 $\kB T$ & \\
brook charr, 40 males & sperm, WGBS & 3.82 $\kB T$ & ICC 0.92 \\
Rimouski Atlantic salmon, 64 fish & fin, WGBS & 3.47 $\kB T$ (F0) & ICC 0.996 \\\bottomrule
\end{tabular}
\end{center}
\measured{} The instrument repeats almost perfectly in every set. In every set the small differences between fish track the
library (conversion failure, duplicate fraction), so no fish-level difference has yet been read. The steelhead red cells, at
about 10~\textdegree C, read close to healthy human cells at 37~\textdegree C on the same statistic, against the fixed-energy
prediction of a higher value in $\kB T$ units. That comparison is not the test: one small study, reduced-representation
sequencing of CpG-dense regions against whole-genome human reads, a nucleated red cell against human granulocytes, and a
body temperature assumed from the water. \openprob

\begin{keybox}[title=The test that decides]
One species, one cell type, one read pipeline, at two recorded temperatures. A rearing-temperature experiment does it; a
whole-genome coho data set already downloaded is the first fish reading on a pipeline comparable to the human one. Until then
the temperature behaviour of the floor is \openprob, and no ectotherm is read on the gauge.
\end{keybox}

\section{Dogs}

Dogs are the natural first mammal after humans: their blood is drawn routinely, they share our environment, and a
methylation array designed for mammals covers conserved CpGs across species~\cite{Arneson2022}. For a dog $M=20.84$ and the
floor moves by about 1~\%. The first step is a canine healthy reference from purified canine blood cells; then sorted healthy
canine cells read on it, held out, should land at 1.00. \prediction
EOF
echo ok