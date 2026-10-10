set -e; cd cbook2/chapters && cat > 06_gauge.tex <<'EOF'
\chapter{One gauge for a cell}\label{ch:gauge}

\section{The reading}

Every reading in this book is an error score read on the same gauge: the reading divided by the same cell type's healthy
reading, on the same instrument,
\begin{equation}
  A=\frac{\text{reading}}{\text{healthy reference}} .
  \label{eq:A}
\end{equation}
$A=1$ is the cell working as itself and sits in the middle. Below it lies $\Hmin$, the floor: the least error the cell's
energy budget can hold against thermal kicks; nothing reads below it. Above 1 the cell makes more error than when healthy
(Figure~\ref{fig:gauge}). The ratio is dimensionless and independent of laboratory, array and person, provided the reading
and the reference are made on the same platform through the same calibration (Chapter~\ref{ch:instrument}). The same
construction reads qubits and chips in the companion volume; for a cell the healthy reference is what a clinician needs,
because what matters is whether this cell is still the cell it should be.

\begin{figure}[t]\centering
\includegraphics[width=\textwidth]{fig_gauge_cell.pdf}
\caption{The cell gauge for neutrophils. Top: Met-A on EPIC arrays. Bottom: IAM-A on single molecules. $A=1$ is the healthy
reference; Normal is $0.95$--$1.05$. $\Hmin$ (IAM-A, $0.910$) is the floor where thermal kicks win. The breach line at
$1.10$ is provisional (class-era scale) and the cancer region beyond it is still to be mapped. The surface is full when every
identity site sits at a coin flip: $3.03$ (Met-A), $4.45$ (IAM-A). \calc}
\label{fig:gauge}
\end{figure}

\section{The points on the gauge}

\begin{center}\small
\begin{tabular}{@{}p{0.2\textwidth}p{0.2\textwidth}p{0.5\textwidth}@{}}\toprule
point & value (neutrophils) & status \\\midrule
$\Hmin$ & IAM-A $0.910$; Met-A not yet defined & IAM-A: form \derived, height \measured{} (Ch.~\ref{ch:iama}); Met-A \openprob \\
Normal & $0.95$--$1.05$ & a design tolerance: healthy is $A=1\pm5$~\%; no population sets it \\
breach & $1.10$ & provisional; carried from the class-era scale, to be mapped on the current scale \\
cancer region & beyond breach & to be mapped as per-person data come in \\
full surface & Met-A $3.03$; IAM-A $4.45$ & \calc{} from the healthy reference alone \\\bottomrule
\end{tabular}
\end{center}

\begin{keybox}[title=What the chain prints today]
Chain v3 prints the reading with one of three words: \emph{Normal}, \emph{above Normal}, \emph{below Normal}. Every line
beyond Normal is withheld until it is measured on the current scale. The earlier tier scheme (Elevated 1.05--1.07, a Warburg
line at 1.07, Breach at 1.10) was set on the class-era scale, where $A=1$ meant a class floor; its two named lines were never
derived. The breach line is drawn on the gauge as a provisional mark; nothing is printed from it.
\end{keybox}

\section{How far a reading moves}

What a reading can show depends on how far a real change moves it. On arrays, a simulated loss of 2~\% of the neutrophil
pattern in whole blood moves tared Met-A by about $+0.06$ (1.052--1.090 on six constructed mixtures); the shift scales with
the neutrophil fraction of the specimen, 0.033 at 40--50~\% neutrophils and 0.064 above 70~\% (DEV-LOWFRAC-01).
\measured{} On molecules, a simulated 2~\% rise in copy error moves IAM-A from about 1.00 to 1.29--1.35. \measured{} The
single-molecule reading is the more sensitive of the two by a factor of about five.

\section{The interval on a reading}

A reading carries three uncertainties. The healthy reference itself: six purified neutrophil arrays read one another, with
the identity sites re-chosen each time, at a standard deviation of 0.020 (Chapter~\ref{ch:meta}). The run: a same-run tare
against healthy references processed alongside the specimen states its own spread. And the fraction: in whole blood a cell
that is a small share of the DNA moves the reading less, so each specimen carries its own \emph{detection limit}, the
smallest loss of its neutrophil pattern it could show (Chapter~\ref{ch:chain}). \measured

\section{Tier words are not diagnoses}
\begin{rulebox}
A word on the gauge is a statement about where a cell population's error sits relative to the healthy cell of its kind. It
is not a diagnosis, a prognosis or a statement about a person. The report does not name conditions; a render-time guard
fails any measurement tab that does (Chapter~\ref{ch:report}).
\end{rulebox}
EOF
cat > 07_meta.tex <<'EOF'
\chapter{Met-A: the cell's own reference on arrays}\label{ch:meta}

\section{The reading}

On a methylation array, a cell is read on its identity sites $\mathcal I$ as the mean per-site entropy (Eq.~\ref{eq:meanH})
over the same quantity measured on purified healthy cells of the same type, on the same platform, through the same
calibration:
\begin{equation}
  \text{Met-A}=\frac{\frac{1}{|\mathcal I|}\sum_{i\in\mathcal I}H(\beta_i)}{H_{\rm ref}},
  \qquad H_{\rm ref}^{\text{EPIC, neutrophil}}=0.330263\ \text{bits}.
  \label{eq:meta}
\end{equation}
\calibrated{} The reference is frozen in \texttt{chain/Runtime Matrices/Met\_A\_Floors/metA\_floors\_v1\_3.json} and read from
there by the chain; it is never typed.

\section{How the healthy reference was measured}

\begin{enumerate}
\item \textbf{Specimens.} Purified neutrophils on EPIC v1 arrays from healthy donors~\cite{Salas2018,Salas2022}, from the raw
      IDATs through the chain's own Stage~1. Two public series deposit the same six physical arrays under two accessions
      (same Sentrix identifiers); each array is counted once.
\item \textbf{Identity sites.} Sites the purified cells hold steadily, on both channels: across-array standard deviation
      $\le0.05$; $\beta$ in 0.75--0.95 (methylated channel) or 0.05--0.25 (unmethylated channel); at most 3,000 per channel.
      The neutrophil set has 6,000 sites, 3,000 on each channel.
\item \textbf{The value.} The mean over the six arrays of the mean $H(\beta)$ on those sites: 0.330263 bits.
\item \textbf{Held out.} Each array read against a reference rebuilt from the other five, with the identity sites re-chosen
      on those five: $A$ 0.983--1.045, standard deviation 0.020, six of six in Normal. On the frozen sites the spread is
      0.993--1.008. \measured
\end{enumerate}

\begin{rulebox}
The reference arrays read 1.00 by construction. A reference array reading 1.00 is therefore not evidence. An array that
played no part in choosing the sites or setting the value, reading 1.00 within tolerance, is.
\end{rulebox}

\section{One reference per cell type and per platform}

A reference measured on one platform does not transfer to another. On 450K arrays, purified neutrophils read 0.932 against
the EPIC reference, monocytes 0.916, NK cells 0.904 (DIAG-450K-01, eight donors). \measured{} The array chemistry differs,
so the entropy the same cell shows differs. Every cell type needs its own reference on every platform. The 450K neutrophil
reference is pending, and chain v3 reads EPIC v1 arrays only.

\section{Whole blood: the person's own expectation}

In whole blood a neutrophil is mixed with every other blood cell, and at the neutrophil's identity sites the mixture pulls
$\beta$ toward one half. The denominator then becomes what this specimen should read if every cell in it were healthy:
\begin{equation}
  \text{Met-A}_{\rm WB}=\frac{\overline{H(\beta_i)}}{\overline{H(e_i)}},\qquad e_i=\sum_g f_g\,\mu_{g,i},
  \label{eq:metawb}
\end{equation}
with $f_g$ this specimen's own fractions of eight purified blood-cell groups and $\mu_{g,i}$ their healthy profiles at the
6,000 sites (\texttt{blood\_composition\_EPIC\_v1.json}). No other person enters: the expectation is built from this person's
composition and the purified profiles. On six DNA mixtures of known composition the expectation built from the true fractions
reads 0.982--1.016 (PROC-WB-NEUT-01), while the neutrophil reference alone reads the same mixtures 1.062--1.118. \measured{}
Below 20~\% neutrophils the reading is withheld and the fraction is printed.

\section{The noise index}

A second laboratory's purified neutrophils read 0.86--1.26 against the reference, and the spread followed the arrays'
noise, not their purity (PROC-NEUT-TEST-01). \measured{} Array noise raises $H$ at every site. The chain therefore measures
it on every specimen: the noise index $N$ is the mean $H(\beta)$ at 48,528 EPIC sites that every purified blood group holds
fixed (every group mean $\le0.03$ or $\ge0.97$, every group SD $\le0.02$). The reference arrays read $N=0.122$--$0.149$;
second-laboratory arrays up to 0.243, and Met-A follows $N$ ($\rho$ 0.79--0.83; DEV-NOISE-01). \measured{} $N$ is used by the
same-run tare (Chapter~\ref{ch:chain}).

\section{The entropy ceiling}

On the methylated channel, Met-A rises with loss only while those sites stay above $\beta=\tfrac12$ (Chapter~\ref{ch:surface}).
A DNA-methyltransferase inhibitor drives cells past that point: on 51 EPIC arrays from three cell lines, vehicle arrays read
0.968--1.048, an inactive analogue 1.002--1.032, and the active drug at $\ge80$~nM 1.16--1.85, near one bit over the reference,
then lower again at the highest doses (PROC-DNMT-01, Part A). \measured{} The chain flags any specimen whose methylated sites
average below one half and tells the reader to read the mean $\beta$ instead.

\section{Where the earlier floors came from, and why they were retired}

The first instrument divided every cell by one floor per \emph{architecture class}: eight classes, eight floors, fitted in
April 2026 by Markov-chain Monte Carlo so that 37 published reference cells read 1.00. The fit converged and reproduced. A
provenance check then found that 8 of the 22 cited Roadmap identifiers name a different cell, that the 37 inputs were not the
genome-wide means of the cited data in 8 of 9 checkable cases, and that class floors read unseen cell types inside Normal only
about half the time on single-molecule data (PROC-CHANNEL-01, section 9). On 1 October 2026 the class floor was retired: every
cell type is read against its own healthy reference, measured on purified cells of that type. \record
EOF
echo ok