# The mini cell (development; a simulation of the model, not data)

## Where IAM's Law is

IAM's Law: every irreversible transition from quantum superposition to classical record pays at least kT ln2 per bit at the nearest
encoding surface, at its local temperature. In a maintained record this sets the ledger every simulator is built on:

| Ledger entry | Relation | Origin |
|---|---|---|
| The coin | M = (energy of one corrective quantum) / kT at the encoding surface | IAM (`coin_M`) |
| The delivery | E_hold = phi M kT | IAM (`delivery_phi`; phi measured) |
| The hold | eps = 1/(1 + exp(E_hold/kT)): the residual error set by what maintenance costs | IAM (`hold_energy`) |
| The floor | kT ln2 per bit | established (Landauer); placed at the encoding surface by IAM |
| The ceiling | the coin flip | established |

## Origin of every number

Every parameter and every reading carries one origin: **IAM** (produced by the law), **established** (known physics), **measured** (from
data), **borrowed** (a method from another field), **ours** (our own method, not physics), **assumed** (a placeholder). A reading may be
labeled IAM only if it names the IAM relation in its calculation path; `simcore.Reading` refuses anything else.

## The IAM switch

Every simulator runs with IAM on or off. Off, it runs on established physics and measured inputs alone. The difference between the two runs
is exactly the IAM claim. In the mini cell, IAM off holds the measured 37 C held error fixed whatever the temperature or drive; IAM on moves
it through the coin and the delivery (a fever to 40 C: 0.0437 to 0.0450; ATP's free energy 54 to 50 kJ/mol: 0.0437 to 0.0544). The 37 C
level itself is measured (E_hold = 3.084 kT), not predicted.

## Analyses (the same in every domain)

| Analysis | In the code | Power-system term |
|---|---|---|
| Operating point | `op` | power flow |
| One lever over a range | `sweep` | a study across a range |
| Sensitivity factors | `sensitivity` | sensitivity analysis |
| A lever tripped or pushed | `contingency` | contingency analysis |
| Spread across cells | `montecarlo` | many base cases |

The full operator translation of every shared term is `simcore.TRANSLATOR`.

## Files

- `simcore.py`: the shared core (model cards, origins, the IAM relations, analyses, translator).
- `minicell.py`: the mini cell (DEV-MINICELL-01): the law's ledger, then maintenance kinetics (CpG pair states; copying, maintenance,
  writing, erasing, lag) that deliver the level, then a bisulfite read simulator with each library's molecule-end fault.
- `selftest.py`: checks that must all pass before any output is used (origin rules, the switch, limits, the floor).

Evidence comes only from a prediction written before its data are read and compared with data the simulator was not tuned on.

**The baseline candidate switch (2026-11-12, PHY-083).** `law_card().set(baseline_candidate=True)` replaces measured phi with the driven
passage: W = SAM_cost x M kT per methyl transfer (SAM_cost 2.0, canon SAM_cost_ATP), held error 0.04454 and 3.066 kT, readings labeled
IAM through the relation `driven_passage` (candidate, its test not yet passed). Off by default; `selftest.py` checks both.
