"""IAM mini-system simulators: the shared core (development; a simulation of the model, not data).

One structure for every simulator, so the nomenclature and the analyses stay the same across them:

  model card   every parameter with its value, units, label (measured / derived / assumed) and source, in the form the field already
               uses (for the cell: maintenance rates per CpG pair)
  analyses     op         operating point            (grid: power flow, the base case)
               sweep      one lever moved over a range (grid: sensitivity analysis)
               contingency one lever tripped or pushed (grid: contingency analysis)
               montecarlo spread across cells (grid: many base cases drawn from the measured spread)
  levers       one vocabulary across domains: restore, loss, writer, eraser, copy, kick (temperature), coin (drive), meter

Every parameter and every reading carries its ORIGIN (IAM, established physics, measured, borrowed method, our own method, assumed).
A reading may be labeled IAM only if it names the IAM relation in its calculation path; the code refuses anything else.

A model's output counts as evidence only where a prediction is written down from it first and then compared with real data it was not
tuned on. Nothing in a model card is fitted to a target reading.
"""
import copy, json, math, os

K_B = 1.380649e-23          # J/K (exact, SI)
H = 6.62607015e-34          # J s (exact, SI)
HBAR = H / (2 * math.pi)
E_CHARGE = 1.602176634e-19  # C (exact, SI)
LN2 = math.log(2.0)
LABELS = ("measured", "derived", "assumed", "definition")

# ORIGIN of every parameter and every reading (author 2026-11-11: always say what our physics produced and what it did not).
#   IAM          produced by IAM's Law (the product; where referees and engineers should press hardest)
#   established  established physics anyone can use (Landauer, Boltzmann, the virial theorem)
#   measured     a number taken from data
#   borrowed     a method borrowed from another field (cosmology, power systems, metrology)
#   ours         our own method or engineering, not physics
#   assumed      a placeholder value, to be replaced by a measurement
ORIGINS = ("IAM", "established", "measured", "borrowed", "ours", "assumed")

# IAM's relations: an output may be labeled IAM only if one of these is in its calculation path (named in Reading.relations)
IAM_RELATIONS = {
    "encoding_surface": "every irreversible quantum-to-classical transition pays at least kT ln2 per bit at the nearest encoding surface, at its LOCAL temperature",
    "coin_M": "the drive in kicks: M = (energy of one corrective quantum) / (kT at the encoding surface)",
    "hold_energy": "a maintained record's residual error is set by what maintenance costs. The form eps = 1/(1 + exp(E_hold/kT)) is established "
                   "(Boltzmann); IAM's content is that E_hold is the share of the drive maintenance delivers (E_hold = phi M kT), so the error moves "
                   "with the drive and the local temperature",
    "delivery_phi": "phi = E_hold / M: the share of each coin delivered as holding",
    "driven_passage": "CANDIDATE (PHY-083): the baseline as restoring paid against thermal kicks, odds = (w/2)/(1-(1-e^-w)/w), w = W/kT, W = (SAM price in coins) x M kT per methyl transfer; candidate, its test not yet passed",
    "virial_half": "a 1/r-bound record releases exactly half its binding energy as the minimum cost of the record it writes (the book's labels apply)",
}

# the shared lever vocabulary: each model maps its own parameters onto these (grid words in TRANSLATOR)
LEVERS = ("restore", "loss", "writer", "eraser", "copy", "kick", "coin", "meter")


class Param:
    """One model-card entry: value, units, label (measured / derived / assumed / definition), source and ORIGIN (see ORIGINS)."""
    def __init__(self, value, units, label, source, lever=None, note="", origin=None):
        assert label in LABELS, f"label must be one of {LABELS}"
        assert lever is None or lever in LEVERS, f"lever must be one of {LEVERS}"
        origin = origin or {"measured": "measured", "assumed": "assumed"}.get(label)
        assert origin in ORIGINS, f"every parameter states its origin, one of {ORIGINS}"
        self.value, self.units, self.label, self.source, self.lever, self.note, self.origin = value, units, label, source, lever, note, origin
    def as_dict(self): return dict(value=self.value, units=self.units, label=self.label, source=self.source, lever=self.lever, note=self.note, origin=self.origin)


class Reading(float):
    """A model output: a number that carries its origin and, if IAM, the IAM relations in its calculation path."""
    def __new__(cls, value, origin, relations=(), note=""):
        assert origin in ORIGINS, f"every reading states its origin, one of {ORIGINS}"
        relations = tuple(relations)
        assert all(r in IAM_RELATIONS for r in relations), "unknown IAM relation"
        assert (origin == "IAM") == bool(relations), "a reading is IAM if and only if it names the IAM relation it used"
        x = float.__new__(cls, value); x.origin, x.relations, x.note = origin, relations, note; return x


def ledger_table(readings):
    """Rows for a report: each reading with its origin and, for IAM readings, the relation it used (what is ours to sell is the IAM rows)."""
    return [dict(reading=k, value=float(v), origin=getattr(v, "origin", "unlabeled"), relations=", ".join(getattr(v, "relations", ())), note=getattr(v, "note", ""))
            for k, v in readings.items()]


class ModelCard(dict):
    """name -> Param. card.v('x') is the value; card.set('x', v) returns a changed copy (the original is never altered)."""
    def __init__(self, name, domain, params):
        super().__init__(params); self.name, self.domain = name, domain
    def v(self, k): return self[k].value
    def set(self, **kw):
        c = ModelCard(self.name, self.domain, {k: copy.copy(p) for k, p in self.items()})
        for k, x in kw.items():
            assert k in c, f"no parameter {k!r} in card {self.name}"
            c[k].value = x
        return c
    def table(self):
        return [dict(name=k, **p.as_dict()) for k, p in self.items()]
    def write(self, path):
        json.dump(dict(name=self.name, domain=self.domain, params={k: p.as_dict() for k, p in self.items()}), open(path, "w"), indent=1)


def op(model, card):
    """Operating point (grid: power flow): the model's steady readings for this card."""
    return model(card)


def sweep(model, card, param, values, readings=None):
    """One lever moved over a range (grid: sensitivity analysis). Returns rows of {param: value, readings...}."""
    out = []
    for x in values:
        r = model(card.set(**{param: x})); out.append({param: x, **({k: r[k] for k in readings} if readings else r)})
    return out


def sensitivity(model, card, readings, rel=0.01):
    """d ln(reading) / d ln(parameter) for every numeric parameter (grid: sensitivity factors), by central differences."""
    base = model(card); S = {}
    for k, p in card.items():
        if not isinstance(p.value, (int, float)) or isinstance(p.value, bool) or p.value == 0: continue
        up = model(card.set(**{k: p.value * (1 + rel)})); dn = model(card.set(**{k: p.value * (1 - rel)}))
        S[k] = {r: (math.log(up[r]) - math.log(dn[r])) / (math.log(1 + rel) - math.log(1 - rel)) if up[r] > 0 and dn[r] > 0 else None for r in readings}
    return base, S


def contingency(model, card, cases):
    """Each named case changes some parameters (grid: contingency analysis). Returns {case: readings}, with 'base' first."""
    out = {"base": model(card)}
    for name, change in cases.items(): out[name] = model(card.set(**change))
    return out


def montecarlo(model, card, spreads, n=200, seed=20261111):
    """Many cards drawn from each parameter's spread (lognormal, relative sd), e.g. many cells."""
    import random
    g = random.Random(seed); out = []
    for _ in range(n):
        ch = {k: card.v(k) * math.exp(g.gauss(0, s) - s * s / 2) for k, s in spreads.items()}; out.append(model(card.set(**ch)))
    return out


# the operator translator: every shared term and analysis in power-system operator language
TRANSLATOR = {
    "restore": "regulation: the control action that pulls a point back to schedule (governors, AGC, the repair crew)",
    "loss": "the disturbance that knocks a point off schedule (load swings, faults, leakage)",
    "writer": "scheduled switching that puts a point into a new position on purpose",
    "eraser": "scheduled switching that takes a point out of position on purpose",
    "copy": "a system split or a new section energized: each new piece starts with only half its telemetry and must be brought to schedule",
    "kick": "temperature: the size of the background disturbance hitting every point (kT)",
    "coin": "the size of one dispatched correction packet relative to one disturbance (M)",
    "meter": "the measurement chain: CTs, PTs, transducers and the RTU; its own error must be told apart from the system's",
    "op": "power flow: the base-case operating point",
    "sweep": "a study varying one quantity across its range",
    "sensitivity": "sensitivity factors: how much each reading moves per unit change of each input",
    "contingency": "contingency analysis: trip or push one element and read the result",
    "montecarlo": "many base cases drawn from the measured spread of the equipment",
    "model card": "the equipment's nameplate and test-report data used by the model",
    "held error": "the share of points found off schedule at any moment",
    "end fault (meter)": "an error at the terminals of the line (the molecule's ends), located by distance like an impedance relay",
        "refresh / maintenance": "a periodic sweep that puts every point back to schedule",
}
