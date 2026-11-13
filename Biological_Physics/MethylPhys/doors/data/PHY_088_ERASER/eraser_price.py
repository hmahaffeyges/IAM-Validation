"""PHY-088, the eraser's price (exploration, desk; 2026-11-13). Where IAM's Law is: the law prices KEEPING a mark (the hold, PHY-083: two
coins per restore against thermal kicks, w = 2M) and gives the floor for RESETTING one (Landauer, kT ln 2 at the encoding surface, 310 K).
The eraser (TET, then TDG and base excision repair) resets marks on purpose; the cell then restores them. This ledger prices one erasure
and its restore in coins (ATP), and asks what share of the record's maintenance energy goes to planned erasure, given the measured shares
of the losses the erasers cause. No number here is fitted. Sources, per line:
  (1) Landauer floor: kT ln 2, with M = dG_ATP / (R T) from CANON (M_cell).
  (2) TET oxidation: each step spends one 2-oxoglutarate and gives succinate + CO2, bypassing 2-oxoglutarate dehydrogenase and
      succinyl-CoA synthetase, so the TCA cycle forgoes one NADH (2.5 ATP at the textbook P/O ratio) and one GTP (1 ATP): 3.5 coins per
      oxidation (Berg, Tymoczko, Stryer, Biochemistry, TCA yields). TDG excises 5fC or 5caC, so a full erasure takes 2 or 3 oxidations.
  (3) Repair: DNA ligase III, one ATP to AMP + PPi = 2 coins (as for Okazaki ligation, Piñeyro et al. PLoS Pathog 2023).
  (4) The new dCTP: salvage (dCMP to dCTP, 2 phosphorylations, plus 1 for deoxycytidine kinase) 3 coins; de novo from glucose 48.7 coins
      (BioNumbers 112129, Lynch and Marinov 2015). A bracket, not a value.
  (5) Restore: the hold's price, 2 coins per restore (PHY-083, CANON eps0_candidate: w = 2M).
Measured shares of loss caused by erasers (1 - eraser-removed over control): all erasers, stem cells, 1 - 0.8642 (DEV-ERASE-01 recount);
TET2 in blood progenitors, 1 - 0.905 (PHY-085, mean of 0.900 and 0.909). Writes eraser_price_summary.txt."""
import os, json, math
H = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(H, "..", "..", "..", "..", ".."))
K = json.load(open(os.path.join(ROOT, "CANON", "iam_canon.json"), encoding="utf-8"))["constants"]; M = float(K["M_cell"]["value"])
landauer = math.log(2) / M
ox, lig, restore = 3.5, 2.0, 2.0; dctp = (3.0, 48.7)
lo = 2 * ox + lig + dctp[0] + restore; hi = 3 * ox + lig + dctp[1] + restore
L = [f"M (CANON M_cell) = {M}; Landauer floor for one reset = ln2 / M = {landauer:.4f} coins ({math.log(2):.3f} kT)",
     f"one erasure + restore: {lo:.1f} to {hi:.1f} coins ({lo * M:.0f} to {hi * M:.0f} kT); {lo / landauer:.0f} to {hi / landauer:.0f} times Landauer's floor",
     f"   of which oxidations {2 * ox:.1f}-{3 * ox:.1f}, ligase {lig:.1f}, dCTP {dctp[0]:.1f}-{dctp[1]:.1f}, restore {restore:.1f}",
     f"a loss from thermal kicks costs one restore: {restore:.1f} coins; an erased mark costs {lo / restore:.1f} to {hi / restore:.1f} times as much"]
for lab, s in (("all erasers, stem cells (DEV-ERASE-01)", 1 - 0.8642), ("TET2 alone, blood progenitors (PHY-085)", 1 - 0.905)):
    f = lambda c: s * c / (s * c + (1 - s) * restore)
    L.append(f"{lab}: erasers cause {100 * s:.1f} % of losses but take {100 * f(lo):.0f} % to {100 * f(hi):.0f} % of the record's maintenance energy")
L.append("Exploration; the dCTP bracket dominates the range (salvage vs de novo). Next: the absolute erasure rate from published 5hmC turnover,")
L.append("against the restore rate the hold implies, as an independent check of the measured shares.")
open(os.path.join(H, "eraser_price_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
