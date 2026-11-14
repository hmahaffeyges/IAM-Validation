#!/usr/bin/env python3
"""Planted cases for CANON/design_check.py: a complete sheet passes, and each of the 2026-11-13 mistakes is refused."""
import os, sys, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import design_check as D
FORM = [{"form": "deposited aligned files (BAM, chain --bam)", "hours": 1, "adequate": True, "chosen": True}, {"form": "raw reads (alignment)", "hours": 5, "adequate": True}]
OK = {"physics": {"where_iam_law": "the candidate's hold against thermal kicks at 310 K", "counterparts": "chips: none yet; qubits: none yet"},
      "data": {"set": "GSE161463", "protocol_read": "sheared, end repaired, methylated adapters, standard directional WGBS from 1 ug",
               "spent_first": "no: spent sets hold no eraser-removed mature B cells", "alternatives": "GSE210026 (PBAT, refused)"},
      "kiss": {"smallest": "two wild type, two knockout: the fewest that measure replicate scatter", "samples_why": "each arm needs two"},
      "machines": {"platform": "github", "github_free_at_design": 17, "s_per_pair": "0.0005 (PHY-085 timing.json)", "pairs": 10000000,
                   "peak_mem_gb": 6, "machine_mem_gb": 16, "parallel_per_machine": 1, "expected_hours": 2.2, "time_limit_hours": 6, "cost": "free"},
      "bars": {"rule": "GUM k = 2; pull within 2 passes, 2 to 3 watch, beyond 3 a real miss (WHY_IAM.md rule 4)"},
      "save": {"resume": "ckpt"}, "watchers": {"count": 1, "stop_condition": "all four runs completed"},
      "canary": {"instrument_check": {"qc_failed_max": 0.05, "mapped_min": 0.95, "s_per_pair_max": 0.0015}},
      "cosmologist": {"method": "forward-model the knockout through the same counter; self-calibrate on the lambda spike", "null_test": "the two wild-type replicates must agree before any ratio is read"},
      "value": {"question": "does removing the erasers bring B cells to the thermal level?", "adequate": "effect 10 % against replicate scatter 1-2 %",
                "options": [{"name": "full libraries on GitHub", "hours": 2.6, "cost_usd": 0, "adequate": True, "chosen": True},
                            {"name": "cosmologist's way: forward-model the knockout through the counter, read only the canary depth", "hours": 1.0, "cost_usd": 0, "adequate": False, "cosmologist": True}]},
      "_cosm": None}
OK["data"]["needs"] = "paired reads, whole-genome spread, genotype label"
OK["data"]["preflight"] = {"checked": "200 blocks: chromosome spread, read pairing, genotype labels", "minutes": 6, "result": "all 21 autosomes present in proportion; paired; labels match"}
OK.pop("_cosm")
OK["data"]["form"] = FORM
OK["explore"] = {"done": "celltype_vs_thermal.py on spent counts: blood types near the thermal level", "rehearsal_pull": -1.16}
OK["prediction"] = {"form": "thermal level / (1 - s), s each type's eraser share", "decided_by": {"s": "the type's own knockout over control, measured independently"},
                    "reading": "zero loose along IAM-A's slope, every laboratory"}
OK["purpose"] = "sealed"
def case(label, edit, expect_refused):
    S = copy.deepcopy(OK); edit(S); F = D.sheet_findings(S, label)
    ok = bool(F) == expect_refused
    print(("ok   " if ok else "FAIL ") + label + ("" if ok else f" (findings: {F})")); return ok
R = [case("complete sheet passes", lambda S: None, False),
     case("missing question refused", lambda S: S["kiss"].pop("smallest"), True),
     case("google without a reason refused", lambda S: S["machines"].update(platform="google"), True),
     case("three 6 GB counts on a 16 GB machine refused (PHY-087, 11-13)", lambda S: S["machines"].update(parallel_per_machine=3), True),
     case("unmeasured memory with 3 per machine refused", lambda S: S["machines"].update(peak_mem_gb="not measured", parallel_per_machine=3), True),
     case("PBAT library without a method answer refused (GSE210026, 11-13)", lambda S: S["data"].update(protocol_read="post-bisulfite adapter tagging, about 500 cells"), True),
     case("1.75 h promised for 3.4 h of aligning refused (Google, 11-13)", lambda S: S["machines"].update(s_per_pair="0.00123", expected_hours=1.75), True),
     case("time limit under 1.3 x expected refused", lambda S: S["machines"].update(time_limit_hours=2.5), True),
     case("two watchers refused", lambda S: S["watchers"].update(count=2), True),
     case("canary without numbers refused", lambda S: S["canary"].update(instrument_check="looks fine"), True),
     case("one option only refused", lambda S: S["value"].update(options=S["value"]["options"][:1]), True),
     case("a faster, cheaper adequate option without a reason refused", lambda S: S["value"]["options"][1].update(adequate=True), True),
     case("no cosmologist's way among the options refused", lambda S: [(o.pop("cosmologist", None), o.update(name="other")) for o in S["value"]["options"][1:]], True),
     case("no preflight refused (the four referee libraries on chr1 and chr2, 11-13)", lambda S: S["data"].pop("preflight"), True),
     case("preflight that found a problem without resolving it refused", lambda S: S["data"]["preflight"].update(result="chr6 missing in 4 libraries"), True),
     case("'quality-failed share' in a clean preflight is not a problem", lambda S: S["data"]["preflight"].update(result="99.8 % mapped, 2.8 % quality-failed share"), False),
     case("'MET only if every type is inside' without the agreed classes refused (PHY-084, 11-13)", lambda S: S["bars"].update(rule="MET if all four types lie inside k = 2"), True),
     case("a sheet without the chip and qubit counterpart refused (11-13)", lambda S: S["physics"].pop("counterparts"), True),
     case("raw reads chosen when aligned files exist, no reason, refused (11-13)", lambda S: [S["data"]["form"][0].update(chosen=False), S["data"]["form"][1].update(chosen=True)], True),
     case("raw reads chosen with a written reason passes", lambda S: [S["data"]["form"][0].update(chosen=False), S["data"]["form"][1].update(chosen=True, why_not_cheaper="the BAM is GRCh38; the loose share needs hg19 tiles")], False),
     case("no data.form refused", lambda S: S["data"].pop("form"), True),
     case("sealing with no rehearsal on spent data refused (11-13)", lambda S: S["explore"].pop("rehearsal_pull"), True),
     case("rehearsal beyond 2 refused: fix the prediction first (11-13)", lambda S: S["explore"].update(rehearsal_pull=-2.45), True),
     case("a branch chosen by the outcome refused (11-13)", lambda S: S["prediction"]["decided_by"].update(s="whichever fits the reading"), True),
     case("no prediction.reading refused (11-13)", lambda S: S["prediction"].pop("reading"), True),
     case("a derivation test naming a set reserved for commissioning refused (11-13)", lambda S: S["data"].update(set="GSE131627 healthy-control monocytes"), True),
     case("an exploration sheet without a rehearsal or prediction passes", lambda S: [S.update(purpose="exploration"), S.pop("prediction"), S["explore"].pop("rehearsal_pull"), S["bars"].update(rule="exploration: no seal; read and reported as found")], False),
     case("an exploration sheet that seals refused", lambda S: [S.update(purpose="exploration"), S["bars"].update(rule="pass within 2, watch 2-3, miss beyond 3")], True),
     case("no purpose refused", lambda S: S.pop("purpose"), True),
     case("count-only canary with its own three numbers passes", lambda S: S["canary"].update(instrument_check={"rows_min": 14000, "halves_corr_min": 0.7, "hours_per_sample_max": 3}), False)]
print("design_check_controls: " + ("ok" if all(R) else "FAILED")); sys.exit(0 if all(R) else 1)
