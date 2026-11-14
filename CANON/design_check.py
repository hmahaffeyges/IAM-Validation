#!/usr/bin/env python3
"""Design check (author 2026-11-13; CANON/TEST_DESIGN_CHECKLIST.md is the question list). A push that ADDS a manifest row, or adds to a doors
record a sealed rule or a '**Synthetic test:**' line, must carry a complete design/<ROW>.json for that row. Refused when: a key is missing or
empty; platform is google without why_not_github; parallel_per_machine x peak_mem_gb > machine_mem_gb (or peak 'not measured' with
parallel > 1); expected_hours below (s_per_pair x pairs / 3600); time_limit_hours < 1.3 x expected_hours; watchers.count != 1; the canary
rule lacks its three numbers; data.protocol_read names PBAT, non-directional, single-cell, low-input, EM-seq or read-end clipping without
data.method_ok explaining how the read path handles it. Applies to rows added from 2026-11-13. Usage: python3 CANON/design_check.py check
[base] | sheet <file>. CANON/design_check_controls.py must pass first."""
import os, sys, re, json, subprocess
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_PREFIX = ("PHY-", "COM-", "DEV-")
KEYS = re.findall(r"^\| ([a-z_]+(?:\.[a-z_]+)?) \|", open(os.path.join(R, "CANON", "TEST_DESIGN_CHECKLIST.md"), encoding="utf-8").read(), re.M)
FLAG = re.compile(r"(?i)\bPBAT\b|post-bisulfite|non-?directional|single[- ]cell|low[- ]input|\b\d{2,4} cells\b|EM-?seq|enzymatic methyl|clip(ped|_r[12])")
def get(d, k):
    for p in k.split("."): d = d.get(p) if isinstance(d, dict) else None
    return d
def num(x):
    m = re.search(r"-?\d+(\.\d+)?(e-?\d+)?", str(x)); return float(m.group(0)) if m else None
def reserve_findings(S, name):
    """2026-11-13 (author: derive first, then commission; preserve data): a set marked reserved in CANON/data_spent.csv may be named only by a
    commissioning row (COMMISSION-, COM-); derivation and development tests refuse it."""
    import csv as _c
    if re.match(r"(?i)(COMMISSION|COM)[-_]", os.path.basename(str(name))): return []
    rows = list(_c.DictReader(open(os.path.join(R, "CANON", "data_spent.csv"), encoding="utf-8")))
    res = [r["dataset"].split()[0] for r in rows if str(r.get("spent_status", "")).lower().startswith("reserved")]
    text = json.dumps(S.get("data", {}).get("set", ""))
    return [f"{name}: data.set names {g}, reserved for commissioning in CANON/data_spent.csv; derivations may not spend it" for g in res if g and g in text]


def explore_findings(S, name):
    """2026-11-13 lessons 2-3: seal only what a rehearsal on spent data supports; branches are set by independent measurements."""
    F = []; E = S.get("explore") or {}; P = S.get("prediction") or {}
    pull = num(E.get("rehearsal_pull"))
    if pull is None: F.append(f"{name}: explore.rehearsal_pull must be a number: the sealed scorer's largest |pull| on spent data")
    elif abs(pull) > 2: F.append(f"{name}: the rehearsal on spent data gives |pull| {abs(pull):.2f} > 2: fix the prediction before spending reserved data")
    d = P.get("decided_by")
    if isinstance(d, dict):
        for k, v in d.items():
            if re.search(r"(?i)\b(outcome|result|the reading itself|whichever|if it reads)\b", str(v)):
                F.append(f"{name}: prediction.decided_by['{k}'] is chosen by the outcome; name an independent measurement")
    return F


def form_findings(S, name):
    """data.form (2026-11-13): a list of {form, hours, adequate, chosen}; the chosen one must be the cheapest adequate, or carry a reason."""
    F = []; L = (S.get("data") or {}).get("form")
    if not isinstance(L, list) or len(L) < 2:
        return [f"{name}: data.form must list at least two forms the data exist in (held counts, .pat, BAM, per-site calls, raw reads) with hours"]
    ok = [x for x in L if isinstance(x, dict) and x.get("adequate") is True and num(x.get("hours")) is not None]
    ch = [x for x in L if isinstance(x, dict) and x.get("chosen")]
    if len(ch) != 1: return F + [f"{name}: data.form must mark exactly one chosen form"]
    c = ch[0]
    if c.get("adequate") is not True: F.append(f"{name}: data.form: the chosen form is not marked adequate")
    cheaper = [x for x in ok if x is not c and num(x["hours"]) < num(c.get("hours") or 1e9)]
    if cheaper and not str(c.get("why_not_cheaper") or "").strip():
        F.append(f"{name}: data.form: '{cheaper[0].get('form')}' is adequate and faster than the chosen '{c.get('form')}'; say why_not_cheaper")
    return F


def sheet_findings(S, name="sheet"):
    F = []
    for k in KEYS:
        if k in ("machines.why_not_github", "value.why_this"): continue
        v = get(S, k)
        if v in (None, "", [], {}): F.append(f"{name}: '{k}' missing ({k} in CANON/TEST_DESIGN_CHECKLIST.md)")
    M = S.get("machines") or {}
    if str(M.get("platform", "")).lower() == "google" and not M.get("why_not_github"): F.append(f"{name}: google without machines.why_not_github")
    par, peak, mem = num(M.get("parallel_per_machine")), num(M.get("peak_mem_gb")), num(M.get("machine_mem_gb"))
    if "not measured" in str(M.get("peak_mem_gb", "")).lower():
        if par and par > 1: F.append(f"{name}: peak memory not measured but {int(par)} jobs per machine")
    elif None not in (par, peak, mem) and par * peak > mem: F.append(f"{name}: {int(par)} x {peak} GB > {mem} GB machine")
    sp, pr, ex, tl = num(M.get("s_per_pair")), num(M.get("pairs")), num(M.get("expected_hours")), num(M.get("time_limit_hours"))
    if None not in (sp, pr, ex) and ex < sp * pr / 3600 * 0.95: F.append(f"{name}: expected_hours {ex} below the measured {sp * pr / 3600:.2f} h of aligning alone")
    if None not in (ex, tl) and tl < 1.3 * ex: F.append(f"{name}: time limit {tl} h under 1.3 x expected {ex} h")
    if num(get(S, "watchers.count")) != 1: F.append(f"{name}: watchers.count must be 1")
    C = get(S, "canary.instrument_check") or {}
    ALIGN, COUNT = ("qc_failed_max", "mapped_min", "s_per_pair_max"), ("rows_min", "halves_corr_min", "hours_per_sample_max")
    if not (isinstance(C, dict) and any(all(num(C.get(k)) is not None for k in ks) for ks in (ALIGN, COUNT))):
        F.append(f"{name}: canary.instrument_check needs numbers: {', '.join(ALIGN)} (alignment jobs) or {', '.join(COUNT)} (count-only jobs)")
    if FLAG.search(str(get(S, "data.protocol_read") or "")) and not get(S, "data.method_ok"):
        F.append(f"{name}: the deposited method is flagged ({FLAG.search(str(get(S, 'data.protocol_read'))).group(0)}) and data.method_ok does not say how the read path handles it")
    br = str(get(S, "bars.rule") or "")
    if br and not (re.search(r"(?i)within 2", br) and re.search(r"(?i)beyond 3", br) and re.search(r"(?i)watch", br)):
        F.append(f"{name}: bars.rule must use the agreed classes (WHY_IAM.md rule 4: within 2 passes; 2 to 3 watch; beyond 3 a real miss)")
    O = get(S, "value.options")
    if not (isinstance(O, list) and len(O) >= 2 and all(isinstance(o, dict) and num(o.get("hours")) is not None and num(o.get("cost_usd")) is not None and "adequate" in o for o in O)):
        F.append(f"{name}: value.options needs at least two options, each with hours, cost_usd and adequate")
    else:
        ch = [o for o in O if o.get("chosen")]
        if len(ch) != 1: F.append(f"{name}: value.options must mark exactly one option chosen")
        else:
            c = ch[0]
            if not c.get("adequate"): F.append(f"{name}: the chosen option is not adequate")
            better = [o.get("name", "?") for o in O if o is not c and o.get("adequate") and num(o["hours"]) <= num(c["hours"]) and num(o["cost_usd"]) <= num(c["cost_usd"]) and (num(o["hours"]) < num(c["hours"]) or num(o["cost_usd"]) < num(c["cost_usd"]))]
            if better and not str(get(S, "value.why_this") or "").strip(): F.append(f"{name}: {better} is adequate and at least as fast and cheap; value.why_this must say why not")
        if not any("cosmolog" in str(o.get("name", "")).lower() or o.get("cosmologist") for o in O): F.append(f"{name}: value.options has no cosmologist's way")
    P = get(S, "data.preflight")
    if not (isinstance(P, dict) and str(P.get("checked", "")).strip() and num(P.get("minutes")) is not None and str(P.get("result", "")).strip()):
        F.append(f"{name}: data.preflight needs checked, minutes and result (run it before the canary)")
    elif num(P.get("minutes")) > 30: F.append(f"{name}: data.preflight took {num(P.get('minutes'))} min; a preflight is a small slice (30 min at most)")
    elif re.search(r"(?i)\b(missing|refused|absent|not present)\b|(?<![-\w])failed\b(?! share)", str(P.get("result"))) and not str(P.get("resolved", "")).strip():
        F.append(f"{name}: data.preflight found a problem and data.preflight.resolved does not say how it was settled")
    if S.get("data", {}).get("form") is not None or not S.get("_after_the_fact"): F += form_findings(S, name)
    if not S.get("_after_the_fact"): F += explore_findings(S, name)
    F += reserve_findings(S, name)
    return F
def git(*a): return subprocess.run(["git", "-C", R] + list(a), capture_output=True, text=True).stdout
if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "sheet":
        S = json.load(open(sys.argv[2])); F = sheet_findings(S, os.path.basename(sys.argv[2]))
        if str(S.get("_written", "")).startswith("AFTER THE FACT") and F:   # a record of a test designed before this check: its misses are listed, not hidden
            print(f"design_check: after the fact, {len(F)} miss(es) recorded"); [print("  miss: " + f) for f in F]; sys.exit(0)
    else:
        base = sys.argv[2] if len(sys.argv) > 2 else "origin/main"
        old = {r.get("id") for r in json.loads(git("show", f"{base}:CANON/manifest.json") or '{"rows": []}')["rows"]}
        new = json.load(open(os.path.join(R, "CANON", "manifest.json"), encoding="utf-8"))["rows"]
        F = []
        for r in new:
            if r.get("id") in old: continue
            if not str(r.get("id", "")).startswith(TEST_PREFIX): continue   # tests only; tooling rows (OTH) are not tests
            if str(r.get("status")) not in ("sealed", "designed", "planned", "synthetic_done", "running", "run"): continue
            p = os.path.join(R, "design", f"{r['id']}.json")
            if not os.path.exists(p): F.append(f"{r['id']}: new row without design/{r['id']}.json (CANON/TEST_DESIGN_CHECKLIST.md)"); continue
            F += sheet_findings(json.load(open(p)), r["id"])
    print("design_check: " + ("ok" if not F else "REFUSED (%d)" % len(F))); [print("  " + f) for f in F]; sys.exit(1 if F else 0)
