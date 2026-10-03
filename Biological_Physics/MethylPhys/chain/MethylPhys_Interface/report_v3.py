#!/usr/bin/env python3
"""Report for conductor v3 (development build, neutrophils only). One self-contained HTML page from the v3 bundle.
The gauge marker is the tared reading (tare.A_rel) whenever Stage T produced one, for whole blood and isolated neutrophils alike.
Untared isolated neutrophils: the own-floor A, labelled untared. Untared whole blood: no gauge position (the number is printed).

Sections (SOP v3 section 2b, stage 13 - the v3 report carries every section the retired v2 report carried that still applies):
  the reading (Stage 0 intake, Stage 1, composition, Met-A, tare, noise gate, C-score, IAM-A when sequencing input is given),
  red flags (STOP / WITHHELD / CAUTION / NOTE, also written into the bundle as `red_flags`), safeguards (rendered-claim scan,
  formula self-test, anchors, deconvolver conformance, atlas separability), troubleshooting, integrity (file hashes), the chain's
  file inventory, run it yourself, the stage table (SOP 2b, 14 stages) and the toolkit table (PASS / FAIL / NOT_RUN / NOT_BUILT).
Every section carries an HTML id listed in REPORT_SECTIONS so a harness can check the page without parsing prose."""
import html, json, os, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); CHAIN = os.path.dirname(HERE)
REPORT_SECTIONS = ("sec-reading-intake", "sec-composition", "sec-met-a", "sec-tare", "sec-noise-gate", "sec-cscore", "sec-red-flags",
                   "sec-safeguards", "sec-troubleshooting", "sec-integrity", "sec-inventory", "sec-run-yourself", "sec-stages",
                   "sec-toolkit", "sec-withheld", "sec-bundle")
# SOP v3 section 2b - the 14 stages and their status on this build (kept in step with the SOP status column)
STAGES_2B = [("0", "Intake", "running"), ("1", "Calibration", "running"), ("2", "Composition, blood groups", "running"),
             ("3", "Atlas deconvolution", "toolkit"), ("4", "NILC component separation", "toolkit"), ("5", "Met-A", "running (neutrophils)"),
             ("6", "C-score", "running (band not set)"), ("7", "IAM-A", "running (development)"), ("8", "Same-run tare", "running"),
             ("9", "Noise gate", "running"), ("10", "Directional decomposition", "toolkit"), ("11", "Sky map", "toolkit"),
             ("12", "Sky statistics", "toolkit"), ("13", "Report", "running"),
             ("3b", "Trace-cell detection", "toolkit"), ("3c", "Foreign-cell detection", "toolkit"), ("11b", "Surface brightness", "toolkit"),
             ("12b", "Difference map", "running with --prior-betas (per-address difference; the sky drawing is not built)")]
# toolkit stages: (stage, name, module, bundle key written when the stage is wired and runs)
TOOLKIT_STAGES = [("3", "Atlas deconvolution", "chain/deconv_v2.py", "atlas_composition"),
                  ("3b", "Trace-cell detection", "chain/stage_2c_trace_detection.py", "trace_detection"),
                  ("3c", "Foreign-cell detection", "chain/toolkit_foreign_detection.py", "foreign_detection"),
                  ("4", "NILC component separation", "chain/nilc_celltype_deconvolver.py", "nilc"),
                  ("10", "Directional decomposition", "chain/Runtime Matrices/Directional Panel/bidirectional_decomposition.py", "directional"),
                  ("11", "Sky map", "chain/stage_4_6_patient_cmb.py", "sky_map"),
                  ("11b", "Surface brightness", "chain/toolkit_surface_brightness.py", "surface_brightness"),
                  ("12", "Sky statistics", "chain/sky_statistics.py", "sky_statistics"),
                  ("12b", "Difference map", "chain/serial_mode.py", "difference_map")]   # 12b wired 2026-10-03 (DEV-TOOLKIT-ADDED-01 passed)
NOT_BUILT = ["look-elsewhere by simulation over the sky", "apodised mask", "beam smoothing", "cell-type covariance in the separation (GLS)",
             "Fisher degeneracy of the composition", "ILC on the residual sky", "per-specimen composition posterior",
             "cross-spectra between cell panels", "difference map drawn as a sky"]
# words the report prose must not carry (no disease, cohort or population vocabulary; the instrument reads one cell against itself)
CLAIM_SCAN = r"\b(cancer|tumou?r|leuka?emia|lymphoma|carcinoma|malignan\w*|diagnos(?!tic test)\w*|disease|cohort|population|percentile|age-matched|risk|prognos\w*|patient)\b"


def _gauge(A, lo=0.80, hi=1.30, w=520, label=""):
    if A is None: return f"<p><i>{html.escape(label) or 'A not reported'}</i></p>"
    x = lambda v: int((min(max(v, lo), hi) - lo) / (hi - lo) * w)
    return (f'<p style="margin:2px 0"><b>{html.escape(label)}</b></p><svg width="{w+20}" height="46"><rect x="10" y="14" width="{w}" height="14" fill="#eee"/>'
            f'<rect x="{10+x(0.95)}" y="14" width="{x(1.05)-x(0.95)}" height="14" fill="#9fd39f"/>'
            f'<line x1="{10+x(A)}" y1="6" x2="{10+x(A)}" y2="36" stroke="#c0392b" stroke-width="3"/>'
            f'<text x="{10+x(0.95)}" y="44" font-size="10">0.95</text><text x="{10+x(1.05)}" y="44" font-size="10">1.05</text></svg>')


def _H(b):
    b = np.clip(np.asarray(b, dtype="float64"), 1e-6, 1 - 1e-6)
    return -(b * np.log2(b) + (1 - b) * np.log2(1 - b))


def _load(rel):
    try: return json.load(open(os.path.join(CHAIN, rel)))
    except Exception: return None


def red_flags(o):
    """STOP / WITHHELD / CAUTION / NOTE, from the bundle only. Returned as a list of {level, code, text}."""
    m, t, it = o.get("met_a") or {}, o.get("tare") or {}, o.get("intake") or {}
    c, q = o.get("met_a_cscore") or {}, o.get("iam_a") or {}
    F = []
    add = lambda lv, code, txt: F.append({"level": lv, "code": code, "text": txt})
    if o.get("refusal"): add("STOP", "PLATFORM_REFUSED", o["refusal"])
    if q.get("refusal"): add("STOP", "IAM_A_REFUSED", q["refusal"])
    if m.get("reason"): add("WITHHELD", "A_WITHHELD", m["reason"])
    if str(m.get("state", "")).startswith("withheld"): add("WITHHELD", "NOISE_GATE", m["state"])
    if m.get("A") is not None and t.get("A_rel") is None and m.get("specimen") == "whole blood":
        add("WITHHELD", "WHOLE_BLOOD_UNTARED", "whole blood without >= 3 same-run references: no gauge position (SOP section 3 rule 2)")
    for w in o.get("withheld", []): add("WITHHELD", "OUT_OF_SCOPE", w)
    if it.get("stage0_verdict") == "PROCEED_WITH_PENALTY": add("CAUTION", "INTAKE_PENALTY", f"Stage 0 borderline: {it.get('stage0_borderline')}")
    if o.get("intake_skipped") or (not it and o.get("met_a")): add("CAUTION", "INTAKE_NOT_RUN", "Stage 0 intake was not run on this specimen")
    for fl in it.get("flags", []) or []:
        if str(fl).startswith("STAGE1_"): add("CAUTION", "STAGE1_QC", str(fl))
    if m.get("past_entropy_ceiling"): add("CAUTION", "ENTROPY_CEILING", "methylated sites past beta 0.5: A falls as loss continues; read beta, not A")
    if m.get("noise_gate") == "above the reference arrays' range" and t.get("A_rel") is not None:
        add("NOTE", "NOISE_ABOVE_RANGE_TARED", f"noise index {m.get('noise_index')} above N_max {m.get('noise_gate_N_max')}; reading is tared, so the gate does not withhold")
    if c.get("C") is not None: add("NOTE", "CSCORE_BAND_NOT_SET", "C-score printed for development; healthy band not set")
    add("NOTE", "DEVELOPMENT_BUILD", str(o.get("build", "development build")) + "; not a diagnostic test")
    return F


def safeguards(o, prose):
    """Rendered-claim scan, formula self-test, anchors, deconvolver conformance, atlas separability. Each PASS / FAIL / NOT_RUN."""
    m, t, a, c = o.get("met_a") or {}, o.get("tare") or {}, o.get("composition") or {}, o.get("met_a_cscore") or {}
    S = []
    hits = sorted(set(x.lower() for x in re.findall(CLAIM_SCAN, prose, flags=re.I)))
    S.append(("rendered-claim scan", "PASS" if not hits else "FAIL", "no disease, cohort or population words in the report prose" if not hits else f"found: {hits}"))
    ok = abs(_H(0.5) - 1) < 1e-9 and abs(_H(0.25) - 0.8112781244591328) < 1e-9 and _H(1e-6) < 3e-5
    det = "H(0.5) = 1, H(0.25) = 0.811278, H(0) -> 0"
    if t.get("A_rel") is not None and m.get("A") is not None and t.get("reference_median"):
        r = abs(round(m["A"] / t["reference_median"], 4) - t["A_rel"]) <= 2e-4; ok &= r; det += f"; A_rel = A / reference median ({'agrees' if r else 'DISAGREES'})"
    if m.get("A") is not None and m.get("state") and t.get("A_rel") is not None:
        st = "Normal" if 0.95 <= t["A_rel"] <= 1.05 else ("above Normal" if t["A_rel"] > 1.05 else "below Normal")
        r = t.get("state") == st; ok &= r; det += f"; tare state matches the band ({'yes' if r else 'NO'})"
    S.append(("formula self-test", "PASS" if ok else "FAIL", det))
    anc, bad = [], []
    try:
        import stage_m_met_a as SM
        anc.append(f"Normal {SM.NORMAL[0]}-{SM.NORMAL[1]}"); (bad.append("Normal band") if tuple(SM.NORMAL) != (0.95, 1.05) else None)
    except Exception as e: bad.append(f"stage_m_met_a not importable ({type(e).__name__})")
    R = _load("Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_1.json"); G = _load("Runtime Matrices/Met_A_Floors/noise_gate_EPIC_v1.json")
    if R is not None and c.get("healthy_baseline") is not None:
        anc.append(f"C-score baseline {R['healthy_clustering_median']}"); (bad.append("C baseline") if R["healthy_clustering_median"] != c["healthy_baseline"] else None)
    if G is not None and m.get("noise_gate_N_max") is not None:
        anc.append(f"N_max {G['N_max']}"); (bad.append("N_max") if G["N_max"] != m["noise_gate_N_max"] else None)
    if m.get("floor") is not None:
        Fl = _load("Runtime Matrices/Met_A_Floors/metA_floors_v1_3.json")
        try:
            f0 = Fl["platforms"]["EPIC"]["neutrophils"]["floor"]; anc.append(f"neutrophil floor {round(f0, 5)}")
            (bad.append("floor") if abs(round(f0, 5) - m["floor"]) > 1e-9 else None)
        except Exception: bad.append("floor not readable")
    S.append(("anchors", "PASS" if anc and not bad else ("FAIL" if bad else "NOT_RUN"), ("; ".join(anc) + (f" | DIFFER: {bad}" if bad else "")) or "no anchor applies"))
    fr = a.get("fractions")
    if fr:
        s, mn = sum(fr.values()), min(fr.values())
        r = abs(s - 1) < 1e-6 and mn >= 0
        S.append(("deconvolver conformance", "PASS" if r else "FAIL", f"fractions sum {s:.6f}, min {mn:.4f}, residual MAE {a.get('residual_mae')}, markers {a.get('n_markers_used')} of {a.get('n_markers_total')}"))
    else:
        S.append(("deconvolver conformance", "NOT_RUN", str(a.get("note") or a.get("reason") or "no composition on this specimen")))
    ac = o.get("atlas_composition") or {}
    if ac.get("separability"):
        S.append(("atlas separability", "PASS" if all(v.get("separable") for v in ac["separability"].values()) else "FAIL", f"{sum(v.get('separable', False) for v in ac['separability'].values())} of {len(ac['separability'])} cells separable"))
    else:
        S.append(("atlas separability", "NOT_RUN", "stage 3 (atlas deconvolution) is not wired into this build"))
    return S


def troubleshooting(o):
    m, t, it = o.get("met_a") or {}, o.get("tare") or {}, o.get("intake") or {}
    T = []
    if o.get("refusal"): T.append(("platform refused", "v3 reads EPIC v1 arrays only; a 450K or EPIC v2 array needs its own frozen floor first"))
    if "fraction" in str(m.get("reason", "")): T.append(("A withheld for fraction", "the neutrophil fraction is below the read line (0.20); the fraction is printed; no action changes this on this specimen"))
    if "sites measured" in str(m.get("reason", "")) or "markers measured" in str(m.get("reason", "")):
        T.append(("too few sites or markers measured", "check Stage 1 detection: probes at background are removed before the reading; re-hybridise if the array is low-signal"))
    if m.get("A") is not None and t.get("A_rel") is None:
        T.append(("untared", "run >= 3 healthy references of the same specimen type on the same slide (else the same batch) and pass their untared A with --slide-ref-A or --slide-ref-table"))
    if str(m.get("state", "")).startswith("withheld"): T.append(("noise gate", "the array's noise index is above the reference arrays' range; tare it against same-run references to get a gauge state"))
    if m.get("past_entropy_ceiling"): T.append(("entropy ceiling", "the methylated sites have fallen past beta 0.5; read the mean beta, not A"))
    if it.get("stage0_verdict") == "PROCEED_WITH_PENALTY": T.append(("intake penalty", f"borderline {it.get('stage0_borderline')}: the reading is printed with this flag"))
    if o.get("intake_skipped"): T.append(("intake not run", "the specimen was read without Stage 0 (beta table or --no-intake); intake hashes and QC are absent"))
    if not T: T.append(("none", "no condition on this specimen needs operator action"))
    return T


def toolkit_table(o):
    out = []
    for st, name, mod, key in TOOLKIT_STAGES:
        rec = o.get(key)
        if rec is None: out.append((st, name, mod, "NOT_RUN", "built; not wired into this build, or not run on this specimen"))
        elif isinstance(rec, dict) and (rec.get("error") or rec.get("status") == "FAIL"): out.append((st, name, mod, "FAIL", str(rec.get("error") or rec.get("reason"))[:160]))
        elif isinstance(rec, dict) and rec.get("status") == "REFUSED": out.append((st, name, mod, "REFUSED", str(rec.get("reason"))[:160]))
        else: out.append((st, name, mod, "PASS", "ran on this specimen" + (f" ({rec.get('status')})" if isinstance(rec, dict) and rec.get("status") else "")))
    out += [("12", n, "-", "NOT_BUILT", "listed under stage 12 as a tool to build") for n in NOT_BUILT]
    return out


def _table(rows, head):
    e = html.escape
    return ("<table border='1' cellpadding='3' style='border-collapse:collapse;font-size:12px'><tr>" + "".join(f"<th>{e(h)}</th>" for h in head) + "</tr>"
            + "".join("<tr>" + "".join(f"<td>{e(str(x))}</td>" for x in r) + "</tr>" for r in rows) + "</table>")


def build(o, out, sid):
    m, c, t, a, intake = o.get("met_a") or {}, o.get("met_a_cscore") or {}, o.get("tare") or {}, o.get("composition") or {}, o.get("intake") or {}
    q = o.get("iam_a") or {}; s1 = intake.get("stage1_qc") or {}
    e = html.escape
    rows = "".join(f"<tr><td>{e(k)}</td><td>{v*100:.1f} %</td></tr>" for k, v in sorted((a.get("fractions") or {}).items(), key=lambda kv: -kv[1]) if v >= 0.01)
    if str(m.get("state", "")).startswith("withheld"): gA, glabel = None, f"gauge not drawn - {m['state']}"   # the noise gate withholds the state: no gauge position
    elif t.get("A_rel") is not None: gA, glabel = t["A_rel"], f"tared: A_rel {t['A_rel']} ({t.get('state')})"
    elif m.get("A") is not None and m.get("specimen") != "whole blood": gA, glabel = m["A"], f"untared: A {m['A']} against the own floor (no same-run references)"
    elif m.get("A") is not None: gA, glabel = None, "whole blood, untared: no gauge position until Stage T (supply >= 3 same-run references)"
    else: gA, glabel = None, ""
    P = [f"<div style='background:#fff3cd;padding:8px;border:1px solid #e0c060'><b>{e(o.get('build',''))}</b>. Not a diagnostic test.</div>",
         f"<h1>Cellular Performance Gauge - {e(sid)}</h1><p>Run {e(str(o.get('run_id')))} | specimen: {e(o.get('specimen',''))} | platform {e(str(o.get('platform','')))} | array type {e(str(o.get('array_type')))} | floors {e(str(o.get('floors_version')))} | reference {e(str(o.get('reference_version')))}</p>",
         (f"<p style='color:#a00'><b>Refused:</b> {e(o['refusal'])}</p>" if o.get("refusal") else ""),
         f"<h2 id='sec-reading-intake'>Stage 0 intake</h2><p>verdict: <b>{e(str(intake.get('stage0_verdict','not run')))}</b> | call rate: {e(str(intake.get('call_rate_status','-')))} {e(str(intake.get('call_rate','')))} | flags: {e(', '.join(map(str,intake.get('flags',[])))[:300])}</p>"
         + (f"<p>Stage 1 (recorded, not gated): poobah detection {e(str(s1.get('detection_qc')))} {e(str(s1.get('pct_probes_detected','')))} | call rate {e(str(s1.get('call_rate_status', s1.get('call_rate_note','-'))))} {e(str(s1.get('call_rate','')))} | controls {e(str(s1.get('ctrl_qc')))}</p>" if s1 else (f"<p>Stage 1 (recorded, not gated): not run - {e(str(o.get('refusal') or 'no Stage 1 record on this specimen'))}</p>" if intake else "")),
         "<h2 id='sec-composition'>Stage 2 composition (8 blood groups)</h2>" + (f"<table>{rows}</table>" if rows else f"<p>{e(str(a.get('note', a.get('reason',''))))}</p>"),
         f"<h2 id='sec-met-a'>Stage 5 Met-A - neutrophils</h2>{_gauge(gA, label=glabel)}<p>A = <b>{m.get('A')}</b> ({e(str(m.get('state', m.get('reason',''))))}); "
         f"neutrophil fraction {m.get('fraction')}; sites {m.get('n_sites')}; {e(str(m.get('expectation') or ('composition-matched healthy expectation (not computed: A withheld)' if m.get('specimen') == 'whole blood' else 'own floor')))}; shift per 1 % loss {m.get('shift_per_1pct_loss')}</p>"
         f"<p>Methylated sites mean beta {m.get('methylated_sites_mean_beta')}" + (" - <b>past the entropy ceiling: A falls as loss continues; read beta, not A</b>" if m.get('past_entropy_ceiling') else "") + "</p>",
         f"<h2 id='sec-tare'>Stage 8 same-run tare</h2><p>A_rel = <b>{t.get('A_rel')}</b> {e(str(t.get('state', t.get('reason',''))))}"
         f" | references {t.get('n_refs')} (median {t.get('reference_median')}) | detection limit: <b>{t.get('detection_limit_pct_loss')}</b> % loss of the neutrophil pattern (reference spread {t.get('reference_spread_sd')})</p>"
         f"<p>Tare method: {e(str(t.get('method', '-')))}</p>",
         f"<h2 id='sec-noise-gate'>Stage 9 noise gate</h2><p>noise index N {m.get('noise_index')} ({m.get('noise_sites_measured')} of {m.get('noise_sites_total')} noise sites) | N_max {m.get('noise_gate_N_max')} | gate: {e(str(m.get('noise_gate')))}</p>",
         f"<h2 id='sec-cscore'>Stage 6 Met-A C-score</h2><p>C = <b>{c.get('C')}</b> (healthy = 1; healthy held-out range {c.get('healthy_range')}); {e(str(c.get('status', c.get('reason',''))))}</p>",
         (f"<h2 id='sec-iam-a'>Stage 7 IAM-A - {e(str(q.get('cell')))}</h2>{_gauge(q.get('A'), label=('IAM-A ' + str(q.get('A'))) if q.get('A') is not None else '')}"
          f"<p>IAM-A = <b>{q.get('A')}</b> ({e(str(q.get('state', q.get('refusal',''))))}); pipeline {e(str(q.get('pipeline')))}; copy error eps {q.get('eps')}; position P {q.get('P')}; "
          f"eps0 {q.get('eps0')}; halves {e(str(q.get('halves')))}; opportunities {q.get('opportunities')}; E = {q.get('E_kT')} kT</p>" if q else ""),
         (f"<h2 id='sec-difference-map'>Stage 12b difference map (two draws of one person)</h2><p>{e(str(o['difference_map'].get('status')))}: "
          + (f"{o['difference_map'].get('n_addresses')} addresses both draws measured; median delta beta {o['difference_map'].get('median_dbeta')}; mean |delta beta| {o['difference_map'].get('mean_abs_dbeta')}; "
             f"q99 |delta beta| {o['difference_map'].get('q99_abs_dbeta')}; prior run {e(str(o['difference_map'].get('prior_run_id')))}. {e(o['difference_map'].get('note', ''))}"
             if o["difference_map"].get("status") == "OK" else e(str(o["difference_map"].get("reason")))) + "</p>" if o.get("difference_map") else ""),
         "<h2 id='sec-withheld'>Withheld</h2><ul>" + "".join(f"<li>{e(w)}</li>" for w in o.get("withheld", [])) + "</ul>"]
    F = red_flags(o); o["red_flags"] = F
    P.append("<h2 id='sec-red-flags'>Red flags</h2>" + _table([(f["level"], f["code"], f["text"]) for f in sorted(F, key=lambda f: ["STOP", "WITHHELD", "CAUTION", "NOTE"].index(f["level"]))], ("level", "code", "what")))
    TS = troubleshooting(o)
    P.append("<h2 id='sec-troubleshooting'>Troubleshooting</h2>" + _table(TS, ("condition", "what to do")))
    v = o.get("versions") or {}
    integ = [("Grn IDAT", intake.get("grn_sha256", "intake not run")), ("Red IDAT", intake.get("red_sha256", "intake not run")),
             ("chain commit", f"{v.get('chain_commit')}{' (uncommitted changes)' if v.get('chain_dirty') else ''}"), ("methylprep", v.get("methylprep")), ("python", v.get("python"))]
    P.append("<h2 id='sec-integrity'>Integrity</h2>" + _table(integ, ("item", "sha256 / version")))
    P.append("<h2 id='sec-inventory'>Chain file inventory</h2>" + _table([(k, d.get("sha256_12"), d.get("bytes")) for k, d in sorted((v.get("inputs") or {}).items())], ("file (relative to chain/)", "sha256 (12)", "bytes")))
    cmd = o.get("command")
    P.append("<h2 id='sec-run-yourself'>Run it yourself</h2><p>From <code>Biological_Physics/MethylPhys/chain/MethylPhys_Interface</code> at the chain commit above:</p><pre>"
             + e("python " + " ".join((f'"{x}"' if " " in str(x) else str(x)) for x in cmd) if cmd else "command not recorded in the bundle") + "</pre>"
             + "<p>The frozen inputs are listed in the inventory with their hashes; a reading made from the same IDATs, commit and inputs is identical.</p>")
    P.append("<h2 id='sec-stages'>The chain (SOP v3 section 2b)</h2>" + _table(STAGES_2B, ("#", "stage", "status on this build")))
    P.append("<h2 id='sec-toolkit'>Toolkit stages</h2>" + _table(toolkit_table(o), ("stage", "name", "module", "status", "note")))
    prose = re.sub(r"<[^>]+>", " ", re.sub(r"<pre>.*?</pre>", " ", "\n".join(P), flags=re.S))   # the command line (flag names) is not prose
    SG = safeguards(o, prose); o["safeguards"] = [{"check": s[0], "result": s[1], "detail": s[2]} for s in SG]
    if any(s[1] == "FAIL" for s in SG):
        F.append({"level": "CAUTION", "code": "SAFEGUARD_FAIL", "text": "; ".join(f"{s[0]}: {s[2]}" for s in SG if s[1] == "FAIL")})
        P[P.index(next(x for x in P if x.startswith("<h2 id='sec-red-flags'>")))] = "<h2 id='sec-red-flags'>Red flags</h2>" + _table([(f["level"], f["code"], f["text"]) for f in sorted(F, key=lambda f: ["STOP", "WITHHELD", "CAUTION", "NOTE"].index(f["level"]))], ("level", "code", "what"))
    P.insert(P.index(next(x for x in P if x.startswith("<h2 id='sec-troubleshooting'>"))), "<h2 id='sec-safeguards'>Safeguards</h2>" + _table(SG, ("safeguard", "result", "detail")))
    H = [f"<html><head><meta charset='utf-8'><title>CPG v3 - {e(sid)}</title></head><body style='font-family:sans-serif;max-width:900px'>"] + P + [
         "<details id='sec-bundle'><summary>bundle (JSON, also written beside this page)</summary><pre>" + e(json.dumps({k: v for k, v in o.items() if k != 'intake'}, indent=1, default=str)[:20000]) + "</pre></details></body></html>"]
    open(out, "w", encoding="utf-8").write("\n".join(H))
    ref = [x for x in (o.get("refusal"), q.get("refusal")) if x]
    return {"out": out, "bytes": sum(len(x) for x in H), "refusals": ref}
