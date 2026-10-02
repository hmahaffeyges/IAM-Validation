#!/usr/bin/env python3
"""Report for conductor v3 (development build, neutrophils only). One self-contained HTML page from the v3 bundle.
The gauge marker is the tared reading (tare.A_rel) whenever Stage T produced one, for whole blood and isolated neutrophils alike.
Untared isolated neutrophils: the own-floor A, labelled untared. Untared whole blood: no gauge position (the number is printed)."""
import html, json
def _gauge(A, lo=0.80, hi=1.30, w=520, label=""):
    if A is None: return f"<p><i>{html.escape(label) or 'A not reported'}</i></p>"
    x = lambda v: int((min(max(v, lo), hi) - lo) / (hi - lo) * w)
    return (f'<p style="margin:2px 0"><b>{html.escape(label)}</b></p><svg width="{w+20}" height="46"><rect x="10" y="14" width="{w}" height="14" fill="#eee"/>'
            f'<rect x="{10+x(0.95)}" y="14" width="{x(1.05)-x(0.95)}" height="14" fill="#9fd39f"/>'
            f'<line x1="{10+x(A)}" y1="6" x2="{10+x(A)}" y2="36" stroke="#c0392b" stroke-width="3"/>'
            f'<text x="{10+x(0.95)}" y="44" font-size="10">0.95</text><text x="{10+x(1.05)}" y="44" font-size="10">1.05</text></svg>')
def build(o, out, sid):
    m, c, t, a, intake = o.get("met_a") or {}, o.get("met_a_cscore") or {}, o.get("tare") or {}, o.get("composition") or {}, o.get("intake") or {}
    q = o.get("iam_a") or {}; s1 = intake.get("stage1_qc") or {}
    e = html.escape
    rows = "".join(f"<tr><td>{e(k)}</td><td>{v*100:.1f} %</td></tr>" for k, v in sorted((a.get("fractions") or {}).items(), key=lambda kv: -kv[1]) if v >= 0.01)
    if t.get("A_rel") is not None: gA, glabel = t["A_rel"], f"tared: A_rel {t['A_rel']} ({t.get('state')})"
    elif m.get("A") is not None and m.get("specimen") != "whole blood": gA, glabel = m["A"], f"untared: A {m['A']} against the own floor (no same-run references)"
    elif m.get("A") is not None: gA, glabel = None, "whole blood, untared: no gauge position until Stage T (supply >= 3 same-run references)"
    else: gA, glabel = None, ""
    H = [f"<html><head><meta charset='utf-8'><title>CPG v3 - {e(sid)}</title></head><body style='font-family:sans-serif;max-width:900px'>",
         f"<div style='background:#fff3cd;padding:8px;border:1px solid #e0c060'><b>{e(o.get('build',''))}</b>. Not a diagnostic test.</div>",
         f"<h1>Cellular Performance Gauge - {e(sid)}</h1><p>Specimen: {e(o.get('specimen',''))} | platform {e(str(o.get('platform','')))} | array type {e(str(o.get('array_type')))} | floors {e(str(o.get('floors_version')))} | reference {e(str(o.get('reference_version')))}</p>",
         (f"<p style='color:#a00'><b>Refused:</b> {e(o['refusal'])}</p>" if o.get("refusal") else ""),
         f"<h2>Stage 0 intake</h2><p>verdict: <b>{e(str(intake.get('stage0_verdict','not run')))}</b> | call rate: {e(str(intake.get('call_rate_status','-')))} {e(str(intake.get('call_rate','')))} | flags: {e(', '.join(map(str,intake.get('flags',[])))[:300])}</p>"
         + (f"<p>Stage 1 (recorded, not gated): poobah detection {e(str(s1.get('detection_qc')))} {e(str(s1.get('pct_probes_detected','')))} | call rate {e(str(s1.get('call_rate_status', s1.get('call_rate_note','-'))))} {e(str(s1.get('call_rate','')))} | controls {e(str(s1.get('ctrl_qc')))}</p>" if s1 else ""),
         "<h2>Stage A composition</h2>" + (f"<table>{rows}</table>" if rows else f"<p>{e(str(a.get('note', a.get('reason',''))))}</p>"),
         f"<h2>Stage M Met-A - neutrophils</h2>{_gauge(gA, label=glabel)}<p>A = <b>{m.get('A')}</b> ({e(str(m.get('state', m.get('reason',''))))}); "
         f"neutrophil fraction {m.get('fraction')}; sites {m.get('n_sites')}; {e(str(m.get('expectation','own floor')))}; shift per 1 % loss {m.get('shift_per_1pct_loss')}</p>",
         f"<h2>Stage T same-run tare</h2><p>A_rel = <b>{t.get('A_rel')}</b> {e(str(t.get('state', t.get('reason',''))))}"
         f" | references {t.get('n_refs')} (median {t.get('reference_median')}) | detection limit: <b>{t.get('detection_limit_pct_loss')}</b> % loss of the neutrophil pattern (reference spread {t.get('reference_spread_sd')})</p>"
         f"<p>Tare method: {e(str(t.get('method', '-')))}" + (f" | fit a {t['fit']['a']}, b {t['fit']['b']}, c {t['fit']['c']} on {t['fit']['n_refs_fitted']} references; prediction {t.get('prediction')}" if t.get('fit') else "")
         + f" | noise index N {m.get('noise_index')} ({m.get('noise_sites_measured')} of {m.get('noise_sites_total')} noise sites)</p>"
         f"<p>Methylated sites mean beta {m.get('methylated_sites_mean_beta')}" + (" - <b>past the entropy ceiling: A falls as loss continues; read beta, not A</b>" if m.get('past_entropy_ceiling') else "") + "</p>",
         f"<h2>Stage MC Met-A C-score</h2><p>C = <b>{c.get('C')}</b> (healthy = 1; healthy held-out range {c.get('healthy_range')}); {e(str(c.get('status', c.get('reason',''))))}</p>",
         (f"<h2>Stage Q IAM-A - {e(str(q.get('cell')))}</h2>{_gauge(q.get('A'), label=('IAM-A ' + str(q.get('A'))) if q.get('A') is not None else '')}"
          f"<p>IAM-A = <b>{q.get('A')}</b> ({e(str(q.get('state', q.get('refusal',''))))}); pipeline {e(str(q.get('pipeline')))}; copy error eps {q.get('eps')}; position P {q.get('P')}; "
          f"eps0 {q.get('eps0')}; halves {e(str(q.get('halves')))}; opportunities {q.get('opportunities')}; E = {q.get('E_kT')} kT</p>" if q else ""),
         "<h2>Withheld</h2><ul>" + "".join(f"<li>{e(w)}</li>" for w in o.get("withheld", [])) + "</ul>",
         "<details><summary>bundle</summary><pre>" + e(json.dumps({k: v for k, v in o.items() if k != 'intake'}, indent=1, default=str)[:20000]) + "</pre></details></body></html>"]
    open(out, "w", encoding="utf-8").write("\n".join(H))
    ref = [x for x in (o.get("refusal"), q.get("refusal")) if x]
    return {"out": out, "bytes": sum(len(x) for x in H), "refusals": ref}
