#!/usr/bin/env python3
"""Report for conductor v3 (development build, neutrophils only). One self-contained HTML page from the v3 bundle."""
import html, json
def _gauge(A, lo=0.80, hi=1.30, w=520):
    if A is None: return "<p><i>A not reported</i></p>"
    x = lambda v: int((min(max(v, lo), hi) - lo) / (hi - lo) * w)
    return (f'<svg width="{w+20}" height="46"><rect x="10" y="14" width="{w}" height="14" fill="#eee"/>'
            f'<rect x="{10+x(0.95)}" y="14" width="{x(1.05)-x(0.95)}" height="14" fill="#9fd39f"/>'
            f'<line x1="{10+x(A)}" y1="6" x2="{10+x(A)}" y2="36" stroke="#c0392b" stroke-width="3"/>'
            f'<text x="{10+x(0.95)}" y="44" font-size="10">0.95</text><text x="{10+x(1.05)}" y="44" font-size="10">1.05</text></svg>')
def build(o, out, sid):
    m, c, t, a, intake = o.get("met_a", {}), o.get("met_a_cscore", {}), o.get("tare", {}), o.get("composition", {}), o.get("intake") or {}
    e = html.escape
    rows = "".join(f"<tr><td>{e(k)}</td><td>{v*100:.1f} %</td></tr>" for k, v in sorted((a.get("fractions") or {}).items(), key=lambda kv: -kv[1]) if v >= 0.01)
    H = [f"<html><head><meta charset='utf-8'><title>CPG v3 - {e(sid)}</title></head><body style='font-family:sans-serif;max-width:900px'>",
         f"<div style='background:#fff3cd;padding:8px;border:1px solid #e0c060'><b>{e(o.get('build',''))}</b>. Not a diagnostic test.</div>",
         f"<h1>Cellular Performance Gauge - {e(sid)}</h1><p>Specimen: {e(o.get('specimen',''))} | platform {e(o.get('platform',''))} | floors {e(str(o.get('floors_version')))} | reference {e(str(o.get('reference_version')))}</p>",
         f"<h2>Stage 0 intake</h2><p>verdict: <b>{e(str(intake.get('stage0_verdict','not run')))}</b> | call rate: {e(str(intake.get('call_rate_status','-')))} | flags: {e(', '.join(map(str,intake.get('flags',[])))[:300])}</p>",
         "<h2>Stage A composition</h2>" + (f"<table>{rows}</table>" if rows else f"<p>{e(str(a.get('note','')))}</p>"),
         f"<h2>Stage M Met-A - neutrophils</h2>{_gauge(t.get('A_rel') if m.get('specimen')=='whole blood' else m.get('A'))}<p>A = <b>{m.get('A')}</b> ({e(str(m.get('state', m.get('reason',''))))}); "
         f"neutrophil fraction {m.get('fraction')}; sites {m.get('n_sites')}; {e(str(m.get('expectation','own floor')))}</p>",
         f"<h2>Stage T slide tare</h2><p>A_rel = <b>{t.get('A_rel')}</b> {e(str(t.get('state', t.get('reason',''))))}</p>",
         f"<h2>Stage MC Met-A C-score</h2><p>C = <b>{c.get('C')}</b> (healthy = 1; healthy held-out range {c.get('healthy_range')}); {e(str(c.get('status', c.get('reason',''))))}</p>",
         "<h2>Withheld</h2><ul>" + "".join(f"<li>{e(w)}</li>" for w in o.get("withheld", [])) + "</ul>",
         "<details><summary>bundle</summary><pre>" + e(json.dumps({k: v for k, v in o.items() if k != 'intake'}, indent=1, default=str)[:20000]) + "</pre></details></body></html>"]
    open(out, "w", encoding="utf-8").write("\n".join(H))
    return {"out": out, "bytes": sum(len(x) for x in H), "refusals": [o["refusal"]] if o.get("refusal") else []}
