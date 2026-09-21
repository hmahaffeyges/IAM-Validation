#!/usr/bin/env python3
"""CPG report v3 (row 9, 2026-09-21) - renders EXACTLY what the commissioned chain measured, from cpg_conductor.run_full's bundle.

Author's specification (Issue 003 p4, RUNBOOK): cells detected and their percentages; the A-score of each cell the chain could calculate
and of each class on the three-layer reference, with band placement and tier; the Stage 5 departure with the laboratory's false-alarm rate;
the patient's sky; every flag raised. It names NO disease and gives NO age in years. Anything the chain marks not reportable is printed
as NOT REPORTABLE with the reason, never as a number. The old cpg_report_builder.py (Stage-8 concordance, disease cards, straw-man
wall, cellular age) is record-side and is not called by the chain.
"""
import os, json, html, base64, datetime
FORBIDDEN = ("cancer","carcinoma","alzheimer","dementia","diagnos","disease","lymphoma","leukemia","tumor","tumour","years old","cellular age")
def _e(x): return html.escape(str(x))
def _f(x, nd=4): return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x,(int,float)) else _e(x))
def _pct(x): return "-" if x is None else f"{100*x:.1f} %"

def build_report_v3(bundle, out_path, sample_id="", plate_png=None):
    ctx=bundle.get("context",{}); cls=bundle.get("classes",{}); cells=bundle.get("cells",[]); comp=bundle.get("composition",{}).get("class",{})
    dep=bundle.get("departure",{}) or {}; sky=bundle.get("patient_sky",{}) or {}; bidir=bundle.get("bidirectional",{}) or {}
    flags=[]
    if bundle.get("lab_zero") is None: flags.append("Laboratory zero UNSET - class gauges not reportable")
    if not sky.get("available"): flags.append(f"Patient sky not available: {_e(sky.get('status','no laboratory residual scale'))}")
    for c,r in cls.items():
        if r.get("present") and not r.get("reportable"): flags.append(f"{c}: gauge not reportable ({_e(r.get('band_status') or r.get('status') or 'no band')})")
        if r.get("tier")=="AT_CEILING": flags.append(f"{c}: A at structural ceiling 1/H_min")
        if r.get("placement") in ("ABOVE_BAND","BELOW_BAND"): flags.append(f"{c}: {r['placement']} (A'' {_f(r.get('A_abs'))}, band {_f(r['band']['p10'])}-{_f(r['band']['p90'])})")
    if dep.get("reportable") and dep.get("mahalanobis_beyond_band"): flags.append(f"Stage 5 departure beyond the laboratory p95 ({_f(dep.get('mahalanobis_distance'),3)} vs {_f(dep.get('alarm_threshold_p95'),3)})")
    for c,r in bidir.items():
        if r.get("flag_bidirectional"): flags.append(f"{c}: bidirectional panel flag ({_e(r.get('interpretation',''))[:80]})")
    for k in ("pending_recalibration",):
        pr=bundle.get(k) or {}
        for st,v in pr.items():
            if v is True: flags.append(f"{st}: pending recalibration on the identity gauge - not reported")
    # ---- HTML ----
    H=[]; A=H.append
    A(f"<!doctype html><html><head><meta charset='utf-8'><title>CPG report {_e(sample_id)}</title><style>body{{font:13px/1.45 -apple-system,Helvetica,Arial;margin:32px;color:#1a1a1a;max-width:1000px}} h1{{font-size:20px;margin:0}} h2{{font-size:15px;margin:22px 0 6px;border-bottom:1px solid #ccc}} table{{border-collapse:collapse;width:100%;font-size:12px}} th,td{{border:1px solid #ddd;padding:4px 6px;text-align:left;vertical-align:top}} th{{background:#f3f3f3}} .m{{color:#666;font-size:11px}} .nr{{color:#8a4b00}} .flag{{background:#fff4e5;border-left:4px solid #e08a00;padding:6px 10px;margin:4px 0}} .ok{{background:#eef7ee;border-left:4px solid #3a8a3a;padding:6px 10px}}</style></head><body>")
    A(f"<h1>Cellular Performance Gauge - measurement report</h1><div class='m'>Sample {_e(sample_id)} · substrate {_e(ctx.get('substrate','whole_blood'))} · age {_e(ctx.get('age','-'))} · laboratory {_e(bundle.get('lab') or sky.get('lab') or '-')} · scale {_e(bundle.get('scale','-'))} · lab zero {_f(bundle.get('lab_zero'))} · rendered {datetime.date.today().isoformat()} · report v3</div>")
    A("<p class='m'>This report states what the chain measured on this sample against a fixed physical reference. It names no condition and gives no age in years. Every line marked NOT REPORTABLE says why.</p>")
    A("<h2>1. Cells detected (Stage 2 composition)</h2><table><tr><th>architecture class</th><th>fraction</th></tr>")
    tot=sum(comp.values()) or 1.0; scale=(1.0/100.0) if tot>1.5 else 1.0   # Stage 2 stores class fractions in percent; cells in fractions
    for c,f in sorted(comp.items(), key=lambda kv:-kv[1]):
        if f*scale>=0.001: A(f"<tr><td>{_e(c)}</td><td>{_pct(f*scale)}</td></tr>")
    A("</table>")
    A("<h2>2. Class gauge (identity loci, mapped, age-referenced, laboratory-zeroed)</h2><table><tr><th>class</th><th>fraction</th><th>A (mapped)</th><th>c(age)</th><th>A''</th><th>healthy band p10-p90</th><th>placement</th><th>tier</th><th>H_min</th><th>loci</th></tr>")
    for c,r in cls.items():
        if not r.get("present"): continue
        if r.get("reportable"):
            b=r.get("band") or {}; A(f"<tr><td>{_e(c)}</td><td>{_pct(r.get('fraction'))}</td><td>{_f(r.get('A_mapped'))}</td><td>{_f(r.get('age_reference_c'))}</td><td><b>{_f(r.get('A_abs'))}</b></td><td>{_f(b.get('p10'))}-{_f(b.get('p90'))}</td><td>{_e(r.get('placement'))}</td><td>{_e(r.get('tier'))}</td><td>{_f(r.get('H_min'),4)}</td><td>{r.get('n_loci','-')}</td></tr>")
        else:
            A(f"<tr><td>{_e(c)}</td><td>{_pct(r.get('fraction'))}</td><td>{_f(r.get('A_mapped'))}</td><td colspan='7' class='nr'>NOT REPORTABLE - {_e(r.get('band_status') or r.get('status') or 'no commissioned band for this class on this substrate')}</td></tr>")
    A("</table><p class='m'>A = H(mean beta over the class's identity loci) / H_min(class). A'' = A - c(decade) - z_lab. Band: identity_band_v3 (four laboratories, n = 1,379 healthy). Tier: tier_breakpoints.json v1.4 (NORMAL = healthy central 95 %).</p>")
    A("<h2>3. Per-cell A-scores (separation surface, cells the deconvolver placed in the sample)</h2><table><tr><th>cell type</th><th>class</th><th>fraction</th><th>A (mean of per-CpG H / H_min)</th></tr>")
    for r in sorted(cells, key=lambda r:-(r.get("fraction") or 0)):
        A(f"<tr><td>{_e(r.get('cell'))}</td><td>{_e(r.get('class'))}</td><td>{_pct(r.get('fraction'))}</td><td>{_f(r.get('A'))}</td></tr>")
    A("</table><p class='m'>Per-cell A is on the discriminative-marker surface (PROC-ANCHOR-01, r = 1.00000 to the sealed foundation cohort). It is a separation statistic, not the class gauge: it carries no band, no tier and no healthy reference on this report, and is shown so the cells the chain found can be logged.</p>")
    A("<h2>4. Departure from the age-matched healthy reference (Stage 5)</h2>")
    if dep.get("reportable"):
        A(f"<p>Mahalanobis distance <b>{_f(dep.get('mahalanobis_distance'),3)}</b> over {dep.get('n_assessable','-')} assessable class(es); laboratory p95 {_f(dep.get('alarm_threshold_p95'),3)}, p99 {_f(dep.get('alarm_threshold_p99'),3)}; beyond p95: <b>{'YES' if dep.get('mahalanobis_beyond_band') else 'no'}</b>.</p>")
        if dep.get("lab_false_alarm_sentence"): A(f"<p class='m'>{_e(dep['lab_false_alarm_sentence'])}</p>")
    else: A(f"<p class='nr'>NOT REPORTABLE - {_e(dep.get('status') or dep.get('reason') or 'not computed')}</p>")
    A("<h2>5. The patient's sky (Stage 4.6)</h2>")
    if sky.get("available"):
        al=sky.get("all",{}); A(f"<p>{al.get('n','-'):,} CpGs; fraction beyond |z| = 2: <b>{_f(al.get('frac_abs_z_gt2'),3)}</b> (healthy on this laboratory's zero and scale: 0.026-0.032); median z {_f(al.get('median_z'),3)}. Scale panel n = {sky.get('scale_panel_n','-')}.</p><table><tr><th>class panel</th><th>status</th><th>fraction beyond |z|=2</th><th>median z</th></tr>")
        for c,r in sky.get("classes",{}).items(): A(f"<tr><td>{_e(c)}</td><td>{_e(r.get('status'))}</td><td>{_f(r.get('frac_abs_z_gt2'),3)}</td><td>{_f(r.get('median_z'),3)}</td></tr>")
        A("</table>")
        if plate_png and os.path.exists(plate_png): A(f"<img src='data:image/png;base64,{base64.b64encode(open(plate_png,'rb').read()).decode()}' style='width:100%;margin-top:8px'>")
    else: A(f"<p class='nr'>NOT AVAILABLE - {_e(sky.get('status','no laboratory residual scale'))}</p>")
    A("<h2>6. Flags</h2>"+("".join(f"<div class='flag'>{f}</div>" for f in flags) if flags else "<div class='ok'>No flags raised.</div>"))
    A("<h2>7. Scope of this report</h2><p class='m'>The instrument reports whether this sample's cellular write process is operating within the healthy range for its age, by architecture class, against a fixed physical zero. It does not name a condition, match a pattern to any signature, or state an age in years (single-array resolution ~50 yr, PROC-AGE-01). Fields from the retired marker-union gauge exist in the bundle and are not shown. This is a measurement record, not a clinical interpretation.</p></body></html>")
    out="".join(H); low=out.lower()
    bad=[w for w in FORBIDDEN if w in low]
    assert not bad, f"forbidden vocabulary in report: {bad}"
    open(out_path,"w",encoding="utf-8").write(out); return {"flags":flags,"path":out_path}
