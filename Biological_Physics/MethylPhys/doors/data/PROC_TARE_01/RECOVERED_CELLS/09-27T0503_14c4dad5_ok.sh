set -e; W="$(pwd)"; MP="$W/iamrepo/Biological_Physics/MethylPhys"; cd "$W"
python3 - <<'PY' 2>&1 | grep -v Deprec
import pandas as pd, numpy as np, json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib as mpl
df=pd.read_parquet("results/tare01/PROC_TARE_01_per_array.parquet").dropna(subset=["A_raw","A_tare"]); R=json.load(open("handoff/tare01_results.json"))
fig,axs=plt.subplots(1,3,figsize=(15,4.6)); labs=["GSE87571","GSE42861","GSE111629","GSE125105"]; col=dict(zip(labs,["#1f5fa8","#c0392b","#2e8b57","#8e44ad"]))
ax=axs[0]
for lab in labs:
    g=df[df.lab==lab]; ax.scatter(g.A_raw,g.A_tare,s=8 if lab=="GSE87571" else 22,alpha=.45 if lab=="GSE87571" else .9,c=col[lab],label=f"{lab} (n={len(g)})",edgecolor="none")
lo,hi=0.85,1.10; ax.plot([lo,hi],[lo,hi],"k-",lw=.8); ax.axvspan(0.95,1.05,color="#9ecae1",alpha=.18); ax.axhspan(0.95,1.05,color="#9ecae1",alpha=.18)
ax.set_xlim(lo,hi); ax.set_ylim(lo,hi); ax.set_xlabel("immune A, raw (pipeline-mapped)"); ax.set_ylabel("immune A after the SNP-probe tare"); ax.legend(frameon=False,fontsize=8,loc="upper left")
ax.set_title("The tare is a constant downward shift\n(median −0.044; 94 % of arrays move down, wherever they sat)",fontsize=10,loc="left")
ax=axs[1]; u=df[df.lab=="GSE87571"]
ax.scatter(u.T_scale,u.A_raw,s=8,alpha=.45,c=col["GSE87571"],edgecolor="none"); ax.set_xlabel("T_scale from the 65 SNP probes (array compression at β = 0 / 1)"); ax.set_ylabel("immune A, raw")
ax.set_title(f"What the probes see does not predict A\n(r = {np.corrcoef(u.T_scale,u.A_raw)[0,1]:+.2f}; T_scale within-chip SD exceeds between-chip)",fontsize=10,loc="left")
ax=axs[2]
for lab in labs:
    g=df[df.lab==lab]; a=np.array(g.A_raw); b=np.array(g.A_tare)
    ax.plot([0,1],[np.median(a),np.median(b)],"-o",c=col[lab],ms=6,label=lab); ax.errorbar([0,1],[np.median(a),np.median(b)],yerr=[[np.median(a)-np.percentile(a,10),np.median(b)-np.percentile(b,10)],[np.percentile(a,90)-np.median(a),np.percentile(b,90)-np.median(b)]],c=col[lab],lw=1,capsize=3,alpha=.7)
ax.axhspan(0.95,1.05,color="#9ecae1",alpha=.18); ax.axhline(1.0,color="k",lw=.6); ax.set_xticks([0,1]); ax.set_xticklabels(["raw","tared"]); ax.set_ylabel("immune A, median (bars p10–p90)")
ax.set_title("Per laboratory: two move toward 1.00, two away;\nspread widens on three of four (B2, B4 failed)",fontsize=10,loc="left"); ax.legend(frameon=False,fontsize=8)
for a_ in axs:
    for s in ("top","right"): a_.spines[s].set_visible(False)
fig.suptitle("PROC-TARE-01 — can the array's own SNP probes tare the instrument? NOT COMMISSIONED: all six bars failed on 768 arrays, four laboratories",y=1.02,fontsize=11)
fig.tight_layout(); fig.savefig("PROC_TARE_01.png",dpi=300,bbox_inches="tight",facecolor="white")
r=fig.canvas.get_renderer(); tx=[(t,t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text) if t.get_text().strip() and t.get_visible()]
print("overlaps:",[(a.get_text()[:16],b.get_text()[:16]) for i,(a,ba) in enumerate(tx) for b,bb in tx[i+1:] if ba.overlaps(bb) and not (a.get_text().replace('.','').replace('−','').isdigit() and b.get_text().replace('.','').replace('−','').isdigit())][:4])
PY
L=$(python3 -c "
import json; R=json.load(open('handoff/tare01_results.json')); L=R['labs']
for k in ('GSE87571','GSE42861','GSE111629','GSE125105'):
    v=L[k]; print(f\"| {k} | {v['n']} | {v['median_A_raw']:.4f} | {v['median_A_tare']:.4f} | {v['sd_raw']:.4f} -> {v['sd_tare']:.4f} | {v['median_T_offset']:+.4f} | {v['median_T_scale']:.4f} | {v['z_lab_commissioned']:+.4f} / {v['z_lab_measured_here']:+.4f} |\")")
D=$(cat handoff/tare01_diag.json)
cat > "$MP/doors/PROC_TARE_01_OUTCOME.md" <<EOF
# PROC-TARE-01 — outcome: NOT COMMISSIONED. The array's SNP probes see a real compression, and it carries almost no information about where a cell sits on the gauge.

**Sealed 2026-09-27** against the bars in [\`PROC_TARE_01_PREREG.md\`](PROC_TARE_01_PREREG.md), fixed before any SNP probe was
read for this purpose. 768 arrays: the full 732-array GSE87571 calibration plus the 12-array panels of GSE42861, GSE111629
and GSE125105 (raw IDATs fetched from GEO for this procedure). Evidence: PROC_TARE_01.json, PROC_TARE_01_per_array.parquet,
PROC_TARE_01_diag.json in the kit results folder; PROC_TARE_01.py in the kit; plate PROC_TARE_01.png.

**Run record.** The first pass started before the panel fetch had finished and its panel arm carried 1 + 12 + 0 pairs; the
GSE87571 arm (732) was complete and is kept. The panel arm was re-run alone on 2026-09-27 with the fetch complete (36 arrays)
and merged; the bars were then scored once, on all 768. Nothing was changed after results were visible.

## The construction, as pre-registered

65 SNP probes on the 450K array read β = 0, 0.5 or 1 by genotype. Per array: cluster to the nearest ideal (two passes),
fit a linear tare β' = (β − T_offset) / T_scale that puts the three clusters back on 0 / 0.5 / 1, apply it to the whole array
before the pipeline map, and read the immune identity gauge raw and tared.

## Result

| laboratory | n | median A raw | median A tared | sd raw -> tared | T_offset | T_scale | z_lab commissioned / measured here |
|---|---|---|---|---|---|---|---|
$L

| bar | requirement | result |
|---|---|---|
| B1 | the raw path reproduces each laboratory's commissioned zero to ±0.005 | **FAILED** - met on GSE87571 (Δ 0.002) and GSE42861 (Δ 0.001); not on GSE111629 (Δ 0.020) or GSE125105 (Δ 0.037), each measured on 12 arrays against a 40-array commissioning |
| B2 | the tare moves every laboratory's median toward 1.00 and shrinks the worst offset by ≥ 50 % | **FAILED** - GSE87571 0.9918 → 0.9475 and GSE125105 1.0043 → 1.0192 move away; worst shrinks 24 % |
| B3 | all four medians inside NORMAL after the tare | **FAILED** - GSE87571 0.9475 |
| B4 | spread tightens or holds on ≥ 3 of 4 | **FAILED** - widens on three (GSE87571 0.024 → 0.036, GSE111629 0.020 → 0.034, GSE125105 0.012 → 0.052) |
| B5 | tare parameters independent of age and sex (\|r\| < 0.10) | **FAILED** - r_age = −0.18 (T_scale), r_sex = −0.04 |
| B6 | per-chip term shrinks ≥ 20 % | **FAILED** - sd of chip medians 0.012 → 0.019 (worse) |
| B7 | instrument unchanged on the 11 commissioning arrays | NOT ASSESSED (nothing in the chain was changed by this procedure) |

## Why, measured after the bars (diagnostics, not bars)

- **The tare is a constant shift.** It moves A by a median of −0.044 on 94 % of arrays regardless of where they sat: 95 % of
  arrays above 1.00 moved toward it and 94 % of arrays below 1.00 moved away. A correction that shifts everyone the same way is
  an offset, and the instrument already has one (the pipeline map); it does not tare anything.
- **T_scale is not a chip property.** Median 0.939 (the array compresses β by ~6 % at the extremes - real, and seen on every
  laboratory: 0.89-0.94), but within-chip SD 0.0098 exceeds between-chip SD 0.0062 over 62 chips. The chip term of row 5b is
  not what the SNP probes measure.
- **It does not predict A.** corr(T_scale, A_raw) = −0.14; corr(T_offset, A_raw) = −0.07 on 732 arrays. Compression at β = 0
  and β = 1 says little about the array's response at β ≈ 0.7, where the identity loci sit.

## What this decides

The question was whether the instrument can be zeroed on a known input carried by every array, instead of on a healthy panel.
With a **linear** tare from the **SNP probes**, no. Two routes remain and are recorded, not started: (i) a nonlinear response
model - the compression is a saturation and a straight rescale is the wrong form for it; the tri-modal SNP clusters constrain
only three points of a curve, so this needs the array's control probes as well; (ii) a physical reference material run
through the laboratory's own pipeline. Until one of these is measured, the pipeline map onto the atlas scale is the instrument's
calibration, and a laboratory zero remains on record and unapplied.

The statement stands as it was before this procedure: healthy is A = 1.00 with the tier scale as tolerance, and no population
defines it. What PROC-TARE-01 failed to do is replace the panel-derived instrument constants with an on-array one.
EOF
cp handoff/tare01_results.json "$MP/kit/results/PROC_TARE_01.json"; cp handoff/tare01_diag.json "$MP/kit/results/PROC_TARE_01_diag.json"
cp results/tare01/PROC_TARE_01_per_array.parquet "$MP/kit/results/"; cp tare01.py "$MP/kit/PROC_TARE_01.py"; cp PROC_TARE_01.png "$MP/plates/"
sed -i.bak '1,3s/^#!\/usr\/bin\/env python3$/#!\/usr\/bin\/env python3\n# INSTRUMENT-TEST: measures a CANDIDATE per-array tare (SNP probes) against the commissioned pipeline map; produces bar results only - no tier, no report./' "$MP/kit/PROC_TARE_01.py" 2>/dev/null; rm -f "$MP/kit/PROC_TARE_01.py.bak"; head -3 "$MP/kit/PROC_TARE_01.py" | cut -c1-100
cp "$MP/doors/PROC_TARE_01_OUTCOME.md" .
echo done