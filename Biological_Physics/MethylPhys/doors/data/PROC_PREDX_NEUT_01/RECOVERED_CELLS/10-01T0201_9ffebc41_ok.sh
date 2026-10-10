cd remote_jobs/predx && cat > PROC_PREDX_NEUT_01_PREREG.md <<'EOF'
# PROC-PREDX-NEUT-01 — pre-registration (written 2026-10-01, before EPIC-Italy is read with the corrected rules)

PROC-PREDX-SEQUENCE-01 was uninformative: the reader used non-450K floors and printed A for minor cells below resolution. DIAG-450K-01 fixed both:
a 450K neutrophil floor from purified 450K neutrophils (GSE88824; H_min = 0.784057, file floors_450k_v1.json), and A printed only for the dominant
cell. Neutrophil A = H(mean of the separated neutrophil β over its identity loci) / 0.784057, separated = (β − Σ_{k≠neu} f_k μ_k)/f_neu.

**Data.** GSE51032 (EPIC-Italy), our Stage 1; evidence only from the 516 arrays not in GSE51057. Specimens with neutrophil fraction < 0.30 are
reported, not scored.

**Predictions.**
- P1 (healthy reads Normal): ≥ 80 % of controls read neutrophil A within 0.95–1.05.
- P2 (immune first, breast): breast cases > 8 years before diagnosis read outside Normal more often than controls (one-sided Fisher, p < 0.05),
  and the same holds among women only.
- P3 (immune first, colorectal): the same for colorectal cases > 8 years before diagnosis, overall and within each sex present.
Descriptive: neutrophil A by years-to-diagnosis bin (> 8, 5–8, 2–5, < 2) for breast and colorectal; age and sex association in controls.

**Stated limits now.** Floor from 8 donors on one 450K batch; EPIC-Italy is a different lab; arrays give the program reading only.
EOF
shasum -a 256 PROC_PREDX_NEUT_01_PREREG.md | cut -c1-16
python3 - <<'PY'
s=open("predx_read.py").read()
a=s.index("def read(g):"); b=s.index("ok=[g for g in G")
new='''FL=json.load(open("floors_450k_v1.json"))["neutrophils"]["H_min_450K"]
AT=R.A; AT.index=AT.index.astype(str); NL=[x for x in R.ID["neutrophils"]["loci"] if x in AT.index]
Hb=lambda b: -(b*np.log2(b)+(1-b)*np.log2(1-b))
def read(g):
    try:
        b=pd.read_parquet(f"{W}/shards/{g}.parquet").iloc[:,0]; b.index=b.index.astype(str); o=R.read(b,specimen="whole blood",n_boot=20)
        fr=o["composition"]["fractions"]; fn=fr.get("neutrophils",0.0); L=[x for x in NL if x in b.index]
        bg=sum(f*AT.loc[L,f"{k}_mean"] for k,f in fr.items() if k!="neutrophils" and f>0 and f"{k}_mean" in AT.columns)
        bs=((b.reindex(L)-bg)/max(fn,1e-3)).clip(0.001,0.999).dropna()
        A=float(Hb(np.clip(bs.mean(),1e-9,1-1e-9))/FL) if fn>=0.30 and len(bs)>=100 else None
        return dict(gsm=g,call_rate=CR.get(g),f_neu=fn,A_neu=A,n_loci=len(bs),fractions={k:round(v,4) for k,v in fr.items() if v>=0.01})
    except Exception as e: return dict(gsm=g,error=str(e)[:120])
'''
s=s[:a]+new+s[b:]
s=s.replace('json.dump(OUT,open("predx_readings.json","w"),default=float)','json.dump(OUT,open("predx_neut_readings.json","w"),default=float)')
s=s.replace("Pool(NC-4)","Pool(max(NC-2,4))")
import ast; ast.parse(s); open("predx_neut.py","w").write(s); print("ok")
PY
grep -n "^import\|^from" predx_neut.py | head