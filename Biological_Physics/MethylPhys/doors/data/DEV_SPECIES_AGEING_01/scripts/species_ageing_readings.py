#!/usr/bin/env python3
"""species_ageing_readings.py -- DEV_SPECIES_AGEING_01. Each species read against its own young adults.

Stage 1 (--stage groups): species selection, adult/juvenile split, young reference, oldest decile -> groups.csv;
         lineage marker CpGs from human sorted blood cells (GSE110554 series matrix) -> immune_marker_probes.csv.
         Then run probe_level_confounds.py on the box (needs the 3.3 GB beta parquet and IDAT detection p-values).
Stage 2 (--stage readings): Met-A per animal, per-species summary, figures.

Inputs: PKG = unpacked species_mmc_package/data (sample_stats_final.csv, GSE223748_sample_annotation.csv,
anage_lookup.csv, anage_dataset.zip, species_table.csv one level up, mammal40_manifest_min.csv);
REF = GSE110554_series_matrix.txt.gz (GEO); BOX = folder holding per_array_probe_level.csv / per_species_probe_level.csv.
Nothing is fitted. No statistic pools species.
"""
import os, sys, math, gzip, zipfile, argparse
import numpy as np, pandas as pd
from scipy.stats import spearmanr

ap = argparse.ArgumentParser()
ap.add_argument('--stage', choices=['groups', 'readings'], required=True)
ap.add_argument('--pkg', default='pkg/species_mmc_package')
ap.add_argument('--ref', default='ref/GSE110554_series_matrix.txt.gz')
ap.add_argument('--box', default='box_outputs')
ap.add_argument('--out', default='.')
a = ap.parse_args()
D = a.pkg + '/data/'
MIN_N, MIN_SPAN = 10, 0.30          # C2
REF_FRAC, REF_MIN = 0.25, 3         # C4
OLD_FRAC, OLD_MIN = 0.10, 2         # C9
MARK_D = 0.40                       # C13
E_ATP = 54000.0 / 6.02214076e23     # J per molecule (C12)
kB = 1.380649e-23

def load():
    ss = pd.read_csv(D + 'sample_stats_final.csv')
    b = ss[ss.tissue == 'Blood'].copy(); b['age'] = pd.to_numeric(b.age, errors='coerce')
    an = pd.read_csv(D + 'GSE223748_sample_annotation.csv')
    an = an[['geo_accession', 'female', 'confidenceinageestimate', 'speciescommonname', 'ageatsexualmaturityyears']].rename(columns={'geo_accession': 'gsm'})
    ag = pd.read_csv(D + 'anage_lookup.csv')
    z = zipfile.ZipFile(D + 'anage_dataset.zip')
    A = pd.read_csv(z.open('anage_data.txt'), sep='\t', encoding='latin1'); A['anage_name'] = A.Genus + ' ' + A.Species
    A = A[['anage_name', 'Female maturity (days)', 'Male maturity (days)', 'Temperature (K)']]
    L = ag.merge(A, on='anage_name', how='left')
    b = b.merge(an, on='gsm', how='left').merge(L, on='species', how='left')
    b['female_mat_y'] = b['Female maturity (days)'] / 365.25; b['male_mat_y'] = b['Male maturity (days)'] / 365.25
    return b

def maturity(r):                    # C3
    f, m = r.female_mat_y, r.male_mat_y
    if r.female == 1: return f if pd.notna(f) else m
    if r.female == 0: return m if pd.notna(m) else f
    return np.nanmax([f, m]) if (pd.notna(f) or pd.notna(m)) else np.nan

if a.stage == 'groups':
    b = load(); b['maturity_y'] = b.apply(maturity, axis=1)
    g = b[b.age.notna() & b.max_lifespan.notna()]
    q = g.groupby('species').agg(n_aged=('gsm', 'size'), age_min=('age', 'min'), age_max=('age', 'max'), max_lifespan=('max_lifespan', 'first'), order=('order', 'first'))
    allsp = b.groupby('species').agg(n_blood=('gsm', 'size'), order=('order', 'first'), max_lifespan=('max_lifespan', 'first'))
    q = allsp.join(q[['n_aged', 'age_min', 'age_max']]); q['n_aged'] = q.n_aged.fillna(0).astype(int)
    q['span_frac'] = (q.age_max - q.age_min) / q.max_lifespan
    q['qualifies'] = (q.n_aged >= MIN_N) & (q.span_frac >= MIN_SPAN)
    q['reason_if_not'] = np.where(q.qualifies, '', np.where(q.max_lifespan.isna(), 'no AnAge max longevity',
                                  np.where(q.n_aged < MIN_N, 'fewer than 10 aged arrays', 'age span < 30% of max lifespan')))
    q.to_csv(os.path.join(a.out, 'species_qualification.csv'))
    B = g[g.species.isin(q.index[q.qualifies])].copy()
    B['maturity_source'] = np.where(B.maturity_y.notna(), 'AnAge sex-specific', 'consortium annotation')
    B.loc[B.maturity_y.isna(), 'maturity_y'] = B.loc[B.maturity_y.isna(), 'ageatsexualmaturityyears']
    B['frac_age'] = B.age / B.max_lifespan; B['adult'] = B.age >= B.maturity_y
    B['role'] = np.where(B.adult, 'adult', 'juvenile'); B['oldest_decile'] = False
    for sp, ga in B[B.adult].groupby('species'):
        n = len(ga); k = max(REF_MIN, math.ceil(REF_FRAC * n)); ko = max(OLD_MIN, math.ceil(OLD_FRAC * n))
        B.loc[ga.sort_values(['age', 'gsm']).index[:k], 'role'] = 'young_reference'
        B.loc[ga.sort_values(['frac_age', 'gsm'], ascending=[False, True]).index[:ko], 'oldest_decile'] = True
    B.to_csv(os.path.join(a.out, 'groups_full.csv'), index=False)
    B[['gsm', 'barcode', 'species', 'age', 'frac_age', 'role', 'oldest_decile', 'female']].to_csv(os.path.join(a.out, 'groups.csv'), index=False)
    # lineage markers from human sorted cells (GSE110554), restricted to mammalian-array probe IDs
    man = pd.read_csv(D + 'mammal40_manifest_min.csv'); mp = set(man.IlmnID.astype(str))
    hdr = {}; rows = []; idx = []
    with gzip.open(a.ref, 'rt') as f:
        for line in f:
            if line.startswith('!Sample_'):
                k, *v = line.rstrip('\n').split('\t'); hdr.setdefault(k, []).append([x.strip('"') for x in v])
            if line.startswith('!series_matrix_table_begin'): break
        cols = [x.strip('"') for x in next(f).rstrip('\n').split('\t')]
        for line in f:
            p = line.split('\t', 1)[0].strip('"')
            if p in mp: idx.append(p); rows.append(line.rstrip('\n').split('\t')[1:])
    gsms = hdr['!Sample_geo_accession'][0]; meta = [{} for _ in gsms]
    for c in hdr['!Sample_characteristics_ch1']:
        for i, x in enumerate(c):
            if ':' in x: kk, vv = x.split(':', 1); meta[i][kk.strip()] = vv.strip()
    meta = pd.DataFrame(meta, index=gsms); ct = meta['cell type']
    R = pd.DataFrame(np.array(rows, dtype=float), index=idx, columns=cols[1:])
    M = pd.DataFrame({k: R[ct.index[ct == k]].mean(1) for k in ['Neu', 'Mono', 'CD4T', 'CD8T', 'Bcell', 'NK']})
    lym, mye = ['CD4T', 'CD8T', 'Bcell', 'NK'], ['Neu', 'Mono']
    dm = M[mye].mean(1) - M[lym].mean(1); dT = M[['CD4T', 'CD8T']].mean(1) - M[mye + ['Bcell', 'NK']].mean(1)
    mk = pd.concat([pd.DataFrame({'set': 'myeloid_hypo', 'probe': dm.index[dm <= -MARK_D]}),
                    pd.DataFrame({'set': 'lymphoid_hypo', 'probe': dm.index[dm >= MARK_D]}),
                    pd.DataFrame({'set': 'T_hypo', 'probe': dT.index[dT <= -MARK_D]})])
    mk.to_csv(os.path.join(a.out, 'immune_marker_probes.csv'), index=False)
    print('qualifying species', int(q.qualifies.sum()), '; markers', mk.set.value_counts().to_dict())
    sys.exit(0)

# ---------------- stage 2: readings ----------------
B = pd.read_csv(os.path.join(a.out, 'groups_full.csv'))
st = pd.read_csv(a.pkg + '/species_table.csv')
for p in ['primary', 'core']:                                  # C7
    ref = B[B.role == 'young_reference'].groupby('species')[f'mean_H_{p}'].median()
    B[f'ref_median_meanH_{p}'] = B.species.map(ref)
    B[f'MetA_{p}'] = B[f'mean_H_{p}'] / B[f'ref_median_meanH_{p}']
pa = pd.read_csv(os.path.join(a.box, 'per_array_probe_level.csv'))
for p in ['primary', 'core']:                                  # recomputation check
    x = pa[pa.panel == p].merge(B[['gsm', f'mean_H_{p}']], on='gsm')
    assert len(x) == len(B) and np.abs(x.mean_H_recomputed - x[f'mean_H_{p}']).max() < 1e-6
mkw = pa[pa.panel.str.startswith('marker_')].pivot(index='gsm', columns='panel', values='marker_mean_beta').reset_index()
B = B.merge(mkw, on='gsm', how='left')
B['chip'] = B.barcode.str.split('_').str[0]
B['M_mahaffey'] = E_ATP / (kB * B['Temperature (K)'])
keep = ['species', 'speciescommonname', 'order', 'gsm', 'barcode', 'chip', 'female', 'age', 'confidenceinageestimate', 'max_lifespan', 'frac_age',
        'maturity_y', 'maturity_source', 'adult', 'role', 'oldest_decile', 'det_rate_cg', 'mean_H_primary', 'ref_median_meanH_primary', 'MetA_primary',
        'mean_H_core', 'ref_median_meanH_core', 'MetA_core', 'marker_myeloid_hypo', 'marker_lymphoid_hypo', 'marker_T_hypo', 'Temperature (K)', 'M_mahaffey']
B[keep].rename(columns={'speciescommonname': 'common_name', 'max_lifespan': 'anage_max_lifespan_y', 'Temperature (K)': 'anage_temperature_K',
                        'marker_myeloid_hypo': 'mean_beta_myeloid_low_CpGs', 'marker_lymphoid_hypo': 'mean_beta_lymphoid_low_CpGs',
                        'marker_T_hypo': 'mean_beta_T_low_CpGs'}).sort_values(['order', 'species', 'frac_age']).to_csv(
    os.path.join(a.out, 'per_animal_readings.csv'), index=False)

def eta2(y, gr):
    y = np.asarray(y, float); m = y.mean(); ss = ((y - m) ** 2).sum()
    return sum(len(v) * (v.mean() - m) ** 2 for v in [y[gr == k] for k in np.unique(gr)]) / ss
ps = pd.read_csv(os.path.join(a.box, 'per_species_probe_level.csv'))
nmk = pa[pa.panel.str.startswith('marker_')].merge(B[['gsm', 'species']], on='gsm').groupby(['species', 'panel']).n_marker_probes.first().unstack()
rows = []
for sp, g in B.groupby('species'):
    ad = g[g.adult]; rf = g[g.role == 'young_reference']; od = g[g.oldest_decile]; o9 = ad[ad.frac_age >= 0.9]
    r = dict(species=sp, common_name=g.speciescommonname.iloc[0], order=g.order.iloc[0], anage_max_lifespan_y=g.max_lifespan.iloc[0],
             n_aged_blood_arrays=len(g), n_adult=len(ad), n_juvenile=int((~g.adult).sum()),
             female_maturity_y=g.female_mat_y.iloc[0], male_maturity_y=g.male_mat_y.iloc[0],
             frac_age_min=g.frac_age.min(), frac_age_max=g.frac_age.max(), n_frac_age_over_1=int((g.frac_age > 1).sum()),
             n_ref=len(rf), ref_age_min_y=rf.age.min(), ref_age_max_y=rf.age.max(), ref_frac_age_max=rf.frac_age.max(),
             n_oldest_decile=len(od), oldest_frac_age_min=od.frac_age.min(), oldest_frac_age_max=od.frac_age.max(), n_adult_frac_age_ge_0p9=len(o9))
    for p in ['primary', 'core']:
        c = f'MetA_{p}'
        r.update({f'{p}_n_probes': int(st.set_index('species').loc[sp, 'n_probes' if p == 'primary' else 'n_probes_core']),
                  f'{p}_ref_median_meanH_bits': rf[f'mean_H_{p}'].median(),
                  f'{p}_ref_MetA_min': rf[c].min(), f'{p}_ref_MetA_max': rf[c].max(), f'{p}_ref_MetA_sd': rf[c].std(),
                  f'{p}_oldest_decile_MetA_median': od[c].median(), f'{p}_oldest_decile_MetA_min': od[c].min(), f'{p}_oldest_decile_MetA_max': od[c].max(),
                  f'{p}_frac_ge_0p9_MetA_median': o9[c].median() if len(o9) else np.nan,
                  f'{p}_spearman_MetA_vs_frac_age_adults': spearmanr(ad.frac_age, ad[c]).correlation})
    r.update(dict(n_female=int((g.female == 1).sum()), n_male=int((g.female == 0).sum()), n_sex_unrecorded=int(g.female.isna().sum()),
                  female_share_ref=(rf.female == 1).mean(), female_share_oldest=(od.female == 1).mean(),
                  primary_MetA_median_female_adults=ad[ad.female == 1].MetA_primary.median(), primary_MetA_median_male_adults=ad[ad.female == 0].MetA_primary.median(),
                  n_chips_adults=ad.chip.nunique(), eta2_chip_on_MetA_primary=eta2(ad.MetA_primary, ad.chip.values),
                  eta2_chip_on_frac_age=eta2(ad.frac_age, ad.chip.values), n_age_confidence_below_100=int((g.confidenceinageestimate < 100).sum()),
                  spearman_detection_rate_vs_frac_age_adults=spearmanr(ad.frac_age, ad.det_rate_cg).correlation))
    for m, lab in [('marker_myeloid_hypo', 'myeloid_low'), ('marker_lymphoid_hypo', 'lymphoid_low'), ('marker_T_hypo', 'T_low')]:
        r[f'{lab}_n_CpGs_used'] = int(nmk.loc[sp, m])
        r[f'{lab}_spearman_beta_vs_frac_age_adults'] = spearmanr(ad.frac_age, ad[m], nan_policy='omit').correlation
        r[f'{lab}_spearman_beta_vs_MetA_primary_adults'] = spearmanr(ad.MetA_primary, ad[m], nan_policy='omit').correlation
    for p in ['primary', 'core']:
        x = ps[(ps.species == sp) & (ps.panel == p)].iloc[0]
        r[f'{p}_median_probe_sd_ref'] = x.median_probe_sd_ref; r[f'{p}_median_probe_sd_oldest'] = x.median_probe_sd_old
        r[f'{p}_share_probes_H_higher_oldest'] = x.share_probes_H_higher_old
    T = g['Temperature (K)'].iloc[0]
    r['anage_temperature_K'] = T; r['M_mahaffey'] = E_ATP / (kB * T) if pd.notna(T) else np.nan
    r['probe_basis'] = st.set_index('species').loc[sp, 'probe_basis']
    rows.append(r)
S = pd.DataFrame(rows).sort_values(['order', 'species'])
S.to_csv(os.path.join(a.out, 'per_species_summary.csv'), index=False)

# ---------------- figures ----------------
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'xtick.labelsize': 6, 'ytick.labelsize': 6})
os.makedirs(os.path.join(a.out, 'figures'), exist_ok=True)
orders = sorted(B.order.unique())
ocol = dict(zip(orders, ['#E69F00', '#56B4E9', '#009E73', '#8C7A00', '#0072B2', '#D55E00', '#CC79A7', '#000000']))
spec_order = S.species.tolist(); Si = S.set_index('species')

def runmed(x, y, w=0.15, minn=5):
    xs = np.linspace(x.min(), x.max(), 60)
    return xs, np.array([np.median(y[np.abs(x - v) <= w / 2]) if (np.abs(x - v) <= w / 2).sum() >= minn else np.nan for v in xs])

for p, pname in [('primary', 'species-specific probe set'), ('core', 'conserved panel')]:
    c = f'MetA_{p}'
    fig, axs = plt.subplots(6, 5, figsize=(10, 11.5), sharex=True, sharey=True)
    for ax, sp in zip(axs.flat, spec_order):
        g = B[B.species == sp]; ad = g[g.adult]; rf = g[g.role == 'young_reference']; od = g[g.oldest_decile]; col = ocol[g.order.iloc[0]]
        ax.axhspan(rf[c].min(), rf[c].max(), color='0.88', lw=0, zorder=0); ax.axhline(1, color='0.5', lw=0.6, zorder=1)
        j = g[~g.adult]; ax.scatter(j.frac_age, j[c], s=7, facecolors='none', edgecolors='0.6', lw=0.5, zorder=2)
        o = ad[~ad.oldest_decile]; ax.scatter(o.frac_age, o[c], s=7, color=col, alpha=0.6, lw=0, zorder=3)
        ax.scatter(od.frac_age, od[c], s=14, color=col, edgecolors='k', lw=0.6, zorder=4)
        flag = ' †' if 'detection only' in Si.loc[sp, 'probe_basis'] else ''
        ax.set_title(f'{sp}{flag}\nn={len(ad)} adults, ρ={Si.loc[sp, f"{p}_spearman_MetA_vs_frac_age_adults"]:+.2f}', fontsize=6, loc='left', style='italic')
        ax.set_xlim(-0.03, 1.25)
    for ax in axs[-1]: ax.set_xlabel('age / max lifespan')
    for ax in axs[:, 0]: ax.set_ylabel('Met-A')
    fig.suptitle(f'Met-A against fraction of own maximum lifespan, each species against its own young adults ({pname})', fontsize=8, x=0.01, ha='left')
    fig.text(0.01, 0.003, 'Grey band: range of the young reference (median = 1, grey line). Hollow grey: juveniles (not in reference or ρ). Black-edged: oldest decile of adults.\n'
             'Colour = AnAge order. ρ = Spearman, adults, description only. † probe set rests on detection only.', fontsize=6, ha='left')
    fig.tight_layout(rect=(0, 0.02, 1, 0.975))
    fn = os.path.join(a.out, f'figures/fig_species_MetA_small_multiples_{p}'); fig.savefig(fn + '.png', dpi=200); fig.savefig(fn + '.pdf'); plt.close(fig)

    LAB = ['Homo sapiens', 'Mus musculus', 'Rattus norvegicus', 'Chlorocebus sabaeus', 'Canis lupus familiaris', 'Bos taurus', 'Delphinapterus leucas', 'Heterocephalus glaber']
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True, gridspec_kw=dict(wspace=0.06))
    ad = B[B.adult]
    for o in orders:
        x = ad[ad.order == o]; axs[0].scatter(x.frac_age, x[c], s=4, color=ocol[o], alpha=0.35, lw=0, label=o)
    ends = []
    for sp in spec_order:
        g = ad[ad.species == sp]; xs, ys = runmed(g.frac_age.values, g[c].values)
        axs[1].plot(xs, ys, color=ocol[g.order.iloc[0]], lw=1.1, alpha=0.9)
        k = np.where(~np.isnan(ys))[0]
        if len(k) and sp in LAB: ends.append([xs[k[-1]], ys[k[-1]], sp, ocol[g.order.iloc[0]]])
    ends.sort(key=lambda e: e[1]); ly = []
    for e in ends:
        y = e[1] if not ly else max(e[1], ly[-1] + 0.013); ly.append(y)
        axs[1].annotate(e[2].split()[0][0] + '. ' + e[2].split()[1], xy=(e[0], e[1]), xytext=(1.12, y), fontsize=6, style='italic', va='center',
                        color=e[3], arrowprops=dict(arrowstyle='-', color=e[3], lw=0.4))
    for ax in axs: ax.axhline(1, color='0.5', lw=0.6); ax.set_xlabel('age / AnAge maximum lifespan')
    axs[0].set_xlim(-0.03, 1.3); axs[1].set_xlim(-0.03, 1.45)
    axs[0].set_ylabel('Met-A (own young adults = 1)')
    axs[0].set_title(f'All adult animals, {len(spec_order)} species', loc='left')
    axs[1].set_title('Per-species running median (window 0.15, ≥5 animals); not a fit', loc='left')
    leg = axs[0].legend(frameon=False, markerscale=3, fontsize=6, loc='upper left', ncol=2)
    for lh in leg.legend_handles: lh.set_alpha(1)
    fig.suptitle(f'Each species read against its own young; {pname}', fontsize=8, x=0.01, ha='left')
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.12, top=0.86)
    fn = os.path.join(a.out, f'figures/fig_overlay_fractional_age_{p}'); fig.savefig(fn + '.png', dpi=200); fig.savefig(fn + '.pdf'); plt.close(fig)
print('done', len(B), 'animals', len(S), 'species')
