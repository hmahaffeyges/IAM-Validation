#!/usr/bin/env python3
"""Stage C: combine Fisher matrices into scenarios, write results/validation tables and the figure.
Reads out/fishers.json (from run_forecast.py). Runs locally in < 1 min (one CAMB call for the figure)."""
import os, sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import iamfisher as F

OUT = 'out'
D = json.load(open(os.path.join(OUT, 'fishers.json')))
FS = {k: (v['names'], np.array(v['F'])) for k, v in D['fishers'].items()}
S8 = D['sigma8_fid']
DR1_AREA = 1900.0                      # deg^2, ESA DR1 timeline (cosmos.esa.int/web/euclid/dr1-timeline, June 2026)
DR1_SCALE = DR1_AREA / F.AREA_DEG2


def fm(fid, grav, probe, scale=1.0):
    n, f = FS[f'{fid}|{grav}|{probe}']
    return n, f * scale


def sig(nf, p, fix=()):
    s, _, _ = F.invert(*nf, fix=fix)
    return s[p]


def combo(fid, grav, parts):
    return F.add_fishers(*[fm(fid, grav, p, sc) for p, sc in parts])


# ---------------------------------------------------------------- scenarios (Planck 2018 fiducial)
SCEN = [
    # name, parts, settings note, published-template reference for scale-matching (key) or None
    ('Euclid DR1 pessimistic (1900 deg2)', [('gcsp_pess', DR1_SCALE), ('xc_pess_lowz', DR1_SCALE)], 'IST:F pessimistic, area x 0.127', None),
    ('Euclid DR1 optimistic (1900 deg2)', [('gcsp_opt', DR1_SCALE), ('xc_opt', DR1_SCALE)], 'IST:F optimistic, area x 0.127', None),
    ('Euclid full pessimistic', [('gcsp_pess', 1), ('xc_pess_lowz', 1)], 'GCsp k<0.25h/Mpc; WL l<1500; GCph,XC l<750, GCph z<0.9', None),
    ('Euclid full optimistic', [('gcsp_opt', 1), ('xc_opt', 1)], 'GCsp k<0.30h/Mpc; WL l<5000; GCph,XC l<3000', None),
    ('Euclid full pessimistic + Planck lensing', [('gcsp_pess', 1), ('xc_pess_lowz', 1), ('cmb', 1)], '+ C_L^kk 8<=L<=400', None),
    ('Euclid full optimistic + Planck lensing', [('gcsp_opt', 1), ('xc_opt', 1), ('cmb', 1)], '+ C_L^kk 8<=L<=400', None),
    ('Euclid full pessimistic + Planck lensing + DESI', [('gcsp_pess', 1), ('xc_pess_lowz', 1), ('cmb', 1), ('desi', 1)], '+ DESI fs8 (kmax 0.1h/Mpc)', None),
    ('Euclid full optimistic + Planck lensing + DESI', [('gcsp_opt', 1), ('xc_opt', 1), ('cmb', 1), ('desi', 1)], '+ DESI fs8 (kmax 0.1h/Mpc)', None),
    # single probes (diagnostic)
    ('GCsp alone, pessimistic', [('gcsp_pess', 1)], 'diagnostic', None),
    ('GCsp alone, optimistic', [('gcsp_opt', 1)], 'diagnostic', None),
    ('3x2pt alone, pessimistic (all 10 GCph bins)', [('xc_pess', 1)], 'diagnostic', None),
    ('3x2pt alone, optimistic', [('xc_opt', 1)], 'diagnostic', None),
    # Albuquerque et al. 2025 settings (for scale matching)
    ('Albuquerque-conservative: GCsp k<0.1/Mpc + 3x2pt k<0.25/Mpc', [('gcsp_alb', 1), ('xc_alb_cons', 1)], 'Albuquerque+25 Sect. 5.1 cuts', 'alb_cons'),
    ('Albuquerque 3x2pt alone k<4/Mpc', [('xc_alb_k4', 1)], 'Albuquerque+25 Sect. 6.1', 'alb_k4'),
    # sensitivity variants
    ('Euclid full pessimistic, GCsp in IST:F h-unit convention', [('gcsp_pess_hu', 1), ('xc_pess_lowz', 1)], 'sensitivity', None),
    ('Euclid full optimistic, GCsp in IST:F h-unit convention', [('gcsp_opt_hu', 1), ('xc_opt', 1)], 'sensitivity', None),
    ('Euclid full optimistic + Planck lensing + DESI (kmax 0.2h/Mpc)', [('gcsp_opt', 1), ('xc_opt', 1), ('cmb', 1), ('desi_k02', 1)], 'sensitivity', None),
    ('Euclid full optimistic + Planck lensing + DESI z<0.9 only (no volume overlap)', [('gcsp_opt', 1), ('xc_opt', 1), ('cmb', 1), ('desi_lowz', 1)], 'sensitivity', None),
]

# ---------------------------------------------------------------- validation 2: template vs Albuquerque+25
pub_alb = {'gcsp': 0.530, 'xc_cons': 1.694, 'comb_cons': 0.233, 'xc_k4': 0.04}
tm = {
    'gcsp': sig(fm('alb', 'tmpl_mu0', 'gcsp_alb'), 'mu0'),
    'xc_cons': sig(fm('alb', 'tmpl_mu0', 'xc_alb_cons'), 'mu0'),
    'xc_cons_dv807': sig(fm('alb', 'tmpl_mu0', 'xc_alb_cons_dv807'), 'mu0'),
    'comb_cons': sig(combo('alb', 'tmpl_mu0', [('gcsp_alb', 1), ('xc_alb_cons', 1)]), 'mu0'),
    'comb_cons_dv807': sig(combo('alb', 'tmpl_mu0', [('gcsp_alb', 1), ('xc_alb_cons_dv807', 1)]), 'mu0'),
    'xc_k4': sig(fm('alb', 'tmpl_mu0', 'xc_alb_k4'), 'mu0'),
    'Sigma_comb_cons': sig(combo('alb', 'tmpl_mu0', [('gcsp_alb', 1), ('xc_alb_cons', 1)]), 'Sigma0'),
    'Sigma_xc_k4': sig(fm('alb', 'tmpl_mu0', 'xc_alb_k4'), 'Sigma0'),
}
SCALE = {'alb_cons': pub_alb['comb_cons'] / tm['comb_cons'], 'alb_k4': pub_alb['xc_k4'] / tm['xc_k4']}

rows = []
for name, parts, note, ref in SCEN:
    for sigcase, fix in [('Sigma=1 fixed', ('Sigma0',)), ('Sigma0 free', ())]:
        try:
            sA1 = sig(combo('planck', 'iam_A1', parts), 'A', fix)
            sA0 = sig(combo('planck', 'iam_A0', parts), 'A', fix)
            sT = sig(combo('planck', 'tmpl_mu0', parts), 'mu0', fix)
        except KeyError:
            continue
        has_lens = any(p[0].startswith(('xc', 'cmb')) for p in parts)
        if sigcase == 'Sigma0 free' and not has_lens:
            continue
        r = dict(scenario=name, sigma_case=sigcase, settings=note,
                 sigma_A_IAMfid=sA1, significance_IAMfid=1 / sA1,
                 sigma_A_GRfid=sA0, significance_GRfid=1 / sA0,
                 template_sigma_mu0=sT, template_naive_significance=0.136 / sT)
        if ref is not None and sigcase == 'Sigma0 free':
            r['scale_factor_pub_over_ours'] = SCALE[ref]
            r['sigma_A_IAMfid_scalematched'] = sA1 * SCALE[ref]
            r['significance_scalematched'] = 1 / (sA1 * SCALE[ref])
        rows.append(r)
res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, 'results.csv'), index=False, float_format='%.5g')


def md_table(df, cols, fmt):
    h = '| ' + ' | '.join(cols) + ' |\n|' + '---|' * len(cols) + '\n'
    for _, r in df.iterrows():
        h += '| ' + ' | '.join((fmt.get(c, '{}').format(r[c]) if pd.notna(r[c]) else '') for c in cols) + ' |\n'
    return h


cols = ['scenario', 'sigma_case', 'sigma_A_IAMfid', 'significance_IAMfid', 'sigma_A_GRfid', 'significance_GRfid',
        'template_sigma_mu0', 'template_naive_significance', 'sigma_A_IAMfid_scalematched', 'significance_scalematched']
for c in cols:
    if c not in res:
        res[c] = np.nan
fmt = {c: '{:.3f}' for c in cols[2:]}
fmt.update({'significance_IAMfid': '{:.2f}', 'significance_GRfid': '{:.2f}', 'template_naive_significance': '{:.2f}',
            'significance_scalematched': '{:.2f}'})
with open(os.path.join(OUT, 'results.md'), 'w') as fh:
    fh.write('# CALCULATED FORECAST - Fisher-matrix sensitivity of Euclid (+Planck CMB lensing, +DESI) to IAM mu(z)\n\n')
    fh.write('mu(z) = 1 + A [mu_IAM(z) - 1]; A = 0 is GR, A = 1 is IAM. Significance = 1/sigma(A). Planck 2018 LCDM background.\n')
    fh.write('Marginalised over Omega_m, Omega_b, h, n_s, sigma8 (GR-normalised; equivalent to A_s) and all nuisance parameters.\n')
    fh.write('template_sigma_mu0 = sigma(mu0) for Euclid\'s mu = 1 + mu0 Omega_DE(z)/Omega_DE(0) from the same pipeline (GR fiducial); ')
    fh.write('template_naive_significance = 0.136/sigma(mu0), i.e. the old template-matching estimate, shown for comparison only.\n')
    fh.write('Scale-matched values (Albuquerque-settings rows only) multiply sigma(A) by (published template sigma / our template sigma).\n\n')
    fh.write(md_table(res, cols, fmt))

# ---------------------------------------------------------------- validation table
val = []
fid_istf = F.FIDUCIALS['istf']
pub_istf = {'gcsp_pess': [0.021, 0.051, 0.0063, 0.014, 0.0094], 'gcsp_opt': [0.013, 0.018, 0.0017, 0.0099, 0.0077],
            'gcsp_pess_hu': [0.021, 0.051, 0.0063, 0.014, 0.0094], 'gcsp_opt_hu': [0.013, 0.018, 0.0017, 0.0099, 0.0077],
            'wl_pess': [0.018, 0.47, 0.21, 0.035, 0.0087], 'wl_opt': [0.012, 0.42, 0.20, 0.030, 0.0061],
            'xc_pess': [0.0081, 0.052, 0.027, 0.0085, 0.0038], 'xc_opt': [0.0028, 0.046, 0.020, 0.0036, 0.0013]}
label = {'gcsp_pess': 'GCsp pessimistic (physical k units)', 'gcsp_opt': 'GCsp optimistic (physical k units)',
         'gcsp_pess_hu': 'GCsp pessimistic (IST:F h-unit convention)', 'gcsp_opt_hu': 'GCsp optimistic (IST:F h-unit convention)',
         'wl_pess': 'WL pessimistic', 'wl_opt': 'WL optimistic', 'xc_pess': 'WL+GCph+XC pessimistic', 'xc_opt': 'WL+GCph+XC optimistic'}
for p, pubv in pub_istf.items():
    s, _, _ = F.invert(*fm('istf', 'gr', p))
    for par, pv in zip(['Om', 'Ob', 'h', 'ns', 's8'], pubv):
        ours = s[par] / fid_istf[par]
        val.append(dict(test='1 IST:F flat LCDM (Blanchard+20 Table 9)', case=label[p], parameter=par,
                        published=pv, ours=ours, ratio=ours / pv))
for k, lab in [('gcsp', 'GCsp k<0.1/Mpc alone, Sigma fixed (pub 53.0%)'),
               ('xc_cons', '3x2pt k<0.25/Mpc alone, US (pub 169.4%)'),
               ('xc_cons_dv807', '3x2pt cut rescaled to 807-element data vector (diagnostic; pub 169.4%)'),
               ('comb_cons', 'GCsp+3x2pt conservative, US (pub 23.3%)'),
               ('comb_cons_dv807', 'GCsp+3x2pt with 807-element 3x2pt cut (diagnostic; pub 23.3%)'),
               ('xc_k4', '3x2pt k<4/Mpc alone (pub ~4%)')]:
    pv = pub_alb['xc_cons'] if 'xc_cons' in k else pub_alb['comb_cons'] if 'comb' in k else pub_alb[k]
    val.append(dict(test='2 Template mu0, Albuquerque+25 Table 5 / Sect. 6.1 (PMG-1, Sigma0 free unless noted)', case=lab,
                    parameter='mu0 (= rel. error on 1+mu0)', published=pv, ours=tm[k], ratio=tm[k] / pv))
for k, lab, pv in [('Sigma_comb_cons', 'GCsp+3x2pt conservative, US (pub 2.6%)', 0.026), ('Sigma_xc_k4', '3x2pt k<4/Mpc (pub ~1%)', 0.01)]:
    val.append(dict(test='2 Template Sigma0, Albuquerque+25', case=lab, parameter='Sigma0', published=pv, ours=tm[k], ratio=tm[k] / pv))
valdf = pd.DataFrame(val)

# growth & halofit checks (recomputed here)
cbp = F.run_camb(F.camb_point('planck'))
thp = dict(F.FIDUCIALS['planck'], s8=cbp['s8'])
mI = F.Model(cbp, dict(thp, A=1.0), 'iam')
mG = F.Model(cbp, dict(thp, A=0.0), 'iam')
zchk = np.array([0, 0.3, 0.5, 1.0])
dfs = 100 * (1 - mI.fs8(zchk) / mG.fs8(zchk))
extra = [dict(test='0 IAM growth (chain values)', case='D(z=0) IAM/LCDM - 1 [%]', parameter='D0', published=-0.78,
              ours=100 * (mI.D[0] / mG.D[0] - 1), ratio=np.nan)]
for z, pv, o in zip(zchk, [4.25, 2.17, 1.35, 0.41], dfs):
    extra.append(dict(test='0 IAM growth (chain values)', case=f'f sigma8 deficit at z={z} [%]', parameter='fs8', published=pv, ours=o, ratio=o / pv))
import camb
pars = camb.CAMBparams()
h = thp['h']
pars.set_cosmology(H0=100 * h, ombh2=thp['Ob'] * h * h, omch2=(thp['Om'] - thp['Ob']) * h * h - 0.06 / 93.14, mnu=0.06,
                   num_massive_neutrinos=1, tau=thp['tau'])
pars.InitPower.set_params(As=thp['As'], ns=thp['ns'])
pars.set_matter_power(redshifts=[0, 1, 2], kmax=60)
pars.NonLinear = camb.model.NonLinear_both
pars.NonLinearModel.set_params(halofit_version='takahashi')
PKn = camb.get_results(pars).get_matter_power_interpolator(nonlinear=True, var1='delta_tot', var2='delta_tot',
                                                            hubble_units=False, k_hunit=False)
kk = np.geomspace(0.01, 10, 50)
mx = 0
for z in [0, 1, 2]:
    j = np.argmin(abs(F.Z_MASTER - z))
    mine = np.exp(np.interp(np.log(kk), F.LNK, np.log(mG.PNL[j])))
    mx = max(mx, np.max(np.abs(mine / PKn.P(z, kk) - 1)))
extra.append(dict(test='0 HALOFIT implementation', case='max |P_NL ours / CAMB takahashi - 1|, k 0.01-10/Mpc, z 0,1,2',
                  parameter='P_NL', published=0.0, ours=mx, ratio=np.nan))
cm = D['meta'].get('planck|iam_A1|cmb', {})
extra.append(dict(test='0 Planck lensing noise', case=f"white N_kk = {cm.get('N', np.nan):.3g} gives S/N", parameter='S/N',
                  published=40.0, ours=cm.get('snr', np.nan), ratio=cm.get('snr', np.nan) / 40.0))
valdf = pd.concat([pd.DataFrame(extra), valdf], ignore_index=True)
valdf.to_csv(os.path.join(OUT, 'validation.csv'), index=False, float_format='%.4g')
with open(os.path.join(OUT, 'validation.md'), 'w') as fh:
    fh.write('# Validation of the forecast pipeline (CALCULATED)\n\nRelative 1-sigma errors (sigma/fiducial) unless stated; ratio = ours/published.\n\n')
    fh.write(md_table(valdf, ['test', 'case', 'parameter', 'published', 'ours', 'ratio'],
                      {'published': '{:.4g}', 'ours': '{:.4g}', 'ratio': '{:.2f}'}))
json.dump(dict(template=tm, published=pub_alb, scale=SCALE, sigma8_fid=S8), open(os.path.join(OUT, 'template_validation.json'), 'w'), indent=1)

# ---------------------------------------------------------------- figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'legend.fontsize': 7,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.spines.top': False, 'axes.spines.right': False,
                     'font.family': 'DejaVu Sans'})
FOCAL, COMP1, COMP2, GREY = '#1f4e9c', '#d08a1f', '#6a9f58', '#8c8c8c'


def headline(sc):
    r = res[(res.scenario == sc) & (res.sigma_case == 'Sigma=1 fixed')].iloc[0]
    return r.sigma_A_IAMfid


sA_opt = headline('Euclid full optimistic + Planck lensing + DESI')
sA_pes = headline('Euclid full pessimistic + Planck lensing + DESI')
zz = np.linspace(0.0, 2.0, 201)
ratio = lambda A: 100 * (F.Model(cbp, dict(thp, A=A), 'iam').fs8(zz) / mG.fs8(zz) - 1)
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
ax = axs[0]
ax.fill_between(zz, ratio(1 - sA_pes), ratio(1 + sA_pes), color=FOCAL, alpha=0.12, lw=0,
                label=f'IAM, 1$\\sigma$ band, pessimistic combination ($\\sigma_A$={sA_pes:.2f})')
ax.fill_between(zz, ratio(1 - sA_opt), ratio(1 + sA_opt), color=FOCAL, alpha=0.30, lw=0,
                label=f'IAM, 1$\\sigma$ band, optimistic combination ($\\sigma_A$={sA_opt:.2f})')
ax.plot(zz, ratio(1.0), color=FOCAL, lw=1.6, label='IAM signal (A = 1)')
ax.axhline(0, color=GREY, lw=0.8, ls='--')
# DESI per-z fractional fs8 errors (published) placed on the IAM curve
dz = np.vstack([F.DESI_T25, F.DESI_T23])
ax.errorbar(dz[:, 0], np.interp(dz[:, 0], zz, ratio(1.0)), yerr=dz[:, 1], fmt='o', ms=2.5, color=COMP1, lw=0.8,
            capsize=0, label='DESI f$\\sigma_8$ errors (2016 forecast, k<0.1 h/Mpc)')
# Euclid GCsp per-bin f sigma8 errors (our Fisher, shape and geometry fixed)
nf = fm('planck', 'gr', 'gcsp_fs8')
s, _, _ = F.invert(*nf)
zc = np.array([(a + b) / 2 for a, b, _, _ in F.GCSP_BINS])
ef = np.array([100 * s[f'gcsp_fx{i}'] for i in range(4)])
ax.errorbar(zc, np.interp(zc, zz, ratio(1.0)), yerr=ef, fmt='s', ms=3, color=COMP2, lw=1.0, capsize=0,
            label='Euclid GCsp f$\\sigma_8$ errors (this Fisher, pessimistic)')
ax.set_xlabel('redshift z')
ax.set_ylabel('f$\\sigma_8$ (IAM) / f$\\sigma_8$ ($\\Lambda$CDM) $-$ 1  [%]')
ax.set_title('IAM suppresses growth mostly at z < 1', loc='left')
ax.set_ylim(-9, 5)
ax.set_xlim(-0.02, 2.02)
ax.legend(frameon=False, loc='lower right', fontsize=6)
ax.text(-0.16, 1.04, 'a', transform=ax.transAxes, fontweight='bold', fontsize=10)

ax = axs[1]
ph = F.Photo(mG, ell_max=5000, lmax_wl=5000, lmax_gc=3000)
CI = ph.cls(mI, ph.nuis_fid())
sigC = ph.sigma_C()
pairs = ph.pairs
fields = ph.fields
sel = [(('L', 4), ('L', 4), 'shear, bin 5 x 5', FOCAL), (('G', 2), ('L', 8), 'galaxy-galaxy lensing, G3 x L9', COMP1),
       (('G', 2), ('G', 2), 'photometric clustering, bin 3 x 3', COMP2)]
edges = np.geomspace(10, 5000, 11)
rmin, rmax = 0, -100
for fa, fb, lab, col in sel:
    a, b = fields.index(fa), fields.index(fb)
    a, b = min(a, b), max(a, b)
    j = pairs.index((a, b))
    r = 100 * (CI[:, a, b] / ph.Cfid[:, a, b] - 1)
    ok = ph.mask[:, j]
    ax.plot(ph.ell[ok], r[ok], color=col, lw=1.4, label=lab, ls='--' if 'clustering' in lab else '-')
    rmin, rmax = min(rmin, r[ok].min()), max(rmax, r[ok].max())
    # Gaussian error per coarse band (inverse-variance combination of the 100 sub-bins in each band)
    ec, eh = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = ok & (ph.ell >= lo) & (ph.ell < hi)
        if m.sum() == 0:
            continue
        rel = sigC[m, j] / np.abs(ph.Cfid[m, a, b])
        ec.append(np.sqrt(lo * hi))
        eh.append(100 / np.sqrt(np.sum(1 / rel**2)))
    ec, eh = np.array(ec), np.array(eh)
    rc = np.interp(np.log(ec), np.log(ph.ell[ok]), r[ok])
    ax.fill_between(ec, rc - eh, rc + eh, color=col, alpha=0.18, lw=0)
ax.axhline(0, color=GREY, lw=0.8, ls='--')
ax.set_xscale('log')
ax.set_xlabel('multipole $\\ell$')
ax.set_ylabel('C$_\\ell$ (IAM) / C$_\\ell$ ($\\Lambda$CDM) $-$ 1  [%]')
ax.set_title(f'3x2pt spectra lowered by {abs(rmax):.1f}-{abs(rmin):.1f} %', loc='left')
ax.set_ylim(-6, 6)
ax.legend(frameon=False, loc='lower left', fontsize=6)
ax.text(-0.16, 1.04, 'b', transform=ax.transAxes, fontweight='bold', fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'iam_signal_forecast.png'), dpi=300)
fig.savefig(os.path.join(OUT, 'iam_signal_forecast.pdf'))
print(res[['scenario', 'sigma_case', 'sigma_A_IAMfid', 'significance_IAMfid', 'sigma_A_GRfid', 'template_sigma_mu0']].to_string())
print(valdf[['case', 'parameter', 'published', 'ours', 'ratio']].to_string())
print('SCALE', SCALE)
