#!/usr/bin/env python3
"""Stage A: CAMB runs (fiducial + 4-point stencil in Om, Ob, h, ns) for 3 fiducial cosmologies, in parallel.
Stage B: Fisher matrices for every (fiducial, gravity model, probe) case; one parallel task per (case, parameter).
Writes out/camb_cache.pkl, out/fishers.json.  Usage: python run_forecast.py [n_jobs]"""
import os, sys, json, time, pickle
import numpy as np
from joblib import Parallel, delayed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import iamfisher as F

OUT = 'out'
COSMO = ['Om', 'Ob', 'h', 'ns', 's8']

# ---------------------------------------------------------------- probes
PROBES = {
    'gcsp_pess':   lambda m: F.GCsp(m, 0.25, 'h/Mpc', nl_mode='istf_pess'),
    'gcsp_opt':    lambda m: F.GCsp(m, 0.30, 'h/Mpc', nl_mode='istf_opt'),
    'gcsp_pess_hu': lambda m: F.GCsp(m, 0.25, 'h/Mpc', nl_mode='istf_pess', hunits=True),
    'gcsp_opt_hu':  lambda m: F.GCsp(m, 0.30, 'h/Mpc', nl_mode='istf_opt', hunits=True),
    'gcsp_alb':    lambda m: F.GCsp(m, 0.10, '1/Mpc', nl_mode='perbin'),
    'gcsp_fs8':    lambda m: F.GCsp(m, 0.25, 'h/Mpc', nl_mode='istf_pess'),   # per-bin f sigma8 (figure)
    'wl_pess':     lambda m: F.Photo(m, ell_max=1500, lmax_wl=1500, probes=('L',)),
    'wl_opt':      lambda m: F.Photo(m, ell_max=5000, lmax_wl=5000, probes=('L',)),
    'xc_pess':     lambda m: F.Photo(m, ell_max=1500, lmax_wl=1500, lmax_gc=750),
    'xc_pess_lowz': lambda m: F.Photo(m, ell_max=1500, lmax_wl=1500, lmax_gc=750, gc_bins=range(5)),
    'xc_opt':      lambda m: F.Photo(m, ell_max=5000, lmax_wl=5000, lmax_gc=3000),
    'xc_alb_cons': lambda m: F.Photo(m, ell_max=5000, n_ell=60, kcut=0.25, drop_above=3000),
    # diagnostic: cut rescaled by 0.119 so that k=0.05/Mpc gives the 807-element data vector quoted by Albuquerque+25
    'xc_alb_cons_dv807': lambda m: F.Photo(m, ell_max=5000, n_ell=60, kcut=0.25 * 0.119, drop_above=3000),
    'xc_alb_k4':   lambda m: F.Photo(m, ell_max=5000, n_ell=60, kcut=4.0, drop_above=3000),
    'cmb':         lambda m: F.CMBLens(m),
    'desi':        lambda m: F.DESI(m, kcol=1),
    'desi_k02':    lambda m: F.DESI(m, kcol=2),
    'desi_lowz':   lambda m: F.DESI(m, kcol=1, zmax=0.9),
}
LENSING = {'wl_pess', 'wl_opt', 'xc_pess', 'xc_pess_lowz', 'xc_opt', 'xc_alb_cons', 'xc_alb_cons_dv807', 'xc_alb_k4', 'cmb'}

# ---------------------------------------------------------------- gravity models: (kind, fiducial MG params)
GRAV = {
    'gr':        ('gr', {}),
    'iam_A1':    ('iam', {'A': 1.0, 'Sigma0': 0.0}),     # IAM fiducial
    'iam_A0':    ('iam', {'A': 0.0, 'Sigma0': 0.0}),     # GR fiducial, IAM direction
    'tmpl_mu0':  ('tmpl', {'mu0': 0.0, 'Sigma0': 0.0}),  # Euclid Omega_DE template, GR fiducial
}


def cases():
    c = []
    for p in ['gcsp_pess', 'gcsp_opt', 'gcsp_pess_hu', 'gcsp_opt_hu', 'wl_pess', 'wl_opt', 'xc_pess', 'xc_opt']:
        c.append(('istf', 'gr', p))                                       # validation 1
    for p in ['gcsp_alb', 'xc_alb_cons', 'xc_alb_cons_dv807', 'xc_alb_k4']:
        c.append(('alb', 'tmpl_mu0', p))                                  # validation 2
    for g in ['iam_A1', 'iam_A0', 'tmpl_mu0']:
        for p in ['gcsp_pess', 'gcsp_opt', 'gcsp_pess_hu', 'gcsp_opt_hu', 'gcsp_alb', 'xc_pess', 'xc_pess_lowz', 'xc_opt', 'xc_alb_cons',
                  'xc_alb_cons_dv807', 'xc_alb_k4', 'cmb', 'desi', 'desi_k02', 'desi_lowz']:
            c.append(('planck', g, p))
    c.append(('planck', 'iam_A1', 'gcsp_fs8'))
    c.append(('planck', 'gr', 'gcsp_fs8'))
    return c


# ---------------------------------------------------------------- stage A
def camb_task(key):
    fid, p, s = key
    return key, F.run_camb(F.camb_point(fid, p, s))


def build_cache(n_jobs):
    keys = []
    for fid in F.FIDUCIALS:
        keys.append((fid, None, 0))
        keys += [(fid, p, s) for p in F.CAMB_PARAMS for s in F.STENCIL]
    res = Parallel(n_jobs=n_jobs, verbose=0)(delayed(camb_task)(k) for k in keys)
    return dict(res)


# ---------------------------------------------------------------- stage B
_CACHE, _PROBES = None, {}


def load_cache():
    global _CACHE
    if _CACHE is None:
        with open(os.path.join(OUT, 'camb_cache.pkl'), 'rb') as fh:
            _CACHE = pickle.load(fh)
    return _CACHE


def fid_theta(fid, grav):
    cache = load_cache()
    th = dict(F.FIDUCIALS[fid])
    if th['s8'] is None:
        th['s8'] = cache[(fid, None, 0)]['s8']
    th.update(GRAV[grav][1])
    return th


def model_at(fid, grav, pname=None, s=0):
    cache = load_cache()
    th = fid_theta(fid, grav)
    key = (fid, None, 0)
    if pname is not None and s != 0:
        if pname in F.CAMB_PARAMS:
            key = (fid, pname, s)
            th[pname] = cache[key]['th'][pname]
        elif pname in th:
            th[pname] = th[pname] + s * F.step_of(pname, th[pname])
    return F.Model(cache[key], th, GRAV[grav][0])


def probe_obj(fid, grav, probe):
    k = (fid, grav, probe)
    if k not in _PROBES:
        _PROBES[k] = PROBES[probe](model_at(fid, grav))
    return _PROBES[k]


def case_params(fid, grav, probe):
    pr = probe_obj(fid, grav, probe)
    mg = [x for x in GRAV[grav][1] if not (x == 'Sigma0' and probe not in LENSING)]
    nu = pr.nuis_fid() if hasattr(pr, 'nuis_fid') else {}
    if probe == 'gcsp_fs8':
        nu = dict(nu, **{f'gcsp_fx{i}': 1.0 for i in range(4)})
        return [], nu
    return COSMO + mg, nu


def deriv_task(case, pname):
    fid, grav, probe = case
    pr = probe_obj(fid, grav, probe)
    cos, nu0 = case_params(*case)
    vals = []
    for s in F.STENCIL:
        if pname in nu0:
            nu = dict(nu0)
            nu[pname] = nu0[pname] + s * F.step_of(pname, nu0[pname])
            m = model_at(fid, grav)
            vals.append(pr.observable(m, nu))
            h = F.step_of(pname, nu0[pname])
        else:
            m = model_at(fid, grav, pname, s)
            vals.append(pr.observable(m, nu0))
            th0 = fid_theta(fid, grav)
            h = F.REL_STEP * abs(th0[pname]) if pname in F.CAMB_PARAMS else F.step_of(pname, th0[pname])
    d = sum(w * v for w, v in zip(F.STENCIL_W, vals)) / h
    return case, pname, d


def fisher_case(case, derivs):
    pr = probe_obj(*case)
    cos, nu = case_params(*case)
    names = cos + list(nu)
    return pr.fisher({n: derivs[n] for n in names})


def main():
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else -1
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    if not os.path.exists(os.path.join(OUT, 'camb_cache.pkl')):
        cache = build_cache(n_jobs)
        with open(os.path.join(OUT, 'camb_cache.pkl'), 'wb') as fh:
            pickle.dump(cache, fh)
        print(f'stage A: {len(cache)} CAMB runs in {time.time() - t0:.1f} s', flush=True)
    cs = cases()
    if len(sys.argv) > 2:          # quick test subset
        cs = [c for c in cs if sys.argv[2] in '|'.join(c)]
    tasks = []
    for c in cs:
        cos, nu = case_params(*c)
        tasks += [(c, p) for p in cos + list(nu)]
    print(f'stage B: {len(cs)} cases, {len(tasks)} derivative tasks', flush=True)
    t1 = time.time()
    res = Parallel(n_jobs=n_jobs, verbose=0, batch_size=1)(delayed(deriv_task)(c, p) for c, p in tasks)
    print(f'derivatives in {time.time() - t1:.1f} s', flush=True)
    by = {}
    for c, p, d in res:
        by.setdefault(c, {})[p] = d
    out = {}
    for c in cs:
        names, Fm = fisher_case(c, by[c])
        out['|'.join(c)] = dict(names=names, F=Fm.tolist())
    # extras used by the analysis (probe metadata)
    meta = {}
    for c in cs:
        pr = probe_obj(*c)
        if isinstance(pr, F.GCsp):
            meta['|'.join(c)] = dict(V=pr.V.tolist(), n=pr.n.tolist(), sv=pr.sv_fid.tolist())
        if isinstance(pr, F.CMBLens):
            meta['|'.join(c)] = dict(N=pr.N, snr=pr.snr)
    with open(os.path.join(OUT, 'fishers.json'), 'w') as fh:
        json.dump(dict(fishers=out, meta=meta, sigma8_fid={f: load_cache()[(f, None, 0)]['s8'] for f in F.FIDUCIALS},
                       camb_version=load_cache()[('planck', None, 0)]['camb_version']), fh)
    print(f'total {time.time() - t0:.1f} s', flush=True)


if __name__ == '__main__':
    main()
