"""Step-size stability check: sigma(A) for Euclid optimistic (GCsp+3x2pt), IAM fiducial, with A-step 0.025/0.05/0.1
and cosmological relative step 0.005/0.01 (needs out/camb_cache.pkl; rebuilt for the 0.005 step)."""
import sys, os, pickle, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import iamfisher as F, run_forecast as R
out = {}
for rel in [0.01, 0.005]:
    F.REL_STEP = rel
    R._CACHE = None; R._PROBES.clear()
    R.OUT = f'out_step{rel}'
    os.makedirs(R.OUT, exist_ok=True)
    if not os.path.exists(os.path.join(R.OUT, 'camb_cache.pkl')):
        keys = [('planck', None, 0)] + [('planck', p, s) for p in F.CAMB_PARAMS for s in F.STENCIL]
        pickle.dump(dict(R.camb_task(k) for k in keys), open(os.path.join(R.OUT, 'camb_cache.pkl'), 'wb'))
    for hA in ([0.025, 0.05, 0.1] if rel == 0.01 else [0.05]):
        F.ABS_STEP['A'] = hA
        nfs = []
        for probe in ['gcsp_opt', 'xc_opt']:
            c = ('planck', 'iam_A1', probe)
            cos, nu = R.case_params(*c)
            der = {p: R.deriv_task(c, p)[2] for p in cos + list(nu)}
            nfs.append(R.fisher_case(c, der))
        nf = F.add_fishers(*nfs)
        out[f'rel{rel}_A{hA}'] = [F.invert(*nf, fix=('Sigma0',))[0]['A'], F.invert(*nf)[0]['A']]
        print(rel, hA, out[f'rel{rel}_A{hA}'], flush=True)
json.dump(out, open('out/step_test.json', 'w'), indent=1)
