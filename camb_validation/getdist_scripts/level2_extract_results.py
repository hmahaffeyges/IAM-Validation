#!/usr/bin/env python3
"""
IAM Level 2 Results Extraction Script
Usage: python3 level2_extract_results.py [chain_dir]
Default chain_dir: ~/CAMB_IAM_L2/chains
"""
import argparse, glob, os, sys
import numpy as np
try:
    from getdist.mcsamples import loadMCSamples
except ImportError:
    print("ERROR: getdist not installed. Run: pip install getdist")
    sys.exit(1)

CORE_PARAMS = ['H0', 'omegabh2', 'omegach2', 'ns', 'tau', 'sigma8', 'logA', 'omegam', 'S8']
CHI2_PARAMS = ['chi2', 'chi2__CMB', 'chi2__planck_2018_lowl.TT', 'chi2__planck_2018_lowl.EE',
               'chi2__planck_NPIPE_highl_CamSpec.TTTEEE', 'chi2__planck_2018_lensing.CMBMarged',
               'chi2__iam_rsd']
PASS_CRITERIA = {
    'sigma8_iam_range': (0.790, 0.810),
    'sigma8_lcdm_range': (0.800, 0.820),
    'delta_chi2_max': 5.0,
    'R_minus_1_max': 0.01,
}

def find_chains(chain_dir):
    txt_files = glob.glob(os.path.join(chain_dir, '*.1.txt'))
    chains = {}
    for f in txt_files:
        name = os.path.basename(f).replace('.1.txt', '')
        chains[name] = f.replace('.1.txt', '')
    return chains

def extract_convergence(chain_dir, chain_name):
    progress_file = os.path.join(chain_dir, chain_name + '.progress')
    if not os.path.exists(progress_file):
        return None, None
    with open(progress_file, 'r') as f:
        lines = f.readlines()
    for line in reversed(lines):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) >= 4:
            try:
                acceptance = float(parts[2])
                r_minus_1 = parts[3]
                if r_minus_1 == 'NaN':
                    continue
                return float(r_minus_1), acceptance
            except (ValueError, IndexError):
                continue
    return None, None

def extract_params(chain_path, chain_name):
    try:
        samples = loadMCSamples(chain_path, settings={'ignore_rows': 0.3})
    except Exception as e:
        print(f"  ERROR loading {chain_name}: {e}")
        return None
    stats = samples.getMargeStats()
    results = {'name': chain_name}
    for p in CORE_PARAMS:
        par = stats.parWithName(p)
        if par:
            results[p] = {'mean': par.mean, 'err': par.err}
    for p in CHI2_PARAMS:
        par = stats.parWithName(p)
        if par:
            results[p] = {'mean': par.mean, 'err': par.err}
    try:
        chi2_samples = samples.getParams().chi2
        results['chi2_min'] = float(np.min(chi2_samples))
    except Exception:
        pass
    return results

def print_comparison(iam_results, lcdm_results, iam_r1, lcdm_r1, iam_acc, lcdm_acc):
    print(f"\n{'='*70}")
    print(f"  IAM Level 2 Results Comparison")
    print(f"  IAM:  {iam_results['name']}")
    print(f"  LCDM: {lcdm_results['name']}")
    print(f"{'='*70}")
    print(f"\n  Convergence:")
    print(f"    {'':20s} {'IAM':>15s} {'LCDM':>15s}")
    if iam_r1 is not None and lcdm_r1 is not None:
        print(f"    {'R-1':20s} {iam_r1:>15.6f} {lcdm_r1:>15.6f}")
    if iam_acc is not None and lcdm_acc is not None:
        print(f"    {'Acceptance':20s} {iam_acc:>15.3f} {lcdm_acc:>15.3f}")
    print(f"\n  Parameter Comparison:")
    print(f"    {'Parameter':12s} {'IAM':>20s} {'LCDM':>20s} {'Shift':>12s} {'sigma':>8s}")
    print(f"    {'-'*12} {'-'*20} {'-'*20} {'-'*12} {'-'*8}")
    for p in CORE_PARAMS:
        if p in iam_results and p in lcdm_results:
            iam = iam_results[p]
            lcdm = lcdm_results[p]
            shift = iam['mean'] - lcdm['mean']
            combined_err = np.sqrt(iam['err']**2 + lcdm['err']**2)
            n_sigma = shift / combined_err if combined_err > 0 else 0
            iam_str = f"{iam['mean']:.5f} +/- {iam['err']:.5f}"
            lcdm_str = f"{lcdm['mean']:.5f} +/- {lcdm['err']:.5f}"
            print(f"    {p:12s} {iam_str:>20s} {lcdm_str:>20s} {shift:>+12.5f} {n_sigma:>+7.2f}s")
    print(f"\n  Chi-Squared Comparison:")
    print(f"    {'Likelihood':45s} {'IAM':>10s} {'LCDM':>10s} {'Delta':>10s}")
    print(f"    {'-'*45} {'-'*10} {'-'*10} {'-'*10}")
    for p in CHI2_PARAMS:
        if p in iam_results and p in lcdm_results:
            iam_val = iam_results[p]['mean']
            lcdm_val = lcdm_results[p]['mean']
            delta = iam_val - lcdm_val
            label = p.replace('chi2__', '')
            print(f"    {label:45s} {iam_val:>10.2f} {lcdm_val:>10.2f} {delta:>+10.2f}")
    if 'chi2_min' in iam_results and 'chi2_min' in lcdm_results:
        iam_bf = iam_results['chi2_min']
        lcdm_bf = lcdm_results['chi2_min']
        delta_bf = iam_bf - lcdm_bf
        print(f"    {'best-fit (minimum)':45s} {iam_bf:>10.2f} {lcdm_bf:>10.2f} {delta_bf:>+10.2f}")
    if 'chi2' in iam_results and 'chi2' in lcdm_results:
        delta_chi2_mean = iam_results['chi2']['mean'] - lcdm_results['chi2']['mean']
        print(f"\n  Delta-chi2 (mean posterior): {delta_chi2_mean:+.2f}")
    if 'chi2_min' in iam_results and 'chi2_min' in lcdm_results:
        delta_chi2_bf = iam_results['chi2_min'] - lcdm_results['chi2_min']
        print(f"  Delta-chi2 (best-fit):       {delta_chi2_bf:+.2f}")
    print(f"\n  {'='*70}")
    print(f"  PASS/FAIL ASSESSMENT")
    print(f"  {'='*70}")
    all_pass = True
    if 'sigma8' in iam_results:
        s8 = iam_results['sigma8']['mean']
        lo, hi = PASS_CRITERIA['sigma8_iam_range']
        passed = lo <= s8 <= hi
        status = "PASS" if passed else "FAIL"
        print(f"    sigma8 (IAM) = {s8:.4f}  in [{lo}, {hi}]?  {status}")
        all_pass = all_pass and passed
    if 'sigma8' in lcdm_results:
        s8 = lcdm_results['sigma8']['mean']
        lo, hi = PASS_CRITERIA['sigma8_lcdm_range']
        passed = lo <= s8 <= hi
        status = "PASS" if passed else "FAIL"
        print(f"    sigma8 (LCDM) = {s8:.4f}  in [{lo}, {hi}]?  {status}")
        all_pass = all_pass and passed
    if 'chi2' in iam_results and 'chi2' in lcdm_results:
        dc2 = abs(iam_results['chi2']['mean'] - lcdm_results['chi2']['mean'])
        passed = dc2 < PASS_CRITERIA['delta_chi2_max']
        status = "PASS" if passed else "FAIL"
        print(f"    |Delta-chi2| = {dc2:.2f}  < {PASS_CRITERIA['delta_chi2_max']}?  {status}")
        all_pass = all_pass and passed
    if iam_r1 is not None:
        passed = iam_r1 < PASS_CRITERIA['R_minus_1_max']
        status = "PASS" if passed else "FAIL"
        print(f"    R-1 (IAM) = {iam_r1:.6f}  < {PASS_CRITERIA['R_minus_1_max']}?  {status}")
        all_pass = all_pass and passed
    if lcdm_r1 is not None:
        passed = lcdm_r1 < PASS_CRITERIA['R_minus_1_max']
        status = "PASS" if passed else "FAIL"
        print(f"    R-1 (LCDM) = {lcdm_r1:.6f}  < {PASS_CRITERIA['R_minus_1_max']}?  {status}")
        all_pass = all_pass and passed
    print(f"\n    Standard parameter stability (|shift| < 0.5 sigma):")
    for p in ['H0', 'omegabh2', 'omegach2', 'ns', 'tau']:
        if p in iam_results and p in lcdm_results:
            shift = iam_results[p]['mean'] - lcdm_results[p]['mean']
            combined_err = np.sqrt(iam_results[p]['err']**2 + lcdm_results[p]['err']**2)
            n_sigma = abs(shift / combined_err) if combined_err > 0 else 0
            passed = n_sigma < 0.5
            status = "PASS" if passed else "CHECK"
            print(f"      {p:12s}: {n_sigma:.2f} sigma  {status}")
            if not passed:
                all_pass = False
    print(f"\n  {'='*70}")
    if all_pass:
        print(f"  OVERALL: ALL CHECKS PASSED")
    else:
        print(f"  OVERALL: SOME CHECKS NEED ATTENTION")
    print(f"  {'='*70}")

def main():
    parser = argparse.ArgumentParser(description='Extract IAM Level 2 MCMC results')
    parser.add_argument('chain_dir', nargs='?', default=os.path.expanduser('~/CAMB_IAM_L2/chains'))
    args = parser.parse_args()
    chain_dir = args.chain_dir
    print(f"Looking for chains in: {chain_dir}")
    chains = find_chains(chain_dir)
    print(f"Found chains: {list(chains.keys())}")
    iam_candidates = [k for k in chains if 'lcdm' not in k.lower() and ('runA' in k or 'run_a' in k.lower())]
    lcdm_candidates = [k for k in chains if 'lcdm' in k.lower() or 'runC' in k]
    rsd_candidates = [k for k in chains if 'runD' in k or 'run_d' in k.lower()]
    if not iam_candidates or not lcdm_candidates:
        print("Could not auto-detect IAM and LCDM chains. Check chain directory.")
        return
    iam_name = iam_candidates[0]
    lcdm_name = lcdm_candidates[0]
    print(f"\nExtracting IAM chain: {iam_name}")
    iam_r1, iam_acc = extract_convergence(chain_dir, iam_name)
    iam_results = extract_params(os.path.join(chain_dir, iam_name), iam_name)
    print(f"Extracting LCDM chain: {lcdm_name}")
    lcdm_r1, lcdm_acc = extract_convergence(chain_dir, lcdm_name)
    lcdm_results = extract_params(os.path.join(chain_dir, lcdm_name), lcdm_name)
    if iam_results and lcdm_results:
        print_comparison(iam_results, lcdm_results, iam_r1, lcdm_r1, iam_acc, lcdm_acc)
    for rsd_name in rsd_candidates:
        rsd_path = os.path.join(chain_dir, rsd_name)
        if os.path.exists(rsd_path + '.1.txt'):
            print(f"\n\nExtracting RSD chain: {rsd_name}")
            rsd_r1, rsd_acc = extract_convergence(chain_dir, rsd_name)
            rsd_results = extract_params(rsd_path, rsd_name)
            if rsd_results and lcdm_results:
                print_comparison(rsd_results, lcdm_results, rsd_r1, lcdm_r1, rsd_acc, lcdm_acc)

if __name__ == '__main__':
    main()
