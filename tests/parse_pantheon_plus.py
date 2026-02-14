#!/usr/bin/env python3
"""
Parse real Pantheon+ data
Extract z and distance modulus
"""

import numpy as np
import pandas as pd

# Load the data
data_file = '../data/pantheon_plus/pantheon_plus_sh0es.dat'

try:
    # Read Pantheon+ data
    # Format: name, zcmb, zhel, mb, dmb, ...
    df = pd.read_csv(data_file, delim_whitespace=True, comment='#')
    
    # Extract what we need
    z = df['zcmb'].values  # CMB frame redshift
    mb = df['mb'].values   # Apparent magnitude
    dmb = df['dmb'].values # Magnitude uncertainty
    
    print(f"Loaded {len(z)} supernovae")
    print(f"Redshift range: {z.min():.4f} to {z.max():.4f}")
    print(f"Median uncertainty: {np.median(dmb):.3f} mag")
    
    # Save in simple format
    np.savez('../data/pantheon_plus/pantheon_plus_processed.npz',
             z=z, mb=mb, dmb=dmb)
    
    print("\nSaved to pantheon_plus_processed.npz")
    print("\nFirst 10 entries:")
    for i in range(10):
        print(f"  z={z[i]:.4f}, mb={mb[i]:.3f} ± {dmb[i]:.3f}")
    
except FileNotFoundError:
    print(f"ERROR: Could not find {data_file}")
    print("Please download Pantheon+ data first!")
    print("\nTry:")
    print("  cd ~/Desktop/IAM-Validation/data")
    print("  mkdir -p pantheon_plus")
    print("  cd pantheon_plus")
    print("  # Download from https://github.com/PantheonPlusSH0ES/DataRelease")
except Exception as e:
    print(f"ERROR: {e}")
