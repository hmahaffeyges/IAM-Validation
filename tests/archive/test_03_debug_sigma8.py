import numpy as np

# Our prediction at z=0.295
fsig8_model = 0.177  # ΛCDM from our calculation
f_model = 0.177 / 0.811  # Assuming sigma8=0.811

# DESI observation
fsig8_desi = 0.470

# What sigma8 would make them match?
sigma8_needed = fsig8_desi / f_model

print("Sigma8 diagnosis:")
print(f"Our prediction: fσ₈ = {fsig8_model:.3f} (with σ₈=0.811)")
print(f"DESI observes:  fσ₈ = {fsig8_desi:.3f}")
print(f"Ratio: {fsig8_desi/fsig8_model:.2f}×")
print(f"\nTo match, we'd need σ₈ = {sigma8_needed:.3f}")
print(f"(Planck 2018: σ₈ = 0.811)")
print(f"\nThis suggests either:")
print("1. Wrong DESI data values")
print("2. Different σ₈ definition/normalization")
print("3. We're comparing to wrong observable")
