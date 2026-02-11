import numpy as np

# At z = 0.295
f_at_z = 0.165 / 0.76  # f ~ fσ8 / σ8 from our calculation
fsig8_observed = 0.452

sigma8_needed = fsig8_observed / f_at_z

print("Diagnostic: What's wrong with fσ₈?")
print(f"\nAt z = 0.295:")
print(f"  Our prediction: fσ₈ = 0.165 (with σ₈ = 0.76)")
print(f"  DESI observes:  fσ₈ = 0.452")
print(f"  Ratio: {fsig8_observed / 0.165:.2f}×")
print(f"\nTo match, we'd need σ₈ = {sigma8_needed:.3f}")
print(f"(We're using σ₈ = 0.76)")
print(f"\nThis is {sigma8_needed / 0.76:.2f}× higher than expected")

print("\n" + "="*60)
print("Possible issues:")
print("1. DESI data values are wrong (wrong observable)")
print("2. σ₈ normalization scale is different")
print("3. We're missing a factor in the calculation")
print("4. The 'fσ₈' from DESI is defined differently")
print("="*60)

# Check if maybe it's σ₁₂ or different scale
for R in [8, 12, 15]:
    scale_factor = (R / 8.0)**(-0.5)  # Approximate scaling
    sigma_R = 0.76 * scale_factor
    print(f"\nIf R = {R} Mpc/h: σ_R = {sigma_R:.3f}")
    fsig8_pred = f_at_z * sigma_R
    print(f"  → fσ_R = {fsig8_pred:.3f} (vs 0.452 observed)")
