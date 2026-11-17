import numpy as np

# S₈ is defined as: S₈ = σ₈ (Ωₘ/0.3)^0.5
# This is what Planck vs weak lensing tension is about

Omega_m = 0.315
sigma8 = 0.76

S8 = sigma8 * (Omega_m / 0.3)**0.5
print(f"S₈ parameter check:")
print(f"  σ₈ = {sigma8}")
print(f"  Ωₘ = {Omega_m}")
print(f"  S₈ = σ₈(Ωₘ/0.3)^0.5 = {S8:.4f}")

# Planck predicts S₈ ~ 0.83
# Weak lensing observes S₈ ~ 0.76
print(f"\nPlanck 2018: S₈ ~ 0.83")
print(f"KiDS+VIKING: S₈ ~ 0.76")
print(f"IAM (with σ₈=0.76): S₈ = {S8:.4f}")

# Check if DESI "fσ₈" might actually be something else
print("\n" + "="*60)
print("Could the 'fσ₈' values actually be:")
print("="*60)

# Check typical fσ₈ values from literature
typical_fsig8_at_z05 = 0.45  # From various surveys
our_prediction = 0.130

print(f"\n1. Actual fσ₈ from real surveys:")
print(f"   BOSS DR12 at z~0.5: fσ₈ ~ 0.45")
print(f"   eBOSS at z~0.7: fσ₈ ~ 0.45")
print(f"   Our ΛCDM: {our_prediction:.3f}")
print(f"   → We're ~3× too low")

print(f"\n2. Maybe it's f·σ₁₂ (12 Mpc/h scale)?")
print(f"   σ₁₂ ~ 0.6 · σ₈ = {0.6 * sigma8:.3f}")
print(f"   Still doesn't explain 3× factor")

print(f"\n3. Maybe it's GROWTH RATE f(z) only, not fσ₈?")
f_typical = 0.7  # f ~ Ωₘ^0.55 ~ 0.5-0.8
print(f"   Typical f(z=0.5) ~ {f_typical:.2f}")
print(f"   But data shows 0.428, which is too low for f alone")

print(f"\n4. Maybe h-dependence issue?")
print(f"   σ₈ vs σ₈·h? Distances in Mpc vs Mpc/h?")

print("\n" + "="*60)
print("CONCLUSION: Need to see the EXACT data source/table")
print("="*60)
