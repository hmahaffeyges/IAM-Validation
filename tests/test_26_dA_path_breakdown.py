import numpy as np

c = 299792.458  # km/s

def angular_diameter_segment(z_low, z_high, H_func):
    z_vals = np.linspace(z_low, z_high, 500)
    integrand = c / H_func(z_vals)
    return np.trapz(integrand, z_vals) / (1 + z_high)

def H_lcdm(z):
    a = 1/(1+z)
    Om0, Om_r, H0 = 0.315, 9.24e-5, 67.38
    Om_L = 1 - Om0 - Om_r
    return H0 * np.sqrt(Om0 * a**-3 + Om_r * a**-4 + Om_L)

def H_iam(z):
    # Use your current/desired IAM H(z) function here
    a = 1/(1+z)
    Om0, Om_r, H0 = 0.315, 9.24e-5, 67.38
    Om_L = 1 - Om0 - Om_r
    beta = 0.179
    activation = np.exp(1 - 1/a)
    return H0 * np.sqrt(Om0 * a**-3 + Om_r * a**-4 + Om_L + beta * activation)

z_breaks = [0, 0.1, 0.5, 1.0, 2.0, 5.0, 10, 100, 1090]
dA_lcdm_cumulative = 0
dA_iam_cumulative = 0

print(f"{'z_low':>6} {'z_high':>7} {'Δd_A(%)':>10} {'Cumulative(%)':>15}")
print("-" * 50)
for i in range(len(z_breaks)-1):
    zl, zh = z_breaks[i], z_breaks[i+1]
    seg_lcdm = angular_diameter_segment(zl, zh, H_lcdm)
    seg_iam  = angular_diameter_segment(zl, zh, H_iam)
    dA_lcdm_cumulative += seg_lcdm
    dA_iam_cumulative  += seg_iam
    seg_diff = 100*(seg_iam-seg_lcdm)/seg_lcdm
    cum_diff = 100*(dA_iam_cumulative-dA_lcdm_cumulative)/dA_lcdm_cumulative
    print(f"{zl:6.1f} {zh:7.1f} {seg_diff:10.3f} {cum_diff:15.3f}")
