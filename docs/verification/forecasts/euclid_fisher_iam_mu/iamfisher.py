"""Fisher forecast of Euclid (+ Planck CMB lensing + DESI) sensitivity to a scale-independent
modified-growth function mu(z), applied to IAM's own mu(z) and to Euclid's Omega_DE template.

Method summary (details in METHODS.md):
  * CAMB (pip) gives the GR linear matter power P_GR(k,z) (delta_tot, 1 massive nu 0.06 eV) and background.
  * Scale-independent mu(z): linear growth ODE in ln a (LCDM background, no radiation, D -> a at a=1e-3,
    i.e. identical early amplitude); P_MG(k,z) = P_GR(k,z) * [D_MG(z)/D_GR(z)]^2, f_MG from the same ODE.
  * Sigma(z) multiplies every lensing kernel (Euclid WL, CMB lensing).
  * Nonlinear: HALOFIT (Takahashi et al. 2012) applied to the growth-rescaled linear spectrum (own implementation).
  * GCsp: IST:F recipe (Blanchard et al. 2020, Eq. 87): AP, Kaiser + Lorentzian FoG, de-wiggled BAO,
    redshift-error damping, shot-noise nuisance; Fisher directly in the final parameters.
  * 3x2pt: Limber C_ell for WL (with eNLA IA), GCph and GGL with IST:F n(z), photo-z model, bias, noise;
    Gaussian covariance; data-vector Fisher with arbitrary element-wise scale cuts.
  * CMB lensing: Limber C_L^kk, L = 8..400, white reconstruction noise calibrated to Planck 2018's 40 sigma.
  * DESI: independent Gaussian f sigma8(z) errors from DESI Collaboration 2016 Tables 2.3 + 2.5.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import erf
from scipy.ndimage import gaussian_filter1d

C_KMS = 299792.458
_trapz = getattr(np, 'trapezoid', None) or np.trapz
Z_MASTER = np.unique(np.round(np.concatenate([np.linspace(0, 3, 301), np.geomspace(3.05, 20, 60)]), 6))
LNK = np.linspace(np.log(1e-5), np.log(400.0), 1000)
K = np.exp(LNK)
DLNK = LNK[1] - LNK[0]

# ----------------------------------------------------------------------------------------------
# fiducial cosmologies
# ----------------------------------------------------------------------------------------------
FIDUCIALS = {
    # Blanchard et al. 2020 (IST:F) Table 1, flat LCDM, sum m_nu = 0.06 eV, tau = 0.058; sigma8 given directly
    'istf': dict(Om=0.32, Ob=0.05, h=0.67, ns=0.96, s8=0.816, mnu=0.06, tau=0.058, As=2.12605e-9),
    # Albuquerque et al. 2025 Table 1 (As fixed, sigma8 derived), m_nu 0.06 fixed
    'alb': dict(Om=0.315, Ob=0.05, h=0.674, ns=0.966, s8=None, mnu=0.06, tau=0.058, As=2.097e-9),
    # Planck 2018 (TT,TE,EE+lowE+lensing) values given in the task; sigma8 derived from As
    'planck': dict(Om=0.3153, Ob=0.02237 / 0.6736**2, h=0.6736, ns=0.9649, s8=None, mnu=0.06, tau=0.0544,
                   As=np.exp(3.044) * 1e-10),
}
CAMB_PARAMS = ['Om', 'Ob', 'h', 'ns']          # parameters that need a new CAMB run
REL_STEP = 0.01                                 # relative finite-difference step for cosmological parameters
STENCIL = (-2, -1, 1, 2)
STENCIL_W = np.array([1, -8, 8, -1]) / 12.0


def camb_key(fid, p=None, s=0):
    return (fid, p, s)


def camb_point(fid, p=None, s=0):
    th = dict(FIDUCIALS[fid])
    if p is not None:
        th[p] = th[p] * (1 + s * REL_STEP)
    return th


def run_camb(th):
    """GR linear P(k,z) (Mpc^3, k in 1/Mpc) of total matter on (Z_MASTER, K), background and CMB-lensing geometry."""
    import camb
    pars = camb.CAMBparams()
    h = th['h']
    omnuh2 = th['mnu'] / 93.14
    pars.set_cosmology(H0=100 * h, ombh2=th['Ob'] * h * h, omch2=(th['Om'] - th['Ob']) * h * h - omnuh2,
                       mnu=th['mnu'], num_massive_neutrinos=1, omk=0.0, tau=th['tau'])
    pars.InitPower.set_params(As=th['As'], ns=th['ns'])
    zs = np.unique(np.concatenate([np.linspace(0, 3, 31), np.geomspace(3.3, 20, 12)]))[::-1]
    pars.set_matter_power(redshifts=list(zs), kmax=60.0, nonlinear=False)
    pars.NonLinear = camb.model.NonLinear_none
    res = camb.get_results(pars)
    PK = res.get_matter_power_interpolator(nonlinear=False, var1='delta_tot', var2='delta_tot',
                                           hubble_units=False, k_hunit=False, extrap_kmax=500.0)
    P = PK.P(Z_MASTER, K)
    zbg = np.linspace(0, 20, 4001)
    chi = res.comoving_radial_distance(zbg)
    H = res.hubble_parameter(zbg)
    zstar = res.get_derived_params()['zstar']
    chistar = float(res.comoving_radial_distance(zstar))
    chi_c = np.linspace(5.0, chistar - 5.0, 900)
    z_c = res.redshift_at_comoving_radial_distance(chi_c)
    # sigma8 of the linear z=0 spectrum (R = 8 Mpc/h), computed here for consistency
    s8 = sigma_R(P[0], 8.0 / h)
    return dict(th=th, P=P.astype(np.float64), zbg=zbg, chi=chi, H=H, zstar=zstar, chistar=chistar,
                chi_c=chi_c, z_c=z_c, s8=s8, camb_version=camb.__version__)


def sigma_R(Pk, R):
    x = K * R
    W = 3 * (np.sin(x) - x * np.cos(x)) / x**3
    return np.sqrt(np.sum(K**3 * Pk / (2 * np.pi**2) * W**2) * DLNK)


# ----------------------------------------------------------------------------------------------
# growth with mu(a)
# ----------------------------------------------------------------------------------------------
def growth(mu, Om, z_out):
    """D (-> a at a=1e-3) and f = dlnD/dlna for mu(a); LCDM background without radiation (as the IAM chain)."""
    OL = 1 - Om

    def rhs(lna, y):
        a = np.exp(lna)
        h2 = Om / a**3 + OL
        dlnh = -1.5 * Om / a**3 / h2
        return [y[1], -(2 + dlnh) * y[1] + 1.5 * Om / a**3 / h2 * mu(a) * y[0]]

    a0 = 1e-3
    s = solve_ivp(rhs, [np.log(a0), 0.0], [a0, a0], dense_output=True, rtol=1e-10, atol=1e-13)
    y = s.sol(np.log(1 / (1 + np.asarray(z_out, float))))
    return y[0], y[1] / y[0]


def mu_function(kind, Om, A=0.0, mu0=0.0):
    OL = 1 - Om
    H2 = lambda a: Om / a**3 + OL
    if kind == 'iam':
        bm = Om / 2.0                                   # beta_m = Omega_m / 2 (IAM, no free parameter)
        mu_iam = lambda a: H2(a) / (H2(a) + bm * np.exp(1 - 1 / a))
        return lambda a: 1 + A * (mu_iam(a) - 1)
    if kind == 'tmpl':                                  # Euclid: mu = 1 + mu0 Omega_DE(z)/Omega_DE(0)
        return lambda a: 1 + mu0 * (OL / H2(a)) / OL
    return lambda a: 1.0


# ----------------------------------------------------------------------------------------------
# HALOFIT, Takahashi et al. 2012 (w = -1)
# ----------------------------------------------------------------------------------------------
def halofit(Plin, z, Om, zmax_nl=6.0):
    """Plin [nz, nk] on K (1/Mpc). Returns P_NL. Linear above zmax_nl (sigma(R)=1 not reached on grid)."""
    PNL = Plin.copy()
    sel = np.where(z <= zmax_nl)[0]
    if len(sel) == 0:
        return PNL
    D2 = K**3 * Plin[sel] / (2 * np.pi**2)
    lo = np.full(len(sel), np.log(1e-4))
    hi = np.full(len(sel), np.log(50.0))
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        s2 = np.sum(D2 * np.exp(-(K[None, :] * np.exp(mid)[:, None])**2), axis=1) * DLNK
        big = s2 > 1
        lo = np.where(big, mid, lo)
        hi = np.where(big, hi, mid)
    R = np.exp(0.5 * (lo + hi))
    y2 = (K[None, :] * R[:, None])**2
    e = np.exp(-y2)
    s0 = np.sum(D2 * e, axis=1) * DLNK
    I1 = np.sum(D2 * y2 * e, axis=1) * DLNK
    I2 = np.sum(D2 * y2**2 * e, axis=1) * DLNK
    n = -3 + 2 * I1 / s0
    C = 4 * I1**2 / s0**2 + 4 * (I1 - I2) / s0
    zz = z[sel]
    H2 = Om * (1 + zz)**3 + 1 - Om
    Omz = Om * (1 + zz)**3 / H2
    an = 10**(1.5222 + 2.8553 * n + 2.3706 * n**2 + 0.9903 * n**3 + 0.2250 * n**4 - 0.6038 * C)
    bn = 10**(-0.5642 + 0.5864 * n + 0.5716 * n**2 - 1.5474 * C)
    cn = 10**(0.3698 + 2.0404 * n + 0.8161 * n**2 + 0.5869 * C)
    gn = 0.1971 - 0.0843 * n + 0.8460 * C
    alpha = np.abs(6.0835 + 1.3373 * n - 0.1959 * n**2 - 5.5274 * C)
    beta = 2.0379 - 0.7354 * n + 0.3157 * n**2 + 1.2490 * n**3 + 0.3980 * n**4 - 0.1682 * C
    nu = 10**(5.2105 + 3.6902 * n)
    f1, f2, f3 = Omz**-0.0307, Omz**-0.0585, Omz**0.0743
    y = K[None, :] * R[:, None]
    fy = y / 4 + y**2 / 8
    DQ = D2 * (1 + D2)**beta[:, None] / (1 + alpha[:, None] * D2) * np.exp(-fy)
    DHp = an[:, None] * y**(3 * f1[:, None]) / (1 + bn[:, None] * y**f2[:, None] + (cn[:, None] * f3[:, None] * y)**(3 - gn[:, None]))
    DH = DHp / (1 + nu[:, None] / y**2)
    PNL[sel] = (DQ + DH) * 2 * np.pi**2 / K**3
    return PNL


# ----------------------------------------------------------------------------------------------
# Eisenstein & Hu 1998 no-wiggle spectrum (for the de-wiggled GCsp model)
# ----------------------------------------------------------------------------------------------
def eh_nowiggle_T(k, Om, Ob, h, Tcmb=2.7255):
    om = Om * h * h
    fb = Ob / Om
    theta = Tcmb / 2.7
    s = 44.5 * np.log(9.83 / om) / np.sqrt(1 + 10 * (Ob * h * h)**0.75)
    aG = 1 - 0.328 * np.log(431 * om) * fb + 0.38 * np.log(22.3 * om) * fb**2
    G = Om * h * (aG + (1 - aG) / (1 + (0.43 * k * s)**4))
    q = k * theta**2 / G * h      # k in 1/Mpc; q = k/(h Mpc^-1) * Theta^2 / Gamma
    L0 = np.log(2 * np.e + 1.8 * q)
    C0 = 14.2 + 731 / (1 + 62.5 * q)
    return L0 / (L0 + C0 * q * q)


def nowiggle(Pk, Om, Ob, h, ns):
    Peh = K**ns * eh_nowiggle_T(K, Om, Ob, h)**2
    r = np.log(Pk / Peh)
    rs = gaussian_filter1d(r, sigma=0.5 / DLNK, mode='nearest')
    return Peh * np.exp(rs)


# ----------------------------------------------------------------------------------------------
# model = CAMB run + sigma8 rescaling + mu/Sigma
# ----------------------------------------------------------------------------------------------
class Model:
    def __init__(self, cb, th, mg):
        """cb: CAMB dict at (Om,Ob,h,ns); th: full parameter dict (incl. s8, A/mu0, Sigma0); mg: kind."""
        self.th, self.cb = th, cb
        Om = th['Om']
        self.Om, self.h = Om, th['h']
        kind = mg
        mu = mu_function(kind, Om, A=th.get('A', 0.0), mu0=th.get('mu0', 0.0))
        D, f = growth(mu, Om, Z_MASTER)
        Dg, fg = growth(lambda a: 1.0, Om, Z_MASTER)
        self.D, self.f, self.Dgr = D, f, Dg
        self.R = D / Dg
        self.Dnorm = D / D[0]                           # model growth normalised to 1 today (IA model)
        amp = (th['s8'] / cb['s8'])**2
        self.Plin = cb['P'] * amp * self.R[:, None]**2
        self.s8z = th['s8'] * D / Dg[0]                 # sigma8(z) of the MG linear field (GR-normalised today)
        Sigma0 = th.get('Sigma0', 0.0)
        OL = 1 - Om
        self.Sigma = lambda z: 1 + Sigma0 / (Om * (1 + z)**3 + OL)   # 1 + Sigma0 Omega_DE(z)/Omega_DE(0)
        self.chi = lambda z: np.interp(z, cb['zbg'], cb['chi'])
        self.H = lambda z: np.interp(z, cb['zbg'], cb['H'])
        self._PNL = None

    @property
    def PNL(self):
        if self._PNL is None:
            self._PNL = halofit(self.Plin, Z_MASTER, self.Om)
        return self._PNL

    def Pz(self, z, nonlinear=False):
        """P(k) at a single redshift, linear interpolation of ln P in z."""
        P = self.PNL if nonlinear else self.Plin
        j = np.clip(np.searchsorted(Z_MASTER, z) - 1, 0, len(Z_MASTER) - 2)
        t = (z - Z_MASTER[j]) / (Z_MASTER[j + 1] - Z_MASTER[j])
        return np.exp((1 - t) * np.log(P[j]) + t * np.log(P[j + 1]))

    def P_on(self, z_arr, nonlinear=True):
        """P on [len(z_arr), nk] (z <= 20)."""
        P = self.PNL if nonlinear else self.Plin
        lP = np.log(P)
        out = np.empty((len(z_arr), len(K)))
        for i in range(len(K)):
            out[:, i] = np.interp(z_arr, Z_MASTER, lP[:, i])
        return np.exp(out)

    def fz(self, z):
        return np.interp(z, Z_MASTER, self.f)

    def fs8(self, z):
        return np.interp(z, Z_MASTER, self.f * self.s8z)


def interp_logk(lP_rows, kq):
    """lP_rows [n, nk] on LNK; kq [m, n] query k (1/Mpc) per row -> P [m, n]; power-law tail beyond grid."""
    x = (np.log(kq) - LNK[0]) / DLNK
    x = np.clip(x, 0, len(LNK) - 1.000001)
    i = np.floor(x).astype(int)
    t = x - i
    cols = np.arange(lP_rows.shape[0])[None, :]
    v = (1 - t) * lP_rows[cols, i] + t * lP_rows[cols, i + 1]
    out = np.exp(v)
    out[kq > K[-1]] = 0.0
    out[kq < K[0]] = 0.0
    return out


# ----------------------------------------------------------------------------------------------
# GCsp (IST:F Sect. 3.2)
# ----------------------------------------------------------------------------------------------
GCSP_BINS = [(0.90, 1.10, 1815.0, 1.46), (1.10, 1.30, 1701.5, 1.61), (1.30, 1.50, 1410.0, 1.75),
             (1.50, 1.80, 940.97, 1.90)]          # zmin, zmax, dN/dOmega dz [deg^-2], b (IST:F Table 3)
AREA_DEG2 = 15000.0
FSKY = AREA_DEG2 / (4 * np.pi * (180 / np.pi)**2)


class GCsp:
    def __init__(self, fid_model, kmax, kmax_units='h/Mpc', kmin_h=0.001, nl_mode='istf_pess', nk=400, nmu=24,
                 hunits=False):
        """nl_mode: 'istf_pess' (global sigma_p, sigma_v factors free), 'istf_opt' (fixed),
        'perbin' (sigma_p, sigma_v free in every bin; Albuquerque et al. 2025)."""
        m = fid_model
        h = m.h
        self.nl_mode = nl_mode
        self.hunits, self.h_ref = hunits, h
        self.zc = np.array([(a + b) / 2 for a, b, _, _ in GCSP_BINS])
        kmax_M = kmax * h if kmax_units == 'h/Mpc' else kmax
        self.k = np.linspace(kmin_h * h, kmax_M, nk)
        x, w = np.polynomial.legendre.leggauss(2 * nmu)
        self.mu, self.wmu = x[x > 0], w[x > 0] * 2          # integrand even in mu
        self.chi_ref = m.chi(self.zc)
        self.H_ref = m.H(self.zc)
        V, n = [], []
        for (a, b, dN, _) in GCSP_BINS:
            v = 4 * np.pi / 3 * (m.chi(b)**3 - m.chi(a)**3) * FSKY
            V.append(v)
            n.append(dN * AREA_DEG2 * (b - a) / v)
        self.V, self.n = np.array(V), np.array(n)
        self.b_fid = np.array([b for *_, b in GCSP_BINS])
        # fiducial sigma_v = sigma_p (IST:F Eq. 81) from the fiducial linear spectrum at each bin centre
        self.sv_fid = np.array([np.sqrt(np.sum(m.Pz(z) * K) * DLNK / (6 * np.pi**2)) for z in self.zc])
        self.s8z_fid = np.interp(self.zc, Z_MASTER, m.s8z)
        self.Pfid = self.pobs(m, self.nuis_fid())
        self.weight = (self.n[:, None, None] * self.Pfid / (1 + self.n[:, None, None] * self.Pfid))**2

    def nuis_fid(self):
        d = {f'gcsp_b{i}': self.b_fid[i] for i in range(4)}
        d.update({f'gcsp_Ps{i}': 0.0 for i in range(4)})
        if self.nl_mode == 'istf_pess':
            d.update({'gcsp_sp': 1.0, 'gcsp_sv': 1.0})
        elif self.nl_mode == 'perbin':
            d.update({f'gcsp_sp{i}': 1.0 for i in range(4)})
            d.update({f'gcsp_sv{i}': 1.0 for i in range(4)})
        return d

    def pobs(self, m, nu):
        out = np.empty((4, len(self.mu), len(self.k)))
        for i, z in enumerate(self.zc):
            qperp = m.chi(z) / self.chi_ref[i]
            qpar = self.H_ref[i] / m.H(z)
            mur = self.mu[:, None]
            kr = self.k[None, :]
            fac = np.sqrt(mur**2 / qpar**2 + (1 - mur**2) / qperp**2)
            k = kr * fac
            mu = mur / qpar / fac
            f = m.fz(z) * nu.get(f'gcsp_fx{i}', 1.0)   # optional per-bin growth-rate scaling (figure only)
            Pl = m.Pz(z)
            Pnw = nowiggle(Pl, m.Om, m.th['Ob'], m.h, m.th['ns'])
            # hunits=True reproduces the IST:F convention in which the model spectrum is tabulated in h/Mpc of the
            # *model* h (k_phys = k_ref[h/Mpc] * h_model / q) and P in (Mpc/h_model)^3; default = physical units
            hr = m.h / self.h_ref if self.hunits else 1.0
            lnk = np.log(k * hr)
            Plk = np.exp(np.interp(lnk, LNK, np.log(Pl)))
            Pnwk = np.exp(np.interp(lnk, LNK, np.log(Pnw)))
            if self.nl_mode == 'istf_pess':
                sp, sv = nu['gcsp_sp'], nu['gcsp_sv']
            elif self.nl_mode == 'perbin':
                sp, sv = nu[f'gcsp_sp{i}'], nu[f'gcsp_sv{i}']
            else:
                sp = sv = 1.0
            sigp = sp * self.sv_fid[i]
            sigv = sv * self.sv_fid[i]
            g = sigv**2 * (1 - mu**2 + mu**2 * (1 + f)**2)
            Pdw = Plk * np.exp(-g * k**2) + Pnwk * (1 - np.exp(-g * k**2))
            b = nu[f'gcsp_b{i}']
            sr = C_KMS * 0.001 * (1 + z) / m.H(z)
            P = (b + f * mu**2)**2 / (1 + (f * k * mu * sigp)**2) * Pdw * np.exp(-(k * mu * sr)**2)
            out[i] = P * hr**3 / (qperp**2 * qpar) + nu[f'gcsp_Ps{i}']
        return out

    def observable(self, m, nu):
        return np.log(self.pobs(m, nu))

    def fisher(self, dobs):
        """dobs: dict name -> d lnP [4, nmu, nk]."""
        names = list(dobs)
        k2 = self.k**2
        wk = np.gradient(self.k)          # uniform grid -> trapezoid-like weights
        wk[0] *= 0.5; wk[-1] *= 0.5
        W = self.weight * (self.wmu[None, :, None]) * (k2 * wk)[None, None, :] * (self.V / (8 * np.pi**2))[:, None, None]
        F = np.zeros((len(names), len(names)))
        for a, na in enumerate(names):
            for b, nb in enumerate(names[a:], a):
                F[a, b] = F[b, a] = np.sum(W * dobs[na] * dobs[nb])
        return names, F


# ----------------------------------------------------------------------------------------------
# photometric 3x2pt (IST:F Sect. 3.3-3.4)
# ----------------------------------------------------------------------------------------------
PHOTO_EDGES = np.array([0.0010, 0.42, 0.56, 0.68, 0.79, 0.90, 1.02, 1.15, 1.32, 1.58, 2.50])
LUM = None


def lum_ratio(z):
    global LUM
    if LUM is None:
        import os
        LUM = np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'scaledmeanlum-E2Sa.dat'))
    return np.interp(z, LUM[:, 0], LUM[:, 1])


def photo_nz(z):
    zm = 0.9
    z0 = zm / np.sqrt(2)
    n = (z / z0)**2 * np.exp(-(z / z0)**1.5)
    cb, zb, sb, co, zo, so, fout = 1.0, 0.0, 0.05, 1.0, 0.1, 0.05, 0.1
    ni = []
    for a, b in zip(PHOTO_EDGES[:-1], PHOTO_EDGES[1:]):
        # integral over z_p in [a,b] of p_ph(z_p|z); c = 1 for both components
        p1 = 0.5 * (erf((z - zb - cb * a) / (np.sqrt(2) * sb * (1 + z))) - erf((z - zb - cb * b) / (np.sqrt(2) * sb * (1 + z))))
        p2 = 0.5 * (erf((z - zo - co * a) / (np.sqrt(2) * so * (1 + z))) - erf((z - zo - co * b) / (np.sqrt(2) * so * (1 + z))))
        ni.append(n * ((1 - fout) * p1 + fout * p2))
    ni = np.array(ni)
    ni /= _trapz(ni, z, axis=1)[:, None]
    return ni


class Photo:
    def __init__(self, fid_model, ell_min=10, ell_max=5000, n_ell=100, lmax_wl=5000, lmax_gc=3000,
                 probes=('L', 'G'), gc_bins=range(10), kcut=None, drop_above=None, nz=300, area=AREA_DEG2):
        """Element-wise scale cuts: IST:F style (lmax_wl, lmax_gc) or Albuquerque (kcut [1/Mpc]:
        ell_max^ij = kcut * min(r(z_i), r(z_j)), z_i bin centre; elements with ell > drop_above removed)."""
        m = fid_model
        self.z = np.linspace(0.001, 2.5, nz)
        self.ni = photo_nz(self.z)
        lam = np.linspace(np.log10(ell_min), np.log10(ell_max), n_ell + 1)
        self.ell = 10**(0.5 * (lam[1:] + lam[:-1]))
        self.dell = 10**lam[1:] - 10**lam[:-1]
        self.fsky = area / (4 * np.pi * (180 / np.pi)**2)
        nbar = 30.0 / 10 * (180 * 60 / np.pi)**2       # per bin, sr^-1
        self.fields = []
        if 'L' in probes:
            self.fields += [('L', i) for i in range(10)]
        if 'G' in probes:
            self.fields += [('G', i) for i in gc_bins]
        nf = len(self.fields)
        self.noise = np.diag([0.30**2 / nbar if t == 'L' else 1 / nbar for t, _ in self.fields])
        self.pairs = [(p, q) for p in range(nf) for q in range(p, nf)]
        zc = 0.5 * (PHOTO_EDGES[1:] + PHOTO_EDGES[:-1])
        rc = m.chi(zc)
        self.b_fid = np.sqrt(1 + zc)
        mask = np.zeros((len(self.ell), len(self.pairs)), bool)
        for j, (p, q) in enumerate(self.pairs):
            (tp, ip), (tq, iq) = self.fields[p], self.fields[q]
            if kcut is not None:
                lm = kcut * min(rc[ip], rc[iq])
            else:
                lm = lmax_wl if (tp == 'L' and tq == 'L') else lmax_gc
            mask[:, j] = self.ell <= lm
            if drop_above is not None:
                mask[:, j] &= self.ell <= drop_above
        self.mask = mask
        self.Cfid = self.cls(m, self.nuis_fid())
        self._prep_cov()

    def nuis_fid(self):
        d = {f'ph_b{i}': self.b_fid[i] for i in range(10)}
        d.update({'A_IA': 1.72, 'eta_IA': -0.41, 'beta_IA': 2.17})
        return d

    def kernels(self, m, nu):
        z = self.z
        chi = m.chi(z)
        H = m.H(z)
        H0 = m.H(0.0)
        dz = np.gradient(z)
        # lensing efficiency  int_z^zmax n_i(z') (1 - chi/chi') dz'
        ratio = 1 - chi[:, None] / chi[None, :]
        tri = np.triu(np.ones((len(z), len(z))))
        eff = np.einsum('iz,yz->iy', self.ni * dz[None, :], np.clip(ratio, 0, None) * tri)
        Wg = 1.5 * m.Om * (H0 / C_KMS)**2 * (1 + z) * chi * eff * m.Sigma(z)[None, :]
        Dn = np.interp(z, Z_MASTER, m.Dnorm)
        AIA = -nu['A_IA'] * 0.0134 * m.Om * (1 + z)**nu['eta_IA'] * lum_ratio(z)**nu['beta_IA'] / Dn
        WIA = self.ni * H / C_KMS * AIA
        WL = Wg + WIA
        WG = np.array([nu[f'ph_b{i}'] for i in range(10)])[:, None] * self.ni * H / C_KMS
        rows = [WL[i] if t == 'L' else WG[i] for t, i in self.fields]
        return np.array(rows), chi, H

    def cls(self, m, nu):
        W, chi, H = self.kernels(m, nu)
        Pz = m.P_on(self.z, nonlinear=True)
        kq = (self.ell[:, None] + 0.5) / chi[None, :]
        Pl = interp_logk(np.log(Pz), kq)             # [nell, nz]
        wz = np.gradient(self.z) * C_KMS / H / chi**2
        return np.einsum('az,bz,lz->lab', W * wz[None, :], W, Pl)

    def observable(self, m, nu):
        C = self.cls(m, nu)
        p = np.array(self.pairs)
        return C[:, p[:, 0], p[:, 1]]

    def _prep_cov(self):
        Ch = self.Cfid + self.noise[None]
        p = np.array(self.pairs)
        self.icov = []
        for l in range(len(self.ell)):
            sel = np.where(self.mask[l])[0]
            if len(sel) == 0:
                self.icov.append((sel, None)); continue
            P, Q = p[sel, 0], p[sel, 1]
            C = Ch[l]
            cov = (C[P[:, None], P[None, :]] * C[Q[:, None], Q[None, :]] +
                   C[P[:, None], Q[None, :]] * C[Q[:, None], P[None, :]])
            cov /= (2 * self.ell[l] + 1) * self.fsky * self.dell[l]
            self.icov.append((sel, np.linalg.inv(cov)))

    def fisher(self, dobs):
        names = list(dobs)
        F = np.zeros((len(names), len(names)))
        for l, (sel, ic) in enumerate(self.icov):
            if ic is None:
                continue
            Dm = np.array([dobs[n][l, sel] for n in names])
            F += Dm @ ic @ Dm.T
        return names, F

    def sigma_C(self):
        """Gaussian 1-sigma error on each element (diagonal of covariance), [nell, npairs]."""
        Ch = self.Cfid + self.noise[None]
        p = np.array(self.pairs)
        P, Q = p[:, 0], p[:, 1]
        var = (Ch[:, P, P] * Ch[:, Q, Q] + Ch[:, P, Q]**2) / ((2 * self.ell + 1) * self.fsky * self.dell)[:, None]
        return np.sqrt(var)


# ----------------------------------------------------------------------------------------------
# Planck CMB lensing
# ----------------------------------------------------------------------------------------------
class CMBLens:
    def __init__(self, fid_model, Lmin=8, Lmax=400, fsky=0.67, snr_target=40.0):
        self.L = np.arange(Lmin, Lmax + 1).astype(float)
        self.fsky = fsky
        self.Cfid = self.observable(fid_model, {})
        # white reconstruction noise N_kk calibrated so that the total S/N = 40 (Planck 2018 VIII abstract)
        def snr(N):
            return np.sqrt(np.sum(self.fsky * (2 * self.L + 1) / 2 * (self.Cfid / (self.Cfid + N))**2))
        lo, hi = 1e-10, 1e-4
        for _ in range(100):
            mid = np.sqrt(lo * hi)
            lo, hi = (mid, hi) if snr(mid) > snr_target else (lo, mid)
        self.N = np.sqrt(lo * hi)
        self.snr = snr(self.N)

    def observable(self, m, nu):
        cb = m.cb
        chi, z = cb['chi_c'], cb['z_c']
        chis = cb['chistar']
        H0 = m.H(0.0)
        W = 1.5 * m.Om * (H0 / C_KMS)**2 * (1 + z) * chi * (chis - chi) / chis * m.Sigma(z)
        zl = np.minimum(z, 20.0)
        Pz = m.P_on(zl, nonlinear=True)
        # above z = 20: matter-era growth scaling of the z = 20 spectrum (MG -> GR there)
        Pz *= np.where(z > 20, ((1 + 20.0) / (1 + z))**2, 1.0)[:, None]
        kq = (self.L[:, None] + 0.5) / chi[None, :]
        Pl = interp_logk(np.log(Pz), kq)
        dchi = np.gradient(chi)
        return np.einsum('z,lz->l', W**2 / chi**2 * dchi, Pl)

    def fisher(self, dobs):
        names = list(dobs)
        w = self.fsky * (2 * self.L + 1) / 2 / (self.Cfid + self.N)**2
        F = np.array([[np.sum(w * dobs[a] * dobs[b]) for b in names] for a in names])
        return names, F


# ----------------------------------------------------------------------------------------------
# DESI f sigma8 (DESI Collaboration 2016, arXiv:1611.00036, Tables 2.3 and 2.5; 14,000 deg^2)
# ----------------------------------------------------------------------------------------------
DESI_T23 = np.array([  # z, sigma_fs8/fs8 [%] kmax 0.1, kmax 0.2   (ELG+LRG+QSO)
    [0.65, 3.31, 1.57], [0.75, 2.10, 1.01], [0.85, 2.12, 1.01], [0.95, 2.09, 0.99], [1.05, 2.23, 1.11],
    [1.15, 2.25, 1.14], [1.25, 2.25, 1.16], [1.35, 2.90, 1.73], [1.45, 3.06, 1.87], [1.55, 3.53, 2.27],
    [1.65, 5.10, 3.61], [1.75, 8.91, 6.81], [1.85, 9.25, 7.07]])
DESI_T25 = np.array([  # BGS
    [0.05, 33.24, 14.08], [0.15, 12.47, 5.25], [0.25, 7.69, 3.25], [0.35, 5.83, 2.60], [0.45, 6.35, 3.77]])


class DESI:
    def __init__(self, fid_model, kcol=1, zmax=None):
        t = np.vstack([DESI_T25, DESI_T23])
        if zmax is not None:
            t = t[t[:, 0] < zmax]
        self.z = t[:, 0]
        self.frac = t[:, kcol] / 100
        self.fid = fid_model.fs8(self.z)

    def observable(self, m, nu):
        return m.fs8(self.z)

    def fisher(self, dobs):
        names = list(dobs)
        w = 1 / (self.frac * self.fid)**2
        F = np.array([[np.sum(w * dobs[a] * dobs[b]) for b in names] for a in names])
        return names, F


# ----------------------------------------------------------------------------------------------
# finite differences
# ----------------------------------------------------------------------------------------------
ABS_STEP = {'A': 0.05, 'mu0': 0.02, 'Sigma0': 0.02, 'eta_IA': 0.02}


def step_of(name, val):
    if name in ABS_STEP:
        return ABS_STEP[name]
    if 'Ps' in name:
        return 100.0
    return REL_STEP * abs(val)


def invert(names, F, keep=None, fix=()):
    idx = [i for i, n in enumerate(names) if n not in fix and (keep is None or n in keep)]
    sub = F[np.ix_(idx, idx)]
    cov = np.linalg.inv(sub)
    return {names[i]: np.sqrt(cov[j, j]) for j, i in enumerate(idx)}, cov, [names[i] for i in idx]


def add_fishers(*nf):
    names = []
    for n, _ in nf:
        names += [x for x in n if x not in names]
    F = np.zeros((len(names), len(names)))
    for n, f in nf:
        ix = [names.index(x) for x in n]
        F[np.ix_(ix, ix)] += f
    return names, F
