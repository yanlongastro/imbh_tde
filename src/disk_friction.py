import numpy as np
from scipy.optimize import brentq

from Constants import Constants
c = Constants()


_ONE_MINUS_ECC_LC_CACHE = {}
_ONE_MINUS_ECC_LC_CACHE_MAXSIZE = 20000
_T2_COEFFS = (-44.15, 112.3, -93.49, 23.88, -0.906, -0.517)
_T3_COEFFS = (-66.53, 168.2, -139.9, 37.64, -1.354, -1.04)
_LOG10_MAX_FLOAT = np.log10(np.finfo(float).max)


def _freeze_cache_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (bool, int, float, str, type(None))):
        return value
    return repr(value)


def _disk_state_key(s):
    disk_items = tuple(
        sorted(
            (key, _freeze_cache_value(val))
            for key, val in s.d.__dict__.items()
            if key != 'c'
        )
    )
    return (type(s.d).__name__, disk_items)


def _one_minus_ecc_lc_disk_cache_key(s, sma, Rs, Ms, tdisk, tidal_capture):
    return (
        _freeze_cache_value(s.Mbh),
        _disk_state_key(s),
        _freeze_cache_value(sma),
        _freeze_cache_value(Rs),
        _freeze_cache_value(Ms),
        _freeze_cache_value(tdisk),
        bool(tidal_capture),
    )


def DeltaE_o_Ecirc_tidal_capture(rp, sma, Mbh, Ms, Rs, ):
    """
    returns dE/E per orbit
    dE: energy loss due to tidal disspation near the BH
    E: defined as G*Mbh*Ms/(2*rp), the energy required at the circular orbit of rp
    ------------
    rp: pericenter distance in cgs
    sma: semi-major axis in cgs
    Mbh: mass of the black hole in cgs
    Ms: mass of the star in cgs
    Rs: stellar radius in cgs
    ------------
    """
    mass_ratio = Mbh / Ms
    rp_over_rs = rp / Rs
    inv_rp_over_rs = Rs / rp

    dE_E = 2.0 * mass_ratio * rp_over_rs
    e = 1.0 - rp / sma
    if e < 0.5:
        dE_E *= e

    eta = np.sqrt(Ms / (Ms + Mbh)) * rp_over_rs**1.5
    eta *= np.sqrt(2.0 / (1.0 + e))
    x = np.log10(eta)

    # Horner evaluation of the Portegies & Meinen 1993 polynomial fits.
    T2 = (((((_T2_COEFFS[0] * x + _T2_COEFFS[1]) * x + _T2_COEFFS[2]) * x
            + _T2_COEFFS[3]) * x + _T2_COEFFS[4]) * x + _T2_COEFFS[5])
    T3 = (((((_T3_COEFFS[0] * x + _T3_COEFFS[1]) * x + _T3_COEFFS[2]) * x
            + _T3_COEFFS[3]) * x + _T3_COEFFS[4]) * x + _T3_COEFFS[5])

    log10_inv_rp_over_rs = np.log10(inv_rp_over_rs)
    log10_c2 = 6.0 * log10_inv_rp_over_rs + T2
    log10_c3 = 8.0 * log10_inv_rp_over_rs + T3

    if log10_c2 < _LOG10_MAX_FLOAT:
        c2 = 10**log10_c2
    else:
        c2 = np.inf

    if log10_c3 < _LOG10_MAX_FLOAT:
        c3 = 10**log10_c3
    else:
        c3 = np.inf

    coeff_sum = c2 + c3
    if not np.isfinite(coeff_sum):
        return np.inf
    return dE_E * coeff_sum

def ome_root_disk(one_minus_ecc, s, sma,Rs,Ms,tdisk=1.e6*c.yr, tidal_capture=True):    
    """
    for given semi-major axis and eccentricy, returns the change in the energy 
    ------------
    one_minus_ecc: 1-eccentricity, the eccentricity of the orbit
    s: star_disk object, contains the disk parameters
    sma: semi-major axis of the orbit
    Rs: stellar radius in cgs
    Ms: stellar mass in cgs
    tdisk: disk lifetime in cgs, default 1e6 years
    tidal_capture: whether to include the tidal capture effect
    """
    Norb=tdisk/s.n.P(sma)
    dm_o_ms = (s.d.sigma(sma*one_minus_ecc)*np.pi*Rs**2 / Ms)
    DeltaEoEcirc = 4* dm_o_ms
    if tidal_capture: # add tidal interaction loss
        DeltaEoEcirc += DeltaE_o_Ecirc_tidal_capture(sma*one_minus_ecc, sma, s.n.Mbh, Ms, Rs, )
    return Norb*DeltaEoEcirc - 1.0

def one_minus_ecc_lc_disk(s, sma,Rs,Ms,tdisk=1.e6*c.yr, tidal_capture=True):
    """
    returns the critical one_minus_eccentricity for the loss cone
    ------------
    s: star_disk object, contains the disk parameters
    sma: semi-major axis of the orbit
    Rs: stellar radius in cgs
    Ms: stellar mass in cgs
    tdisk: disk lifetime in cgs, default 1e6 years
    tidal_capture: whether to include the tidal capture effect
    ------------
    """
    cache_key = _one_minus_ecc_lc_disk_cache_key(s, sma, Rs, Ms, tdisk, tidal_capture)
    cached_value = _ONE_MINUS_ECC_LC_CACHE.get(cache_key)
    if cached_value is not None:
        return cached_value

    lower = 1.e-30
    upper = 1.0
    f_lower = ome_root_disk(lower, s, sma, Rs, Ms, tdisk, tidal_capture)
    if f_lower == 0.0:
        result = lower
    else:
        f_upper = ome_root_disk(upper, s, sma, Rs, Ms, tdisk, tidal_capture)
        if f_upper == 0.0:
            result = upper
        elif f_lower * f_upper > 0.0:
            result = 1.0
        else:
            result = brentq(
                ome_root_disk,
                lower,
                upper,
                args=(s, sma, Rs, Ms, tdisk, tidal_capture),
                xtol=1e-8,
                rtol=1e-6,
                maxiter=100,
            )

    if not np.isfinite(result):
        result = 1.0

    if len(_ONE_MINUS_ECC_LC_CACHE) >= _ONE_MINUS_ECC_LC_CACHE_MAXSIZE:
        _ONE_MINUS_ECC_LC_CACHE.clear()
    _ONE_MINUS_ECC_LC_CACHE[cache_key] = result
    return result

def r_lc_disk(sma, s, Rs,Ms,tdisk=1.e6*c.yr, tidal_capture=True):
    """
    returns the loss cone radius for the disk
    ------------
    sma: semi-major axis of the orbit
    s: star_disk object, contains the disk parameters
    Rs: stellar radius in cgs
    Ms: stellar mass in cgs
    tdisk: disk lifetime in cgs, default 1e6 years
    tidal_capture: whether to include the tidal capture effect
    ------------
    """
    return sma*one_minus_ecc_lc_disk(s, sma,Rs,Ms,tdisk, tidal_capture)
r_lc_disk = np.vectorize(r_lc_disk)

def r_lc_mixed(sma, s, Rs, Ms, Mbh,tdisk=1.e6*c.yr, tidal_capture=True, binary_seperation_in_solar=None):
    """
    returns the redefined loss cone radius, which is the larger of the disk lc and the tidal lc
    ------------
    sma: semi-major axis of the orbit
    s: star_disk object, contains the disk parameters
    Rs: stellar radius in cgs
    Ms: stellar mass in cgs
    Mbh: mass of the black hole in cgs
    tdisk: disk lifetime in cgs, default 1e6 years
    tidal_capture: whether to include the tidal capture effect
    binary_seperation_in_solar
    ------------
    """
    # r_lc = max(r_t, r_lc_disk)
    rd = sma*one_minus_ecc_lc_disk(s, sma,Rs,Ms,tdisk, tidal_capture)
    rt = 1*(Mbh/Ms)**(1./3.)*Rs
    if binary_seperation_in_solar is not None:
            rt *= binary_seperation_in_solar # larger TDE radius for binaries
    rg = 2*c.G*Mbh/(c.c**2)  # Schwarzschild radius
    return max(rd, rt, rg)
r_lc_mixed = np.vectorize(r_lc_mixed)


def fR_frac(R,R0):
    tol=1e-50
    return np.where(R0>tol,np.log(R/R0)/(np.log(1/R0)-1.0+R0),1.0)

def fR_frac_integral(R_min, R_max, R0):
    """
    Analytic integral of fR_frac dR between R_min and R_max.
    This avoids a nested numerical quadrature inside the E integral.
    """
    tol = 1e-50
    if R_max <= R_min:
        return 0.0
    if R0 <= tol:
        return R_max - R_min
    norm = np.log(1.0 / R0) - 1.0 + R0
    return (
        (R_max * np.log(R_max / R0) - R_max)
        - (R_min * np.log(R_min / R0) - R_min)
    ) / norm

def flux_lc_disk(E, s, rlc_tidal, tdisk=1e6*c.yr, tidal_capture=True):
    """flux into the TDE lc in the presence of a disk"""
    # disk loss cone radius, and R0
    rlcd = r_lc_disk(s.n.a(E),s, s.rs,s.ms, tdisk, tidal_capture)
    Rlc_disk = s.n.Rlc(E,rlcd)
    R0_d = s.n.R0(E,rlcd)
    R0_t = s.n.R0(E,rlc_tidal)
    R0 = max(R0_d,R0_t)
    Rlc_tidal = s.n.Rlc(E,rlc_tidal)
    
    if (R0_d>=Rlc_tidal):
        return 0.0
    else:
        ang_int = fR_frac_integral(R0, Rlc_tidal, R0)
        fE = ang_int * s.n.fE(E) * (1-R0) # last term, depletion by disk
        return 4.*np.pi**2 * s.n.Jc(s.n.a(E))**2 * fE 
flux_lc_disk = np.vectorize(flux_lc_disk)

def flux_lc_mixed(E, s, rlc, tde=False):
    """flux into the TDE lc + disk lc"""
    # disk loss cone radius, and R0
    if not tde:
        Rlc_disk = s.n.Rlc(E,rlc)
    else:
        rt = 1*(s.Mbh/s.ms)**(1./3.)*s.rs  # specify the tidal radius
        rg = 2*c.G*s.Mbh/(c.c**2)
        if rt<rg: return 0.0
        Rlc_disk = s.n.Rlc(E, rt)
    R0 = s.n.R0(E,rlc)
    
    if (R0>=Rlc_disk):
        return 0.0
    else:
        ang_int = fR_frac_integral(R0, Rlc_disk, R0)
        fE = ang_int * s.n.fE(E) * (1-R0) # last term, depletion by disk
        return 4.*np.pi**2 * s.n.Jc(s.n.a(E))**2 * fE 
flux_lc_mixed = np.vectorize(flux_lc_mixed)

def flux_lc_mixed_(E, s, rlc_tidal, tdisk=1e6*c.yr, tidal_capture=True, tde=False):
    """flux into the TDE lc + disk lc; the version for integration"""
    # disk loss cone radius, and R0
    rlcd = r_lc_disk(s.n.a(E),s, s.rs,s.ms, tdisk, tidal_capture)
    rg = 2*c.G*s.Mbh/(c.c**2)  # Schwarzschild radius
    rlc = max(rlc_tidal, rlcd, rg)
    if not tde:
        Rlc_disk = s.n.Rlc(E,rlc)
    else:
        Rlc_disk = s.n.Rlc(E,rlc_tidal)
        if rlc_tidal<rg: return 0.0
    R0 = s.n.R0(E,rlc)
    
    if (R0>=Rlc_disk):
        return 0.0
    else:
        ang_int = fR_frac_integral(R0, Rlc_disk, R0)
        fE = ang_int * s.n.fE(E) * (1-R0) # last term, depletion by disk
        return 4.*np.pi**2 * s.n.Jc(s.n.a(E))**2 * fE 
flux_lc_mixed_ = np.vectorize(flux_lc_mixed_)


def flux_lc_tidal(E, s, rlc_tidal):
    """Flux into a normal lc, eg the tidal lc"""
    # TDE loss cone radius, and R0
    R0 = s.n.R0(E,rlc_tidal)
    Rlc_tidal = s.n.Rlc(E,rlc_tidal)
    
    ang_int = fR_frac_integral(R0, Rlc_tidal, R0)
    fE = s.n.fE(E) * ang_int
    return 4.*np.pi**2 * s.n.Jc(s.n.a(E))**2 * fE 
flux_lc_tidal = np.vectorize(flux_lc_tidal)



def flux_lc_diskcap(E, s, rlc_tidal, tidal_capture=True):
    """flux into the disk capture LC in the presence of a disk"""
    # disk loss cone radius, and R0
    rlcd = r_lc_disk(s.n.a(E), s, s.rs,s.ms, tidal_capture)
    Rlc_disk = s.n.Rlc(E,rlcd)
    R0_d = s.n.R0(E,rlcd)
    rt = 1*(s.Mbh/s.ms)**(1./3.)*s.rs
    R0_t = s.n.R0(E,rt)
    R0 = max(R0_d,R0_t)
    Rlc_tidal = s.n.Rlc(E,rlc_tidal)
    Rmin = max(R0,Rlc_tidal)
    
    if (Rlc_tidal >= Rlc_disk):
        return 0.0
    else:
        ang_int = fR_frac_integral(Rmin, Rlc_disk, R0)
        fE = ang_int * s.n.fE(E) * (1-R0) 
        # last term, depletion by disk of overall normalization
        return 4.*np.pi**2 * s.n.Jc(s.n.a(E))**2 * fE 

flux_lc_diskcap = np.vectorize(flux_lc_diskcap)


def simpson_integrate_flux_lc_mixed(E_min, E_max, s, rlc_tidal, tdisk=1e6*c.yr,
                                    tidal_capture=True, tde=False, nE=101):
    """
    Fixed-grid Simpson integration of flux_lc_mixed_ over energy.
    Uses the same integrand and bounds as the previous quad-based version.
    """
    if E_max <= E_min:
        return 0.0
    if nE % 2 == 0:
        nE += 1
    E_grid = np.geomspace(E_min, E_max, nE)
    flux_grid = flux_lc_mixed_(E_grid, s, rlc_tidal, tdisk, tidal_capture, tde)
    return integrate.simpson(flux_grid, x=E_grid)



# disk friction boosts the diffusion coefficient
def R0_disk(E, s, rlc, Rs, Ms, rp_in_rlc=1):
    q = s.n.qE(E, rlc) # two-body relaxation
    
    # then we calculate the disk friction
    rp = rlc*rp_in_rlc
    dm_o_ms = (s.d.sigma(rp)*np.pi*Rs**2 / Ms)
    q_ = (dm_o_ms)**2 *(rp/rlc)
    # add them up, and get the R0
    q += q_
    alpha = (q**4 + q**2)**0.25
    
    #alpha = (q>=1)*q + (q<1)*(0.186*q+0.824*q**0.5) # Cohn & Kulsrud approximation
    
    return s.n.Rlc(E,rlc)*np.exp(-alpha)

import sys
sys.path.append('/Users/yanlong/Library/CloudStorage/OneDrive-Personal/Science/MyCode')
from sylpy import stellar_evolution as se # or check https://github.com/yanlongastro/sylpy/blob/main/stellar_evolution.py
from star_disk import star_disk
from scipy import integrate

def Flc_imf(imf, Mbh=1e3*c.msun, nbin=50, lambda_disk=1, mass_weighted=False, Mdisk=None, tdisk=1e6*c.yr, gamma=1.5,
            gamma_disk=None, rmax_in_rh=None, sigma_h=None, full_loss_cone=False, tidal_capture=True, tde=False, 
            binary_mass_in_solar=None, binary_seperation_in_solar=None, nE=101):
    """
    returns:
    - semi-major axis of the orbits considered in calculation
    - the flux, as a function of semi-major axis: dF/dE, in 1/yr/(m/s)^2; or Msun/yr/(m/s)^2 if mass_weighted
    - the flux intergrated over energy, in 1/(yr), or Msun/yr is mass_weighted
    - the nsc model class used here
    - masses of the stars, if an IMF is given
    - the flux as a function of mass: dF/dm
    ----------
    Mbh : centeral BH mass in cgs
    nbin : bins of the imf
    lambda_disk : 
    mass_weighted : if true, we get the mass flux, otherwise we get the event rate
    Mdisk : disk mass in cgs
    tdisk : disk lifetime in cgs
    gamma : density slope of the cluster inside the BH influcence, can be a function of stellar mass
    gamma_disk : density slope of the spherical disk outside the tidal radius
    rmax_in_rh : how far the spherical disk extends, in half-mass radius
    sigma_h : velocity dispersion around the BH, if None, use the Komendy & Ho 2013
    full_loss_cone : special case that returns the full loss cone solution
    tidal_capture : whether to include the tidal capture effect
    tde : if True, return the TDE rate assuming stars in the disk are abalted; otherwise return the total rates
    binary_mass_in_solar : speical case for binary distribution functions
    binary_seperation_in_solar: only for binaries
    nE : number of energy samples for Simpson integration of the total rate
    """
    if binary_mass_in_solar is not None or binary_seperation_in_solar is not None:
        assert imf is None
    if imf is None:
        m_list = np.array([1])
        if binary_mass_in_solar is not None:
            m_list *= binary_mass_in_solar
        ms_mean = None
    else:
        m_list = np.logspace(np.log10(imf.m_min), np.log10(imf.m_max), num=nbin+1)
        ms_mean = imf.mean_mass*c.msun
    res = 0.
    res_f = 0.
    imf_norm = 0.
    masses = []
    dF_dm = []
    
    for i in range(max(1, len(m_list)-1)):
        if imf is not None:
            m = np.sqrt(m_list[i]*m_list[i+1])
        else:
            m = m_list[i]

        if callable(gamma):
            gamma_cluster = gamma(m)
        else:
            gamma_cluster = gamma
        
        r = se.stellar_radius_tout96(m, 0.02)
        s = star_disk(Mbh=Mbh,
                    ms=m*c.msun,
                    rs=r*c.rsun,
                    lambda_disk=lambda_disk,
                    alpha_disk =1e-3,
                    Q_disk=1/lambda_disk,
                    ms_mean=ms_mean,
                    gamma=gamma_cluster,
                    Mdisk=Mdisk,
                    gamma_disk=gamma_disk,
                    rmax_in_rh=rmax_in_rh,
                    sigma_h=sigma_h,
                    )           
        rt = 1*(s.Mbh/s.ms)**(1./3.)*s.rs  # specify the tidal radius
        if binary_seperation_in_solar is not None:
            rt *= binary_seperation_in_solar # larger TDE radius for binaries
        rt_sun = 1*(s.Mbh/c.msun)**(1./3.)*c.rsun  # specify the tidal radius
        alist = np.logspace(max(np.log10(s.n.r_h/s.n.c.pc)-4, np.log10(rt_sun/s.n.c.pc)+3), 
                            np.log10(s.n.r_h/s.n.c.pc)+3, 50)*s.n.c.pc
        if not full_loss_cone:
            rlcd = r_lc_mixed(alist, s, s.rs, s.ms, s.Mbh, tdisk, tidal_capture, binary_seperation_in_solar) # newly defined loss cone radius
            dr = flux_lc_mixed(s.n.E(alist), s, rlcd, tde=tde) # flux dF/dE
            drf = simpson_integrate_flux_lc_mixed(
                s.n.E(s.n.r_h),
                s.n.E(alist[0]),
                s,
                rt,
                tdisk=tdisk,
                tidal_capture=tidal_capture,
                tde=tde,
                nE=nE,
            ) * s.n.c.yr
            drf = max(drf, 0) # for safety, avoid negative rates due to numerical issues
        else:
            dr = s.n.flux_flc(s.n.E(alist), rt)
            drf = 0 # not meaningful for full loss cone, ignore it
        if imf is not None:
            imf_frac = imf.imf_integral(m_list[i], m_list[i+1])
            dr *= imf_frac
            drf *= imf_frac
            imf_norm += imf_frac
            # print(m, imf_frac)
        if mass_weighted:
            dr *= m
            drf *= m
            if binary_mass_in_solar is not None and binary_seperation_in_solar is not None:
                # for binaires, only one of the stars get swallowed by the IMBH (Hills mechanism)
                dr /= 2
                drf /= 2
        res += dr
        res_f += drf
        if imf is not None:
            masses.append(m)
            dF_dm.append(drf/(m_list[i+1]-m_list[i]))
    if imf is not None:
        res /= imf_norm
        res_f /= imf_norm
    return alist, res, res_f, s, masses, dF_dm
