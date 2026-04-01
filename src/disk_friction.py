import numpy as np
from scipy import integrate
from scipy.optimize import brentq

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from Constants import Constants
from star_disk import star_disk
c = Constants()

import stellar_evolution as se


_ONE_MINUS_ECC_LC_CACHE = {}
_ONE_MINUS_ECC_LC_CACHE_MAXSIZE = 20000
_T2_COEFFS = (-44.15, 112.3, -93.49, 23.88, -0.906, -0.517)
_T3_COEFFS = (-66.53, 168.2, -139.9, 37.64, -1.354, -1.04)
_MAX_FLOAT = np.finfo(float).max
_LOG10_MAX_FLOAT = np.log10(np.finfo(float).max)
_LOG10_2 = np.log10(2.0)
_LOG10_ONE_MINUS_ECC_MIN = -30.0
_LOG10_ONE_MINUS_ECC_MAX = 0.0


# Cache and broadcast helpers
def _freeze_cache_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (bool, int, float, str, type(None))):
        return value
    return repr(value)


def _disk_state_key(s):
    cached_key = getattr(s, '_disk_friction_cache_disk_state_key', None)
    if cached_key is not None:
        return cached_key

    disk_items = tuple(
        sorted(
            (key, _freeze_cache_value(val))
            for key, val in s.d.__dict__.items()
            if key != 'c'
        )
    )
    cached_key = (type(s.d).__name__, disk_items)
    s._disk_friction_cache_disk_state_key = cached_key
    return cached_key


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


# Tidal capture and loss-cone roots
@njit(cache=True)
def _deltae_o_ecirc_tidal_capture_scalar(rp, sma, Mbh, Ms, Rs):
    """Numba-compiled scalar kernel for the tidal-capture energy loss."""
    t2_0, t2_1, t2_2, t2_3, t2_4, t2_5 = _T2_COEFFS
    t3_0, t3_1, t3_2, t3_3, t3_4, t3_5 = _T3_COEFFS

    mass_ratio = Mbh / Ms
    rp_over_rs = rp / Rs
    inv_rp_over_rs = Rs / rp

    dE_E = 2.0 * mass_ratio * rp_over_rs
    e = 1.0 - rp / sma
    if e < 0.5:
        dE_E *= e

    x = (
        0.5 * np.log10(Ms / (Ms + Mbh))
        + 1.5 * np.log10(rp_over_rs)
        + 0.5 * (_LOG10_2 - np.log10(1.0 + e))
    )

    T2 = (((((t2_0 * x + t2_1) * x + t2_2) * x + t2_3) * x + t2_4) * x + t2_5)
    T3 = (((((t3_0 * x + t3_1) * x + t3_2) * x + t3_3) * x + t3_4) * x + t3_5)

    log10_inv_rp_over_rs = np.log10(inv_rp_over_rs)
    log10_c2 = 6.0 * log10_inv_rp_over_rs + T2
    log10_c3 = 8.0 * log10_inv_rp_over_rs + T3

    if not np.isfinite(dE_E):
        return np.inf

    log10_coeff_max = log10_c2 if log10_c2 >= log10_c3 else log10_c3
    log10_coeff_min = log10_c3 if log10_c2 >= log10_c3 else log10_c2
    if log10_coeff_max >= _LOG10_MAX_FLOAT:
        return np.inf

    coeff_sum = 10**log10_coeff_max
    if log10_coeff_min > -np.inf:
        coeff_sum *= 1.0 + 10**(log10_coeff_min - log10_coeff_max)

    abs_dE_E = abs(dE_E)
    scale = abs_dE_E if abs_dE_E >= 1.0 else 1.0
    if coeff_sum > _MAX_FLOAT / scale:
        return np.inf
    return dE_E * coeff_sum


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
    rp_arr = np.asarray(rp)
    sma_arr = np.asarray(sma)
    if rp_arr.ndim == 0 and sma_arr.ndim == 0:
        return _deltae_o_ecirc_tidal_capture_scalar(
            float(rp_arr), float(sma_arr), float(Mbh), float(Ms), float(Rs)
        )

    rp_b, sma_b = np.broadcast_arrays(rp_arr, sma_arr)
    result = np.empty(rp_b.shape, dtype=float)
    flat_rp = rp_b.reshape(-1)
    flat_sma = sma_b.reshape(-1)
    flat_result = result.reshape(-1)
    Mbh_f = float(Mbh)
    Ms_f = float(Ms)
    Rs_f = float(Rs)
    for i in range(flat_rp.size):
        flat_result[i] = _deltae_o_ecirc_tidal_capture_scalar(
            float(flat_rp[i]), float(flat_sma[i]), Mbh_f, Ms_f, Rs_f
        )
    return result

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
    if not np.isfinite(Norb) or not np.isfinite(DeltaEoEcirc):
        return np.inf
    if DeltaEoEcirc > _MAX_FLOAT / max(abs(Norb), 1.0):
        return np.inf
    return Norb*DeltaEoEcirc - 1.0


def log_ome_root_disk(log10_one_minus_ecc, s, sma, Rs, Ms, tdisk=1.e6*c.yr, tidal_capture=True):
    """
    Same root equation as ome_root_disk, parameterized by log10(1-e).
    """
    return ome_root_disk(10**log10_one_minus_ecc, s, sma, Rs, Ms, tdisk, tidal_capture)


def _make_ome_root_disk_func(s, sma, Rs, Ms, tdisk, tidal_capture):
    """Build a scalar root function with precomputed orbit-independent factors."""
    Norb = tdisk / s.n.P(sma)
    sigma_func = s.d.sigma
    area_over_mass = np.pi * Rs**2 / Ms
    Mbh = s.n.Mbh
    max_delta_e = _MAX_FLOAT / max(abs(Norb), 1.0)

    if tidal_capture:
        def root_func(one_minus_ecc):
            rp = sma * one_minus_ecc
            DeltaEoEcirc = 4.0 * sigma_func(rp) * area_over_mass
            DeltaEoEcirc += DeltaE_o_Ecirc_tidal_capture(rp, sma, Mbh, Ms, Rs)
            if not np.isfinite(DeltaEoEcirc) or DeltaEoEcirc > max_delta_e:
                return np.inf
            return Norb * DeltaEoEcirc - 1.0
    else:
        def root_func(one_minus_ecc):
            rp = sma * one_minus_ecc
            DeltaEoEcirc = 4.0 * sigma_func(rp) * area_over_mass
            if not np.isfinite(DeltaEoEcirc) or DeltaEoEcirc > max_delta_e:
                return np.inf
            return Norb * DeltaEoEcirc - 1.0

    return root_func


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

    root_func = _make_ome_root_disk_func(s, sma, Rs, Ms, tdisk, tidal_capture)
    lower = _LOG10_ONE_MINUS_ECC_MIN
    upper = _LOG10_ONE_MINUS_ECC_MAX
    f_lower = root_func(10**lower)
    if f_lower == 0.0:
        result = 10**lower
    else:
        f_upper = root_func(10**upper)
        if f_upper == 0.0:
            result = 10**upper
        elif f_lower * f_upper > 0.0:
            result = 1.0
        else:
            log10_result = brentq(
                lambda log10_one_minus_ecc: root_func(10**log10_one_minus_ecc),
                lower,
                upper,
                xtol=1e-2,
                rtol=1e-5,
                maxiter=30,
            )
            result = 10**log10_result

    if not np.isfinite(result):
        result = 1.0

    if len(_ONE_MINUS_ECC_LC_CACHE) >= _ONE_MINUS_ECC_LC_CACHE_MAXSIZE:
        _ONE_MINUS_ECC_LC_CACHE.clear()
    _ONE_MINUS_ECC_LC_CACHE[cache_key] = result
    return result


# Array wrappers around scalar kernels
def _map_broadcast_scalar_func(func, *args):
    """Apply a scalar kernel over broadcastable arguments."""
    arrays = [np.asarray(arg) for arg in args]
    if all(arr.ndim == 0 for arr in arrays):
        return func(*[arr.item() for arr in arrays])

    broadcasted = np.broadcast_arrays(*arrays)
    result = np.empty(broadcasted[0].shape, dtype=float)
    for idx in np.ndindex(result.shape):
        result[idx] = func(*[arr[idx].item() for arr in broadcasted])
    return result


def _wrap_unary_scalar(scalar_func, arg):
    return _map_broadcast_scalar_func(scalar_func, arg)


def _wrap_binary_scalar(scalar_func, arg1, arg2):
    return _map_broadcast_scalar_func(scalar_func, arg1, arg2)


# Loss-cone radii
def _r_lc_disk_scalar(sma, s, Rs,Ms,tdisk=1.e6*c.yr, tidal_capture=True):
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


def r_lc_disk(sma, s, Rs,Ms,tdisk=1.e6*c.yr, tidal_capture=True):
    sma_arr = np.asarray(sma)
    if sma_arr.ndim == 0:
        return _r_lc_disk_scalar(sma_arr.item(), s, Rs, Ms, tdisk, tidal_capture)
    result = np.empty(sma_arr.shape, dtype=float)
    flat_sma = sma_arr.reshape(-1)
    flat_result = result.reshape(-1)
    for i in range(flat_sma.size):
        flat_result[i] = _r_lc_disk_scalar(flat_sma[i].item(), s, Rs, Ms, tdisk, tidal_capture)
    return result


def _r_lc_mixed_scalar(sma, s, Rs, Ms, Mbh,tdisk=1.e6*c.yr, tidal_capture=True, binary_seperation_in_solar=None):
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


def r_lc_mixed(sma, s, Rs, Ms, Mbh,tdisk=1.e6*c.yr, tidal_capture=True, binary_seperation_in_solar=None):
    sma_arr = np.asarray(sma)
    if sma_arr.ndim == 0:
        return _r_lc_mixed_scalar(
            sma_arr.item(), s, Rs, Ms, Mbh, tdisk, tidal_capture, binary_seperation_in_solar
        )
    result = np.empty(sma_arr.shape, dtype=float)
    flat_sma = sma_arr.reshape(-1)
    flat_result = result.reshape(-1)
    for i in range(flat_sma.size):
        flat_result[i] = _r_lc_mixed_scalar(
            flat_sma[i].item(), s, Rs, Ms, Mbh, tdisk, tidal_capture, binary_seperation_in_solar
        )
    return result


# Angular-momentum integrals and fluxes
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


def _fR_frac_integral_array(R_min, R_max, R0):
    """Array version of fR_frac_integral with elementwise masking."""
    R_min = np.asarray(R_min, dtype=float)
    R_max = np.asarray(R_max, dtype=float)
    R0 = np.asarray(R0, dtype=float)
    result = np.zeros(np.broadcast(R_min, R_max, R0).shape, dtype=float)
    R_min_b, R_max_b, R0_b = np.broadcast_arrays(R_min, R_max, R0)

    valid_range = R_max_b > R_min_b
    tol_mask = valid_range & (R0_b <= 1e-50)
    result[tol_mask] = R_max_b[tol_mask] - R_min_b[tol_mask]

    general_mask = valid_range & (R0_b > 1e-50)
    if np.any(general_mask):
        R_min_g = R_min_b[general_mask]
        R_max_g = R_max_b[general_mask]
        R0_g = R0_b[general_mask]
        norm = np.log(1.0 / R0_g) - 1.0 + R0_g
        result[general_mask] = (
            (R_max_g * np.log(R_max_g / R0_g) - R_max_g)
            - (R_min_g * np.log(R_min_g / R0_g) - R_min_g)
        ) / norm
    return result

def _flux_lc_disk_scalar(E, s, rlc_tidal, tdisk=1e6*c.yr, tidal_capture=True):
    """flux into the TDE lc in the presence of a disk"""
    # disk loss cone radius, and R0
    rlcd = _r_lc_disk_scalar(s.n.a(E), s, s.rs, s.ms, tdisk, tidal_capture)
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


def flux_lc_disk(E, s, rlc_tidal, tdisk=1e6*c.yr, tidal_capture=True):
    E_arr, rlc_arr = np.broadcast_arrays(np.asarray(E), np.asarray(rlc_tidal))
    if E_arr.ndim == 0:
        return _flux_lc_disk_scalar(E_arr.item(), s, rlc_arr.item(), tdisk, tidal_capture)
    result = np.empty(E_arr.shape, dtype=float)
    flat_E = E_arr.reshape(-1)
    flat_rlc = rlc_arr.reshape(-1)
    flat_result = result.reshape(-1)
    for i in range(flat_E.size):
        flat_result[i] = _flux_lc_disk_scalar(flat_E[i].item(), s, flat_rlc[i].item(), tdisk, tidal_capture)
    return result

def _flux_lc_mixed_scalar(E, s, rlc, tde=False):
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


def flux_lc_mixed(E, s, rlc, tde=False):
    E_arr, rlc_arr = np.broadcast_arrays(np.asarray(E), np.asarray(rlc))
    if E_arr.ndim == 0:
        return _flux_lc_mixed_scalar(E_arr.item(), s, rlc_arr.item(), tde)
    result = np.empty(E_arr.shape, dtype=float)
    flat_E = E_arr.reshape(-1)
    flat_rlc = rlc_arr.reshape(-1)
    flat_result = result.reshape(-1)
    for i in range(flat_E.size):
        flat_result[i] = _flux_lc_mixed_scalar(flat_E[i].item(), s, flat_rlc[i].item(), tde)
    return result

def _flux_lc_mixed_scalar_integrand(E, s, rlc_tidal, tdisk=1e6*c.yr, tidal_capture=True, tde=False):
    """flux into the TDE lc + disk lc; the version for integration"""
    # disk loss cone radius, and R0
    rlcd = _r_lc_disk_scalar(s.n.a(E), s, s.rs, s.ms, tdisk, tidal_capture)
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


def flux_lc_mixed_(E, s, rlc_tidal, tdisk=1e6*c.yr, tidal_capture=True, tde=False):
    E_arr, rlc_arr = np.broadcast_arrays(np.asarray(E), np.asarray(rlc_tidal))
    if E_arr.ndim == 0:
        return _flux_lc_mixed_scalar_integrand(E_arr.item(), s, rlc_arr.item(), tdisk, tidal_capture, tde)
    result = np.empty(E_arr.shape, dtype=float)
    flat_E = E_arr.reshape(-1)
    flat_rlc = rlc_arr.reshape(-1)
    flat_result = result.reshape(-1)
    for i in range(flat_E.size):
        flat_result[i] = _flux_lc_mixed_scalar_integrand(
            flat_E[i].item(), s, flat_rlc[i].item(), tdisk, tidal_capture, tde
        )
    return result


def _flux_lc_mixed_grid(E_grid, s, rlc_tidal, tdisk=1e6*c.yr, tidal_capture=True, tde=False):
    """
    Grid-oriented version of flux_lc_mixed_ that reuses array computations across E.
    """
    E_grid = np.asarray(E_grid, dtype=float)
    a_grid = s.n.a(E_grid)
    rlcd_grid = r_lc_disk(a_grid, s, s.rs, s.ms, tdisk, tidal_capture)
    rg = 2 * c.G * s.Mbh / (c.c**2)
    rlc_grid = np.maximum(np.maximum(rlc_tidal, rlcd_grid), rg)

    if not tde:
        Rlc_disk = s.n.Rlc(E_grid, rlc_grid)
    else:
        if rlc_tidal < rg:
            return np.zeros_like(E_grid, dtype=float)
        Rlc_disk = s.n.Rlc(E_grid, rlc_tidal)

    R0 = s.n.R0(E_grid, rlc_grid)
    active = R0 < Rlc_disk
    result = np.zeros_like(E_grid, dtype=float)
    if not np.any(active):
        return result

    E_active = E_grid[active]
    a_active = a_grid[active]
    R0_active = R0[active]
    Rlc_active = Rlc_disk[active]
    ang_int = _fR_frac_integral_array(R0_active, Rlc_active, R0_active)
    fE = ang_int * s.n.fE(E_active) * (1.0 - R0_active)
    result[active] = 4.0 * np.pi**2 * s.n.Jc(a_active)**2 * fE
    return result


def _flux_lc_tidal_scalar(E, s, rlc_tidal):
    """Flux into a normal lc, eg the tidal lc"""
    # TDE loss cone radius, and R0
    R0 = s.n.R0(E,rlc_tidal)
    Rlc_tidal = s.n.Rlc(E,rlc_tidal)
    
    ang_int = fR_frac_integral(R0, Rlc_tidal, R0)
    fE = s.n.fE(E) * ang_int
    return 4.*np.pi**2 * s.n.Jc(s.n.a(E))**2 * fE 


def flux_lc_tidal(E, s, rlc_tidal):
    E_arr, rlc_arr = np.broadcast_arrays(np.asarray(E), np.asarray(rlc_tidal))
    if E_arr.ndim == 0:
        return _flux_lc_tidal_scalar(E_arr.item(), s, rlc_arr.item())
    result = np.empty(E_arr.shape, dtype=float)
    flat_E = E_arr.reshape(-1)
    flat_rlc = rlc_arr.reshape(-1)
    flat_result = result.reshape(-1)
    for i in range(flat_E.size):
        flat_result[i] = _flux_lc_tidal_scalar(flat_E[i].item(), s, flat_rlc[i].item())
    return result



def _flux_lc_diskcap_scalar(E, s, rlc_tidal, tidal_capture=True):
    """flux into the disk capture LC in the presence of a disk"""
    # disk loss cone radius, and R0
    rlcd = _r_lc_disk_scalar(s.n.a(E), s, s.rs, s.ms, 1.e6*c.yr, tidal_capture)
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


def flux_lc_diskcap(E, s, rlc_tidal, tidal_capture=True):
    E_arr, rlc_arr = np.broadcast_arrays(np.asarray(E), np.asarray(rlc_tidal))
    if E_arr.ndim == 0:
        return _flux_lc_diskcap_scalar(E_arr.item(), s, rlc_arr.item(), tidal_capture)
    result = np.empty(E_arr.shape, dtype=float)
    flat_E = E_arr.reshape(-1)
    flat_rlc = rlc_arr.reshape(-1)
    flat_result = result.reshape(-1)
    for i in range(flat_E.size):
        flat_result[i] = _flux_lc_diskcap_scalar(flat_E[i].item(), s, flat_rlc[i].item(), tidal_capture)
    return result


# Energy integration and IMF-averaged rates
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
    flux_grid = _flux_lc_mixed_grid(E_grid, s, rlc_tidal, tdisk, tidal_capture, tde)
    return integrate.simpson(flux_grid, x=E_grid)

# Disk-friction diffusion coefficient
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
        mass_bins = m_list.copy()
        imf_fracs = None
    else:
        m_list = np.logspace(np.log10(imf.m_min), np.log10(imf.m_max), num=nbin+1)
        ms_mean = imf.mean_mass*c.msun
        mass_bins = np.sqrt(m_list[:-1] * m_list[1:])
        imf_fracs = np.array(
            [imf.imf_integral(m_list[i], m_list[i+1]) for i in range(len(m_list) - 1)]
        )
    n_mass_bins = max(1, len(m_list) - 1)
    if callable(gamma):
        gamma_clusters = [gamma(m) for m in mass_bins]
    else:
        gamma_clusters = [gamma] * n_mass_bins
    stellar_radii = np.array([se.stellar_radius_tout96(m, 0.02) for m in mass_bins])

    alpha_disk = 1e-3
    q_disk = 1 / lambda_disk
    binary_mass_factor = 0.5 if (
        binary_mass_in_solar is not None and binary_seperation_in_solar is not None
    ) else 1.0

    res = 0.
    res_f = 0.
    imf_norm = 0.
    masses = mass_bins.tolist() if imf is not None else []
    dF_dm = np.empty(n_mass_bins, dtype=float) if imf is not None else []
    for i in range(n_mass_bins):
        m = mass_bins[i]
        gamma_cluster = gamma_clusters[i]
        r = stellar_radii[i]
        s = star_disk(Mbh=Mbh,
                    ms=m*c.msun,
                    rs=r*c.rsun,
                    lambda_disk=lambda_disk,
                    alpha_disk=alpha_disk,
                    Q_disk=q_disk,
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
            imf_frac = imf_fracs[i]
            dr *= imf_frac
            drf *= imf_frac
            imf_norm += imf_frac
        if mass_weighted:
            dr *= m
            drf *= m
            dr *= binary_mass_factor
            drf *= binary_mass_factor
        res += dr
        res_f += drf
        if imf is not None:
            dF_dm[i] = drf / (m_list[i+1] - m_list[i])
    if imf is not None:
        res /= imf_norm
        res_f /= imf_norm
        dF_dm = dF_dm.tolist()
    return alist, res, res_f, s, masses, dF_dm
