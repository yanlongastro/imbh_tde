import numpy as np
from scipy.integrate import quad
from scipy.optimize import bisect

from Constants import Constants
c = Constants()


def DeltaE_o_Ecirc_tidal_capture(rp, sma, Mbh, Ms, Rs, ):
    """
    returns dE/E
    dE: energy loss due to tidal disspation near the BH
    E: defined as G*Mbh*Ms/(2*rp), the energy required at the circular orbit of rp
    """
    rt = 1*(Mbh/Ms)**(1./3.)*Rs  # tidal radius
    # rp = (rp>rt)*rp + (rp<=rt)*rt  # rp should be larger than Rs
    dE = Ms**2/Rs *(Mbh/Ms)**2
    E = Mbh*Ms/2/rp
    dE_E = dE/E
    eta = (Ms/(Ms+Mbh))**0.5 *(rp/Rs)**1.5
    e = 1-rp/sma
    dE_E = (e>=0.5)*dE_E + (e<0.5)*dE_E*e
    eta = eta*(2/(1+e))**0.5 #((1+0.5*np.abs((eta-2)/2)**1.5)/2) # correction due to eccentric orbits
    x = np.log10(eta)
    #x = min(x, 1) # for safety
    # Portegies & Meinen 1993, for n=2
    # ell = 2
    cs = [-0.517, -0.906, 23.88, -93.49, 112.3, -44.15]
    #cs = [-0.397, 1.678, 1.277, -12.42, 9.446, -5.55]
    T2 = 0
    for i in range(6):
        T2 += cs[i]*x**i
    c2 = (Rs/rp)**(2*2+2)*10**T2
    # ell = 3
    cs = [-1.04, -1.354, 37.64, -139.9, 168.2, -66.53]
    #cs = [-0.909, 1.574, 12.37, -57.4, 80.1, -46.43]
    T3 = 0
    for i in range(6):
        T3 += cs[i]*x**i
    c3 = (Rs/rp)**(2*3+2)*10**T3
    return dE_E*(c2+c3)

def ome_root_disk(one_minus_ecc, s, sma,Rs,Ms,tdisk=1.e6*c.yr, tidal_capture=True):    
    Norb=tdisk/s.n.P(sma)
    dm_o_ms = (s.d.sigma(sma*one_minus_ecc)*np.pi*Rs**2 / Ms)
    DeltaEoEcirc = 4* dm_o_ms
    if tidal_capture: # add tidal interaction loss
        DeltaEoEcirc += DeltaE_o_Ecirc_tidal_capture(sma*one_minus_ecc, sma, s.n.Mbh, Ms, Rs, )
    return Norb*DeltaEoEcirc - 1.0

def one_minus_ecc_lc_disk(s, sma,Rs,Ms,tdisk=1.e6*c.yr, tidal_capture=True):
    try:
        return bisect(ome_root_disk,1.e-30,1,args=(s, sma,Rs,Ms,tdisk, tidal_capture)) 
    except:
        return 1.0

def r_lc_disk(sma, s, Rs,Ms,tdisk=1.e6*c.yr, tidal_capture=True):
    return sma*one_minus_ecc_lc_disk(s, sma,Rs,Ms,tdisk, tidal_capture)
r_lc_disk = np.vectorize(r_lc_disk)

def r_lc_mixed(sma, s, Rs, Ms, Mbh,tdisk=1.e6*c.yr, tidal_capture=True, binary_seperation_in_solar=None):
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
        # integrate over angular momenta
        fE = quad(fR_frac,R0,Rlc_tidal,args=(R0))[0] * s.n.fE(E) * (1-R0) # last term, depletion by disk
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
        # integrate over angular momenta
        fE = quad(fR_frac, R0, Rlc_disk,args=(R0))[0] * s.n.fE(E) * (1-R0) # last term, depletion by disk
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
        # integrate over angular momenta
        fE = quad(fR_frac, R0, Rlc_disk,args=(R0))[0] * s.n.fE(E) * (1-R0) # last term, depletion by disk
        return 4.*np.pi**2 * s.n.Jc(s.n.a(E))**2 * fE 
flux_lc_mixed_ = np.vectorize(flux_lc_mixed_)


def flux_lc_tidal(E, s, rlc_tidal):
    """Flux into a normal lc, eg the tidal lc"""
    # TDE loss cone radius, and R0
    R0 = s.n.R0(E,rlc_tidal)
    Rlc_tidal = s.n.Rlc(E,rlc_tidal)
    
    # integrate over angular momenta
    fE = s.n.fE(E)*quad(fR_frac,R0,Rlc_tidal,args=(R0))[0]
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
        # integrate over angular momenta
        fE = quad(fR_frac,Rmin,Rlc_disk,args=(R0))[0] * s.n.fE(E) * (1-R0) 
        # last term, depletion by disk of overall normalization
        return 4.*np.pi**2 * s.n.Jc(s.n.a(E))**2 * fE 

flux_lc_diskcap = np.vectorize(flux_lc_diskcap)



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
from sylpy import stellar_evolution as se
from star_disk import star_disk
from scipy import integrate

def Flc_imf(imf, Mbh=1e3*c.msun, nbin=50, lambda_disk=1, mass_weighted=False, Mdisk=None, tdisk=1e6*c.yr, gamma=1.5,
            gamma_disk=None, rmax_in_rh=None, sigma_h=None, full_loss_cone=False, tidal_capture=True, tde=False, 
            binary_mass_in_solar=None, binary_seperation_in_solar=None):
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
            drf = integrate.quad(flux_lc_mixed_,s.n.E(max(alist[-1], rt)),s.n.E(alist[0]),args=(s, rt, tdisk, tidal_capture, tde))[0]*s.n.c.yr # flux integrated over E
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