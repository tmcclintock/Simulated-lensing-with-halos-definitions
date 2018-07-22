"""This contains the routines that return both the DeltaSigma and boost factor
profiles.
"""
import numpy as np
import cluster_toolkit as ct

def model_swap(params, args):
    """Take in the params array and the args dictionary, which contains
    the name of the model. Return the full set of parameters used
    in the modeling functions. This makes it seemless to switch between
    different model configurations (for instance between the full model
    and one with an M-c relation).

    Args:
        params (float or array like): Model parameters
        args (dictionary): Dictionary containing extra arguments, including the model name
    
    Returns:
        array like: Full parameter list
    """
    if args['name'] == 'full':
        return params
    raise Exception("Model %s not implemented yet."%name)

def get_delta_sigma_profile(log10M, c, tau, fmis, Am, B0, Rs, args):
    """Return the DeltaSigma(R) profile. Note that this does not compute
    the profile in averaged angular bins. It computes the continuous profile
    at a finely sampled range of radii in units of Mph/h, supplied in
    the args dictionary.

    Args:
        log10M (float): Log base 10 of the Mass (Msun/h)
        c (float): Concentration (r200/r_scale)
        tau (float): Miscentering length that describes how large the
            distribution is of miscentered clusters (Rmis/R_lambda)
        fmis (float): Fraction of clusters that are miscentered
        Am (float): Multiplicative bias (shear+photoz)
        B0 (float): Boost factor amplitude
        Rs (float): Boost factor scale radius (Mpc physical)
        args (dict): Contains all extra information used for the computation

    Returns:
        array like: DeltaSigma(R) lensing profile (h Msun/pc^2 comoving)
    """
    M = 10**log10M #Mass Msun/h
    r = args['r'] #3d radii Mpc/h comoving
    Rp = args['Rp'] #projected radii Mpc/h comoving
    k = args['k'] #wavenumber h/Mpc
    Plin = args['Plin'] #linear power spectrum (Mpc/h)^3
    Pnl = args['Pnl'] #nonlinear power spectrum (Mpc/h)^3
    xi_mm = args['xi_nl'] #matter power spectrum
    Rlam = args['Rlam'] #Richness radii; Mpc/h comoving
    z = args['z'] #Redshift
    h = args['h'] #Reduced hubble constant
    Omega_m = args['Omega_m'] #Omega_matter
    Sigma_crit_inv = args['Sigma_crit_inv'] #pc^2/hMsun comoving

    #Compute the halo-matter correlation function
    xi_nfw = ct.xi.xi_nfw_at_R(r, M, c, Omega_m) #NFW
    bias = ct.bias.bias_at_M(M, k, Plin, Omega_m) #halo bias
    xi_2halo = ct.xi.xi_2halo(bias, xi_mm) #2halo term
    xi_hm = ct.xi.xi_hm(xi_nfw, xi_2halo)

    #Compute projected profiles; hMsun/pc^2
    Sigma = ct.deltasigma.Sigma_at_R(Rp, r, xi_hm, M, c, Omega_m)
    DeltaSigma = ct.deltasigma.DeltaSigma_at_R(Rp, Rp, Sigma, M, c, Omega_m)

    #Compute miscentered projected profiles; hMsun/pc^2
    Rmis = tau*Rlam #Mpc/h comoving
    Sigma_mis = ct.miscentering.Sigma_mis_at_R(Rp, Rp, Sigma, M, c, Omega_m, Rmis, kernel='exponential') #use the kernel from DES Y1
    DeltaSigma_mis = ct.miscentering.DeltaSigma_mis_at_R(Rp, Rp, Sigma_mis)

    #Combine the centered and miscentered parts
    full_Sigma = (1-fmis)*Sigma + fmis*Sigma_mis
    full_DeltaSigma = (1-fmis)*DeltaSigma + fmis*DeltaSigma_mis

    #Apply the multiplicative bias and reduced shear
    full_DeltaSigma *= Am/(1 - full_Sigma*Sigma_crit_inv)

    #Compute and apply the boost factor
    #Note the unit change for Rs
    boost = ct.boostfactors.boost_nfw_at_R(Rp, B0, Rs*h*(1+z))
    full_DeltaSigma /= boost

    #Average within the bins
    Redges = args['Redges'] #Mpc/h comoving
    ave_DeltaSigma = ct.averaging.average_profile_in_bins(Redges, Rp, full_DeltaSigma)

    #Change units
    ave_DeltaSigma *= h*(1+z)**2 #Msun/pc^2 physical

    #Pick out the useful indices given the scale cuts, and return
    inds = args['inds']
    return ave_DeltaSigma[inds] #Msun/pc^2 physical

def get_boost_model_NFW(B0, Rs, Rp):
    """Return the NFW boost factor model. Note that it is assumed that
    Rs and Rp have either the same units.
    
    Args:
        B0 (float): boost factor amplitude
        Rs (float): boost factor scale radius. Units are the same as Rp
        Rp (float or array like): projected radii. Units are the same as Rs
    
    Returns:
        float or array like: NFW boost factor profile
    """
    return ct.boostfactors.boost_nfw_at_R(Rp, B0, Rs)
