"""This file contains the prior and posterior that make up the final likelihood.
"""
import numpy as np
from models import *

def lnprior(params, args):
    """Log prior on the parameters, P(theta)

    Args:
        params (array like): Modeling parameters
        args (dictionary): Extra arguments

    Returns:
        float: Natural log of the prior
    """
    log10M, c, tau, fmis, Am, B0, Rs = params
    #Prior edges
    if log10M < 11. or log10M > 18 or c <= 0. or Am <= 0. or tau <= 0. or fmis <= 0. or fmis >= 1.: return -np.inf
    if Rs <= 0.0 or B0 < 0.0 or Rs > 100.: return -np.inf
    #Gaussian priors on fmis, tau and Am
    Am_prior = args['Am_prior']
    Am_prior_var = args['Am_prior_var']
    LPfmis = (0.25 - fmis)**2/0.08**2 #Y1 
    LPtau  = (0.17 - tau)**2/0.04**2 #Y1
    LPA    = (Am_prior - Am)**2/Am_prior_var #Y1
    return -0.5*(LPfmis + LPtau + LPA)

def lnlike(params, args):
    """Log likelihood of the data given the parameters, P(D|theta)

    Args:
        params (array like): Modeling parameters
        args (dictionary): Extra arguments

    Returns:
        float: Natural log of the likelihood
    """
    #Pull out all parameters
    log10M, c, tau, fmis, Am, B0, Rs = params
    
    #Compute the log likelihood for the DeltaSigma part first
    DS = args['ds'] #DeltaSigma data; Msun/pc^2 physical
    icov = args['icov'] #Inverse covariance matrix
    DS_model = get_delta_sigma_profile(log10M, c, tau, fmis, Am, B0, Rs, args)
    X = DS-DS_model
    LLDS = -0.5*np.dot(X, np.dot(icov, X))

    #Compute the log likelihood for the boost factor part
    Bp1 = args['Bp1'] #1+B (see McClintock paper)
    icov_B = args['iBcov'] #inverse covariance matrix for boost factors
    Rb = args['Rb'] #Boost factor radii, Mpc physical
    Bp1_model = get_boost_model_NFW(B0, Rs, Rb) #Note: Rs has units of Mpc physical
    Xb = Bp1 - Bp1_model
    LLb = -0.5*np.dot(Xb, np.dot(icov_B, Xb))

    #Return the sum of the log likelihoods, not counting ln(det(Cov))
    return LLDS + LLb

def lnpost(params, args):
    """Log posterior of the parameters given the data, P(theta|D)

    Args:
        params (array like): Modeling parameters
        args (dictionary): Extra arguments

    Returns:
        float: Natural log of the posterior, without including P(D)
    """
    full_params = model_swap(params, args)
    lpr = lnprior(full_params, args)
    if not np.isfinite(lpr): return -1e99
    return lpr + lnlike(full_params, args)
