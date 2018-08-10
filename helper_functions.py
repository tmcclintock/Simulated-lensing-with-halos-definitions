"""This file is designed to help set up the arguments
for the analysis.
"""
import numpy as np
import cluster_toolkit as ct

#The cosmologies
foxcosmo = {'h'      : 0.6704,
            'om'     : 0.318,
            'ode'    : 0.682,
            'ob'     : 0.049,
            'ok'     : 0.0,
            'sigma8' : 0.835,
            'ns'     : 0.962}
#Choose the fox cosmology
cosmo = foxcosmo

def get_arguments(perc_index, fox_z_index, fox_lambda_index, zi, lj):
    """Creates the arguments dictionary that is passed around
    the various functions.

    Args:
        perc_index (int): index of the percolation model (0-3)
        fox_z_index (int): z-index for the fox snapshots (0-3)
        fox_lambda_index (int): lambda-index for the fox richness bins (0-6)
        zi (int): z-index of the DES Y1 data set (0-1), zi=2 should NOT be used in fox
        lj (int): lambda-index of the DES Y1 data set (3-6)
    """
    if not (0 <= fox_z_index <= 3): raise Exception("Invalid fox_z_inded value.")
    if not (0 <=fox_lambda_index<= 3): raise Exception("Invalid fox_lambda_inded value.")
    if not (0 <= zi <= 1): raise Exception("Invalid zi value.")
    if not (3 <= lj <= 6): raise Exception("Invalid lj value.")
    #Create a dictionary to add things
    print "Creating arguments dictionary"
    args = {}

    #Add the redshift
    zs = [1.0, 0.5, 0.25, 0.0]
    z = zs[fox_z_index]
    args['z'] = z
    print "\tz = %.2f"%z

    #Add the richnesses and compure R_lambda (Rlam) in Mpc/h comoving
    lams = np.loadtxt("Percolation_data/lambdas")
    lam = lams[fox_lambda_index]
    Rlams = (lams/100.)**0.2 #Mpc/h comoving
    Rlam = Rlams[zi, lj]
    args['Rlam'] = Rlam
    print "lambda = %.1f"%lam
    
    #Add the cosmology
    h = cosmo['h'] #to be used in this script
    args['h'] = h
    args['Omega_m'] = cosmo['om']
    print "\th = %.3f"%h
    print "\tOmega_m = %.3f"%args['Omega_m']

    #Add the prior for the multiplicative bias
    deltap1 = np.loadtxt("photoz_calibration/Y1_deltap1.txt")[zi, lj]
    deltap1_var = np.loadtxt("photoz_calibration/Y1_deltap1_var.txt")[zi, lj]
    Am_prior = deltap1 + 0.012 #photo-z + shear
    Am_prior_var = deltap1_var + 0.013**2 #photo-z + shear variance
    args['Am_prior'] = Am_prior
    args['Am_prior_var'] = Am_prior_var
    print "\tAm = %.3f +- %.3f"%(args['Am_prior'], args['Am_prior_var'])

    #Add the Sigma_crit_inverse value, after converting to pc^2/hMsun comoving
    SCI = np.loadtxt("photoz_calibration/sigma_crit_inv.tx")[zi, lj] * h*(1+z)**2
    args['Sigma_crit_inv'] = SCI

    #Add on radial bins
    r = np.logspace(-2, 3, num=1000, base=10) #3d radii, Mpc/h comoving
    Rp = np.logspace(-2, 2.4, num=1000, base=10) #perpendicular radii, Mpc/h comoving
    args['r'] = r
    args['Rp'] = Rp

    #Compute the power spectrum and xi_mm
    #First, make a 'CLASS-formatted' cosmology dictionary
    from classy import Class
    params = {
        "output": "mPk",
        "h":h,
        "sigma8": 0.835,
        "n_s":cosmo["ns"],
        "Omega_b":cosmo["ob"],
        "Omega_cdm":cosmo["om"] - cosmo["ob"],
        "YHe":0.24755048455476272,#By hand, default value
        "P_k_max_h/Mpc":3000.,
        "z_max_pk":1.0,
        "non linear":"halofit"}
    class_cosmo = Class()
    class_cosmo.set(params)
    print "\tConfiguring CLASS cosmology"
    class_cosmo.compute()
    print "\tComputing P(k,z) using CLASS"
    k = np.logspace(-5, 3, base=10, num=4000) #1/Mpc, apparently
    kh = k/h #h/Mpc
    args['k'] = kh
    #Call class to compute P(k,z)
    Pnl  = np.array([cosmo.pk(ki, z) for ki in k])
    Plin = np.array([cosmo.pk_lin(ki, z) for ki in k])
    Pnl *= h**3 #matter power spectrum, (Mpc/h)^3
    Plin *= h**3 #linear matter power spectrum, (Mpc/h)^3
    args['Pnl'] = Pnl
    args['Plin'] = Plin
    #Call the toolkit to compute xi(r,z) from P(k,z)
    print "\tcalling the toolkit to compute xi_mm"
    xi_nl  = ct.xi.xi_mm_at_R(r, k, Pnl)
    xi_lin = ct.xi.xi_mm_at_R(r, k, Plin)
    args['xi_nl'] = xi_nl
    args['xi_lin'] = xi_lin

    #Add the edges of the radial bins (Y1 version), Mpc/h comoving
    #Note that the bin edges are defined in Mpc physical
    Nbins = 15
    Redges = np.logspace(np.log10(0.0323), np.log10(30.), num=Nbins+1)
    args['Redges'] = Redges*h*(1+z) #converted to Mpc/h comoving

    #Add the indices used for scale cuts
    #Note: no upper limit scale cut
    inds = (Redges > 0.2) * (Redges < 9999) #0.2 Mpc physical scale cut
    args['inds'] = inds

    #Add the covariance matrix
    fullbase = "/Users/tmcclintock/Data" #laptop
    #fullbase = "/calvin1/tmcclintock/DES_DATA_FILES" #calvin
    y1base = fullbase+"/DATA_FILES/y1_data_files/FINAL_FILES/"
    covpath = y1base+"SACs/SAC_z%d_l%d.txt"%(zi, lj)
    cov = np.genfromtxt(covpath)
    cov = cov[inds]
    cov = cov[:,inds]
    #Apply the hartlap correction
    Njk = 100.
    D = len(Redges[inds])
    cov *= (Njk-1)/(Njk-D-2)
    print "\tlen(DeltaSigma) = %d\n\tNjk = %d"%(D, Njk)
    icov = np.linalg.inv(cov)
    args['icov'] = icov #(pc^2/Msun physical)^2

    #Add the boost factor data
    boostpath = y1base+"FINAL_FILES/full-unblind-v2-mcal-zmix_y1clust_l%d_z%d_zpdf_boost.dat"%(lj, zi)
    bcovpath = y1base+"FINAL_FILES/full-unblind-v2-mcal-zmix_y1clust_l%d_z%d_zpdf_boost_cov.dat"%(lj, zi)
    Bcov = np.loadtxt(bcovpath)
    Rb, Bp1, Be = np.genfromtxt(boostpath, unpack=True) #Rb is Mpc physical
    Becut = (Be > 1e-6)*(Rb > 0.2) #Some boost factors are 0. Exclude those
    Bp1 = Bp1[Becut]
    Rb  = Rb[Becut]
    Be  = Be[Becut]
    Bcov = Bcov[Becut]
    Bcov = Bcov[:,Becut]
    Njk = 100.
    D = len(Rb)
    Bcov *= (Njk-1.)/(Njk-D-2) #Apply the Hartlap correction
    icov_B = np.linalg.inv(Bcov)
    args['Rb'] = Rb
    args['iBcov'] = icov_B
    args['Bp1'] = Bp1

    #Add the DeltaSigma data vector
    mass_lim_labels = ["3e12_5e12", "5e12_9e12", "9e12_2e13", "2e13_6e13", "6e13_2e14", "2e14_5e14", "5e14_5e15"]
    lab = mass_lim_labels[fox_lambda_index]
    DS = np.loadtxt("fixedmeanDeltaSigma_i%d_%s_z%.1f"%(perc_index, lab, z)) #h Msun/pc^2 comoving
    DS *= h*(1+z)**2 #Msun/pc^2 physical
    args['ds'] = DS[inds] #take only the useful indices

    return args
