"""
This is the script to run to obtain mass estimates for the lensing profiles.
"""
import numpy as np
from likelihoods import *
from helper_functions import *
import scipy.optimize as op
import emcee

def main():
    #Specify which percolation model
    perc_index = 0

    #Specify which fox snapshot by index
    #0,1,2,3 are for z=[1.0, 0.5, 0.25, 0.0]
    fox_z_index = 3

    #Specify which mass bin 0-7
    fox_lambda_index = 6

    print "Starting the analysis for:\n\tpercolation model %d"%(perc_index)
    print "\tfox_z_index %d\n\tlambda bin %d"%(fox_z_index, fox_lambda_index)

    #This maps the snaphot and mass bins to specific DES indices for z and lambda
    #These are labeled zi and lj, respectively
    zmap = [1, 1, 0 ,0]
    zi = zmap[fox_z_index]
    lambda_edges = [0, 5, 10, 14, 20, 30, 45, 60, 999]
    lams = np.loadtxt("Percolation_data/lambdas")
    ljs = np.digitize(lams, lambda_edges) - 2
    ljs[(ljs < 3)] = 3 #Gotta do this annoying trick since digitize is finicky
    #Note: in DES we only have covariance for lambda index lj >= 3.
    lj = ljs[fox_lambda_index]

    #Get the arguments
    print "\tAssembling the arguments"
    args = get_arguments(perc_index, fox_z_index, fox_lambda_index, zi, lj)

    #Specify which model we are using
    #The basic_sim has only mass and concentration as free parameters
    args['name'] = "basic_sim"

    print "\tArguments dictionary assembled"

    print "Running a test call to lnposterior()"
    mass = np.loadtxt("Percolation_data/truemass")[fox_lambda_index]
    concentration = 5 #a first guess
    log10mass = np.log10(mass)
    guess = np.array([log10mass, concentration])
    print "\tTest call of lnpost()=",lnpost(guess, args)

    print "Finding best fit parameters"
    nll = lambda *args: -lnpost(*args)
    result = op.minimize(nll, guess, args=(args,), tol=1e-3)
    print "\tsuccess = %s"%(result['success'])
    print "\tparams  = ",result['x']

    print "Best fit compared to truth (log10M in Msun/h)"
    print "\tBest mass = %.2f"%result['x'][0]
    print "\tTrue mass = %.2f"%np.log10(mass)

    print "Performing MCMC"
    nwalkers = 8
    nsteps = 2
    ndim = len(guess)
    pos = [result['x'] + 1e-3*np.random.randn(ndim) for k in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost, args=(args,), threads=4)
    sampler.run_mcmc(pos, nsteps)
    print "\tMCMC complete"
    print "\tnwalkers = %d\n\tnsteps per = %d"%(nwalkers, nsteps)
    np.save("chains/mcmc_chains_perc%d_fox-z-index%d_lambda-index%d"%(perc_index, fox_z_index, fox_lambda_index), sampler.flatchain)
    np.save("chains/mcmc_likes_perc%d_fox-z-index%d_lambda-index%d"%(perc_index, fox_z_index, fox_lambda_index), sampler.flatlnprobability)
    
    
if __name__ == "__main__":
    main()
