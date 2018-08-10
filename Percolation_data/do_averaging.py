"""
This short script shows how to take in the DeltaSigma profile and average it in radial bins by calling the cluster toolkit.

Since I don't have real data, I wasn't able to debug this. Apologies if it is broken, but it is very short and easy to fix if so.
"""
import numpy as np
import cluster_toolkit as ct

#Hubble constant
h = 0.6704 #fox cosmology

#Redshift
z = 0.0

#The DES bin edges are defined in Mpc physical coordinates
Nbins = 15
R_edges = np.logspace(np.log10(0.0323), np.log10(30), Nbins+1)
#+1 since we have 1 more edge than bins

#Change units to Mpc/h comoving coordinates
R_edges *= h*(1+z)

#Here, pretend that we got R and DeltaSigma from a data file
#Pretend that R is in Mpc/h comoving and DeltaSigma is in h*Msun/pc^2 comoving
R = thing1
DeltaSigma = thing2

#Call the toolkit
averaged_DeltaSigma = ct.averaging.average_profile_in_bins(R_edges, R, DeltaSigma)
#Done!
