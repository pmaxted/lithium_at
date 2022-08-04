# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 10:04:07 2022

@authors: Richard Jackson, Rob Jeffries
"""

import sys
import getopt
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# This will get the name of this file
script_name = os.path.basename(__file__)

##
# @brief Help document for this script.  See the main function below.
#
help = f'''
    {script_name} input_file output_file  [-c] [-s] [-p prior] [-z constant] [--lagesmin min_age] [--lagesmax max_age] [--lapkmin min_peak_age] [--nage age_steps] [-h]"
    
    Estimates the ages for a set of stars or a cluster based on their lithium EWs and Teff.
    Results are output for individual stars in a csv table. The posterior probability distribution
    for an individual star or a cluster are written to an ascii file. Summary plots can be saved
    as pdf files.
    
    input_file is an ascii file containing >=1 row and 4 or 5 columns (with no header). .
        Col 1 is a string identifier with no spaces,
        Col 2 is an effective temperature in K,
        Col 3 is an *optional* 1-sigma error in Teff in K (see paper)
        Col 4 is Li equivalent width in milli-angstroms,
        Col 5 is the error in equivalent width in milli-angstroms. 
        
    output_file is the stem for the output files, with no suffix.
        output_file.csv is a csv contain the individual result for each star i input_file.
        output_file.dat is an ascii file containing P(log age) of the (combined) posterior.
    
    examples:
        Read data for a single star and estimate its age (input_file would have one row), 
        based on a prior age probability that is flat in age, saving the output plots
        
            {script_name} input_file output_file -s -p 1
    
        Read data for a cluster of stars and calculate their combined posterior age
        probability distribution using a prior that is flat in log age
        
            {script_name} input_file output_file -c -s
    
    -c 
    --cluster
        Indicates that the list of targets belong to a coeval cluster and their log 
        likelihoods will be summed to produce the final result. NB: output_file.csv 
         will still contain the results for each star individually.\n\
    -s 
    --save
        Indicates that the plots should be saved: <output_file>_prob.pdf gives 
        the P(log age); <output_file>_iso.pdf shows the EWLi vs Teff isochrone.
        
    -p 
    --prior
        Sets the prior age probability: 
        0 = flat in log age; 1 = flat in log age.
        
        The default prior is 0 .
        
    -z 
        A likelihood regularisation constant to avoid outliers skewing the results
        badly in clusters
        
        The default value of z is 1.0e-8 .
        
    --lagesmin 
        Is the minimum log age/yr for the prior 
        
        The default value of lagesmin is 6.

    --lagesmax 
        Is the maximum log age/yr for the prior 
        
        The default value of lagesmax is 10.1.
        
    --lapkmin 
        Is the minimum log age/yr at which the likelihood is calculated.
        Likelihoods at lower ages are set to the value at this age.
        
        The default value of lapkmin is 6.699 .
        
    --nage 
        The number of log ages at which the posterior probability is calculated 
        between lagemin and lagemax.
        
        The default value of nage is 820 .

    -h 
    --help
        prints this message.
'''

def print_help():
    print(help, file=sys.stderr)
    exit()


def AT2EWm(lTeff, lAge):
    
    # constants defining model fit
    lTc   = 3.52118
    CC0   = -5.69534
    CC1   = 3.92675
    AAc   = 301.12
    AA1   = 18859.7
    BBc   = 0.1194
    BB1   = -189.19
    CCc   = 1.13922

    # calculate parameters
    AM    = AAc - AA1*(lTeff -lTc)*(lTeff -lTc)/(2*lTc)
    BM    = BBc - BB1*(lTeff -lTc)*(lTeff -lTc)/(2*lTc)
    CM    = np.zeros(np.shape(lTeff)[0])
    
    index = np.nonzero(lTeff <= lTc)
    if np.shape(index)[1] > 0 : 
        CM[index] = CCc + CC0*(lTeff[index]-lTc)
    index = np.nonzero(lTeff > lTc)
    if np.shape(index)[1] > 0 :
        CM[index] = CCc + CC1*(lTeff[index]-lTc)
        
    # Model Lithium EW
    return  AM*(1-np.sinh((lAge-CM)/BM)/np.cosh((lAge-CM)/BM))

def eAT2EWm(lTeff, lAge):
    
    # constants defining model fit
    lTc   = 3.52118
    CC0   = -5.69534
    CC1   = 3.92675
    AAc   = 301.12
    AA1   = 18859.7
    BBc   = 0.1194
    BB1   = -189.19
    CCc   = 1.13922

    EE0   = 114.19
    EE1   = 15.976
    EE2   = 1.0696
    FF0   = 0.0895934

    # calculate parameters
    AM    = AAc - AA1*(lTeff -lTc)*(lTeff -lTc)/(2*lTc)
    BM    = BBc - BB1*(lTeff -lTc)*(lTeff -lTc)/(2*lTc)
    CM    = np.zeros(np.shape(lTeff)[0])
    
    index = np.nonzero(lTeff <= lTc)
    if np.shape(index)[1] > 0 : 
        CM[index] = CCc + CC0*(lTeff[index]-lTc)
    index = np.nonzero(lTeff > lTc)
    if np.shape(index)[1] > 0 :
        CM[index] = CCc + CC1*(lTeff[index]-lTc)
    
    # model dispersion
    eEEM   = EE0/np.exp(EE2*lAge) + EE1
    eFFM  = FF0*(AM/BM)/np.cosh((lAge-CM)/BM)**2
    return np.sqrt(eEEM**2 + eFFM**2)
        


def bounds3(lAge, like, prior=None, lApkmin=None, pcuthi=None, limcut=None):
    
    #INPUT
    # lAge    - array of uniformly spaced log ages
    # like    - likelihood at each value of lage
    #OUTPUT (Returns)
    # prob    - likelihood weighted by prior
    # lApk    - log Age at aximum of weighted probability
    # siglo   - offset in logage of 68% lower bound
    # siglo   - offset in logage of 68% lower bound
    # sighi   - offset in logage of 68% upper bound
    # limup   - 95% upper limit where no clear peak
    # limlo   - 95% lower limit where no clear peak
    #    value set to -1 if  parameter  undefined
    #OPTIONS
    # prior   - constant defining weighting of age scale
    #         - 0 for uniform with log10(age) 1 for uniform with age
    # lApkmin - lowest age for valid model of EW_Li
    #         - Likelihood is constant below this age
    # pcuthi  - level defining resolvablelog age decrease at limit
    # limcut  - fraction defineing upper/lower limit    

    # default limits
    if prior is None:
        prior = 0
    if lApkmin is None:
        lApkmin = np.log10(5)
    if pcuthi is None:
        pcuthi = np.exp(-0.5)
    if limcut is None:
        limcut = 0.95

    # default values of bounds
    sighi = -1
    siglo = -1
    limlo = -1
    limup = -1   
 
    # fill low end of likelihood curve
    Nmax = np.shape(lAge)[0]
    index = np.nonzero(lAge < lApkmin)
    Nlo = np.shape(index)[1]
    if Nlo > 0 :
        like[0:Nlo] = like[Nlo-1]

    # scale likelihood by the prior
    prob = like*10**(prior*lAge) 

    # handle case of very low lApk
    Npk = np.argmax(prob)
    if Nlo > Npk :
        Npk = Nlo 
            
    # define max values
    ##### POSSIBLE BUG - should Npk be calculated again?
    Npk = np.argmax(prob)
    Pmax = prob[Npk]
    lApk = lAge[Npk]
    lAstep =  lAge[Nmax-1]-lAge[Nmax-2]
    # print('lApk, Nlo', lApk, Nlo, lAge)
    # case with resolvable peak
    if prob[Nlo]/Pmax < pcuthi and prob[Nmax-1]/Pmax < pcuthi :
        
        # get lower bound
        plo = prob[0:Npk][::-1]  # reverses a slice of prob
        cdflo = np.cumsum(plo)/np.sum(plo)
        index = np.nonzero(cdflo > 0.68)
        indlo = np.shape(index)[1]
        siglo = lAge[Npk] - lAge[indlo] + lAstep

        # get upper bound
        phi = prob[Npk:Nmax]
        cdfhi = np.cumsum(phi)/np.sum(phi)
        index = np.nonzero(cdfhi <= 0.68)
        indhi = np.shape(index)[1]
        sighi = lAge[indhi + Npk] - lAge[Npk] + lAstep
      
#        plt.xlim([np.amin(lAge)+6, np.amax(lAge)+6])
#        plt.xlabel('log Age/yr')
#        plt.ylabel('P(log Age)')
#        plt.plot(lAge[Npk:Nmax-1]+6, cdfhi, 'g--')
#        plt.plot(lAge[Npk:Nmax-1]+6, phi/Pmax, 'r-')
#        plt.plot(lAge+6, prob/Pmax, 'b-')
#        plt.plot(lAge[0:Npk]+6, cdflo[::-1], 'g--')
#        plt.plot(lAge+6, np.cumsum(prob)/np.sum(prob), 'r--')
#        plt.show()

    # case of upper limit
    if prob[Nlo]/Pmax > pcuthi and prob[Nmax-1]/Pmax < pcuthi :
        cdf = np.cumsum(prob)/np.sum(prob)     
        index = np.nonzero(cdf <= limcut)
        Ncut = np.shape(index)[1]
        limup = lAge[Ncut] + lAstep
        lApk = -1
        
    # case of lower limit
    if prob[Nlo]/Pmax < pcuthi and prob[Nmax-1]/Pmax > pcuthi :
        cdf = np.cumsum(prob)/np.sum(prob)
        index = np.nonzero(cdf <= 1.0-limcut)
        Ncut = np.shape(index)[1]
        limlo = lAge[Ncut] - lAstep
        lApk = -1
        
    return prob, lApk, siglo, sighi, limup, limlo 
   
def age_fit3(lTeff, LiEW, eLiEW, lAges, z, lApkmin, nTeff, prior=None, elTeff=None):

    # default value of prior
    if prior is None:
        prior = 0      
    nStar = np.shape(LiEW)[0]
    nAges = np.shape(lAges)[0] 
    pfit = np.zeros(nAges) 
    dpfit = np.zeros(nAges)

    if elTeff is None:
        # get raw log likelihood at each log age 
        for k in range(0, nAges):
            EWm = AT2EWm(lTeff, lAges[k])
            xLiEW = eAT2EWm(lTeff, lAges[k])
            sLiEW = np.sqrt(xLiEW**2 + eLiEW**2)
            pstar = 1.0/np.sqrt(2.0*np.pi)/sLiEW* \
                np.exp(-(LiEW-EWm)*(LiEW-EWm)/(2.0*sLiEW*sLiEW)) + z
            pfit[k] = np.sum(np.log(pstar))
    else:
        # need to loop over a set of temperature AND over a set of ages
        dlTeff = 4.0*elTeff/nTeff
        for j in range(0, nTeff):
            # newTeff is the new (log) Teff to evaluate the likelihood
            newTeff = lTeff + (j - (nTeff-1)/2)*dlTeff
            # factor is the number by which the log likelihood is decreased
            # at that Teff
            factor = -1.0*(lTeff-newTeff)**2/(2.0*elTeff**2)
            for k in range(0, nAges):
                EWm = AT2EWm(newTeff, lAges[k] )
                xLiEW = eAT2EWm(newTeff, lAges[k])
                sLiEW = np.sqrt(xLiEW**2 + eLiEW**2)
                pstar = 1.0/np.sqrt(2.0*np.pi)/sLiEW* \
                    np.exp(-(LiEW-EWm)*(LiEW-EWm)/(2.0*sLiEW*sLiEW)) + z
                dpfit[k] = np.sum(np.log(pstar))
            pfit = pfit + np.exp(dpfit)*np.exp(factor)   
        pfit = np.log(pfit)
            
            
    # handle low likelihoods
    llike = pfit - np.amax(pfit)
    index = np.nonzero(llike < -70)
    if np.shape(index)[1] > 0:
        llike[index] = -70

    # handle low ages
    like = np.exp(llike - np.amax(llike))
    index = np.nonzero(lAges < lApkmin)
    Nlo = np.shape(index)[1]
    if Nlo > 0:
        like[0:Nlo-1] = like[Nlo-1]
    like = like/np.sum(like)
    
    # get peak age, bounds and limits
    prob, lApk, siglo, sighi, limup, limlo = bounds3(lAges, like, prior=prior)
    
    # log probability
    lprob = np.log(prob) - np.amax(np.log(prob))
    
    # get reduced chisqr
    if lApk > -1 and nStar > 1 :
        xLiEW = eAT2EWm(lTeff, lApk)
        sLiEW = np.sqrt(xLiEW**2 + eLiEW**2)
        dLiEW = (AT2EWm(lTeff, lApk) - LiEW)
        chisq = np.sum(dLiEW**2/sLiEW**2)/nStar
    else:
        chisq = -1
    
    # normalise peak of log likelihood to zero
    llike = np.log(like) - np.amax(np.log(like))  
    
    return llike, lprob, siglo, lApk, sighi, limup, limlo, chisq


def get_li_age(LiEW, eLiEW, Teff, eTeff=None, lagesmin=6.0, lagesmax=10.1, 
               lApkmin=6.699, nAge=820, z=1.0e-8, nTeff=21, prior=None) :
    
 
    # setup default values
        # default value of prior
    if prior is None:
        prior = 0     
    
    # bounds for Teff
    Teffmin = 3000.0
    Teffmax = 6500.0
    
    lagesmin = lagesmin - 6 # fitted function is in terms of log Age/Myr
    lagesmax = lagesmax - 6 # rather than log Age/yr
    lApkmin = lApkmin - 6
    nStar = np.shape(LiEW)[0] # number of stars in file

    if (np.amax(Teff) > Teffmax or np.amin(Teffmin) < Teffmin):
        raise RuntimeError("All temperatures must be between 3000K and 6500K")
    
    if (np.amax(LiEW-eLiEW) > 800 or np.amin(LiEW+eLiEW) < -200):
        raise RuntimeError("LiEW outside a sensible range")

# set array of ages and temperatures
    lAges = lagesmin+(np.arange(nAge)+0.5)*(lagesmax-lagesmin)/float(nAge)
    lTeff = np.log10(Teff)    
    if eTeff is not None:
        elTeff = 0.5*(np.log10(Teff+eTeff)-np.log10(Teff-eTeff))
    else:
        elTeff=None
  
# get the ages and limits
    llike, lprob, siglo, lApk, sighi, limup, limlo, chisq = \
      age_fit3(lTeff, LiEW, eLiEW, lAges, z, lApkmin, nTeff, prior=prior, elTeff=elTeff)

    return lAges, llike, lprob, siglo, lApk, sighi, limup, limlo, chisq


def make_plots(lAges, lprob, siglo, lApk, sighi, limup, limlo, chisq, \
              lagesmin, lagesmax, LiEW, eLiEW, Teff, filename, is_cluster, savefig):

    nStar = np.shape(LiEW)[0]
    plt.xlabel('log Age/yr')
    plt.ylabel('P(log Age)')
    #get position of peak
    
    # produce a probability plot, either for the cluster or for an individual star
    if is_cluster and nStar > 1 : # case of a cluster of >1 stars
        index = np.nonzero(lprob > -20)
        nlo = np.amin(index)
        nhi = np.amax(index)
        plt.xlim([lAges[nlo]+6, lAges[nhi]+6])
    else: # case of a single star
        plt.xlim([lagesmin,lagesmax])
    
    
    ax = plt.gca()
    plt.ylim([-0.05, 1.1])
    plt.plot(lAges+6, np.exp(lprob), c='b')
    # handle cases where a peak is found and also limits
    if lApk != -1:
        plt.fill_between(lAges+6, np.exp(lprob), \
            where = (lAges > lApk-siglo) & (lAges < lApk+sighi), color='0.7' )
        ax.text(0.02,0.94, "Age: {:.1f} +{:.1f}/-{:.1f} Myr".format(10**lApk, 10**(lApk+sighi)-10**lApk, 10**lApk-10**(lApk-siglo)), transform=ax.transAxes)    
    if limup != -1:
        plt.fill_between(lAges+6, np.exp(lprob), \
            where = (lAges < limup), color='0.7' )
        ax.text(0.02,0.94, "Age < {:.1f} Myr".format(10**limup), transform=ax.transAxes)
    if limlo != -1:
        plt.fill_between(lAges+6, np.exp(lprob), \
            where = (lAges > limlo), color='0.7' )
        ax.text(0.02,0.94, "Age > {:.1f} Myr".format(10**limlo), transform=ax.transAxes)
 
    if savefig:    
        plt.savefig(filename+'_prob.pdf')
    plt.show()
    
    # produce a plot of LiEW vs Teff with the best-fitting isochrone
    if is_cluster and nStar > 1:    
        
        lTeff = np.arange(3.4771, 3.8130, 0.001) #between 3000K and 6500K
        
        ax = plt.gca()
        # handle cases where age is found or just limits
        if lApk != -1:
            ax.text(0.02,0.92, "Age: {:.1f} +{:.1f}/-{:.1f} Myr".format(10**lApk, 10**(lApk+sighi)-10**lApk, 10**lApk-10**(lApk-siglo)), transform=ax.transAxes)
            ax.text(0.02,0.85, "Chisqr: {:.2f}".format(chisq), transform=ax.transAxes)
            EWm = AT2EWm(lTeff, lApk)
            EWm_hi = AT2EWm(lTeff, lApk+sighi)
            EWm_lo = AT2EWm(lTeff, lApk-siglo)
        if limup != -1:
            ax.text(0.02,0.92, "Age < {:.1f} Myr".format(10**limup), transform=ax.transAxes)
            EWm_hi = AT2EWm(lTeff, limup)
            EWm_lo = AT2EWm(lTeff, lagesmin-6)
        if limlo != -1:
            ax.text(0.02,0.92, "Age > {:.1f} Myr".format(10**limlo), transform=ax.transAxes)
            EWm_hi = AT2EWm(lTeff, lagesmax-6)
            EWm_lo = AT2EWm(lTeff, limlo)
        plt.xlabel('Teff (K)')
        plt.ylabel('LiEW (mA)')
        plt.xlim([6600, 2900])
        plt.fill_between(10**lTeff, EWm_lo, EWm_hi, color='0.8')
        plt.errorbar(Teff, LiEW, yerr=eLiEW, color='b', fmt='.')
        if lApk != -1:
            plt.plot(10**lTeff, EWm, color='k' )
        
        if savefig:
            plt.savefig(filename+"_iso.pdf")
        plt.show()

# Main code
def main():
    
    argv = sys.argv[1:]

    if len(argv) == 0:
        print ("""Usage: lithium_at input_file output_file  [-c] [-s] [-p prior]
        [-z constant] [--lagesmin min_age] [--lagesmax max_age]
        [--lapkmin min_peak_age] [--nage age_steps] [-h]""")
        exit()

    # default values
    prior=0
    lagesmin = 6.0
    lagesmax = 10.1
    lApkmin = np.log10(5)+6    
    nAge = 820
    nTeff = 21 # not currently an optional parameter
    z = 1.0e-8
    eTeff = None # is over-ridden if input has 5 columns
    is_cluster = False
    savefig = False
    
    # check there are at least two arguments
    if (len(argv) < 2 or '-h' in argv):
        print_help()
        sys.exit(())


    # open the input file and read the data
    inputfile = argv[0]
    if os.path.isfile(inputfile): # check file exists
                    
        with open(inputfile) as f:
            line = f.readline()
            ncol = len(line.split())
            f.close()
            if ncol == 4 or ncol == 5: # check there are 4 or 5 columns
                if ncol == 4:
                    cols = ['ID', 'Teff', 'LiEW', 'eLiEW']  
                            
                if ncol == 5:
                    cols = ['ID', 'Teff', 'eTeff', 'LiEW', 'eLiEW']
                       
                df = pd.read_csv(inputfile, header=None, delim_whitespace=True, names=cols) 
                Teffraw = df[['Teff']].to_numpy()
                Teff = np.reshape(Teffraw, len(Teffraw))
                LiEWraw = df[['LiEW']].to_numpy()
                LiEW = np.reshape(LiEWraw, len(LiEWraw))
                eLiEWraw = df[['eLiEW']].to_numpy()
                eLiEW = np.reshape(eLiEWraw, len(eLiEWraw))
                nStar = np.shape(LiEW)[0]
                if ncol == 5:
                    eTeffraw = df[['eTeff']].to_numpy()
                    eTeff = np.reshape(eTeffraw, len(eTeffraw))
                        
            else:
                print("Wrong number of columns in input file!\n")
                sys.exit()
                    
    else:
        raise Exception("Input file does not exist") 

    # get the output filename stem; NB no check for valid path
    filename = argv[1]

    try:
        options, args = getopt.getopt(argv[2:], "p:z:csh", \
                    ["help", "cluster", "save", "prior=", "lagesmin=", "lagesmax=", "lapkmin=", "nage="])

    except:
        print_help()
    
    try:
        for name, value in options:
        
            if name in ['-c', '--cluster']:
                is_cluster = True
                if nStar < 2:
                    print("Warning: a cluster should have more than one star!")
       
            if name in ['-s', '--save']:
                savefig = True
            
            if name in ['-p', '--prior']:
                prior = int(value)
                if prior != 0 and prior != 1:
                    print("Prior must be either 0 or 1\n")
                    sys.exit()
                    
            if name in ['-h', '--help']:
                print_help()
                sys.exit()
    
            if name in ['-z']:
                z = float(value)
                print("Note that z will be reset to zero if the input file contains only 1 row\n")
    
            if name in ['--nage']:
                nAge = int(value)
                
            if name in ['--lagemin']:
                lagemin = float(value)
                
            if name in ['--lagemax']:
                lagemax = float(value)
                if lagemax > 10.1 :
                    print("Warning: lagemax is older than the Galaxy!\n")
            
            if name in ['lapkmin']:
                lApkmin = float(value)
                if lApkmin < 6.3 :
                    print("Warning: likelihood is completely unconstrained below 2 Myr!\n")
  
    except (IndexError, ValueError):
        print_help()
     
    
    
    # setup the output lists
    l_lApk = []
    l_siglo = []
    l_sighi = []
    l_limup = []
    l_limlo = []
    
    
    # Do the calculations in two passes
    # First treat each star as an individual and write the results into
    # lists that will be output to filename.csv
    
    for i in range(0, nStar):
        if ncol == 4:
            lAges, llike, lprob, siglo, lApk, sighi, limup, limlo, chisq = \
            get_li_age(LiEW[i:i+1], eLiEW[i:i+1], Teff[i:i+1], lagesmax=lagesmax, lagesmin=lagesmin, \
                       lApkmin=lApkmin, z=0.0, nAge=nAge, prior=prior)
        
        if ncol == 5:
            lAges, llike, lprob, siglo, lApk, sighi, limup, limlo, chisq = \
            get_li_age(LiEW[i:i+1], eLiEW[i:i+1], Teff[i:i+1], lagesmax=lagesmax, lagesmin=lagesmin, \
                       lApkmin=lApkmin, z=z, nAge=nAge, prior=prior, eTeff=eTeff[i:i+1])
 
        l_lApk.append(lApk)
        l_siglo.append(siglo)
        l_sighi.append(sighi)
        l_limlo.append(limlo)
        l_limup.append(limup)
 
    #Then if it is a cluster run again with the full list combined   
 
    if is_cluster and nStar > 1:
        
        if ncol == 4:
            lAges, llike, lprob, siglo, lApk, sighi, limup, limlo, chisq = \
                get_li_age(LiEW, eLiEW, Teff, lagesmax=lagesmax, lagesmin=lagesmin, \
                       lApkmin=lApkmin, z=z, nAge=nAge, prior=prior)
       
        if ncol == 5:
            lAges, llike, lprob, siglo, lApk, sighi, limup, limlo, chisq = \
                get_li_age(LiEW, eLiEW, Teff, lagesmax=lagesmax, lagesmin=lagesmin, \
                       lApkmin=lApkmin, z=z, nAge=nAge, prior=prior, eTeff=eTeff)

        print("***********************************************")   
        print('chi-squared of fit = %6.2f' % chisq)
        
    if is_cluster or nStar == 1 :
       
        if lApk != -1 :
            print('log (Age/yr) = %5.3f +%5.3f/-%5.3f' % (lApk+6, sighi, siglo))
            print('Age (Myr) = %6.1f +%6.1f/-%6.1f' % (10**lApk, 10**(lApk+sighi)-10**lApk, 
                                            10**lApk-10**(lApk-siglo)))

        if limup != -1 :
            print('log (Age/yr) < %4.2f' % (limup+6.0))
            print('Age (Myr) < %6.1f (95 per cent limit)' % 10**(limup))

        if limlo != -1 :
            print('log (Age/yr) > %4.2f' % (limlo+6.0))
            print('Age (Myr) > %6.1f (95 per cent limit)' % 10**(limlo))
        
        print("***********************************************\n")
    
    # make the plots
    
    make_plots(lAges, lprob, siglo, lApk, sighi, limup, limlo, chisq, \
              lagesmin, lagesmax, LiEW, eLiEW, Teff, filename, is_cluster, savefig)

    # output results for individual stars by appending new columns to the input csv file
    df['lApk'] = l_lApk
    df['siglo'] = l_siglo
    df['sighi'] = l_sighi
    df['limup'] = l_limup
    df['limlo'] = l_limlo
    
    # add 6 to lApk, limup, limlo (where they are not -1) to convert to log Age/Myr
    
    df['lApk'] = np.where(df['lApk'] != -1, df['lApk']+6, -1)
    df['limup'] = np.where(df['limup'] != -1, df['limup']+6, -1)
    df['limlo'] = np.where(df['limlo'] != -1, df['limlo']+6, -1)
    df.to_csv(filename+'.csv', index=False) 
    
    # write out the posterior probability to an ascii file
    # format is log Age/yr probability (normalised to 1 at its peak)
    data = np.column_stack([lAges+6, np.exp(lprob)])
    np.savetxt(filename+"_pos.txt", data, fmt=['%7.4f', '%7.3e'], header='log (Age/yr)  Probability')

###THE END###



if  __name__ == "__main__":   
    main(sys.argv[1:])


#some test code
# set up 
#an array of temperatures and an age, get the EWm and eEWm from the functions
# and write them into a pandas data array and output as a csv file

Age = 15.4
lAge = np.log10(Age)
lTeff = np.arange(3.4771, 3.8130, 0.001)

gvelew = AT2EWm(lTeff, lAge)
gveleew = eAT2EWm(lTeff, lAge)

#plt.plot(10**lTeff, gvelew)
#plt.plot(10**lTeff, gvelew+gveleew)
#plt.plot(10**lTeff, gvelew - gveleew)

df = pd.DataFrame()

df['Teff'] = lTeff.tolist()
df['EWLi'] = gvelew.tolist()
df['eEWLi'] = gveleew.tolist()

df.to_csv('gvelmodel.csv', index=False)




