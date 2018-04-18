#!/usr/bin/env python
#
#
## Author: I. Czekala and Michael Gully
#
# Modification History:
#
# ZJ Zhang (Apr. 13th, 2018)
#
#################################################

# ----
# Goal:
# plot model fitting results to observations (Gully's hack)
# ----


import multiprocessing
import time
import numpy as np
import Starfish
from Starfish.model import ThetaParam, PhiParam

import argparse
parser = argparse.ArgumentParser(prog="plot_many_mix_models_marley.py", description="Plot many mixture models.")
parser.add_argument("--config", action='store_true', help="Use config file instead of emcee.")
parser.add_argument("--f_burnin", type=float, default=None, help="The fraction of the entire emcee output chain that need to be removed before parameter inferences.")
args = parser.parse_args()

import os

import Starfish.grid_tools
from Starfish.spectrum import DataSpectrum, Mask, ChebyshevSpectrum
from Starfish.emulator import Emulator
import Starfish.constants as C
from Starfish.covariance import get_dense_C, make_k_func, make_k_func_region

from scipy.special import j1
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.linalg import cho_factor, cho_solve
from numpy.linalg import slogdet
from astropy.stats import sigma_clip

import gc
import logging

from itertools import chain
#from collections import deque
from operator import itemgetter
import yaml
import shutil
import json
import pandas as pd


### output path setups - ZJ Zhang
## plots
plotdir = os.path.expandvars(Starfish.config["plotdir"])
# star_inference
sfinfer_plotdir = plotdir + "star_inference/"
## debug files
debug_outdir = os.path.expandvars(Starfish.config["outdir"]) + "star_debug/"
## output files
star_outdir = os.path.expandvars(Starfish.config["outdir"]) + "star_inference/"
###################################


Starfish.routdir = ""

# list of keys from 0 to (norders - 1)
order_keys = np.arange(1)
DataSpectra = [DataSpectrum.open(os.path.expandvars(file), orders=Starfish.data["orders"]) for file in Starfish.data["files"]]
# list of keys from 0 to (nspectra - 1) Used for indexing purposes.
spectra_keys = np.arange(len(DataSpectra))

#Instruments are provided as one per dataset
Instruments = [eval("Starfish.grid_tools." + inst)() for inst in Starfish.data["instruments"]]


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s", filename="{}log.log".format(
    Starfish.routdir), level=logging.DEBUG, filemode="w", datefmt='%m/%d/%Y %I:%M:%S %p')

class Order:
    def __init__(self, debug=False):
        '''
        This object contains all of the variables necessary for the partial
        lnprob calculation for one echelle order. It is designed to first be
        instantiated within the main processes and then forked to other
        subprocesses. Once operating in the subprocess, the variables specific
        to the order are loaded with an `INIT` message call, which tells which key
        to initialize on in the `self.initialize()`.
        '''
        self.lnprob = -np.inf
        self.lnprob_last = -np.inf

        self.debug = debug

    def initialize(self, key):
        '''
        Initialize to the correct chunk of data (echelle order).

        :param key: (spectrum_id, order_key)
        :param type: (int, int)

        This method should only be called after all subprocess have been forked.
        '''

        self.id = key
        spectrum_id, self.order_key = self.id
        # Make sure these are ints
        self.spectrum_id = int(spectrum_id)

        self.instrument = Instruments[self.spectrum_id]
        self.dataSpectrum = DataSpectra[self.spectrum_id]
        self.wl = self.dataSpectrum.wls[self.order_key]
        self.fl = self.dataSpectrum.fls[self.order_key]
        self.sigma = self.dataSpectrum.sigmas[self.order_key]
        self.ndata = len(self.wl)
        self.mask = self.dataSpectrum.masks[self.order_key]
        self.order = int(self.dataSpectrum.orders[self.order_key])

        self.logger = logging.getLogger("{} {}".format(self.__class__.__name__, self.order))
        if self.debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

        self.logger.info("Initializing model on Spectrum {}, order {}.".format(self.spectrum_id, self.order_key))

        self.npoly = Starfish.config["cheb_degree"]
        self.chebyshevSpectrum = ChebyshevSpectrum(self.dataSpectrum, self.order_key, npoly=self.npoly)

        # If the file exists, optionally initiliaze to the chebyshev values
        fname = Starfish.specfmt.format(self.spectrum_id, self.order) + "phi.json"
        if os.path.exists(fname):
            self.logger.debug("Loading stored Chebyshev parameters.")
            phi = PhiParam.load(fname)
            self.chebyshevSpectrum.update(phi.cheb)

        #self.resid_deque = deque(maxlen=500) #Deque that stores the last residual spectra, for averaging
        self.counter = 0

        self.emulator = Emulator.open()
        self.emulator.determine_chunk_log(self.wl)

        self.pca = self.emulator.pca

        self.wl_FFT = self.pca.wl

        # The raw eigenspectra and mean flux components
        self.EIGENSPECTRA = np.vstack((self.pca.flux_mean[np.newaxis,:], self.pca.flux_std[np.newaxis,:], self.pca.eigenspectra))

        self.ss = np.fft.rfftfreq(self.pca.npix, d=self.emulator.dv)
        self.ss[0] = 0.01 # junk so we don't get a divide by zero error

        # Holders to store the convolved and resampled eigenspectra
        self.eigenspectra = np.empty((self.pca.m, self.ndata))
        self.flux_mean = np.empty((self.ndata,))
        self.flux_std = np.empty((self.ndata,))
        self.flux_scalar = None

        self.sigma_mat = self.sigma**2 * np.eye(self.ndata)
        self.mus, self.C_GP, self.data_mat = None, None, None
        self.Omega = None

        self.lnprior = 0.0 # Modified and set by NuisanceSampler.lnprob

        # self.nregions = 0
        # self.exceptions = []

        # Update the outdir based upon id
        self.noutdir = Starfish.routdir + "{}/{}/".format(self.spectrum_id, self.order)

    def evaluate(self):
        '''
        Return the lnprob using the current version of the C_GP matrix, data matrix,
        and other intermediate products.
        '''

        self.lnprob_last = self.lnprob

        X = (self.chebyshevSpectrum.k * self.flux_std * np.eye(self.ndata)).dot(self.eigenspectra.T)

        part1 = X.dot(self.C_GP.dot(X.T))
        part2 = self.data_mat
        CC = part2 #+ part2
        np.save(debug_outdir+'CC_new.npy', CC)

        try:
            factor, flag = cho_factor(CC)
        except np.linalg.linalg.LinAlgError:
            print("Spectrum:", self.spectrum_id, "Order:", self.order)
            self.CC_debugger(CC)
            np.save(debug_outdir+'X.npy', X)
            np.save(debug_outdir+'part1.npy', part1)
            np.save(debug_outdir+'part2.npy', part2)
            np.save(debug_outdir+'cheb.npy', self.chebyshevSpectrum.k)
            np.save(debug_outdir+'flux_mean.npy', self.flux_mean)
            np.save(debug_outdir+'flux_std.npy', self.flux_std)
            np.save(debug_outdir+'C_GP.npy', self.C_GP)
            raise

        try:

            model1 = (self.chebyshevSpectrum.k * self.flux_mean + X.dot(self.mus))
            R = self.fl - model1

            logdet = np.sum(2 * np.log((np.diag(factor))))
            self.lnprob = -0.5 * (np.dot(R, cho_solve((factor, flag), R)) + logdet)

            self.logger.debug("Evaluating lnprob={}".format(self.lnprob))
            return self.lnprob

        # To give us some debugging information about what went wrong.
        except np.linalg.linalg.LinAlgError:
            print("Spectrum:", self.spectrum_id, "Order:", self.order)
            raise


    def draw_save(self):
        '''
        Return the lnprob using the current version of the C_GP matrix, data matrix,
        and other intermediate products.
        '''

        self.lnprob_last = self.lnprob

        X = (self.chebyshevSpectrum.k * self.flux_std * np.eye(self.ndata)).dot(self.eigenspectra.T)
        model1 = (self.chebyshevSpectrum.k * self.flux_mean + X.dot(self.mus))

        part1 = X.dot(self.C_GP.dot(X.T))
        part2 = self.data_mat
        CC = part2 #+ part2
        np.save('CC_new.npy', CC)

        return model1

    def update_Theta(self, p):
        '''
        Update the model to the current Theta parameters.

        :param p: parameters to update model to
        :type p: model.ThetaParam
        '''

        # durty HACK to get fixed logg
        # Simply fixes the middle value to be 4.29
        # Check to see if it exists, as well
        fix_logg = Starfish.config.get("fix_logg", None)
        if fix_logg is not None:
            p.grid[1] = fix_logg
        #print("grid pars are", p.grid)

        self.logger.debug("Updating Theta parameters to {}".format(p))

        # Store the current accepted values before overwriting with new proposed values.
        self.flux_mean_last = self.flux_mean.copy()
        self.flux_std_last = self.flux_std.copy()
        self.eigenspectra_last = self.eigenspectra.copy()
        self.mus_last = self.mus
        self.C_GP_last = self.C_GP

        # Local, shifted copy of wavelengths
        wl_FFT = self.wl_FFT * np.sqrt((C.c_kms + p.vz) / (C.c_kms - p.vz))

        # If vsini is less than 0.2 km/s, we might run into issues with
        # the grid spacing. Therefore skip the convolution step if we have
        # values smaller than this.
        # FFT and convolve operations
        if p.vsini < 0.0:
            raise C.ModelError("vsini must be positive")
        elif p.vsini < 0.2:
            # Skip the vsini taper due to instrumental effects
            eigenspectra_full = self.EIGENSPECTRA.copy()
        else:
            FF = np.fft.rfft(self.EIGENSPECTRA, axis=1)

            # Determine the stellar broadening kernel
            ub = 2. * np.pi * p.vsini * self.ss
            sb = j1(ub) / ub - 3 * np.cos(ub) / (2 * ub ** 2) + 3. * np.sin(ub) / (2 * ub ** 3)
            # set zeroth frequency to 1 separately (DC term)
            sb[0] = 1.

            # institute vsini taper
            FF_tap = FF * sb

            # do ifft
            eigenspectra_full = np.fft.irfft(FF_tap, self.pca.npix, axis=1)

        # Spectrum resample operations
        if min(self.wl) < min(wl_FFT) or max(self.wl) > max(wl_FFT):
            raise RuntimeError("Data wl grid ({:.2f},{:.2f}) must fit within the range of wl_FFT ({:.2f},{:.2f})".format(min(self.wl), max(self.wl), min(wl_FFT), max(wl_FFT)))

        # Take the output from the FFT operation (eigenspectra_full), and stuff them
        # into respective data products
        for lres, hres in zip(chain([self.flux_mean, self.flux_std], self.eigenspectra), eigenspectra_full):
            interp = InterpolatedUnivariateSpline(wl_FFT, hres, k=5)
            lres[:] = interp(self.wl)
            del interp

        # Helps keep memory usage low, seems like the numpy routine is slow
        # to clear allocated memory for each iteration.
        gc.collect()

        # Adjust flux_mean and flux_std by Omega
        #Omega = 10**p.logOmega
        #self.flux_mean *= Omega
        #self.flux_std *= Omega

        # Now update the parameters from the emulator
        # If pars are outside the grid, Emulator will raise C.ModelError
        self.emulator.params = p.grid
        self.mus, self.C_GP = self.emulator.matrix
        self.flux_scalar = self.emulator.absolute_flux
        self.Omega = 10**p.logOmega
        self.flux_mean *= (self.Omega*self.flux_scalar)
        self.flux_std *= (self.Omega*self.flux_scalar)
        


class SampleThetaPhi(Order):

    def initialize(self, key):
        # Run through the standard initialization
        super().initialize(key)

        # for now, start with white noise
        self.data_mat = self.sigma_mat.copy()
        self.data_mat_last = self.data_mat.copy()

        #Set up p0 and the independent sampler
        fname = Starfish.specfmt.format(self.spectrum_id, self.order) + "phi.json"
        phi = PhiParam.load(fname)

        # Set the regions to None, since we don't want to include them even if they
        # are there
        phi.regions = None

        #Loading file that was previously output
        # Convert PhiParam object to an array
        self.p0 = phi.toarray()

        jump = Starfish.config["Phi_jump"]
        cheb_len = (self.npoly - 1) if self.chebyshevSpectrum.fix_c0 else self.npoly
        cov_arr = np.concatenate((Starfish.config["cheb_jump"]**2 * np.ones((cheb_len,)), np.array([jump["sigAmp"], jump["logAmp"], jump["l"]])**2 ))
        cov = np.diag(cov_arr)

        def lnfunc(p):
            # Convert p array into a PhiParam object
            ind = self.npoly
            if self.chebyshevSpectrum.fix_c0:
                ind -= 1

            cheb = p[0:ind]
            sigAmp = p[ind]
            ind+=1
            logAmp = p[ind]
            ind+=1
            l = p[ind]

            par = PhiParam(self.spectrum_id, self.order, self.chebyshevSpectrum.fix_c0, cheb, sigAmp, logAmp, l)

            self.update_Phi(par)

            # sigAmp must be positive (this is effectively a prior)
            # See https://github.com/iancze/Starfish/issues/26
            if not (0.0 < sigAmp): 
                self.lnprob_last = self.lnprob
                lnp = -np.inf
                self.logger.debug("sigAmp was negative, returning -np.inf")
                self.lnprob = lnp # Same behavior as self.evaluate()
            else:
                lnp = self.evaluate()
                self.logger.debug("Evaluated Phi parameters: {} {}".format(par, lnp))

            return lnp


    def update_Phi(self, p):
        self.logger.debug("Updating nuisance parameters to {}".format(p))

        # Read off the Chebyshev parameters and update
        self.chebyshevSpectrum.update(p.cheb)

        # Check to make sure the global covariance parameters make sense
        #if p.sigAmp < 0.1:
        #   raise C.ModelError("sigAmp shouldn't be lower than 0.1, something is wrong.")

        max_r = 6.0 * p.l # [km/s]

        # Create a partial function which returns the proper element.
        k_func = make_k_func(p)

        # Store the previous data matrix in case we want to revert later
        self.data_mat_last = self.data_mat
        self.data_mat = get_dense_C(self.wl, k_func=k_func, max_r=max_r) + p.sigAmp*self.sigma_mat

################ MAIN BODY ################

## 0. Preparation
# id
spectrum_id, order_key = 0, 0 # ??? what is spectrum id and order_key?
# fix c0 of Chebyshev function
fix_c0 = True
# --

# 1. Run the program.
model = SampleThetaPhi(debug=True)
model.initialize((spectrum_id, order_key))
# observed spectrum
obs_wls = model.wl
obs_fls = model.fl
obs_sigmas = model.sigma

def lnprob_all(p):
    pars1 = ThetaParam(grid=p[0:2], vz=p[2], vsini=p[3], logOmega=p[4])
    model.update_Theta(pars1)
    # hard code npoly=3 (for fixc0 = True with npoly=4)
    pars2 = PhiParam(0, 0, True, p[5:8], p[8], p[9], p[10])
    model.update_Phi(pars2)
    draw = model.draw_save()
    return draw

if args.config:
    df_out = pd.DataFrame({'wl':wl, 'data':data})

    with open('s0_o0phi.json') as f:
        s0phi = json.load(f)

    psl = (Starfish.config['Theta']['grid']+
          [Starfish.config['Theta'][key] for key in ['vz', 'vsini', 'logOmega']] +
           s0phi['cheb'] +
          [s0phi['sigAmp']] + [s0phi['logAmp']] + [s0phi['l']])
    ps = np.array(psl)
    df_out['model_composite'] = lnprob_all(ps)  
    
    df_out.to_csv('spec_config.csv', index=False) 

else:
    draws = []
    # load chain
    star_chain = np.load(star_outdir+"emcee_chain.npy")
    param_keywords = ["Teff", "log-g", "vz", "vsini", "logOmega", "Cheb_1", "Cheb_2", "Cheb_3", "sigAmp", "logAmp", "l"]
    # remove the burn-in phase
    nwalkers, nsamples, ndim = star_chain.shape
    if (args.f_burnin is not None) and (0 <= args.f_burnin < 1):
        burned_starchain = star_chain[:, int(nsamples * args.f_burnin):, :]
    else:
        print("warning: f_burnin should be in [0, 1) - using the entire chain...")
        burned_starchain = star_chain
    flatchain_star = burned_starchain.reshape((-1, ndim))

    # derive final parameters
    star_pars = np.median(flatchain_star, axis=0)
    star_pars_upper_error = np.percentile(flatchain_star, 84, axis=0) - star_pars
    star_pars_lower_error = star_pars - np.percentile(flatchain_star, 16, axis=0)

    # triangle plot for the burned-in flatchain
    import corner
    star_fig = corner.corner(flatchain_star[:,0:5], labels=param_keywords[:5], show_titles=True)
    star_fig.savefig(sfinfer_plotdir+'star_chain_corner.png', dpi=1000)
    nuisance_fig = corner.corner(flatchain_star[:,5:], labels=param_keywords[5:], show_titles=True)
    nuisance_fig.savefig(sfinfer_plotdir+'nuisance_chain_corner.png', dpi=1000)

    # model flux
    mod_fls = lnprob_all(star_pars)
    # compare with correct answer
    comp_star_pars = 1.0 * star_pars
    comp_star_pars[0] = 810
    comp_star_pars[1] = 5.15
    comp_mod_fls = lnprob_all(comp_star_pars)

    # write an output data file containing observed and model spectra
    df_star = pd.DataFrame({'obs_wls':obs_wls, 'obs_fls':obs_fls, 'mod_fls':mod_fls, 'comp_mod_fls':comp_mod_fls})
    df_star.to_csv(star_outdir+'osbsmod_spec_comparison.csv', index=False)

    # claim the fitting results
    print("\nFitting results -")
    for item_index in range(0, ndim):
        print("%-10s: %10.2f (+ %5.2f) (- %5.2f)"%(param_keywords[item_index], star_pars[item_index], star_pars_upper_error[item_index], star_pars_lower_error[item_index]))

    # generate json files for plotting - following the original plot_star.py
    my_dict = {"wl": obs_wls.tolist(), "data":obs_fls.tolist(), "sigma":obs_sigmas.tolist(), "model": mod_fls.tolist(), "resid":(obs_fls-mod_fls).tolist(), "spectrum_id":spectrum_id, "order":order_key}
    spec_json_file = Starfish.specfmt.format(spectrum_id, order_key) + "spec.json"
    with open(spec_json_file, 'w') as f:
        json.dump(my_dict, f, indent=2)
