#!/usr/bin/env python
#
#
## Author: ZJ Zhang
#
# Modification History:
#
# ZJ Zhang (Apr 24th, 2018)
#
#################################################

# import standard package
import os
import astropy.units as u
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import corner
import h5py
import numpy as np
import pandas as pd
np.seterr(divide='ignore', invalid='ignore')


# import Starfish packages
import Starfish
from star_class import SampleThetaPhi
from Starfish.model_BD import ThetaParam, PhiParam
from Starfish.utils import random_draws, sigma_envelope


import argparse
parser = argparse.ArgumentParser(prog="summarize_star_MarleyMod.py", description="summarize the fitting results from star_MarleyMod.py, including observed spectra (wls + fls), best-fit parameters with uncertainties, best-fit model flux, and covariance matrix.")
parser.add_argument("--mode", type=str, default='std', help="mode of running this program, including the standard mode ('std') and user-input mode ('usr'); under the 'usr' mode, the input chain files and output files are defined by the user")
parser.add_argument("--f_burnin", type=float, default=0.5, help="burn-in fraction of the emcee chains.")
parser.add_argument("--chain", action="store_true", help="plot the chain values as a function of sampling steps.")
parser.add_argument("--corner", action="store_true", help="plot the corner plots for the chains with the given f_burnin.")
parser.add_argument("--store", action="store_true", help="save the fitting results into a HDF5 file.")
parser.add_argument("--spec", action="store_true", help="plot and compare the observed and best-fit spectra; run `--store` before this option.")
parser.add_argument("--num_nd", type=int, default=100, help="number of noise draw when producing noise spectrum.")
parser.add_argument("--format", type=str, default='png', help="format of output figures")
parser.add_argument("--dpi", type=str, default=None, help="dpi of output figures")
# specially for the "usr" mode
parser.add_argument("--object", type=str, default='object', help="user-defined object name.")
parser.add_argument("--chain_file", type=str, default=None, help="emcee chain files.")
parser.add_argument("--plotdir", type=str, default='./', help="directory for output figures.")
parser.add_argument("--resdir", type=str, default='./', help="directory for output results files.")
parser.add_argument("--resfile", type=str, default='./resfile.hdf5', help="path of resulting HDF5 file (a summary of all fitting results).")
parser.add_argument("--specfile", type=str, default='./specfile.csv', help="path of resulting csv file (observed and model spectra information).")
args = parser.parse_args()


#####################################
# Assistant Parameters and Functions
#####################################

### default plotting labels
labels = [r'$T_{\mathrm{eff}}$', r'$\log{g}$',r'$v_z$', r'$v\sin{i}$', r'$\log{\Omega}$',
          r'$c^1$', r'$c^2$', r'$c^3$', r'sigAmp', r'logAmp', r'$l$']
print_labels = ['Teff', 'log-g', 'vz', 'vsini', 'logOmega', 'c1', 'c2', 'c3', 'sigAmp', 'logAmp', 'l']
default_sn_spl_id = 5  # stellar parameters for <5 and nuisance for >=5 in the final chain
# --

### range of wavelengths with smaller noise, therefore could be used to define the range of plots
wls_range = [9000, 18000]
y_scalar = 1.2
# --

def burnin_flat(chain, f_burnin=0.5):
    ''' return the flatchain for the given chain and burn-in fraction'''
    ### chain shape
    nwalkers, nsamples, ndim = chain.shape
    ### burned-in flat chain
    if (f_burnin is not None) and (0 <= f_burnin < 1):
        burned_chain = chain[:, int(nsamples * f_burnin):, :]
    else:
        print("warning: f_burnin should be in [0, 1) - using the entire chain...")
        burned_chain = np.copy(chain)
    flatchain = burned_chain.reshape((-1, ndim))
    return flatchain
# ----


def plot_star_chain(object, chain_file, plotdir, f_burnin=0.5, format='png', dpi=None):
    ''' plot the chain values as a function of sampling steps'''
    ### 1. load chains
    chain = np.load(chain_file)
    nwalkers, nsamples, ndim = chain.shape
    print("- %s: %d walkers, %d samples, %d parameters"%(object, nwalkers, nsamples, ndim))
    ### 2. plotting
    figure, axes = plt.subplots(ndim, 1, sharex=True, figsize=(8, 14))
    for i in range(0, ndim):
        axes[i].plot(chain[:, :, i].T, color="k", alpha=0.3)
        axes[i].yaxis.set_major_locator(MaxNLocator(5))
        axes[i].set_xlim([0,nsamples])
        axes[i].set_ylabel(labels[i])
        axes[i].axvline(int(nsamples*f_burnin), color='red', linestyle='-') # burn-in cut-off
    axes[ndim-1].set_xlabel("step number")
    figure.suptitle(object, fontsize=20)
    # save plot
    chain_figure_path = os.path.expandvars(plotdir + "star_inference/star_chain.%s"%(format))
    figure.savefig(chain_figure_path, format=format, dpi=dpi)
    print("chain plot finished!")
# ----


def plot_star_corner(object, chain_file, plotdir, f_burnin=0.5, format='png', dpi=None):
    ''' plot corner figures for stellar and nuisance parameters'''
    ### 1. obtain burned-in flat chain
    flatchain = burnin_flat(np.load(chain_file), f_burnin=f_burnin)
    ### 2. corner for stellar parameters
    star_fig = corner.corner(flatchain[:, 0:default_sn_spl_id], labels=labels[:default_sn_spl_id], show_titles=True)
    star_fig.suptitle(object, fontsize=20)
    # save plot
    star_corner_figure_path = os.path.expandvars(plotdir + "star_inference/star_stellar_corner.%s"%(format))
    star_fig.savefig(star_corner_figure_path, format=format, dpi=dpi)
    ### 3. corner for nuisance parameters:
    nuisance_fig = corner.corner(flatchain[:, default_sn_spl_id:], labels=labels[default_sn_spl_id:], show_titles=True)
    nuisance_fig.suptitle(object, fontsize=20)
    # save plot
    nuisance_corner_figure_path = os.path.expandvars(plotdir + "star_inference/star_nuisance_corner.%s"%(format))
    nuisance_fig.savefig(nuisance_corner_figure_path, format=format, dpi=dpi)
    print("corner plot finished!")
# ----


def store_star_MarleyMod(object, chain_file, resfile, specfile, f_burnin=0.5, num_noisedraw=1000):
    ''' store star_MarleyMod.py parameters into HDF5 files, including:
        - observed wavelength
        - observed flux
        - best-fit model flux
        - covariance matrix
        - best-fit stellar & nuisance parameters with uncertainties
        '''
    ### 0. spectrum id and order key
    spectrum_id = 0
    order_key = 0
    fix_c0 = True
    ### 1. obtain the best-fit parameters
    flatchain = burnin_flat(np.load(chain_file), f_burnin=f_burnin)
    star_pars = np.median(flatchain, axis=0)
    star_pars_upp_err = np.percentile(flatchain, 84, axis=0) - star_pars
    star_pars_low_err = star_pars - np.percentile(flatchain, 16, axis=0)
    # final parameters:
    final_param = np.vstack((star_pars, star_pars_upp_err, star_pars_low_err))
    Teff = final_param[:, 0]
    logg = final_param[:, 1]
    vz = final_param[:, 2]
    vsini = final_param[:, 3]
    logOmega = final_param[:, 4]
    Cheb_c1 = final_param[:, 5]
    Cheb_c2 = final_param[:, 6]
    Cheb_c3 = final_param[:, 7]
    nuis_sigAmp = final_param[:, 8]
    nuis_logAmp = final_param[:, 9]
    nuis_l = final_param[:, 10]
    # claim the fitting results
    print("Fitting results -")
    for item_index in range(0, len(star_pars)):
        print("%-10s: %10.2f (+ %5.2f) (- %5.2f)"
              %(print_labels[item_index], star_pars[item_index],star_pars_upp_err[item_index], star_pars_low_err[item_index]))
    ### 2. obtain best-fit model flux and covariance matrix
    star_model = SampleThetaPhi(debug=True)
    star_model.initialize((spectrum_id, order_key))
    bestfit_theta = ThetaParam(grid=star_pars[0:2],
                               vz=star_pars[2],
                               vsini=star_pars[3],
                               logOmega=star_pars[4])
    star_model.update_Theta(bestfit_theta)
    bestfit_phi = PhiParam(spectrum_id=spectrum_id,
                           order=order_key,
                           fix_c0=fix_c0,
                           cheb=star_pars[5:8],
                           sigAmp=star_pars[8],
                           logAmp=star_pars[9],
                           l=star_pars[10])
    star_model.update_Phi(bestfit_phi)
    mod_fls, mod_Covmat = star_model.drawmod_fls_covmat()
    ### 3. obtain the observed spectrum
    obs_wls = star_model.wl
    obs_fls = star_model.fl
    obs_sigmas = star_model.sigma
    obs_mask = star_model.mask
    # residual
    resid_fls = obs_fls - mod_fls
    ### 4. draw a standard noise spectrum from covriance matrix
    print("generating noise...")
    drawn_fls = random_draws(mod_Covmat, num_noisedraw)
    noise_1s_low, noise_1s_upp = sigma_envelope(drawn_fls, num_sigma=1)
    noise_2s_low, noise_2s_upp = sigma_envelope(drawn_fls, num_sigma=2)
    noise_3s_low, noise_3s_upp = sigma_envelope(drawn_fls, num_sigma=3)
    ### 5. save into a HDF5 resfile (results file)
    resfile_load = h5py.File(os.path.expandvars(resfile), "w")
    # spec id & orders
    resfile_load.create_dataset('spectrum_id', data=spectrum_id)
    resfile_load.create_dataset('order', data=order_key)
    # observed spectra
    resfile_load.create_dataset('obs_wls', data=obs_wls)
    resfile_load.create_dataset('obs_fls', data=obs_fls)
    resfile_load.create_dataset('obs_sigmas', data=obs_sigmas)
    resfile_load.create_dataset('obs_mask', data=obs_mask)
    # model spectrum and noise in residual
    resfile_load.create_dataset('mod_fls', data=mod_fls)
    resfile_load.create_dataset('resid_fls', data=resid_fls)
    resfile_load.create_dataset('noise_1s_upp', data=noise_1s_upp)
    resfile_load.create_dataset('noise_1s_low', data=noise_1s_low)
    resfile_load.create_dataset('noise_2s_upp', data=noise_2s_upp)
    resfile_load.create_dataset('noise_2s_low', data=noise_2s_low)
    resfile_load.create_dataset('noise_3s_upp', data=noise_3s_upp)
    resfile_load.create_dataset('noise_3s_low', data=noise_3s_low)
    # best-fit model
    resfile_load.create_dataset('Teff', data=Teff)
    resfile_load.create_dataset('logg', data=logg)
    resfile_load.create_dataset('vz', data=vz)
    resfile_load.create_dataset('vsini', data=vsini)
    resfile_load.create_dataset('logOmega', data=logOmega)
    resfile_load.create_dataset('Cheb_c1', data=Cheb_c1)
    resfile_load.create_dataset('Cheb_c2', data=Cheb_c2)
    resfile_load.create_dataset('Cheb_c3', data=Cheb_c3)
    resfile_load.create_dataset('nuis_sigAmp', data=nuis_sigAmp)
    resfile_load.create_dataset('nuis_logAmp', data=nuis_logAmp)
    resfile_load.create_dataset('nuis_l', data=nuis_l)
    resfile_load.create_dataset('mod_Covmat', data=mod_Covmat)
    # save
    resfile_load.close()
    ### 6. additionally save spectra into a csv file
    specfile_load = pd.DataFrame({'obs_wls': obs_wls,
                                 'obs_fls': obs_fls,
                                 'obs_sigmas': obs_sigmas,
                                 'obs_mask': obs_mask,
                                 'mod_fls': mod_fls,
                                 'resid_fls': resid_fls,
                                 'noise_1s_upp': noise_1s_upp,
                                 'noise_1s_low': noise_1s_low,
                                 'noise_2s_upp': noise_2s_upp,
                                 'noise_2s_low': noise_2s_low,
                                 'noise_3s_upp': noise_3s_upp,
                                 'noise_3s_low': noise_3s_low})
    specfile_load.to_csv(os.path.expandvars(specfile), index=False)
# ----




def comp_star_spec(object, resdir, resfile, format='png', dpi=None):
    ''' compare the observed and model spectra by reading the resfile'''
    ### 1. check if resfile exists
    if os.path.isfile(os.path.expandvars(resfile))==False:
        print("error: resfile not found - %s\n- please run '--store' first."%(resfile))
    else:
        ### 2. read fitting results
        resfile_read = h5py.File(os.path.expandvars(resfile), "r")
        obs_wls = resfile_read['obs_wls'].value
        obs_fls = resfile_read['obs_fls'].value
        mod_fls = resfile_read['mod_fls'].value
        resid_fls = resfile_read['resid_fls'].value
        noise_1s_upp = resfile_read['noise_1s_upp'].value
        noise_1s_low = resfile_read['noise_1s_low'].value
        noise_2s_upp = resfile_read['noise_2s_upp'].value
        noise_2s_low = resfile_read['noise_2s_low'].value
        noise_3s_upp = resfile_read['noise_3s_upp'].value
        noise_3s_low = resfile_read['noise_3s_low'].value
        ### 3. define plotting y-axis range
        id_xrange = np.where((obs_wls >= wls_range[0]) & (obs_wls <= wls_range[-1]))
        comp_spec_min = y_scalar * np.min([ 0, np.nanmin(obs_fls[id_xrange]), np.nanmin(mod_fls[id_xrange]) ])
        comp_spec_max = y_scalar * np.max([ np.nanmax(obs_fls[id_xrange]), np.nanmax(mod_fls[id_xrange]) ])
        resid_spec_min = y_scalar * np.min([ 0, np.nanmin(resid_fls[id_xrange]), np.nanmin(noise_3s_low[id_xrange]) ])
        resid_spec_max = y_scalar * np.max([ np.nanmax(resid_fls[id_xrange]), np.nanmax(noise_3s_upp[id_xrange]) ])
        ### 3. comparison
        figure, axes = plt.subplots(nrows=2, figsize=(10, 8), sharex=True)
        # observation vs. model
        axes[0].plot(obs_wls, obs_fls, color='k', linestyle='-', linewidth=2, zorder=1, label='data')
        axes[0].plot(obs_wls, mod_fls, color='r', linestyle='-', linewidth=2, zorder=2, label='model')
        axes[0].set_ylabel(r'$f_\lambda [erg/s/cm2/A]$')
        axes[0].legend(loc='upper right')
        axes[0].set_xlim(obs_wls[0], obs_wls[-1])
        axes[0].set_ylim(comp_spec_min, comp_spec_max)
        # residual
        axes[1].plot(obs_wls, resid_fls, 'k', linestyle='-', linewidth=2, zorder=4, label='residual')
        axes[1].fill_between(obs_wls, noise_1s_low, noise_1s_upp, zorder=3, color='#e6550d')
        axes[1].fill_between(obs_wls, noise_2s_low, noise_2s_upp, zorder=2, color='#fdae6b')
        axes[1].fill_between(obs_wls, noise_3s_low, noise_3s_upp, zorder=1, color='#fee6ce')
        axes[1].set_xlabel(r'$\lambda$ [AA]')
        axes[1].set_ylabel(r'residual $f_\lambda [erg/s/cm2/A]$')
        axes[1].legend(loc='upper right')
        axes[1].set_xlim(obs_wls[0], obs_wls[-1])
        axes[1].set_ylim(resid_spec_min, resid_spec_max)
        # title
        figure.suptitle(object, fontsize=20)
        # save
        figure.subplots_adjust()
        compspec_figure_path = os.path.expandvars(resdir + "%s_spec.%s"%(object, format))
        figure.savefig(compspec_figure_path, format=format, dpi=dpi)
        print("spectra comparison finished!")
# ----


######################
# OFFICIAL PROCEDURES
######################
## 0. load basic information
if args.mode=='usr':
    print("- 'usr' mode: user-defined input/output (not available by Apr25,2018)")
    object = args.object
    plotdir = args.plotdir
    resdir = args.resdir
    resfile = args.resfile
    specfile = args.specfile
    chain_file = os.path.expandvars(args.chain_file)
else:
    if args.mode=='std':
        print("- 'std' mode: input/output from config.yaml")
    else:
        print("warning: mode should be either 'std' or 'usr'. assuming the 'std' mode anyway...")
    object = Starfish.config["name"]
    outdir = Starfish.config["outdir"]
    plotdir = Starfish.config["plotdir"]
    resdir = Starfish.config["resdir"]["path"]
    resfile = resdir + Starfish.config["resdir"]["resfile"]
    specfile = resdir + Starfish.config["resdir"]["specfile"]
    chain_file = os.path.expandvars(outdir + "star_inference/emcee_chain.npy")

## check the output chain file
if os.path.isfile(chain_file)==False:
    print("error: %s is not found... run star_MarleyMod.py first!"%(chain_file))
else:
    ## 1. plot chain
    if args.chain:
        plot_star_chain(object, chain_file, plotdir, f_burnin=args.f_burnin, format=args.format, dpi=args.dpi)
    ## 2. plot corner
    if args.corner:
        plot_star_corner(object, chain_file, plotdir, f_burnin=args.f_burnin, format=args.format, dpi=args.dpi)
    ## 3. store results
    if args.store:
        store_star_MarleyMod(object, chain_file, resfile, specfile, f_burnin=args.f_burnin, num_noisedraw=args.num_nd)
    ## 4. compare spectra
    if args.spec:
        comp_star_spec(object, resdir, resfile, format=args.format, dpi=args.dpi)
# ----



