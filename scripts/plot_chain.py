#!/usr/bin/env python
#
#
## Author: ZJ Zhang
#
# Modification History:
#
# ZJ Zhang (Apr 23th, 2018)
#
#################################################

import Starfish

import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import corner
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

import argparse
parser = argparse.ArgumentParser(prog="plot_chain.py", description="plot the emcee chains as a function of sampling steps; also produce corner plots upon optional request.")
parser.add_argument("--keyword", type=str, default=None, help="types of emcee output chains: 'emulator' and 'star_inference'")
parser.add_argument("--labels", type=list, default=None, help="y-axis labels for each parameter/dimension of emcee chains")
parser.add_argument("--f_burnin", type=float, default=0.5, help="burn-in fraction of the emcee chains")
parser.add_argument("--corner", action="store_true", help="plot the corner plot (only supporting the 'star_inference' mode at the moment)")
args = parser.parse_args()



#####################################
# Assistant Parameters and Functions
#####################################

# default settings:
default_figpath = 'ZJ_test.pdf'
default_fburnin = 0.5
default_labels = None
default_sn_spl_id = 5  # stellar parameters for <5 and nuisance for >=5 in the final chain
# chain file name
chain_file_name = {"emulator": "eparams_emcee.npy",
                   "star_inference": "emcee_chain.npy"}
# y-axis label
labels_option = {"star_inference": [r"$T_{\mathrm{eff}}$", r"$\log{g}$",r"$v_z$", r"$v\sin{i}$", r"$\log{\Omega}$",
                                   r"$c^1$", r"$c^2$", r"$c^3$", r"sigAmp", r"logAmp", r"$l$"]}


def evolution_chain(objname, chain_file, figure_path=default_figpath, labels=default_labels, f_burnin=default_fburnin):
    ### load chains
    chain = np.load(chain_file)
    nwalkers, nsamples, ndim = chain.shape
    print("- %s: %d walkers, %d samples, %d parameters"%(objname, nwalkers, nsamples, ndim))
    ### labels
    if labels is None:
        labels = ['undefined'] * ndim
    ### plotting
    figure, axes = plt.subplots(ndim, 1, sharex=True, figsize=(8, 14))
    for i in range(0, ndim):
        axes[i].plot(chain[:, :, i].T, color="k", alpha=0.3)
        axes[i].yaxis.set_major_locator(MaxNLocator(5))
        axes[i].set_ylabel(labels[i])
        axes[i].set_xlim([0,nsamples])
        axes[i].axvline(int(nsamples*f_burnin), color='red', linestyle='-') # burn-in cut-off
    axes[ndim-1].set_xlabel("step number")
    # save plot
    if figure_path is not None:
        figure.savefig(figure_path, format='pdf', dpi=1000)
# ----


def starinfer_corner(chain_file, star_corner_figure_path, nuisance_corner_figure_path, labels=default_labels, f_burnin=default_fburnin):
    ''' plot corner figures for stellar and nuisance parameters of the output chains from star_inference emcee'''
    ### load chains
    chain = np.load(chain_file)
    nwalkers, nsamples, ndim = chain.shape
    ### burned-in flat chain
    if (f_burnin is not None) and (0 <= f_burnin < 1):
        burned_chain = chain[:, int(nsamples * f_burnin):, :]
    else:
        print("warning: f_burnin should be in [0, 1) - using the entire chain...")
        burned_chain = chain
    flatchain = burned_chain.reshape((-1, ndim))
    ### labels
    if labels is None:
        labels = ['undefined'] * ndim
    ### corner for stellar parameters
    star_fig = corner.corner(flatchain[:, 0:default_sn_spl_id], labels=labels[:default_sn_spl_id], show_titles=True)
    star_fig.savefig(star_corner_figure_path, dpi=1000)
    ### corner for nuisance parameters:
    nuisance_fig = corner.corner(flatchain[:, default_sn_spl_id:], labels=labels[default_sn_spl_id:], show_titles=True)
    nuisance_fig.savefig(nuisance_corner_figure_path, dpi=1000)
# ----



######################
# OFFICIAL PROCEDURES
######################
## 0. load basic information
object = Starfish.config["name"]
outdir = Starfish.config["outdir"]
plotdir = Starfish.config["plotdir"]

## 1. plotting's
chain_file = os.path.expandvars(outdir + "%s/%s"%(args.keyword, chain_file_name[args.keyword]))
if os.path.isfile(chain_file)==False:
    print("error: %s is not found... (hint: 'keyword' should be either 'emulator' or 'star_inference')."%(chain_file))
else:
    ## 1.1 obtain figure path
    chainevo_figure_path = os.path.expandvars(plotdir + "%s/%s_chain_evo.pdf"%(args.keyword, args.keyword))
    ## 1.2 obtain label
    if args.keyword=="star_inference":
        labels = labels_option[args.keyword]
    ## 1.3 plot emcee walkers as a function of sampling steps
    evolution_chain(object, chain_file, figure_path=chainevo_figure_path, labels=labels, f_burnin=args.f_burnin)
    print("chain evolution plot finished!")
    ## 1.4 plot corner plots
    if args.corner==True:
        if args.keyword=="star_inference":
            # figure path
            star_corner_figure_path = os.path.expandvars(plotdir + "%s/star_corner.pdf"%(args.keyword))
            nuisance_corner_figure_path = os.path.expandvars(plotdir + "%s/nuisance_corner.pdf"%(args.keyword))
            # plot corner plots for star_inference emcee results
            starinfer_corner(chain_file, star_corner_figure_path, nuisance_corner_figure_path, labels=labels, f_burnin=args.f_burnin)
            print("chain corner plot finished!")
        else:
            print("warning: chain corner plot only available for 'star_inference' keyword... no plot produced.")
# ----




















