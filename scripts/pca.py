#!/usr/bin/env python
#
#
## Author: I. Czekala
#
# Modification History:
#
# M. Gully (2015-2017)
# ZJ Zhang (Jul. 21st, 2017)   [ADD --- "corner" module in "args.plot == "emcee""]
# ZJ Zhang (Feb. 23th, 2018)   [REVISE --- replace the burn-in phase by adding a user-defined burn-in fraction "f_burnin"; the "resume" keyword will concatenate the output chains automatically]
#
#################################################


import argparse
parser = argparse.ArgumentParser(description="Create and manipulate a PCA decomposition of the synthetic spectral library.")
parser.add_argument("--create", action="store_true", help="Create a PCA decomposition.")

parser.add_argument("--plot", choices=["reconstruct", "eigenspectra", "priors", "emcee",
                                       "emulator", "chain"], help="reconstruct: plot the original synthetic spectra vs. the PCA reconstructed spectra.\n priors: plot the chosen priors on the parameters.\n emcee: plot the triangle diagram for the result of the emcee optimization.\n emulator: plot weight interpolations.\n chain: plot the chain values as a function of sampling steps.")

parser.add_argument("--f_burnin", type=float, default=None, help="The fraction of the entire emcee output chain that need to be removed before parameter inferences.")

parser.add_argument("--optimize", choices=["fmin", "emcee"], help="Optimize the emulator using either a downhill simplex algorithm or the emcee ensemble sampler algorithm.")

parser.add_argument("--resume", action="store_true", help="Designed to be used with the --optimize flag to continue from the previous set of parameters. If this is left off, the chain will start from your initial guess specified in config.yaml.")

parser.add_argument("--samples", type=int, default=100, help="Number of samples to run the emcee ensemble sampler.")
parser.add_argument("--params", choices=["fmin", "emcee"], help="Which optimized parameters to use.")

parser.add_argument("--store", action="store_true", help="Store the optimized emulator parameters to the HDF5 file. Use with the --params=fmin or --params=emcee to choose.")
args = parser.parse_args()

import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import itertools
import Starfish
from Starfish import emulator
from Starfish.grid_tools import HDF5Interface
from Starfish.emulator import PCAGrid, Gprior, Glnprior, Emulator
from Starfish.covariance import Sigma
import os


### output path setups - ZJ Zhang
## plots
plotdir = os.path.expandvars(Starfish.config["plotdir"])
# pca
pca_plotdir = plotdir + "pca/"
# emulator
emulator_plotdir = plotdir + "emulator/"
## output files
emulator_outdir = os.path.expandvars(Starfish.config["outdir"]) + "emulator/"
###################################



if args.create:
    myHDF5 = HDF5Interface()
    my_pca = PCAGrid.create(myHDF5)
    my_pca.write()

if args.plot == "reconstruct":
    my_HDF5 = HDF5Interface()
    my_pca = PCAGrid.open()

    recon_fluxes = my_pca.reconstruct_all()

    # we need to apply the same normalization to the synthetic fluxes that we
    # used for the reconstruction
    fluxes = np.empty((my_pca.M, my_pca.npix))
    for i, spec in enumerate(my_HDF5.fluxes):
        fluxes[i,:] = spec

    # Normalize all of the fluxes to an average value of 1
    # In order to remove uninteresting correlations
    fluxes = fluxes/np.average(fluxes, axis=1)[np.newaxis].T

    data = zip(my_HDF5.grid_points, fluxes, recon_fluxes)

    # Define the plotting function
    def plot(data):
        par, real, recon = data
        fig, ax = plt.subplots(nrows=2, figsize=(8, 8))
        ax[0].plot(my_pca.wl, real)
        ax[0].plot(my_pca.wl, recon)
        ax[0].set_ylabel(r"$f_\lambda$")

        ax[1].plot(my_pca.wl, real - recon)
        ax[1].set_xlabel(r"$\lambda$ [AA]")
        ax[1].set_ylabel(r"residual $f_\lambda$")

        fmt = "=".join(["{:.2f}" for i in range(len(Starfish.parname))])
        name = fmt.format(*[p for p in par])
        ax[0].set_title(name+"_averes_%.3f_stdres_%.3f"%(np.mean(real - recon), np.std(real - recon)))
        fig.savefig(pca_plotdir + "PCA_" + name + ".png")
        plt.close("all")

    p = mp.Pool(mp.cpu_count())
    p.map(plot, data)

if args.plot == "eigenspectra":
    my_HDF5 = HDF5Interface()
    my_pca = PCAGrid.open()

    row_height = 3 # in
    margin = 0.5 # in

    fig_height = my_pca.m * (row_height + margin) + margin
    fig_width = 14 # in

    fig = plt.figure(figsize=(fig_width, fig_height))

    for i in range(my_pca.m):

        ax = plt.subplot2grid((my_pca.m, 4), (i, 0), colspan=3)
        ax.plot(my_pca.wl, my_pca.eigenspectra[i])
        ax.set_xlabel(r"$\lambda$ [AA]")
        ax.set_ylabel(r"$\xi_{}$".format(i))

        ax = plt.subplot2grid((my_pca.m, 4), (i, 3))
        ax.hist(my_pca.w[i], histtype="step", normed=True)
        ax.set_xlabel(r"$w_{}$".format(i))
        ax.set_ylabel("count")

    fig.subplots_adjust(wspace=0.3, left=0.1, right=0.98, bottom=0.1, top=0.98)
    fig.savefig(pca_plotdir + "eigenspectra.png")


if args.plot == "priors":
    # Read the priors on each of the parameters from Starfish config.yaml
    priors = Starfish.PCA["priors"]
    for i,par in enumerate(Starfish.parname):
        s, r = priors[i]
        mu = s/r
        x = np.linspace(0.01, 2 * mu)
        prob = Gprior(x, s, r)
        plt.plot(x, prob)
        plt.xlabel(par)
        plt.ylabel("Probability")
        plt.savefig(pca_plotdir + "prior_" + par + ".png")
        plt.close("all")

# If we're doing optimization, period, set up some variables and the lnprob
if args.optimize:
    my_pca = emulator.PCAGrid.open()
    PhiPhi = np.linalg.inv(emulator.skinny_kron(my_pca.eigenspectra, my_pca.M))
    priors = Starfish.PCA["priors"]

    def lnprob(p, fmin=False):
        '''
        :param p: Gaussian Processes hyper-parameters
        :type p: 1D np.array

        Calculate the lnprob using Habib's posterior formula for the emulator.
        '''

        # We don't allow negative parameters.
        if np.any(p < 0.):
            if fmin:
                return 1e99
            else:
                return -np.inf

        lambda_xi = p[0]
        hparams = p[1:].reshape((my_pca.m, -1))

        # Calculate the prior for parname variables
        # We have two separate sums here, since hparams is a 2D array
        # hparams[:, 0] are the amplitudes, so we index i+1 here
        lnpriors = 0.0
        for i in range(0, len(Starfish.parname)):
            lnpriors += np.sum(Glnprior(hparams[:, i+1], *priors[i]))

        h2params = hparams**2
        #Fold hparams into the new shape
        Sig_w = Sigma(my_pca.gparams, h2params)

        C = (1./lambda_xi) * PhiPhi + Sig_w

        sign, pref = np.linalg.slogdet(C)

        central = my_pca.w_hat.T.dot(np.linalg.solve(C, my_pca.w_hat))

        lnp = -0.5 * (pref + central + my_pca.M * my_pca.m * np.log(2. * np.pi)) + lnpriors

        # Negate this when using the fmin algorithm
        if fmin:
            print("lambda_xi", lambda_xi)
            for row in hparams:
                print(row)
            print()
            print(lnp)

            return -lnp
        else:
            return lnp


if args.optimize == "fmin":

    if args.resume:
        p0 = np.load(emulator_outdir+"eparams_fmin.npy")

    else:
        amp = 100.
        # Use the mean of the gamma distribution to start
        eigpars = np.array([amp] + [s/r for s,r in priors])
        p0 = np.hstack((np.array([1., ]), #lambda_xi
        np.hstack([eigpars for i in range(my_pca.m)])))

    from scipy.optimize import fmin
    func = lambda p : lnprob(p, fmin=True)
    result = fmin(func, p0, maxiter=10000, maxfun=10000)
    print(result)
    np.save(emulator_outdir+"eparams_fmin.npy", result)

if args.optimize == "emcee":

    import emcee

    ndim = 1 + (1 + len(Starfish.parname)) * my_pca.m
    nwalkers = 4 * ndim # about the minimum per dimension we can get by with

    # Assemble p0 based off either a guess or the previous state of walkers
    if args.resume:
        p0 = np.load(emulator_outdir+"walkers_emcee.npy")
    else:
        p0 = []
        # p0 is a (nwalkers, ndim) array
        amp = [10.0, 150]

        p0.append(np.random.uniform(0.01, 1.0, nwalkers))
        for i in range(my_pca.m):
            p0 +=   [np.random.uniform(amp[0], amp[1], nwalkers)]
            for s,r in priors:
                # Draw randomly from the gamma priors
                p0 += [np.random.gamma(s, 1./r, nwalkers)]

        p0 = np.array(p0).T

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=mp.cpu_count())

    # actual run without burn-in at this stage
    pos, prob, state = sampler.run_mcmc(p0, args.samples)

    # Save the last position of the walkers
    np.save(emulator_outdir+"walkers_emcee.npy", pos)

    # Save the emcee chain
    if args.resume:
        prev_chain = np.load(emulator_outdir+"eparams_emcee.npy")
        new_chain = np.hstack((prev_chain, sampler.chain))
        np.save(emulator_outdir+"eparams_emcee.npy", new_chain)
    else:
        np.save(emulator_outdir+"eparams_emcee.npy", sampler.chain)


if args.plot == "chain":
    ''' plot chain values as a function of sampling steps
        '''
    ### load chains
    chain = np.load(emulator_outdir+"eparams_emcee.npy")
    nwalkers, nsamples, ndim = chain.shape
    print("Emulator training chains: %d walkers, %d samples, %d parameters"%(nwalkers, nsamples, ndim))
    ### plotting
    figure, axes = plt.subplots(ndim, 1, sharex=True, figsize=(8, 14))
    # labels
    label = [r"$\lambda_{\xi}$"]
    for i_eigen in range(0,int((ndim-1)/3)):
        label = label + [r"Amp_%d"%(i_eigen), r"l_Temp_%d"%(i_eigen), r"l_logg_%d"%(i_eigen)]
    # plot
    for i in range(0, ndim):
        axes[i].plot(chain[:, :, i].T, color="k", alpha=0.2)
        axes[i].yaxis.set_major_locator(MaxNLocator(5))
        axes[i].set_ylabel(label[i])
        axes[i].set_xlim([0,nsamples])
        axes[i].axvline(int(nsamples * args.f_burnin), color='red', linestyle='-') # burn-in cut-off
    axes[ndim-1].set_xlabel("step number")
    # save plot
    figure.savefig(emulator_plotdir+'emulator_chain.png', format='png')
# ----


if args.plot == "emcee":
    # Make a triangle plot of the samples
    my_pca = emulator.PCAGrid.open()
    chain = np.load(emulator_outdir+"eparams_emcee.npy")
    # remove the burn-in phase
    nwalkers, nsamples, ndim = chain.shape
    if (args.f_burnin is not None) and (0 <= args.f_burnin < 1):
        burned_chain = chain[:, int(nsamples * args.f_burnin):, :]
    else:
        print("warning: f_burnin should be in [0, 1) - using the entire chain...")
        burned_chain = chain
    flatchain = burned_chain.reshape((-1, ndim))

    try:
        import corner as triangle
    except:
        import triangle

    # figure out how many separate triangle plots we need to make
    npar = len(Starfish.parname) + 1
    labels = ["amp"] + Starfish.parname

    # Make a histogram of lambda xi
    plt.hist(flatchain[:,0], histtype="step", normed=True)
    plt.title(r"$\lambda_\xi$")
    plt.xlabel(r"$\lambda_\xi$")
    plt.ylabel("prob")
    plt.savefig(emulator_plotdir + "triangle_lambda_xi.png")

    # Make a triangle plot for each eigenspectrum independently
    for i in range(my_pca.m):
        start = 1 + i * npar
        end = 1 + (i + 1) * npar
        figure = triangle.corner(flatchain[:, start:end], quantiles=[0.16, 0.5, 0.84],
            plot_contours=True, plot_datapoints=False, show_titles=True, labels=labels)
        figure.savefig(emulator_plotdir + "triangle_{}.png".format(i))

if args.plot == "emulator":

    my_pca = PCAGrid.open()

    if args.params == "fmin":
        eparams = np.load(emulator_outdir+"eparams_fmin.npy")
    elif args.params == "emcee":
        # load chains
        chain = np.load(emulator_outdir+"eparams_emcee.npy")
        # remove the burn-in phase
        nwalkers, nsamples, ndim = chain.shape
        if (args.f_burnin is not None) and (0 <= args.f_burnin < 1):
            burned_chain = chain[:, int(nsamples * args.f_burnin):, :]
        else:
            print("warning: f_burnin should be in [0, 1) - using the entire chain...")
            burned_chain = chain
        flatchain = burned_chain.reshape((-1, ndim))
        # final parameters
        eparams = np.median(flatchain, axis=0)
        print("Using emcee median")
    else:
        import sys
        sys.exit()

    # Print out the emulator parameters in an easily-readable format
    lambda_xi = eparams[0]
    hparams = eparams[1:].reshape((my_pca.m, -1))
    print("Emulator parameters are:")
    print("lambda_xi", lambda_xi)
    for row in hparams:
        print(row)

    emulator = Emulator(my_pca, eparams)

    # We will want to produce interpolated plots spanning each parameter dimension,
    # for each eigenspectrum.

    # Create a list of parameter blocks.
    # Go through each parameter, and create a list of all parameter combination of
    # the other two parameters.
    unique_points = [np.unique(my_pca.gparams[:, i]) for i in range(len(Starfish.parname))]
    blocks = []
    for ipar, pname in enumerate(Starfish.parname):
        upars = unique_points.copy()
        dim = upars.pop(ipar)
        ndim = len(dim)

        # use itertools.product to create permutations of all possible values
        par_combos = itertools.product(*upars)

        # Now, we want to create a list of parameters in the original order.
        for static_pars in par_combos:
            par_list = []
            for par in static_pars:
                par_list.append( par * np.ones((ndim,)))

            # Insert the changing dim in the right location
            par_list.insert(ipar, dim)

            blocks.append(np.vstack(par_list).T)


    # Now, this function takes a parameter block and plots all of the eigenspectra.
    npoints = 40 # How many points to include across the active dimension
    ndraw = 8 # How many draws from the emulator to use

    def plot_block(block):
        # block specifies the parameter grid points
        # fblock defines a parameter grid that is finer spaced than the gridpoints

        # Query for the weights at the grid points.
        ww = np.empty((len(block), my_pca.m))
        for i,param in enumerate(block):
            weights = my_pca.get_weights(param)
            ww[i, :] = weights

        # Determine the active dimension by finding the one that has unique > 1
        uni = np.array([len(np.unique(block[:, i])) for i in range(len(Starfish.parname))])
        active_dim = np.where(uni > 1)[0][0]

        ublock = block.copy()
        ablock = ublock[:,active_dim]
        ublock = np.delete(ublock, active_dim, axis=1)
        nactive = len(ablock)

        fblock = []
        for par in ublock[0, :]:
            # Create a space of the parameter the same length as the active
            fblock.append(par * np.ones((npoints,)))

        # find min and max of active dim. Create a linspace of `npoints` spanning from
        # min to max
        active = np.linspace(ablock[0], ablock[-1], npoints)

        fblock.insert(active_dim, active)
        fgrid = np.vstack(fblock).T

        # Draw multiple times at the location.
        weight_draws = []
        for i in range(ndraw):
            weight_draws.append(emulator.draw_many_weights(fgrid))

        # Now make all of the plots
        for eig_i in range(my_pca.m):
            fig, ax = plt.subplots(nrows=1, figsize=(6,6))

            x0 = block[:, active_dim] # x-axis
            # Weight values at grid points
            y0 = ww[:, eig_i]
            ax.plot(x0, y0, "bo")

            x1 = fgrid[:, active_dim]
            for i in range(ndraw):
                y1 = weight_draws[i][:, eig_i]
                ax.plot(x1, y1)

            ax.set_ylabel(r"$w_{:}$".format(eig_i))
            ax.set_xlabel(Starfish.parname[active_dim])

            fstring = "w{:}".format(eig_i) + Starfish.parname[active_dim] + "".join(["{:.1f}".format(ub) for ub in ublock[0, :]])

            fig.savefig(emulator_plotdir + fstring + ".png")

            plt.close('all')


    # Create a pool of workers and map the plotting to these.
    p = mp.Pool(mp.cpu_count() - 1)
    p.map(plot_block, blocks)

if args.store:
    if args.params == "fmin":
        eparams = np.load(emulator_outdir+"eparams_fmin.npy")
    elif args.params == "emcee":
        # load chains
        chain = np.load(emulator_outdir+"eparams_emcee.npy")
        # remove the burn-in phase
        nwalkers, nsamples, ndim = chain.shape
        if (args.f_burnin is not None) and (0 <= args.f_burnin < 1):
            burned_chain = chain[:, int(nsamples * args.f_burnin):, :]
        else:
            print("warning: f_burnin should be in [0, 1) - using the entire chain...")
            burned_chain = chain
        flatchain = burned_chain.reshape((-1, ndim))
        # final parameters
        eparams = np.median(flatchain, axis=0)
        print("Using emcee median")
    else:
        import sys
        sys.exit()

    import h5py
    filename = os.path.expandvars(Starfish.PCA["path"])
    hdf5 = h5py.File(filename, "r+")

    # check to see whether the dataset already exists
    if "eparams" in hdf5.keys():
        pdset = hdf5["eparams"]
    else:
        pdset = hdf5.create_dataset("eparams", eparams.shape, compression="gzip", compression_opts=9, dtype="f8")

    pdset[:] = eparams
    hdf5.close()
