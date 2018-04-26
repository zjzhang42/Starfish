#!/usr/bin/env python
#
#
## Author: I. Czekala and Michael Gully
#
# Modification History:
#
# ZJ Zhang (Apr. 24th, 2018)
#
#################################################

# ----
# Goal:
# fit models to observations and obtain physical/nuisance/covariance-hyper parameters
# ----

# import master package
from star_class import SampleThetaPhi

# import standard packages
import numpy as np
import os
from numpy.polynomial import Chebyshev as Ch
import emcee
import multiprocessing
import time

# import Starfish packages
import Starfish
from Starfish.model_BD import ThetaParam, PhiParam

import argparse
parser = argparse.ArgumentParser(prog="star_marley.py", description="Run Starfish fitting model in single order mode with many walkers.")
parser.add_argument("--samples", type=int, default=5, help="How many samples to run?")
parser.add_argument("--incremental_save", type=int, default=100, help="How often to save incremental progress of MCMC samples.")
parser.add_argument("--resume", action="store_true", help="Continue from the last sample. If this is left off, the chain will start from your initial guess specified in config.yaml.")
args = parser.parse_args()


### output path setups - ZJ Zhang
## output files
emulator_outdir = os.path.expandvars(Starfish.config["outdir"]) + "emulator/"
star_outdir = os.path.expandvars(Starfish.config["outdir"]) + "star_inference/"
###################################



## 0. Preparation
# id
spectrum_id, order_key = 0, 0 # ??? what is spectrum id and order_key?
# fix c0 of Chebyshev function
fix_c0 = True
# --

## 1. Initial parameter settings
''' initial parameters

    grid [Teff, logg]
    vz
    vsini
    logOmega
    Chebyshev function coefficients (we assume c0 is fixed at 1.0 so there are one less free parameters. e.g., 4-th order Chebyshev function will only have 3 free parameters with c0=1)
    sigAmp
    logAmp
    l
    '''
# 1.1 Load config.yaml
# - Theta
theta_start = Starfish.config["Theta"]
theta_jump = Starfish.config["Theta_jump"]
# - Cheb & Phi
cheb_start = np.zeros((Starfish.config["cheb_degree"]-1,)) if fix_c0 else np.zeros((Starfish.config["cheb_degree"],))
cheb_jump = np.ones_like(cheb_start) * Starfish.config["cheb_jump"]
phi_start = [Starfish.config["Phi"]["sigAmp"], Starfish.config["Phi"]["logAmp"], Starfish.config["Phi"]["l"]]
phi_jump = [Starfish.config["Phi_jump"]["sigAmp"], Starfish.config["Phi_jump"]["logAmp"], Starfish.config["Phi_jump"]["l"]]
# --
# 1.2 Combine initial settings to feed emcee
p0 = np.hstack((np.array(theta_start["grid"]),
                theta_start["vz"],
                theta_start["vsini"],
                theta_start["logOmega"],
                cheb_start,
                np.array(phi_start)
                ))
p0_std = np.hstack((np.array(theta_jump["grid"]),
                    theta_jump["vz"],
                    theta_jump["vsini"],
                    theta_jump["logOmega"],
                    cheb_jump,
                    np.array(phi_jump)
                    ))
# --
# 1.3 Save initial settings
# - Theta
theta_par = ThetaParam(grid=p0[0:2], vz=p0[2], vsini=p0[3], logOmega=p0[4])
theta_par.save()
# - Phi (including Cheb)
phi_par = PhiParam(spectrum_id, order_key, fix_c0, p0[5:8], p0[8], p0[9], p0[10])
phi_par.save()
# --


# 2. Run the program.
model = SampleThetaPhi(debug=True)
model.initialize((spectrum_id, order_key))


## 3. Posterior:
# 3.1 Likelihood function
def lnlike(p):
    try:
        pars1 = ThetaParam(grid=p[0:2], vz=p[2], vsini=p[3], logOmega=p[4])
        model.update_Theta(pars1)
        # hard code npoly=3 (for fixc0 = True with npoly=4)
        pars2 = PhiParam(0, 0, True, p[5:8], p[8], p[9], p[10])
        model.update_Phi(pars2)
        lnp = model.evaluate()
        return lnp
    except C.ModelError:
        model.logger.debug("ModelError in stellar parameters, sending back -np.inf {}".format(p))
        return -np.inf
# --
# 3.2 Prior function
def lnprior(p):
    if not (p[10] > 0):
        return -np.inf
    else:
        return 0
# Try to load a user-defined prior
try:
    sourcepath_env = Starfish.config['Theta_priors']
    sourcepath = os.path.expandvars(sourcepath_env)
    with open(sourcepath, 'r') as f:
        sourcecode = f.read()
    code = compile(sourcecode, sourcepath, 'exec')
    exec(code)
    lnprior = user_defined_lnprior
    print("Using the user defined prior in {}".format(sourcepath_env))
except:
    print("Don't you want to use a user defined prior??")
    raise

# 3.3 Chebyshev priors
x_vec = np.arange(-1, 1, 0.01)
def cheb_prior(p):
    ch_tot = Ch([0, p[5], p[6], p[7]])
    ch_spec = ch_tot(x_vec)
    if not ( (np.max(ch_spec) < 0.01) and
             (np.min(ch_spec) > -0.01) ):
        return -np.inf
    return 0.0
# --
# 3.4 posterior probability:
def lnprob(p):
    lp = lnprior(p) + cheb_prior(p)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(p)
# --
# ----

## 4. Start MCMC fitting
ndim = len(p0)
nwalkers = 4 * ndim

if args.resume:
    p0_ball = np.load(star_outdir+"emcee_chain.npy")[:,-1,:]
else:
    p0_ball = emcee.utils.sample_ball(p0, p0_std, size=nwalkers)

n_threads = multiprocessing.cpu_count()
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=n_threads)


nsteps = args.samples
ninc = args.incremental_save
for i, (pos, lnp, state) in enumerate(sampler.sample(p0_ball, iterations=nsteps)):
    if (i+1) % ninc == 0:
        time.ctime() 
        t_out = time.strftime('%Y %b %d,%l:%M %p') 
        print("{0}: {1:}/{2:} = {3:.1f}%".format(t_out, i, nsteps, 100 * float(i) / nsteps))
        np.save(star_outdir+'temp_emcee_chain.npy',sampler.chain)

# Save the emcee chain
if args.resume:
    prev_chain = np.load(emulator_outdir+"emcee_chain.npy")
    new_chain = np.hstack((prev_chain, sampler.chain))
    np.save(emulator_outdir+"emcee_chain.npy", new_chain)
else:
    np.save(star_outdir+'emcee_chain.npy',sampler.chain)

print("The end.")
