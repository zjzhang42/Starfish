#!/usr/bin/env python
#
#
## Author: ZJ Zhang (Jul. 24th, 2017)
#
# Modification History:
#
# ZJ Zhang (Jul. 24th, 2017)
#
#################################################

# ----
# Goal:
# plot parameter values as a function of step in star_BD.py (emcee pipeline), and then test if the system has converged.
# ----


# import modules
import numpy as np
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import Starfish

## 1. Load emcee chain
chain = np.load('emcee_chain.npy')
# parameter space dimension
nwalkers, n_samples, ndim = chain.shape
# --

## 2. Plot
# 2.1 label settings
label = ["$T_{\mathrm{eff}}$", "$\log{g}$","$v_z$", "$v\sin{i}$", "$\log{\Omega}$", "$c^1$", "$c^2$", "$c^3$", "sigAmp", "logAmp", "$l$"]
print("Convergence test for output emcee_chain of star_BD.py")
# 2.2 plotting
figure, axes = plt.subplots(ndim, 1, sharex=True, figsize=(8, 14))
for i in range(0, ndim, 1):
    axes[i].plot(chain[:, :, i].T, color="k", alpha=0.2)
    axes[i].yaxis.set_major_locator(MaxNLocator(5))
    axes[i].set_ylabel(label[i])
axes[ndim-1].set_xlabel("step number")
# 2.3 funeral
figure.tight_layout(h_pad=0.0)
figure_path = Starfish.config["plotdir"] + "converge_test_star_BD.png"
figure.savefig(figure_path)
print("figure created - %s.\n"%(figure_path))







