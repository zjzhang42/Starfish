#!/usr/bin/env python

# All of the argument parsing is done in the `parallel.py` module.

import numpy as np
import Starfish
from Starfish import parallel
from Starfish.parallel import args
from Starfish.model import ThetaParam, PhiParam
import time
from emcee.utils import sample_ball


if args.generate:
    model = parallel.OptimizeTheta(debug=True)

    # Now that the different processes have been forked, initialize them
    pconns, cconns, ps = parallel.initialize(model)

    pars = ThetaParam.from_dict(Starfish.config["Theta"])

    for ((spectrum_id, order_id), pconn) in pconns.items():
        #Parse the parameters into what needs to be sent to each Model here.
        pconn.send(("LNPROB", pars))
        pconn.recv() # Receive and discard the answer so we can send the save
        pconn.send(("SAVE", None))

    # Kill all of the orders
    for pconn in pconns.values():
        pconn.send(("FINISH", None))
        pconn.send(("DIE", None))

    # Join on everything and terminate
    for p in ps.values():
        p.join()
        p.terminate()

    import sys;sys.exit()


if args.optimize == "Theta":

    # Check to see if the order JSONs exist, if so, then recreate the noise structure according to these.

    # Otherwise assume white noise.
    model = parallel.OptimizeTheta(debug=True)

    # Now that the different processes have been forked, initialize them
    pconns, cconns, ps = parallel.initialize(model)

    def fprob(p):

        # Assume p is [temp, logg, Z, vz, vsini, logOmega]

        #pars = ThetaParam(grid=p[0:3], vz=p[3], vsini=p[4], logOmega=p[5])
        pars = ThetaParam(grid=p[0:3], vz=p[3], vsini=p[4], logOmega=p[5], teff2=p[6], logOmega2=p[7])

        #Distribute the calculation to each process
        for ((spectrum_id, order_id), pconn) in pconns.items():
            #Parse the parameters into what needs to be sent to each Model here.
            pconn.send(("LNPROB", pars))

        #Collect the answer from each process
        lnps = np.empty((len(Starfish.data["orders"]),))
        for i, pconn in enumerate(pconns.values()):
            lnps[i] = pconn.recv()

        s = np.sum(lnps)

        #print(pars, "lnp:", s)
        print("{Teff: >8.1f} {logg: 8.3f} {feh: >8.3f}".format(Teff=pars.grid[0], logg=pars.grid[1], feh=pars.grid[2]), end=' ')
        print("{vz: >8.2f} {vsini: 8.2f} {logOm: >8.4f}".format(vz=pars.vz, vsini=pars.vsini, logOm=pars.logOmega), end=' ')
        print("{Teff2: >8.1f} {logOm2: >8.4f}".format(Teff2=pars.teff2, logOm2=pars.logOmega2), end=' ')
        print("{s: >12.1f}".format(s=s))

        if s == -np.inf:
            return 1e99
        else:
            return -s

    start = Starfish.config["Theta"]
    p0 = np.array(start["grid"] + [start["vz"], start["vsini"], start["logOmega"]])
    if 'teff2' in start.keys():
        p0 = np.append(p0, start["teff2"])
    if 'logOmega2' in start.keys():
        p0 = np.append(p0, start["logOmega2"])

    from scipy.optimize import fmin
    p = fmin(fprob, p0, maxiter=10000, maxfun=10000)
    #print(p)
    pars = ThetaParam(grid=p[0:3], vz=p[3], vsini=p[4], logOmega=p[5], teff2=p[6], logOmega2=p[7])
    pars.save()

    # Kill all of the orders
    for pconn in pconns.values():
        pconn.send(("FINISH", None))
        pconn.send(("DIE", None))

    # Join on everything and terminate
    for p in ps.values():
        p.join()
        p.terminate()

    import sys;sys.exit()

if args.initPhi:
    # Figure out how many models and orders we have
    i_last = len(Starfish.data["orders"]) - 1

    for spec_id in range(len(Starfish.data["files"])):
        for i, order in enumerate(Starfish.data["orders"]):
            fix_c0 = True if i==i_last else False
            if fix_c0:
                cheb = np.zeros((Starfish.config["cheb_degree"] - 1,))
            else:
                cheb = np.zeros((Starfish.config["cheb_degree"],))

            # For each order, create a Phi with these values
            # Automatically reads all of the Phi parameters from config.yaml
            phi = PhiParam(spectrum_id=spec_id, order=int(order), fix_c0=fix_c0, cheb=cheb)
            # Write to CWD using predetermined format string
            phi.save()

if args.optimize == "Cheb":

    model = parallel.OptimizeCheb(debug=True)

    # Now that the different processes have been forked, initialize them
    pconns, cconns, ps = parallel.initialize(model)

    # Initialize to the basics
    pars = ThetaParam.from_dict(Starfish.config["Theta"])

    #Distribute the calculation to each process
    for ((spectrum_id, order_id), pconn) in pconns.items():
        #Parse the parameters into what needs to be sent to each Model here.
        pconn.send(("LNPROB", pars))
        pconn.recv() # Receive and discard the answer so we can send the optimize
        pconn.send(("OPTIMIZE_CHEB", None))

    # Kill all of the orders
    for pconn in pconns.values():
        pconn.send(("FINISH", None))
        pconn.send(("DIE", None))

    # Join on everything and terminate
    for p in ps.values():
        p.join()
        p.terminate()

    import sys;sys.exit()

if args.sample == "ThetaCheb" or args.sample == "ThetaPhi" or args.sample == "ThetaPhiLines":

    if args.sample == "ThetaCheb":
        model = parallel.SampleThetaCheb(debug=True)
    if args.sample == "ThetaPhi":
        model = parallel.SampleThetaPhi(debug=True)
    if args.sample == "ThetaPhiLines":
        model = parallel.SampleThetaPhiLines(debug=True)


    pconns, cconns, ps = parallel.initialize(model)

    # These functions store the variables pconns, cconns, ps.
    def lnprob(p):
        pars = ThetaParam(grid=p[0:3], vz=p[3], vsini=p[4], logOmega=p[5], teff2=p[6], logOmega2=p[7])
        #Distribute the calculation to each process
        for ((spectrum_id, order_id), pconn) in pconns.items():
            pconn.send(("LNPROB", pars))

        #Collect the answer from each process
        lnps = np.empty((len(Starfish.data["orders"]),))
        for i, pconn in enumerate(pconns.values()):
            lnps[i] = pconn.recv()

        result = np.sum(lnps) # + lnprior
        #print("proposed:", p, result)
        print("{Teff: >8.1f} {logg: 8.3f} {feh: >8.3f}".format(Teff=pars.grid[0], logg=pars.grid[1], feh=pars.grid[2]), end=' ')
        print("{vz: >8.2f} {vsini: 8.2f} {logOm: >8.4f}".format(vz=pars.vz, vsini=pars.vsini, logOm=pars.logOmega), end=' ')
        print("{Teff2: >8.1f} {logOm2: >8.4f}".format(Teff2=pars.teff2, logOm2=pars.logOmega2), end=' ')
        print("{result: >12.1f}".format(result=result), end=' ')
        return result

    def query_lnprob():
        for ((spectrum_id, order_id), pconn) in pconns.items():
            pconn.send(("GET_LNPROB", None))

        #Collect the answer from each process
        lnps = np.empty((len(Starfish.data["orders"]),))
        for i, pconn in enumerate(pconns.values()):
            lnps[i] = pconn.recv()

        result = np.sum(lnps) # + lnprior
        #print("queried:", result)
        return result

    def acceptfn():
        print("{:->10}".format("Accept"))
        for ((spectrum_id, order_id), pconn) in pconns.items():
            pconn.send(("DECIDE", True))

    def rejectfn():
        print("{:-<10}".format("Reject"))
        for ((spectrum_id, order_id), pconn) in pconns.items():
            pconn.send(("DECIDE", False))

    from Starfish.samplers import StateSampler

    start = Starfish.config["Theta"]
    p0 = np.array(start["grid"] + [start["vz"], start["vsini"], start["logOmega"]])
    if 'teff2' in start.keys():
        p0 = np.append(p0, start["teff2"])
    if 'logOmega2' in start.keys():
        p0 = np.append(p0, start["logOmega2"])

    jump = Starfish.config["Theta_jump"]
    cov_on_diags = np.array(jump["grid"] + [jump["vz"], jump["vsini"], jump["logOmega"], jump["teff2"], jump["logOmega2"]])
    cov = np.diag(cov_on_diags**2)

    if args.use_cov:
        try:
            cov = np.load('opt_jump.npy')
            print("Found a local optimal jump matrix.")
        except FileNotFoundError:
            print("No optimal jump matrix found, using diagonal jump matrix.")

    p0_ball = sample_ball(p0, cov_on_diags)
    p0_ball.shape =(-1)

    sampler = StateSampler(lnprob, p0_ball, cov, query_lnprob=query_lnprob, acceptfn=acceptfn, rejectfn=rejectfn, debug=True, outdir=Starfish.routdir)

    start = time.time()
    p, lnprob, state = sampler.run_mcmc(p0_ball, N=args.samples, incremental_save=args.incremental_save)
    end = time.time()
    dtime = end - start
    sampler.write()
    print("{:-^50}".format("Final:"))
    print("Total time : {:8.2f} minutes".format(dtime/60.0))
    print("N samples  : {:8.1f}".format(args.samples))
    print("Time/sample: {:8.3f} samples / second".format(args.samples/dtime))
    print("Accept frac: {:8.1%}".format(sampler.acceptance_fraction))
    

    # Kill all of the orders
    for pconn in pconns.values():
        pconn.send(("FINISH", None))
        pconn.send(("DIE", None))

    # Join on everything and terminate
    for p in ps.values():
        p.join()
        p.terminate()

    import sys;sys.exit()
