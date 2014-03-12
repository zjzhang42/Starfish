from StellarSpectra.model import Model, SamplerStellarCheb
from StellarSpectra.spectrum import DataSpectrum
from StellarSpectra.grid_tools import TRES, HDF5Interface, InterpolationError
import numpy as np

myDataSpectrum = DataSpectrum.open("/home/ian/Grad/Research/Disks/StellarSpectra/tests/WASP14/WASP-14_2009-06-15_04h13m57s_cb.spec.flux", orders=np.array([21,22,23]))
myInstrument = TRES()
myHDF5Interface = HDF5Interface("/home/ian/Grad/Research/Disks/StellarSpectra/libraries/PHOENIX_submaster.hdf5")

myModel = Model(myDataSpectrum, myInstrument, myHDF5Interface,
                stellar_tuple=("temp", "logg", "Z", "vsini", "vz", "log_Omega"), Cov_tuple=("sig_amp", "amp", "l"))

def lnprob_Model(p):
    params = myModel.zip_stellar_p(p)
    try:
        myModel.update_Model(params)
        return myModel.evaluate()
    except InterpolationError:
        return -np.inf

def lnprob_Cheb(p):
    myModel.update_Cheb(p)
    return myModel.evaluate()

def lnprob_Cov(p):
    params = myModel.zip_Cov_p(p)
    myModel.update_Cov(params)
    return myModel.evaluate()

mySampler = SamplerStellarCheb(lnprob_Model, {"temp":(6000, 6200), "logg":(3.9, 4.2), "Z":(-0.6, -0.1),
                            "vsini":(3, 7), "vz":(12, 14), "log_Omega":(-19.9, -19.5)}, 100)
mySampler.burn_in()
mySampler.run(100)

mySampler.plot()