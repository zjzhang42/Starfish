import numpy as np
from numpy.fft import fft, ifft, fftfreq
from astropy.io import ascii,fits
import warnings

import multiprocessing as mp

from scipy.interpolate import InterpolatedUnivariateSpline, interp1d, UnivariateSpline
from scipy.integrate import trapz
from scipy.special import j1

import sys
import gc
import bz2
import h5py
from functools import partial
import itertools
from collections import OrderedDict

from StellarSpectra.model import Base1DSpectrum, LogLambdaSpectrum
import StellarSpectra.constants as C


grid_parameters = frozenset(("temp", "logg", "Z", "alpha")) #Allowed grid parameters
pp_parameters = frozenset(("vsini", "FWHM", "vz", "Av", "Omega")) #Allowed "post processing parameters"
all_parameters = grid_parameters | pp_parameters #the union of grid_parameters and pp_parameters
#Dictionary of allowed variables with default values
var_default = {"temp":5800, "logg":4.5, "Z":0.0, "alpha":0.0, "vsini":0.0, "FWHM": 0.0, "vz":0.0, "Av":0.0, "Omega":1.0}

def dict_to_tuple(mydict):
    '''Take a parameter dictionary and convert it to a tuple in the aggreed upon order.'''
    if "alpha" in mydict.keys():
        return (mydict["temp"], mydict['logg'], mydict['Z'], mydict['alpha'])
    else:
        return (mydict["temp"], mydict['logg'], mydict['Z'], var_default['alpha'])


class GridError(Exception):
    def __init__(self, msg):
        self.msg = msg

class InterpolationError(Exception):
    def __init__(self, msg):
        self.msg = msg

class Base:
    def __init__(self, parameters):
        '''Parameters are given as a dictionary, which is cycled through and instantiates self.parameters.
         If the parameter is not found allowed parameters, raise a warning and do not instantiate it. Further down the
         line, if a method needs a value for a parameter, it can query the default value, otherwise it can act as
         a short circuit if this parameter is not in the dictionary.'''
        self.parameters = set([])
        assert type(parameters) is dict, "Parameters must be a dictionary"
        for key, value in parameters.items():
            if key in var_default.keys():
                self.parameters[key] = value
            else:
                warnings.warn("{0} is not an allowed parameter, skipping".format(key), UserWarning)

    def __str__(self):
        prtstr = "".join(["{0} : {1:.2f} \n".format(key, value) for key,value in self.parameters.items()])
        return prtstr


class RawGridInterface:
    '''Takes in points, which is a dictionary with key values as parameters and values the sets of grid points.'''
    def __init__(self, name, points, air=True, wl_range=[3000,13000], base=None):
        self.name = name
        self.points = {}
        assert type(points) is dict, "points must be a dictionary."
        for key, value in points.items():
            if key in grid_parameters:
                self.points[key] = value
            else:
                warnings.warn("{0} is not an allowed parameter, skipping".format(key), UserWarning)

        self.air = air #read files in air wavelengths?
        self.wl_range = wl_range #values to truncate grid
        self.base = base

    def check_params(self, parameters):
        '''Checks to see if parameter dict is a subset of allowed parameters, otherwise raises an AssertionError,
        which can be chosen to be handled later..'''
        if not set(parameters.keys()) <= grid_parameters:
            raise GridError("{} not in allowable grid parameters {}".format(parameters.keys(), grid_parameters))

        for key,value in parameters.items():
            if value not in self.points[key]:
                raise GridError("{} not in the grid points {}".format(value, sorted(self.points[key])))

    def load_file(self, parameters, norm=True):
        '''Designed to be overwritten by an extended class'''
        self.check_params(parameters)
        #loads file
        #truncates to wl_range
        #returns a spectrum object defined by subclass
        #also includes metadata from the FITS header

class PHOENIXGridInterface(RawGridInterface):
    def __init__(self, air=True, norm=True, base="raw_grids/PHOENIX/"):
        super().__init__(name="PHOENIX",
        points={"temp":
      np.array([2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 4000, 4100, 4200,
      4300, 4400, 4500, 4600, 4700, 4800, 4900, 5000, 5100, 5200, 5300, 5400, 5500, 5600, 5700, 5800, 5900, 6000, 6100,
      6200, 6300, 6400, 6500, 6600, 6700, 6800, 6900, 7000, 7200, 7400, 7600, 7800, 8000, 8200, 8400, 8600, 8800, 9000,
      9200, 9400, 9600, 9800, 10000, 10200, 10400, 10600, 10800, 11000, 11200, 11400, 11600, 11800, 12000]),
        "logg":np.arange(0.0, 6.1, 0.5),
        "Z":np.arange(-1., 1.1, 0.5),
        "alpha":np.array([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8])},
        air=air, wl_range=[3000, 13000], base=base)

        self.norm = norm #Normalize to 1 solar luminosity?
        self.Z_dict = {-1: '-1.0', -0.5:'-0.5', 0.0: '-0.0', 0.5: '+0.5', 1: '+1.0'}
        self.alpha_dict = {-0.2:".Alpha=-0.20", 0.0: "", 0.2:".Alpha=+0.20", 0.4:".Alpha=+0.40", 0.6:".Alpha=+0.60",
                           0.8:".Alpha=+0.80"}

        #if air is true, convert the normally vacuum file to air wls.
        try:
            wl_file = fits.open(self.base + "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")
        except OSError:
            raise GridError("Wavelength file improperly specified.")

        w_full = wl_file[0].data
        wl_file.close()
        if self.air:
            self.wl_full = vacuum_to_air(w_full)
        else:
            self.wl_full = w_full

        self.ind = (self.wl_full >= self.wl_range[0]) & (self.wl_full <= self.wl_range[1])
        self.wl = self.wl_full[self.ind]
        self.rname = self.base + "Z{Z:}{alpha:}/lte{temp:0>5.0f}-{logg:.2f}{Z:}{alpha:}" \
                     ".PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"

    def load_file(self, parameters):
        super().load_file(parameters) #Check to make sure that the keys are allowed and that the values are in the grid

        str_parameters = parameters.copy()
        #Rewrite Z
        Z = parameters["Z"]
        str_parameters["Z"] = self.Z_dict[Z]

        #Rewrite alpha, allow alpha to be missing from parameters
        try:
            alpha = parameters["alpha"]
        except KeyError:
            alpha = 0.0
            parameters["alpha"] = alpha
        str_parameters["alpha"] = self.alpha_dict[alpha]

        fname = self.rname.format(**str_parameters)

        #Still need to check that file is in the grid, otherwise raise a GridError
        #Read all metadata in from the FITS header, and append to spectrum
        try:
            flux_file = fits.open(fname)
            f = flux_file[0].data
            hdr = flux_file[0].header
            flux_file.close()
        except OSError:
            raise GridError("{} is not on disk.".format(fname))

        #If we want to normalize the spectra, we must do it now since later we won't have the full EM range
        if self.norm:
            f *= 1e-8 #convert from erg/cm^2/s/cm to erg/cm^2/s/A
            F_bol = trapz(f, self.wl_full)
            f = f * (C.F_sun / F_bol) #bolometric luminosity is always 1 L_sun

        #Add temp, logg, Z, alpha, norm to the metadata
        header = parameters
        header["norm"] = self.norm
        #Keep only the relevant PHOENIX keywords, which start with PHX
        for key, value in hdr.items():
            if key[:3] == "PHX":
                header[key] = value

        return Base1DSpectrum(self.wl, f[self.ind], metadata=header, air=self.air)

class KuruczGridInterface:
    def __init__(self):
        super().__init__("Kurucz", "Kurucz/",
        temp_points = np.arange(3500, 9751, 250),
        logg_points = np.arange(1.0, 5.1, 0.5),
        Z_points = np.arange(-0.5, 0.6, 0.5))

        self.Z_dict = {-0.5:"m05", 0.0:"p00", 0.5:"p05"}
        self.wl_full = np.load("wave_grids/kurucz_raw.npy")
        self.rname = None

    def load_file(self, temp, logg, Z):
        '''Includes an interface that can map a queried number to the actual string'''
        super().load_file(temp, logg, Z)

class BTSettlGridInterface:
    def __init__(self):
        pass

class HDF5GridCreator:
    '''Take a GridInterface, load all spectra in specified ranges (default all), and then stuff them into an HDF5
    file with the proper attributes. '''
    def __init__(self, GridInterface, filename, wldict, ranges={"temp":(0,np.inf),
                 "logg":(-np.inf,np.inf), "Z":(-np.inf, np.inf), "alpha":(-np.inf, np.inf)},
                 parallel=True, nprocesses = 4, chunksize=1):
        self.GridInterface = GridInterface
        self.filename = filename #only store the name to the HDF5 file, because the object cannot be parallelized
        self.flux_name = "t{temp:.0f}g{logg:.1f}z{Z:.1f}a{alpha:.1f}"
        self.parallel = parallel
        self.nprocesses = nprocesses
        self.chunksize = chunksize

        #Discern between HDF5GridCreator points and GridInterface points using ranges
        self.points = {}
        for key, value in ranges.items():
            valid_points  = self.GridInterface.points[key]
            low,high = value
            ind = (valid_points >= low) & (valid_points <= high)
            self.points[key] = valid_points[ind]

        #wldict is the output from create_log_lam_grid, containing CRVAL1, CDELT1, etc...
        wldict = wldict.copy()
        self.wl = wldict.pop("wl")
        self.wl_params = wldict

        with h5py.File(self.filename, "w") as hdf5:
            hdf5.attrs["grid_name"] = GridInterface.name
            hdf5.flux_group = hdf5.create_group("flux")
            hdf5.flux_group.attrs["unit"] = "erg/cm^2/s/A"
            self.create_wl(hdf5)

        #The HDF5 master grid will always have alpha in the name, regardless of whether GridIterface uses it.

    def create_wl(self, hdf5):
        wl_dset = hdf5.create_dataset("wl", (len(self.wl),), dtype="f8", compression='gzip', compression_opts=9)
        wl_dset[:] = self.wl
        for key, value in self.wl_params.items():
            wl_dset.attrs[key] = value
            wl_dset.attrs["air"] = self.GridInterface.air

    #@profile
    def process_flux(self, parameters):
        #'''Assumes that it's going to get parameters (temp, logg, Z, alpha), regardless of whether
        #the GridInterface actually has alpha or not.'''
        assert len(parameters.keys()) == 4, "Must pass dictionary with keys (temp, logg, Z, alpha)"
        print("Processing", parameters)
        try:
            spec = self.GridInterface.load_file(parameters)
            spec.resample_to_grid(self.wl)
            sys.stdout.flush()
            return (parameters,spec)

        except GridError as e:
            print("No file with parameters {}. GridError: {}".format(parameters, e))
            sys.stdout.flush()
            return (None,None)

    #@profile
    def process_grid(self):
        #Take all parameter permutations in self.points and create a list
        param_list = [] #list of parameter dictionaries
        keys,values = self.points.keys(),self.points.values()

        #use itertools.product to create permutations of all possible values
        for i in itertools.product(*values):
            param_list.append(dict(zip(keys,i)))

        if self.parallel:
            pool = mp.Pool(self.nprocesses)
            M = lambda x,y : pool.imap_unordered(x, y, chunksize=self.chunksize)
        else:
            M = map

        for parameters, spec in M(self.process_flux, param_list): #lazy map
            if parameters is None:
                continue
            with h5py.File(self.filename, "r+") as hdf5:
                flux = hdf5["flux"].create_dataset(self.flux_name.format(**parameters), shape=(len(spec.fl),),
                                                      dtype="f", compression='gzip', compression_opts=9)
                flux[:] = spec.fl

                #Store header keywords as attributes in HDF5 file
                for key,value in spec.metadata.items():
                    flux.attrs[key] = value

class HDF5Interface:
    '''Connect to an HDF5 file that stores spectra.'''
    def __init__(self, filename, mode=None):
        self.filename = filename
        self.flux_name = "t{temp:.0f}g{logg:.1f}z{Z:.1f}a{alpha:.1f}"

        with h5py.File(self.filename, "r") as hdf5:
            self.name = hdf5.attrs["grid_name"]
            self.wl = hdf5["wl"][:]
            self.wl_header = dict(hdf5["wl"].attrs.items())

            grid_points = []
            for key in hdf5["flux"].keys():
                #assemble all temp, logg, Z, alpha keywords into a giant list
                hdr = hdf5['flux'][key].attrs
                grid_points.append({k: hdr[k] for k in grid_parameters})
            self.list_grid_points = grid_points

        #determine the bounding regions of the grid by sorting the grid_points
        temp, logg, Z, alpha = [],[],[],[]
        for param in self.list_grid_points:
            temp.append(param['temp'])
            logg.append(param['logg'])
            Z.append(param['Z'])
            alpha.append(param['alpha'])

        self.bounds = {"temp": (min(temp),max(temp)), "logg": (min(logg), max(logg)), "Z": (min(Z), max(Z)),
        "alpha":(min(alpha),max(alpha))}
        self.points = {"temp": np.unique(temp), "logg": np.unique(logg), "Z": np.unique(Z), "alpha": np.unique(alpha)}

    def load_file(self, parameters):
        '''Loads a file and returns it as a LogLambdaSpectrum. (Does it have to assume something about the keywords?
        Perhaps there is a EXFlag or disp type present in the HDF5 attributes.'''
        key = self.flux_name.format(**parameters)
        with h5py.File(self.filename, "r") as hdf5:
            fl = hdf5['flux'][key][:]
            hdr = dict(hdf5['flux'][key].attrs)

        hdr.update(self.wl_header) #add the flux metadata to the wl data

        return LogLambdaSpectrum(self.wl, fl, metadata=hdr)


    def write_to_FITS(self):
        pass

class IndexInterpolator:
    '''Index interpolator should return fractional index between two points (0 - 1) and the low and high values. For
    example, given "temp":6010, the index interpolator should return (0.1, 6000, 6100). Then we create a structure
     that has {"temp": (0.1, 6000, 6100), "logg": (0.7, 3.5, 4.0), etc.... }'''
    def __init__(self, parameter_list):
        self.parameter_list = np.unique(parameter_list)
        self.index_interpolator = interp1d(self.parameter_list, np.arange(len(self.parameter_list)), kind='linear')
        pass

    def __call__(self, value):
        try:
            index = self.index_interpolator(value)
        except ValueError as e:
            raise InterpolationError("Requested value {} is out of bounds. {}".format(value, e))
        high = np.ceil(index)
        low = np.floor(index)
        frac_index = index - low
        return ((self.parameter_list[low], self.parameter_list[high]), ((1 - frac_index), frac_index))

class Interpolator:
    '''Naturally interfaces to the HDF5Grid in its own way, built for model evaluation.'''

    #Takes an HDF5Interface object
    def __init__(self, interface):
        self.interface = interface

        #If alpha only includes one value, then do trilinear interpolation
        (alow, ahigh) = self.interface.bounds['alpha']
        if alow == ahigh:
            self.interpolate = self.trilinear
            self.parameters = grid_parameters - set("alpha")
        else:
            self.interpolate = self.quadlinear
            self.parameters = grid_parameters

        self.setup_index_interpolators()
        self.cache = OrderedDict([])

    def __call__(self, parameters):
        return self.interpolate(parameters)


    def setup_index_interpolators(self):
        #create an interpolator between grid points indices. Given a temp, produce fractional index between two points
        self.index_interpolators = {key:IndexInterpolator(self.interface.points[key]) for key in self.parameters}

        lenF = len(self.interface.wl)
        self.fluxes = np.empty((2**len(self.parameters), lenF))



    def trilinear(self, parameters):
        '''Return a function that will take temp, logg, Z as arguments and do trilinear interpolation on it.'''
        #Because we have three parameters, store both the high and low bordering value

        indexes = {key:self.index_interpolators[key](value) for key,value in parameters.items()}
        print(indexes)


    def quadlinear(self, parameters):
        edges = {key:self.index_interpolators[key](value) for key,value in parameters.items()}
        names = [key for key in edges.keys()]
        params = [edges[key][0] for key in names]
        weights = [edges[key][1] for key in names]

        param_combos = itertools.product(*params)
        weight_combos = itertools.product(*weights)

        parameter_list = [dict(zip(names, param)) for param in param_combos]
        key_list = [self.interface.flux_name.format(**param) for param in parameter_list]
        weight_list = [np.prod(weight) for weight in weight_combos]
        #For each spectrum, want to extract a {"temp":5000, "logg":4.5, "Z":0.0, "alpha":0.0} and weight= 0.1 * 0.4 * .05 * 0.1

        assert np.allclose(np.sum(weight_list), np.array(1.0)), "Sum of weights must equal 1, {}".format(np.sum(weight_list))

        #Assemble flux vector from cache
        for i,param in enumerate(parameter_list):
            key = key_list[i]
            if key not in self.cache.keys():
                self.cache[key] = self.interface.load_file(param).fl
            self.fluxes[i,:] = self.cache[key]*weight_list[i]

        return np.sum(self.fluxes, axis=0)



class MasterToInstrumentProcessor:
    #Take an Interpolator, an instrument object, and some grid points, and create a new HDF5 file processed
    # to that instrument.
    #Will also need to do vsini ranges. If vsini = 0, you can skip.
    #Might also need to interpolate.
    #Will need to update keywords for each flux object
    #Loads files
    def __init__(self, hdf5interface, instrument, points, chunksize=20):
        #points is a dictionary with which values to spit out
        self.instrument = instrument
        self.hdf5 = hdf5interface
        self.points = points


    def process_all(self):
        pass

    def process_spectrum(self, var_combos):
        '''This depends on what you want to do (resample, convolve, etc) and must be defined in a subclass.'''
        spec = self.grid.load_file(var_combos)


    def resample_and_convolve(f, wg_raw, wg_fine, wg_coarse, wg_fine_d=0.35, sigma=2.89):
        '''Take a full-resolution PHOENIX model spectrum `f`, with raw spacing wg_raw, resample it to wg_fine
        (done because the original grid is not log-linear spaced), instrumentally broaden it in the Fourier domain,
        then resample it to wg_coarse. sigma in km/s.'''

        #resample PHOENIX to 0.35km/s spaced grid using InterpolatedUnivariateSpline. First check to make sure there
        #are no duplicates and the wavelength is increasing, otherwise the spline will fail and return NaN.
        wl_sorted, ind = np.unique(wg_raw, return_index=True)
        fl_sorted = f[ind]
        interp_fine = InterpolatedUnivariateSpline(wl_sorted, fl_sorted)
        f_grid = interp_fine(wg_fine)

        #Fourier Transform
        out = fft(f_grid)
        #The frequencies (cycles/km) corresponding to each point
        freqs = fftfreq(len(f_grid), d=wg_fine_d)

        #Instrumentally broaden the spectrum by multiplying with a Gaussian in Fourier space (corresponding to FWHM 6.8km/s)
        taper = np.exp(-2 * (np.pi ** 2) * (sigma ** 2) * (freqs ** 2))
        tout = out * taper

        #Take the broadened spectrum back to wavelength space
        f_grid6 = ifft(tout)
        #print("Total of imaginary components", np.sum(np.abs(np.imag(f_grid6))))

        #Resample the broadened spectrum to a uniform coarse grid
        interp_coarse = InterpolatedUnivariateSpline(wg_fine, np.abs(f_grid6))
        f_coarse = interp_coarse(wg_coarse)

        del interp_fine
        del interp_coarse
        gc.collect() #necessary to prevent memory leak!

        return f_coarse



class Instrument:
    def __init__(self, name, FWHM, wl_range=(-np.inf, np.inf), oversampling=3.5):
        self.name = name
        self.FWHM = FWHM #km/s
        self.oversampling = oversampling
        self.wl_range = wl_range

    def __str__(self):
        return "Instrument Name: {}, FWHM: {:.1f}, oversampling: {}, wl_range: {}".format(self.name, self.FWHM,
                                                                              self.oversampling, self.wl_range)

class TRES(Instrument):
    def __init__(self):
        super().__init__(name="TRES", FWHM=6.8, wl_range=(-np.inf, np.inf))
        #sets the FWHM and wl_range

class Reticon(Instrument):
    def __init__(self):
        super().__init__(name="Reticon", FWHM=8.5, wl_range=(5150,5250))

class KPNO(Instrument):
    def __init__(self):
        super().__init__(name="KPNO", FWHM=14.4)



#wl_file = fits.open("raw_grids/PHOENIX/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")
#w_full = wl_file[0].data
#wl_file.close()
#ind = (w_full > 3000.) & (w_full < 13000.) #this corresponds to some extra space around the
# shortest U and longest z band

#global w
#w = w_full[ind]
#len_p = len(w)

#wave_grid_raw_PHOENIX = np.load("wave_grids/PHOENIX_raw_trim_air.npy")
#wave_grid_fine = np.load('wave_grids/PHOENIX_0.35kms_air.npy')
#wave_grid_coarse = np.load('wave_grids/PHOENIX_2kms_air.npy')
#wave_grid_kurucz_raw = np.load("wave_grids/kurucz_raw.npy")
#wave_grid_2kms_kurucz = np.load("wave_grids/kurucz_2kms_air.npy") #same wl as PHOENIX_2kms_air, but trimmed

grids = {"kurucz": {'T_points': np.arange(3500, 9751, 250),
                    'logg_points': np.arange(1.0, 5.1, 0.5), 'Z_points': ["m05", "p00", "p05"]},
         'BTSettl': {'T_points': np.arange(3000, 7001, 100), 'logg_points': np.arange(2.5, 5.6, 0.5),
                     'Z_points': ['-0.5a+0.2', '-0.0a+0.0', '+0.5a+0.0']}}


def create_wave_grid(v=1., start=3700., end=10000):
    '''Returns a grid evenly spaced in velocity'''
    size = 9000000 #this number just has to be bigger than the final array
    lam_grid = np.zeros((size,))
    i = 0
    lam_grid[i] = start
    vel = np.sqrt((C.c_kms + v) / (C.c_kms - v))
    while (lam_grid[i] < end) and (i < size - 1):
        lam_new = lam_grid[i] * vel
        i += 1
        lam_grid[i] = lam_new
    return lam_grid[np.nonzero(lam_grid)][:-1]


def create_fine_and_coarse_wave_grid():
    wave_grid_2kms_PHOENIX = create_wave_grid(2., start=3050., end=11322.2) #chosen for 3 * 2**16 = 196608
    wave_grid_fine = create_wave_grid(0.35, start=3050., end=12089.65) # chosen for 9 * 2 **17 = 1179648

    np.save('wave_grid_2kms.npy', wave_grid_2kms_PHOENIX)
    np.save('wave_grid_0.35kms.npy', wave_grid_fine)
    print(len(wave_grid_2kms_PHOENIX))
    print(len(wave_grid_fine))


def create_coarse_wave_grid_kurucz():
    start = 5050.00679905
    end = 5359.99761468
    wave_grid_2kms_kurucz = create_wave_grid(2.0, start + 1, 5333.70 + 1)
    #8192 = 2**13
    print(len(wave_grid_2kms_kurucz))
    np.save('wave_grid_2kms_kurucz.npy', wave_grid_2kms_kurucz)


@np.vectorize
def vacuum_to_air(wl):
    '''CA Prieto recommends this as more accurate than the IAU standard. Ciddor 1996.'''
    sigma = (1e4 / wl) ** 2
    f = 1.0 + 0.05792105 / (238.0185 - sigma) + 0.00167917 / (57.362 - sigma)
    return wl / f

def calculate_n(wl):
    sigma = (1e4 / wl) ** 2
    f = 1.0 + 0.05792105 / (238.0185 - sigma) + 0.00167917 / (57.362 - sigma)
    new_wl = wl / f
    n = wl/new_wl
    print(n)


@np.vectorize
def vacuum_to_air_SLOAN(wl):
    '''Takes wavelength in angstroms and maps to wl in air.
    from SLOAN website
     AIR = VAC / (1.0 + 2.735182E-4 + 131.4182 / VAC^2 + 2.76249E8 / VAC^4)'''
    air = wl / (1.0 + 2.735182E-4 + 131.4182 / wl ** 2 + 2.76249E8 / wl ** 4)
    return air


@np.vectorize
def air_to_vacuum(wl):
    sigma = 1e4 / wl
    vac = wl + wl * (6.4328e-5 + 2.94981e-2 / (146 - sigma ** 2) + 2.5540e-4 / (41 - sigma ** 2))
    return vac


def get_wl_kurucz():
    '''The Kurucz grid is already convolved with a FWHM=6.8km/s Gaussian. WL is log-linear spaced.'''
    sample_file = "Kurucz/t06000g45m05v000.fits"
    flux_file = fits.open(sample_file)
    hdr = flux_file[0].header
    num = len(flux_file[0].data)
    p = np.arange(num)
    w1 = hdr['CRVAL1']
    dw = hdr['CDELT1']
    wl = 10 ** (w1 + dw * p)
    return wl


@np.vectorize
def idl_float(idl):
    '''Take an idl number and convert it to scientific notation.'''
    #replace 'D' with 'E', convert to float
    return np.float(idl.replace("D", "E"))


def load_BTSettl(temp, logg, Z, norm=False, trunc=False, air=False):
    rname = "BT-Settl/CIFIST2011/M{Z:}/lte{temp:0>3.0f}-{logg:.1f}{Z:}.BT-Settl.spec.7.bz2".format(temp=0.01 * temp,
                                                                                                   logg=logg, Z=Z)
    file = bz2.BZ2File(rname, 'r')

    lines = file.readlines()
    strlines = [line.decode('utf-8') for line in lines]
    file.close()

    data = ascii.read(strlines, col_starts=[0, 13], col_ends=[12, 25], Reader=ascii.FixedWidthNoHeader)
    wl = data['col1']
    fl_str = data['col2']

    fl = idl_float(fl_str) #convert because of "D" exponent, unreadable in Python
    fl = 10 ** (fl - 8.) #now in ergs/cm^2/s/A

    if norm:
        F_bol = trapz(fl, wl)
        fl = fl * (C.F_sun / F_bol)
        #this also means that the bolometric luminosity is always 1 L_sun

    if trunc:
        #truncate to only the wl of interest
        ind = (wl > 3000) & (wl < 13000)
        wl = wl[ind]
        fl = fl[ind]

    if air:
        wl = vacuum_to_air(wl)

    return [wl, fl]


def load_flux_full(temp, logg, Z, alpha=None, norm=False, vsini=0, grid="PHOENIX"):
    '''Load a raw PHOENIX or kurucz spectrum based upon temp, logg, and Z. Normalize to C.F_sun if desired.'''

    if grid == "PHOENIX":
        if alpha is not None:
            rname = "raw_grids/PHOENIX/Z{Z:}{alpha:}/lte{temp:0>5.0f}-{logg:.2f}{Z:}{alpha:}" \
                ".PHOENIX-ACES-AGSS-COND-2011-HiRes.fits".format(Z=Z, temp=temp, logg=logg, alpha=alpha)
        else:
            rname = "raw_grids/PHOENIX/Z{Z:}/lte{temp:0>5.0f}-{logg:.2f}{Z:}" \
                    ".PHOENIX-ACES-AGSS-COND-2011-HiRes.fits".format(Z=Z, temp=temp, logg=logg)
    elif grid == "kurucz":
        rname = "raw_grids/Kurucz/TRES/t{temp:0>5.0f}g{logg:.0f}{Z:}v{vsini:0>3.0f}.fits".format(temp=temp,
                                                                                       logg=10 * logg, Z=Z, vsini=vsini)
    else:
        print("No grid %s" % (grid))
        return 1

    flux_file = fits.open(rname)
    f = flux_file[0].data

    if norm:
        f *= 1e-8 #convert from erg/cm^2/s/cm to erg/cm^2/s/A
        F_bol = trapz(f, w_full)
        f = f * (C.F_sun / F_bol)
        #this also means that the bolometric luminosity is always 1 L_sun
    if grid == "kurucz":
        f *= C.c_ang / wave_grid_kurucz_raw ** 2 #Convert from f_nu to f_lambda

    flux_file.close()
    #print("Loaded " + rname)
    return f


@np.vectorize
def gauss_taper(s, sigma=2.89):
    '''This is the FT of a gaussian w/ this sigma. Sigma in km/s'''
    return np.exp(-2 * np.pi ** 2 * sigma ** 2 * s ** 2)


def resample_and_convolve(f, wg_raw, wg_fine, wg_coarse, wg_fine_d=0.35, sigma=2.89):
    '''Take a full-resolution PHOENIX model spectrum `f`, with raw spacing wg_raw, resample it to wg_fine
    (done because the original grid is not log-linear spaced), instrumentally broaden it in the Fourier domain,
    then resample it to wg_coarse. sigma in km/s.'''

    #resample PHOENIX to 0.35km/s spaced grid using InterpolatedUnivariateSpline. First check to make sure there
    #are no duplicates and the wavelength is increasing, otherwise the spline will fail and return NaN.
    wl_sorted, ind = np.unique(wg_raw, return_index=True)
    fl_sorted = f[ind]
    interp_fine = InterpolatedUnivariateSpline(wl_sorted, fl_sorted)
    f_grid = interp_fine(wg_fine)

    #Fourier Transform
    out = fft(f_grid)
    #The frequencies (cycles/km) corresponding to each point
    freqs = fftfreq(len(f_grid), d=wg_fine_d)

    #Instrumentally broaden the spectrum by multiplying with a Gaussian in Fourier space (corresponding to FWHM 6.8km/s)
    taper = np.exp(-2 * (np.pi ** 2) * (sigma ** 2) * (freqs ** 2))
    tout = out * taper

    #Take the broadened spectrum back to wavelength space
    f_grid6 = ifft(tout)
    #print("Total of imaginary components", np.sum(np.abs(np.imag(f_grid6))))

    #Resample the broadened spectrum to a uniform coarse grid
    interp_coarse = InterpolatedUnivariateSpline(wg_fine, np.abs(f_grid6))
    f_coarse = interp_coarse(wg_coarse)

    del interp_fine
    del interp_coarse
    gc.collect() #necessary to prevent memory leak!

    return f_coarse


def resample(f, wg_input, wg_output):
    '''Take a TRES spectrum and resample it to 2km/s binning. For the kurucz grid.'''

    # check to make sure there are no duplicates and the wavelength is increasing,
    # otherwise the spline will fail and return NaN.
    wl_sorted, ind = np.unique(wg_input, return_index=True)
    fl_sorted = f[ind]

    interp = InterpolatedUnivariateSpline(wl_sorted, fl_sorted)
    f_output = interp(wg_output)
    del interp
    gc.collect()
    return f_output


def process_spectrum_PHOENIX(pars, convolve=True):
    temp, logg, Z, alpha = pars
    try:
        f = load_flux_full(temp, logg, Z, alpha, norm=True, grid="PHOENIX")[ind]
        if convolve:
            flux = resample_and_convolve(f, wave_grid_raw_PHOENIX, wave_grid_fine, wave_grid_coarse)
        else:
            flux = resample(f, wave_grid_raw_PHOENIX, wave_grid_fine)
        print("PROCESSED: %s, %s, %s %s" % (temp, logg, Z, alpha))
    except OSError:
        print("FAILED: %s, %s, %s, %s" % (temp, logg, Z, alpha))
        flux = np.nan
    return flux


def process_spectrum_kurucz(pars):
    temp, logg, Z = pars
    try:
        f = load_flux_full(temp, logg, Z, norm=False, grid="kurucz")
        flux = resample(f, wave_grid_kurucz_raw, wave_grid_2kms_kurucz)
    except OSError:
        print("%s, %s, %s does not exist!" % (temp, logg, Z))
        flux = np.nan
    return flux


def process_spectrum_BTSettl(pars, convolve=True):
    temp, logg, Z = pars
    try:
        wl, f = load_BTSettl(temp, logg, Z, norm=True, trunc=True, air=True)
        if convolve:
            flux = resample_and_convolve(f, wl, wave_grid_fine, wave_grid_coarse)
        else:
            flux = resample(f, wl, wave_grid_fine)
        print("PROCESSED: %s, %s, %s" % (temp, logg, Z))
    except FileNotFoundError: #on Python2 gives IOError, Python3 use FileNotFoundError
        print("FAILED: %s, %s, %s" % (temp, logg, Z))
        flux = np.nan
    return flux


process_routines = {"PHOENIX": process_spectrum_PHOENIX, "kurucz": process_spectrum_kurucz,
                    "BTSettl": process_spectrum_BTSettl}


def create_grid_parallel(ncores, hdf5_filename, grid_name, convolve=True):
    '''create an hdf5 file of the stellar grid. Go through each T point, if the corresponding logg exists,
    write it. If not, write nan. Each spectrum is normalized to the bolometric flux at the surface of the Sun.'''
    f = h5py.File(hdf5_filename, "w")

    #Grid parameters
    grid = grids[grid_name]
    T_points = grid['T_points']
    logg_points = grid['logg_points']
    Z_points = grid['Z_points']
    alpha_points = grid['alpha_points']

    if grid_name == 'kurucz':
        process_spectrum = process_spectrum_kurucz
        wave_grid_out = wave_grid_2kms_kurucz
    elif (grid_name == 'PHOENIX') or (grid_name == "BTSettl"):
        process_spectrum = {"PHOENIX": partial(process_spectrum_PHOENIX, convolve=convolve),
                            "BTSettl": partial(process_spectrum_BTSettl, convolve=convolve)}[grid_name]
        if convolve:
            wave_grid_out = np.load("wave_grids/PHOENIX_2kms_air.npy")
        else:
            wave_grid_out = np.load("wave_grids/PHOENIX_0.35kms_air.npy")
    else:
        print("No grid %s" % grid_name)
        return 1

    shape = (len(T_points), len(logg_points), len(Z_points), len(alpha_points), len(wave_grid_out))
    dset = f.create_dataset("LIB", shape, dtype="f", compression='gzip', compression_opts=9)

    # A thread pool of P processes
    pool = mp.Pool(ncores)

    index_combos = []
    var_combos = []
    for t, temp in enumerate(T_points):
        for l, logg in enumerate(logg_points):
            for z, Z in enumerate(Z_points):
                for a, A in enumerate(alpha_points):
                    index_combos.append([t, l, z, a])
                    var_combos.append([temp, logg, Z, A])

    spec_gen = pool.imap(process_spectrum, var_combos, chunksize=20)

    for i, spec in enumerate(spec_gen):
        t, l, z, a = index_combos[i]
        dset[t, l, z, a, :] = spec
        print("Writing ", var_combos[i], "to HDF5")

    f.close()


@np.vectorize
def v(ls, lo):
    return C.c_kms * (lo ** 2 - ls ** 2) / (ls ** 2 + lo ** 2)

def create_FITS_wavegrid(wl_start, wl_end, vel_spacing):
    '''Taking the desired wavelengths, output CRVAL1, CDELT1, NAXIS1 and the actual wavelength array.
    vel_spacing in km/s, wavelengths in angstroms.'''
    CRVAL1 = np.log10(wl_start)
    CDELT1 = np.log10(vel_spacing/C.c_kms + 1)
    NAXIS1 = int(np.ceil((np.log10(wl_end) - CRVAL1)/CDELT1)) + 1
    p = np.arange(NAXIS1)
    wl = 10 ** (CRVAL1 + CDELT1 * p)
    return [wl, CRVAL1, CDELT1, NAXIS1]


def create_fits(filename, fl, CRVAL1, CDELT1, dict=None):
    '''Assumes that wl is already log lambda spaced'''

    hdu = fits.PrimaryHDU(fl)
    head = hdu.header
    head["DISPTYPE"] = 'log lambda'
    head["DISPUNIT"] = 'log angstroms'
    head["CRPIX1"] = 1.

    head["CRVAL1"] = CRVAL1
    head["CDELT1"] = CDELT1
    head["DC-FLAG"] = 1

    if dict is not None:
        for key, value in dict.items():
            head[key] = value

    hdu.writeto(filename)

def process_PHOENIX_to_grid(temp, logg, Z, alpha, vsini, instFWHM, air=True):
    #Create the wave_grid
    out_grid, CRVAL1, CDELT1, NAXIS = create_FITS_wavegrid(6200, 6700, 2.)

    #Load the raw file
    flux = load_flux_full(temp, logg, Z, alpha, norm=True, grid="PHOENIX")[ind]

    global w
    if air:
        w_new = vacuum_to_air(w)

    #resample to equally spaced v grid, convolve w/ instrumental profile,
    f_coarse = resample_and_convolve(flux, w_new, wave_grid_fine, out_grid, wg_fine_d=0.35, sigma=instFWHM/2.35)

    ss = np.fft.fftfreq(len(out_grid), d=2.) #2km/s spacing for wave_grid
    ss[0] = 0.01 #junk so we don't get a divide by zero error
    ub = 2. * np.pi * vsini * ss
    sb = j1(ub) / ub - 3 * np.cos(ub) / (2 * ub ** 2) + 3. * np.sin(ub) / (2 * ub ** 3)
    #set zeroth frequency to 1 separately (DC term)
    sb[0] = 1.
    FF = fft(f_coarse)
    FF *= sb

    #do ifft
    f_lam = np.abs(ifft(FF))

    #convert to f_nu
    f_nu = out_grid**2/C.c_ang * f_lam

    filename = "t{temp:0>5.0f}g{logg:.0f}p00v{vsini:0>3.0f}.fits".format(temp=temp,
    logg=10 * logg, vsini=vsini)

    create_fits(filename, f_nu, CRVAL1, CDELT1, {"BUNIT": ('erg/s/cm^2/Hz', 'Unit of flux'),
                                                                "AUTHOR": "Ian Czekala",
                                                                "COMMENT" : "Adapted from PHOENIX"})

#@profile
def main():
    print("Starting main")
    mygrid = PHOENIXGridInterface(air=True, norm=True)


    spec = mygrid.load_file({"temp":5000, "logg":3.5, "Z":0.0,"alpha":0.0})
    #spec.save("test_spec.npy")


    wldict = spec.calculate_log_lam_grid()


    out_path = "PHOENIX_master.hdf5"

    HDF5Creator = HDF5GridCreator(mygrid, filename=out_path, wldict=wldict, nprocesses=4, chunksize=1, parallel=True,
                                ranges={"temp":(5000, 6000), "logg":(4.5,5.5), "Z":(0.0,0.0), "alpha":(0.0,0.0)})

    HDF5Creator.process_grid()


    pass


if __name__ == "__main__":
    main()
