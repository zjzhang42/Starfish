#!/usr/bin/env python
#
#
## Author: ZJ Zhang (Jul. 23rd, 2017)
#
# Modification History:
#
# ZJ Zhang (Jul. 23rd, 2017)
# ZJ Zhang (Jul. 26th, 2017)    (ADD --- flux calibration in the *args choice)
#
#################################################

# ----
# Goal:
# convert input data file into the hdf5 format
# ----


## import modules
import Starfish
import numpy as np
from astropy.io import fits
import h5py
import os
from scipy.interpolate import interp1d
# ZJ's modules
from Catalog_Tool.server_param import local_dropbox
# ----


## argument choices
import argparse
parser = argparse.ArgumentParser(prog="gendata_hdf5.py", description="Convert a user-defined input data spectra into a HDF5 file.")
parser.add_argument("--flux_cal_H", type=float, default=-99., help="spectral flux calibration based on H-band photometry")
args = parser.parse_args()
# ----



#####################################
# Assistant Parameters and Functions
#####################################
# filter response curves for various photometric systems
Phot_Filters_path = local_dropbox + 'Laniakea/ZhangDirac/Observation_Calc_Tool/Phot_Filters/'


def spec_read(infile, wl_unit='um'):
    ''' extract wavelength, flux, and flux_err from the input file
        
        We assume the wavelength units are "um" in the input file.

        different methods would be applied for files with different extensions:
        - fits
        - ascii/dat/txt
        '''
    in_name, in_fmt = os.path.splitext(infile)
    if in_fmt==".fits":
        Spec_Load = fits.open(infile)
        Spec_Content = Spec_Load[0].data
        wls = Spec_Content[0]
        fls = Spec_Content[1]
        sigmas = Spec_Content[2]
    elif in_fmt==".ascii" or in_fmt==".dat" or in_fmt==".txt":
        Spec_Content = np.genfromtxt(infile, dtype=None)
        wls = Spec_Content[:,0]
        fls = Spec_Content[:,1]
        sigmas = Spec_Content[:,2]
    else:
        print("(ZJ-Style)Error: format of input files not found! It should be one of the follow: fits, ascii, dat ,txt.")
    return wls, fls, sigmas
# --


def Spec_Phot_filter(wavelength, flux, filter, band_step=0.0001):
    ''' given a wavelength, flux, and filter response curve, calculate the flux

        flux = int{flux * filter} / int{filter}

        unit of the wavelength should be: 1um

        '''
    ### load filter
    filter_Content = np.genfromtxt(filter)
    filter_wavelength = filter_Content[:,0] * 1.0e-4
    filter_response = filter_Content[:,1]
    ### check if the filter is covered by the spectrum (no extrapolation)
    if (filter_wavelength[0] > wavelength[-1]) or (filter_wavelength[-1] < wavelength[0]):
        # final flux
        filter_flux = np.nan
    else:
        ### dividing band
        band_array = np.arange(filter_wavelength[0], filter_wavelength[-1], band_step)
        # interpolate the spectra and filter response curve
        f_interp_spec = interp1d(wavelength, flux, assume_sorted=True, bounds_error=False)
        flux_array = f_interp_spec(band_array)
        f_interp_filter = interp1d(filter_wavelength, filter_response, assume_sorted=True, bounds_error=False)
        response_array = f_interp_filter(band_array)
        # spectra with filter
        flux_times_response_array = flux_array * response_array
        # integration
        int_spec_filter = ( 2.0 * np.nansum(flux_times_response_array[1:-1]) + flux_times_response_array[0] + flux_times_response_array[-1] ) * band_step / 2
        int_filter = ( 2.0 * np.nansum(response_array[1:-1]) + response_array[0] + response_array[-1] ) * band_step / 2
        # final flux
        filter_flux = int_spec_filter / int_filter
    return filter_flux
# --


def speccal_H_2MASS(wls, fls, sigmas, Hmag):
    ''' spectral flux calibration using H-band 2MASS photometry'''
    print("flux calibration based on H-band photometry.")
    # load H_2MASS filter
    filter_H_2MASS = Phot_Filters_path + 'filter_H_2MASS.dat'
    # load H_2MASS zero-point flux
    H_2MASS_zp_flux = 1.144e-10 # unit=ergss-1cm-2A-1
    # object's H_2MASS mag
    H_spec_flux = Spec_Phot_filter(wls, fls, filter_H_2MASS)
    H_mag_flux = H_2MASS_zp_flux * 10**(-0.4 * Hmag)
    # calibration scaling
    cal_scaling = H_mag_flux / H_spec_flux
    # return calibrated spectrum
    return wls, fls*cal_scaling, sigmas*cal_scaling
# ----



#############
# Main Body
#############
### 1. Obtain input file and output name
input_list = Starfish.data["infiles"]
hdf5_list = Starfish.data["files"]
# --

### 2. Genereate output hdf5 files
for (infile, outfile) in zip(input_list, hdf5_list):
    # load spectra
    wls, fls, sigmas = spec_read(infile)
    # check flux calibration
    if args.flux_cal_H!=-99:
        wls, fls, sigmas = speccal_H_2MASS(wls, fls, sigmas, args.flux_cal_H)
    # convert wls in units of Angstrom
    wls *= 1.0e4
    # save into a HDF5 file
    hd_file = h5py.File(outfile, "w")
    dset_wls = hd_file.create_dataset('wls', data=wls)
    dset_fls = hd_file.create_dataset('fls', data=fls)
    dset_sigmas = hd_file.create_dataset('sigmas', data=sigmas)
    hd_file.close()
# --

### 3. Funeral
print("HDF5 files generated.")
# ----
# END






