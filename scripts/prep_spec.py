#!/usr/bin/env python
#
#
## Author: ZJ Zhang
#
# Modification History:
#
# ZJ Zhang (Feb 24th, 2018)
#
#################################################

import Starfish
import os
from scipy.interpolate import interp1d
from astropy.io import fits
import astropy.units as u
import h5py

import numpy as np
np.seterr(divide='ignore', invalid='ignore')

import argparse
parser = argparse.ArgumentParser(prog="prep_spec.py", description="Convert a raw spectra data file into a HDF5 format, with a required flux calibration based on user-defined J/H/K magnitudes.")
parser.add_argument("--create", action="store_true", help="Create the HDF5 spectra file")
parser.add_argument("--u_wls", type=str, default='micron', help="units of wavelength; chosen from [micron], [Angstrom]")
parser.add_argument("--u_fls", type=str, default='erg/s/cm2/A', help="units of flux and flux_err; chosen from [erg/s/cm2/A], [W/m2/m]")
parser.add_argument("--phot", type=str, default=None, help="photometric system for provided magnitudes")
parser.add_argument("--Jcal", type=float, default=None, help="flux calibration based on J-band photometry")
parser.add_argument("--Hcal", type=float, default=None, help="flux calibration based on H-band photometry")
parser.add_argument("--Kcal", type=float, default=None, help="flux calibration based on K-band photometry")
parser.add_argument("--calbase", type=str, default=None, help="which band should the calibrated spectra be based on")
args = parser.parse_args()


#####################################
# Assistant Parameters and Functions
#####################################
### filter response curves
filter_dir = os.path.expandvars('$Starfish/data/filter_resp/')
# ----
### calibration magnitudes
cal_mag = {"J": args.Jcal,
           "H": args.Hcal,
           "K": args.Kcal}
cal_key = list(cal_mag.keys())
# ----
### flux zero points (erg/cm2/s/A) of each band (from http://ssc.spitzer.caltech.edu/warmmission/propkit/pet/magtojy/ref.html)
flux_zp = {"J_2MASS": 3.14e-10,
           "H_2MASS": 1.11e-10,
           "K_2MASS": 4.29e-11,
           "J_MKO":   3.07e-10,
           "H_MKO":   1.12e-10,
           "K_MKO":   4.07e-11}
# ----


def spec_read(infile):
    ''' extract wavelength, flux, and flux_err from the input file

        different methods would be applied for files with different extensions:
        - fits
        - ascii/dat/txt
        '''
    in_name, in_fmt = os.path.splitext(infile)
    if in_fmt==".fits":
        Spec_Load = fits.open(infile)
        Spec_Content = Spec_Load[0].data
        try:
            wls = Spec_Content[0]
            fls = Spec_Content[1]
            sigmas = Spec_Content[2]
        except TypeError:
            Spec_Content = Spec_Load[1].data
            wls = Spec_Content['wls']
            fls = Spec_Content['fls']
            sigmas = Spec_Content['sigmas']
    elif in_fmt==".ascii" or in_fmt==".dat" or in_fmt==".txt":
        Spec_Content = np.genfromtxt(infile, dtype=None)
        wls = Spec_Content[:,0]
        fls = Spec_Content[:,1]
        sigmas = Spec_Content[:,2]
    else:
        print("warning: format of input files not found! Valid formats include: fits, ascii, dat ,txt.")
        wls, fls, sigmas = None, None, None
    return wls, fls, sigmas
# ----


def spec_cal(wls, fls, sigmas, phot_sys, cal_key, cal_mag, calpath):
    ''' calibrate the spectrum based on each given magnitude with a given photometric system
        and save the calibrated spectrum in to a file

        the output spectra file should have the following units:
        - wavelength: um
        - flux: erg/s/cm2/A
        - flux err: erg/s/cm2/A
        '''
    print("flux calibration")
    for band in cal_key:
        if cal_mag[band] is None:
            print("please provide magnitudes for %s_%s band."%(band, phot_sys))
        else:
            ## load filter response
            fil_content = np.genfromtxt(filter_dir + 'filter_%s_%s.dat'%(band, phot_sys), comments='#')
            fil_wls = fil_content[:,0]
            fil_resp = fil_content[:,1]
            if (fil_wls[-1] < wls[0]) or (fil_wls[0] > wls[-1]):
                print("warning: spectra not covered by the %s_%s-band filter."%(band, phot_sys))
                fls_cal = fls * np.nan
                sigmas_cal = sigmas * np.nan
            else:
                # interpolator of spectra and filter
                spec_interp = interp1d(wls, fls, fill_value=np.nan)
                fil_interp = interp1d(fil_wls, fil_resp, fill_value=np.nan)
                # integration
                band_step = 0.0001 # step for integration
                int_wls = np.arange(fil_wls[0], fil_wls[-1], band_step)
                int_resp = fil_interp(int_wls)
                int_flsresp = spec_interp(int_wls) * int_resp
                # pseudo flux with arbitrary unit
                arb_specfil = ( 2.0 * np.nansum(int_flsresp[1:-1]) + int_flsresp[0] + int_flsresp[-1] ) * band_step / 2
                arb_fil = ( 2.0 * np.nansum(int_resp[1:-1]) + int_resp[0] + int_resp[-1] ) * band_step / 2
                arb_flux = arb_specfil / arb_fil
                ## astrophysical flux
                astro_flux = flux_zp[band+"_"+phot_sys] * 10**(-0.4 * cal_mag[band])
                ## calibrated flux
                fls_cal = fls * (astro_flux / arb_flux)
                sigmas_cal = sigmas * (astro_flux / arb_flux)
            ## save spectra
            col_wls = fits.Column(name='wls', format='D', array=wls)
            col_fls = fits.Column(name='fls', format='D', array=fls_cal)
            col_sigmas = fits.Column(name='sigmas', format='D', array=sigmas_cal)
            combo_col = fits.ColDefs([col_wls, col_fls, col_sigmas])
            HDU = fits.BinTableHDU.from_columns(combo_col)
            HDU.writeto(calpath + 'flux_cal/' + 'cal_%s_%s.fits'%(band, phot_sys), overwrite=True)
# ----


def HDF5_converter(outfile, wls, fls, sigmas, u_wls='micron', u_fls='erg/s/cm2/A'):
    ''' save the spectrum into the HDF5 file
        
        the Starfish spectra should have the following units:
        - wls: Angstrom
        - fls/sigmas: erg/s/cm2/A
        
        we assume the spectrum has the following units:
        - wls: micron
        - fls/sigmas: erg/s/cm2/A
        
        we support the following units:
        - wls: [micron], [Anstrom]
        - fls/sigmas: [erg/s/cm2/A], [W/m2/m]
        '''
    ### unit conversion
    # wavelength
    if u_wls == 'micron':
        Sf_wls = (wls * u.micron).to(u.Angstrom).value
    elif u_wls == 'Angstrom':
        Sf_wls = wls
    else:
        print("warning: the '%s' unit is not incorporated for wls... simply assuming wls has unit of 'Angstrom'."%(u_wls))
        Sf_wls = wls
    # flux
    if u_fls == 'erg/s/cm2/A':
        Sf_fls = fls
        Sf_sigmas = sigmas
    elif u_fls == 'W/m2/m':
        Sf_fls = (fls * u.W/u.m**2/u.m).to(u.erg/u.s/u.cm**2/u.Angstrom).value
        Sf_sigmas = (sigmas * u.W/u.m**2/u.m).to(u.erg/u.s/u.cm**2/u.Angstrom).value
    else:
        print("warning: the '%s' unit is not incorporated for fls... simply assuming wls has unit of 'erg/s/cm2/A'."%(u_fls))
        Sf_fls = fls
        Sf_sigmas = sigmas
    # adjust bad flux uncertainties - use the absolute flux value as sigma's if the given sigma is None or <=0
    id_bad_sigmas = np.where(np.isnan(Sf_sigmas) | (Sf_sigmas <= 0))
    Sf_sigmas[id_bad_sigmas] = np.abs(Sf_fls[id_bad_sigmas])
    ### nominal mask array - all one's (no mask)
    Sf_mask = np.ones(len(Sf_wls), dtype=int)
    ### save to HDF5
    hdf5_load = h5py.File(outfile, 'w')
    hdf5_load.create_dataset('wls', data=Sf_wls)
    hdf5_load.create_dataset('fls', data=Sf_fls)
    hdf5_load.create_dataset('sigmas', data=Sf_sigmas)
    hdf5_load.create_dataset('masks', data=Sf_mask)
    hdf5_load.close()
    print("HDF5 file created.")
# ----


######################
# OFFICIAL PROCEDURES
######################
## load basic information
cal_dir = Starfish.data["path"]
raw_spec = Starfish.data["raw_files"]
hdf5_spec = Starfish.data["files"]

## preparation
for (rel_calpath, rel_infile, rel_outfile) in zip(cal_dir, raw_spec, hdf5_spec):
    # obtain real paths
    calpath = os.path.expandvars(rel_calpath)
    infile = os.path.expandvars(rel_infile)
    outfile = os.path.expandvars(rel_outfile)
    # load spectra
    wls, fls, sigmas = spec_read(infile)
    # flux calibration
    if args.phot is not None:
        spec_cal(wls, fls, sigmas, args.phot, cal_key, cal_mag, calpath)
        args.u_wls = 'micron'
        args.u_fls = 'erg/s/cm2/A'
    # HDF5 conversion
    if args.create==True:
        calbase_file = calpath + 'flux_cal/cal_%s.fits'%(args.calbase)
        # load the calbase file if existed
        if (args.calbase is not None) and (os.path.isfile(calbase_file)):
            wls, fls, sigmas = spec_read(calbase_file)
        # convert to HDF5 file
        HDF5_converter(outfile, wls, fls, sigmas, u_wls=args.u_wls, u_fls=args.u_fls)
    else:
        print("no HDF5 file created.")
# --








