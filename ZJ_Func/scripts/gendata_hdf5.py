#!/usr/bin/env python
#
#
## Author: ZJ Zhang (Jul. 23rd, 2017)
#
# Modification History:
#
# ZJ Zhang (Jul. 23rd, 2017)
#
#################################################

# ----
# Goal:
# convert input data file into the hdf5 format
# ----


# import modules
import Starfish
import numpy as np
from astropy.io import fits
import h5py
import os


def spec_read(infile, wl_unit='um'):
    ''' extract wavelength, flux, and flux_err from the input file
        
        We assume the wavelength units are "um" in the input file; and we need to convert them into Angstrom in the output file.

        different methods would be applied for files with different extensions:
        - fits
        - ascii/dat/txt
        '''
    in_name, in_fmt = os.path.splitext(infile)
    if in_fmt==".fits":
        Spec_Load = fits.open(infile)
        Spec_Content = Spec_Load[0].data
        wls = Spec_Content[0] * 1.0e4
        fls = Spec_Content[1]
        sigmas = Spec_Content[2]
    elif in_fmt==".ascii" or in_fmt==".dat" or in_fmt==".txt":
        Spec_Content = np.genfromtxt(infile, dtype=None)
        wls = Spec_Content[:,0] * 1.0e4
        fls = Spec_Content[:,1]
        sigmas = Spec_Content[:,2]
    else:
        print("(ZJ-Style)Error: format of input files not found! It should be one of the follow: fits, ascii, dat ,txt.")
    return wls, fls, sigmas
# ----


### 1. Obtain input file and output name
input_list = Starfish.data["infiles"]
hdf5_list = Starfish.data["files"]

### 2. Genereate output hdf5 files
for (infile, outfile) in zip(input_list, hdf5_list):
    wls, fls, sigmas = spec_read(infile)
    hd_file = h5py.File(outfile, "w")
    dset_wls = hd_file.create_dataset('wls', data=wls)
    dset_fls = hd_file.create_dataset('fls', data=fls)
    dset_sigmas = hd_file.create_dataset('sigmas', data=sigmas)
    hd_file.close()
# ----

### 3. Funeral
print("HDF5 files generated.")
# ----
# END



