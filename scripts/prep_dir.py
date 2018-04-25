#!/usr/bin/env python
#
#
## Author: ZJ Zhang
#
# Modification History:
#
# ZJ Zhang (Apr 17th, 2018)
#
#################################################

import os
from Sys_Tool.Script_command import *


import argparse
parser = argparse.ArgumentParser(prog="prep_dir.py", description="Prepare directories for running Starfish")
parser.add_argument("--root_dir", type=str, default='./', help="units of wavelength; chosen from [micron], [Angstrom]")
args = parser.parse_args()

######################
# OFFICIAL PROCEDURES
######################
## load basic information
main_path = os.path.expandvars(args.root_dir)
print("The root dir is:\n    ", main_path)

## create sub-folders
# 1. notebooks/
# after Apr 20, 2018, this program will not create the notebooks/ directory. The `config.yaml` will be directly in the main_path
# 2. workplace/
find_dir(main_path + 'workplace/')
find_dir(main_path + 'workplace/data/')
find_dir(main_path + 'workplace/data/flux_cal/')
find_dir(main_path + 'workplace/libraries/')
find_dir(main_path + 'workplace/priors')
find_dir(main_path + 'workplace/plots/')
find_dir(main_path + 'workplace/plots/processed_grids/')
find_dir(main_path + 'workplace/plots/pca/')
find_dir(main_path + 'workplace/plots/emulator/')
find_dir(main_path + 'workplace/plots/star_inference/')
find_dir(main_path + 'workplace/output/')
find_dir(main_path + 'workplace/output/emulator/')
find_dir(main_path + 'workplace/output/star_debug/')
find_dir(main_path + 'workplace/output/star_inference/')
find_dir(main_path + 'workplace/results')
# 3. finish
print("Setup finish!")




