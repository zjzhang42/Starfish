#!/usr/bin/env python
import h5py
import numpy as np

import argparse
parser = argparse.ArgumentParser(prog="last_mc.py", description="Print the last mc value to put into config.yaml. Defaults to Theta.")
parser.add_argument("--phi", action="store_true", help="The mc.hdf5 file contains phi calibration parameters.")
parser.add_argument("--dir", type=str, default='blank', help="Which directory?")
parser.add_argument("--last20k", type=str, default='blank', help="Which directory?")
args = parser.parse_args()

if args.dir != 'blank':
	f = h5py.File('{}/mc.hdf5'.format(args.dir), 'r')
	vals = f['samples'][-1:].squeeze()
	print("The last sample was:")
	print('------------------------')
	for val in vals[:-4]:
	    print("{:.4f},".format(val))
	print("{:.4f}".format(vals[-4]))
	print('"l": {:.4f},'.format(vals[-1]))
	print('"logAmp": {:.4f},'.format(vals[-2]))
	print('"sigAmp": {:.4f},'.format(vals[-3]))
	print('------------------------')
	f.close()
else:
	f = h5py.File('mc.hdf5', 'r')
	vals = f['samples'][-1:].squeeze()
	grid = vals[0:3]
	print("The last sample was:")
	print('------------------------')
	print("grid : [{0:.1f}, {1:1.2f}, {2:1.2f}]".format(grid[0], grid[1], grid[2]))
	print("vz : {:.3f}".format(vals[3]))
	print("vsini : {:.3f}".format(vals[4]))
	print("logOmega : {:.3f}".format(vals[5]))
	print('------------------------')
	f.close()

if args.last20k:
	f = h5py.File('mc.hdf5', 'r')
	vals = f['samples'][-20000:-1, 0].squeeze()
	lb, med, ub = np.percentile(vals, [5,50,95])
	print('------------------------')
	print(lb, med, ub)
	print('------------------------')
	f.close()