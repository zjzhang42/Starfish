#!/usr/bin/env python
import h5py
import numpy as np
import os

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
if True:
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
	cwd=os.getcwd()
	try:
		f = h5py.File('mc.hdf5', 'r')
		vals = f['samples'][-10000:-1, 0].squeeze()
		lb, med, ub = np.percentile(vals, [5,50,95])
		vals_lg = np.percentile(f['samples'][-10000:-1, 1].squeeze(), [5,50,95])
		vals_fe = np.percentile(f['samples'][-10000:-1, 2].squeeze(), [5,50,95])
		vals_vz = np.percentile(f['samples'][-10000:-1, 3].squeeze(), [5,50,95])
		vals_vi = np.percentile(f['samples'][-10000:-1, 4].squeeze(), [5,50,95])
		vals_lo = np.percentile(f['samples'][-10000:-1, 5].squeeze(), [5,50,95])
		f2 = h5py.File('s0_o0/mc.hdf5', 'r')
		vals_c1 = np.percentile(f2['samples'][-10000:-1, 0].squeeze(), [5,50,95])
		vals_c2 = np.percentile(f2['samples'][-10000:-1, 1].squeeze(), [5,50,95])
		vals_c3 = np.percentile(f2['samples'][-10000:-1, 2].squeeze(), [5,50,95])
		vals_SA = np.percentile(f2['samples'][-10000:-1, 3].squeeze(), [5,50,95])
		vals_LA = np.percentile(f2['samples'][-10000:-1, 4].squeeze(), [5,50,95])
		vals_ll = np.percentile(f2['samples'][-10000:-1, 5].squeeze(), [5,50,95])

		n_samps = len(f['samples'][:,1])
		if n_samps > 19000:
			print(cwd[-9:-6], ', ', lb,', ', med, ',' , ub, ', ',
			vals_lg[0], ', ', vals_lg[1], ', ', vals_lg[2], ', ',
			vals_fe[0], ', ', vals_fe[1], ', ', vals_fe[2], ', ',
			vals_vz[0], ', ', vals_vz[1], ', ', vals_vz[2], ', ',
			vals_vi[0], ', ', vals_vi[1], ', ', vals_vi[2], ', ',
			vals_lo[0], ', ', vals_lo[1], ', ', vals_lo[2], ', ',
			vals_c1[0], ', ', vals_c1[1], ', ', vals_c1[2], ', ',
			vals_c2[0], ', ', vals_c2[1], ', ', vals_c2[2], ', ',
			vals_c3[0], ', ', vals_c3[1], ', ', vals_c3[2], ', ',
			vals_SA[0], ', ', vals_SA[1], ', ', vals_SA[2], ', ',	        
			vals_LA[0], ', ', vals_LA[1], ', ', vals_LA[2], ', ',
			vals_ll[0], ', ', vals_ll[1], ', ', vals_ll[2])
		else:
			print('---')
		f.close()
		f2.close()
	except:
		print(cwd[-9:-6])
		f.close()
		f2.close()
