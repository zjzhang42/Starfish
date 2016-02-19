#!/bin/bash

source activate starfish

sed -i 's/Users/home/g' config.yaml

mkdir libraries
mkdir plots
mkdir output

grid.py --create

pca.py --create

time pca.py --optimize=emcee --samples=10

pca.py --store --params=emcee

star.py --initPhi

star.py --generate

star.py --optimize=Theta

theta_into_config.py

star.py --optimize=Cheb

star.py --run_index 1
echo "-------------------------------------"
echo "The end."
echo "-------------------------------------"
