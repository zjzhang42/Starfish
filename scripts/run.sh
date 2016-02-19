#!/bin/bash

source activate starfish
sed -i 's/Users/home/g' config.yaml

star.py --initPhi

star.py --optimize=Theta
theta_into_config.py

star.py --optimize=Cheb

mkdir plots

time star.py --sample=ThetaPhi --samples=10

time star.py --sample=ThetaPhi --samples=10000

