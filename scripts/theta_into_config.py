#!/usr/bin/env python

import json
import yaml

with open('theta.json', mode='r') as theta_file:
    new_theta = json.load(theta_file)

f = open("config.yaml")
config = yaml.load(f)
f.close()

config['Theta'] = new_theta

with open("config.yaml", mode='w') as outfile:
    outfile.write(yaml.dump(config))
    print('Revised the config.yaml file with the optimized theta values.')