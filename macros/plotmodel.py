#!/usr/bin/env python 

import logging
import math
import os
import socket
import sys
import time

import h5py
import numpy as np
import keras

import matplotlib

from keras.utils import plot_model
from keras.models import load_model

from optparse import OptionParser
p = OptionParser()

p.add_option('-o', '--outdir',   type='string',                      default='models/')

(options,args) = p.parse_args()

#======================================================================================================  
fname = args[0]
model = load_model(fname)

plot_model(model, to_file='model.png', show_shapes=True)

