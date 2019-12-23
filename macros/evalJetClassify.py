#!/usr/bin/env python

import os
import socket
import sys
import time
import logging

import h5py
import numpy as np
import keras

import matplotlib
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout

from optparse import OptionParser
p = OptionParser()

p.add_option('-o', '--outdir',   type='string',                      default='models/')
p.add_option('-w', '--weight',   type='string',                      default=None)
p.add_option('-n', '--nevent',   type='int',                         default=None)
p.add_option('-d', '--debug',    action='store_true',  dest='debug', default=False)

(options,args) = p.parse_args()

#======================================================================================================        
def getLog(name, level='INFO', debug=False, print_time=False):

    if print_time:
        f = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
    else:
        f = logging.Formatter('%(message)s')
        
    sh = logging.StreamHandler()
    sh.setFormatter(f)

    logging._warn_preinit_stderr = 0
    
    log = logging.getLogger(name)

    log.handlers  = []
    log.propagate = False
    
    log.addHandler(sh)
    
    if debug:
        log.setLevel(logging.DEBUG)
    else:
        if level == 'DEBUG':   log.setLevel(logging.DEBUG)
        if level == 'INFO':    log.setLevel(logging.INFO)
        if level == 'WARNING': log.setLevel(logging.WARNING)    
        if level == 'ERROR':   log.setLevel(logging.ERROR)

    return log

#======================================================================================================        
log = getLog(os.path.basename(__file__), print_time=False)

#======================================================================================================        
def countEvents(labels, select_label):

    if select_label == None:
        return len(labels)

    icount = 0

    for i in range(len(labels)):
        if labels[i] == select_label:
            icount += 1

    return icount

#======================================================================================================
def getOutName(name):
    
    if not options.outdir:
        return None
    
    outdir = '%s/' %(options.outdir.rstrip('/'))
    
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    return '%s/%s' %(outdir.rstrip('/'), name)

#======================================================================================================        
def saveModel(model, test_data=[]):

    outkey = "first_DNN"

    if outkey == None:
        return

    fname_pred   = getOutName('%s_predictions_all.csv' %outkey)
    
    #   
    # Save model prediction
    #   
    log.info('saveModel - save model predictions for test data')

    pdata = model.predict(test_data)
    
    out_dict = {0:1,
                1:4,
                2:5,
                3:21
                }

    with open(fname_pred, 'w') as f:
         for pdict in pdata:
             max_index = np.argmax(pdict)
             f.write('%d\n'%out_dict[max_index])

#======================================================================================================        
def main():

    if len(args) != 1:
        log.warning('Wrong number of command line arguments: %s' %len(args))
        return

    fname = args[0]

    if not os.path.isfile(fname):
        log.warning('Input file does not exist: %s' %fname)
        return    
 
    if not options.weight:
        log.warning('missing input weight.h5 file')  
        return
    # 
    # Get the csv input file
    #
    ifile = np.genfromtxt(fname, dtype=float, delimiter=',', names=True)

    all_data = []

    for var_name in ['number_of_particles_in_this_jet', 'jet_px', 'jet_py', 'jet_pz', 'jet_energy', 'jet_mass']:
        if len(all_data) == 0:
            all_data = ifile[var_name]
        else:
            all_data = np.vstack((all_data, ifile[var_name]))

    all_data = all_data.T

    model = keras.models.load_model(options.weight)

    saveModel(model, all_data)
        
#======================================================================================================        
if __name__ == '__main__':

    timeStart = time.time()

    log.info('Start job at %s:%s' %(socket.gethostname(), os.getcwd()))
    log.info('Current time: %s' %(time.asctime(time.localtime())))
    
    main()

    log.info('Local time: %s' %(time.asctime(time.localtime())))
    log.info('Total time: %.1fs' %(time.time()-timeStart))
