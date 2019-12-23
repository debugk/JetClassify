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
def oneHotLabel(var_labels):

    one_hot_labels = []

    for i in var_labels:
        if i == 1:
            one_hot = [1, 0, 0, 0]
        elif i == 4:
            one_hot = [0, 1, 0, 0]
        elif i == 5:
            one_hot = [0, 0, 1, 0]
        else:
            one_hot = [0, 0, 0, 1]

        if len(one_hot_labels) == 0:
            one_hot_labels = np.array(one_hot)
        else:
            one_hot_labels = np.vstack((one_hot_labels, np.array(one_hot)))

    return one_hot_labels


#======================================================================================================        
def oneHotLabelFast(var_labels):

    one_hot_labels = []

    for i in var_labels:
        if i == 1:
            one_hot = 0 
        elif i == 4:
            one_hot = 1
        elif i == 5:
            one_hot = 2
        else:
            one_hot = 3

        one_hot_labels += [one_hot]

    return np.array(one_hot_labels)

#======================================================================================================
def getOutName(name):
    
    if not options.outdir:
        return None
    
    outdir = '%s/' %(options.outdir.rstrip('/'))
    
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    return '%s/%s' %(outdir.rstrip('/'), name)

#======================================================================================================        
def trainMLP(train_data, train_labels, title):

    '''Train simple MLP algorithm'''
    #train_labels_bin = oneHotLabel(train_labels)
    #test_labels_bin  = oneHotLabel(test_labels)
    train_labels_bin = keras.utils.to_categorical(oneHotLabelFast(train_labels))

    log.info('trainMLP - with title=%s' %(title))
    log.info('   train_data   len=%s, shape=%s, dtype=%s'     %(len(train_data),       train_data  .shape,     train_data  .dtype))
    log.info('   train_labels len=%s, shape=%s, dtype=%s'     %(len(train_labels),     train_labels.shape,     train_labels.dtype))    
    log.info('   train_labels_bin len=%s, shape=%s, dtype=%s' %(len(train_labels_bin), train_labels_bin.shape, train_labels_bin.dtype))    

    log.info('Will configure model...')
    
    model = Sequential()
    model.add(Dense(64, input_dim=6, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(4, activation='sigmoid'))

    log.info('Will compile model...')
    
    csv_logger = keras.callbacks.CSVLogger(getOutName('train_DNN_loss.csv'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    log.info('Will fit model...')
    
    model.fit(train_data, train_labels_bin,
              validation_split=0.1,
              epochs=100,
              callbacks=[csv_logger])

    return model

#======================================================================================================        
def saveModel(model, train_data=[], test_data=[], lfile = None):

    outkey = "first_DNN"

    if outkey == None:
        return

    fname_model  = getOutName('%s_model.h5'        %outkey)
    fname_weight = getOutName('%s_weights.h5'      %outkey)
    fname_arch   = getOutName('%s_arch.json'       %outkey)
    fname_pred   = getOutName('%s_predictions.csv' %outkey)

    if model == None:
        #   
        # Just check that output files do not already exist
        #   
        for f in [fname_model, fname_weight, fname_arch, fname_pred]:
            if os.path.isfile(f):
                raise Exception('saveModel - file already exists: \"%s\"' %f) 

        return 
    
    #   
    # Save model, weights and architecture as json
    #   
    model.save_weights(fname_weight)
    model.save        (fname_model)
    
    with open(fname_arch, 'w') as arch:
        arch.write(model.to_json(indent=2))    

    log.info('saveModel - saved model to:        %s' %fname_model)
    log.info('saveModel - saved weights to:      %s' %fname_weight)
    log.info('saveModel - saved architecture to: %s' %fname_arch)

    log.info('saveModel - save model predictions for test and training data')

    pdata = model.predict(test_data)
    
    #np.savetxt(fname_pred, pdata, fmt='%.18e',delimiter=None)

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

    all_labels = ifile['label']

    train_data   = all_data[:]    
    train_labels = all_labels[:]  

    if len(train_data) != len(train_labels):
        raise Exception('Length of test data and labels do not match')
    
    log.info('Select indices for test events...')

    model = trainMLP(train_data, train_labels, 'MLP')  
    
    lfile = np.genfromtxt('tem_train.csv', dtype=float, delimiter=',', names=True)
    #lfile = np.genfromtxt('tem_test_1000.csv', dtype=float, delimiter=',', names=True)
    #lfile = np.genfromtxt('jet_simple_data/simple_test_R04_jet.csv', dtype=float, delimiter=',', names=True)

    all_test_data = []

    for var_name in ['number_of_particles_in_this_jet', 'jet_px', 'jet_py', 'jet_pz', 'jet_energy', 'jet_mass']:
        if len(all_test_data) == 0:
            all_test_data = lfile[var_name]
        else:
            all_test_data = np.vstack((all_test_data, lfile[var_name]))

    all_test_data = all_test_data.T


    saveModel(model, train_data, all_test_data, lfile)
        
#======================================================================================================        
if __name__ == '__main__':

    timeStart = time.time()

    log.info('Start job at %s:%s' %(socket.gethostname(), os.getcwd()))
    log.info('Current time: %s' %(time.asctime(time.localtime())))
    
    main()

    log.info('Local time: %s' %(time.asctime(time.localtime())))
    log.info('Total time: %.1fs' %(time.time()-timeStart))
