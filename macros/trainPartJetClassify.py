#!/usr/bin/env python

import os
import socket
import sys
import time
import logging
import math

import h5py
import numpy as np
import keras
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.models import load_model

import tensorflow as tf

from optparse import OptionParser
p = OptionParser()

p.add_option('-o', '--outdir',   type='string',                      default='models/')
p.add_option('-t', '--testfile', type='string',                      default=None)
p.add_option('-w', '--weight',   type='string',                      default=None)
p.add_option('--inputjet',       type='string',                      default=None)
p.add_option('--model',          type='string',                      default=None)
p.add_option('--nparticle',      type='int',                         default=50)
p.add_option('-n', '--nevent',   type='int',                         default=None)
p.add_option('-d', '--debug',    action='store_true',  dest='debug', default=False)
p.add_option('--plot-model',     action='store_true',  default=False)


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
def oneHotLabelFast(var_labels):

    one_hot_labels = [] 
    one_hot_dict = {1:0, 4:1, 5:2, 21:3} 

    for i in var_labels:

        one_hot_labels += [one_hot_dict[i]]

    return np.array(one_hot_labels)

#======================================================================================================        
def sub_oneHotLabelFast(var_label):

    one_hot_dict = {1:0, 4:1, 5:2, 21:3}

    x = one_hot_dict[var_label]

    return x

#======================================================================================================
def sub_process(x):

    out_dict = {0:1, 1:4, 2:5, 3:21}

    x = out_dict[x]

    return x

#======================================================================================================
def get_theta(arrays, axis='x'):

    if axis == 'x':
        axis = np.array([[1], [0], [0]])
    elif axis == 'y':
        axis = np.array([[0], [1], [0]])
    elif axis == 'z':
        axis = np.array([[0], [0], [1]])
    else:
        log.error('get_theta - unknow axis = %s'%axis)
        
    a = (arrays @ axis).ravel()
    b = np.sqrt((arrays*arrays).sum(1))
    return np.arccos(a / b)  

#======================================================================================================
def getOutName(name):
    
    if not options.outdir:
        return None
    
    outdir = '%s/' %(options.outdir.rstrip('/'))
    
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    return '%s/%s' %(outdir.rstrip('/'), name)

#======================================================================================================
def mycrossentropy(y_true, y_pred):
    new_y_pred = tf.clip_by_value(y_pred, 1e-15, 1.0)
    return keras.backend.categorical_crossentropy(y_true, new_y_pred)

#======================================================================================================
# Particle variable processing
#======================================================================================================
def particle_charge(x):

    if x in [211, 321, 2212, -11, -13]:
        return 1
    elif x in [-211, -321, -2212, 11, 13]:
        return -1
    else:
        return 0
    

#======================================================================================================
def trainRNN(train_data, aux_train_vars, train_labels):

    '''Train RNN algorithm'''

    timeStart = time.time()

    train_labels_bin = keras.utils.to_categorical(train_labels)
    #train_labels_bin = keras.utils.to_categorical(oneHotLabelFast(train_labels))

    log.info('trainRNN - start')
    log.info('   train_data       len=%s, shape=%s, dtype=%s' %(len(train_data),       train_data      .shape, train_data      .dtype))
    log.info('   aux_train_vars   len=%s, shape=%s, dtype=%s' %(len(aux_train_vars),   aux_train_vars  .shape, aux_train_vars  .dtype))
    log.info('   train_labels     len=%s, shape=%s, dtype=%s' %(len(train_labels),     train_labels    .shape, train_labels    .dtype))
    log.info('   train_labels_bin len=%s, shape=%s, dtype=%s' %(len(train_labels_bin), train_labels_bin.shape, train_labels_bin.dtype))

    log.info('Will configure model...')

    timeSteps = train_data.shape[1] # number of jets
    nFeatures = train_data.shape[2] # number of variables per jets

    log.info('trainRNN - number of time steps (tracks):          %d' %timeSteps)
    log.info('trainRNN - number of features   (track variables): %d' %nFeatures)

    #-----------------------------------------------------------------------------------
    # Create and connect graphs
    #
    # 1. Prepare Bidirectional LSTM layer
    part_inputs = keras.layers.Input(shape=(timeSteps, nFeatures), name="jets_inputs")

    masked_input = keras.layers.Masking()(part_inputs)

    log.info('part_inputs shape=%s' %part_inputs.shape)
    
    lstm = keras.layers.Bidirectional(LSTM(30, return_sequences=False, name='LSTM'), merge_mode='concat', weights=None)(masked_input)
    
    # 2. Prepare extra input layer of lepton input variables
    nAuxFeatures = aux_train_vars.shape[1]
    
    aux_inputs = keras.layers.Input(shape=(nAuxFeatures, ), name="aux_inputs")

    # 3. Combine Bidirectional LSTM output layer and extra input layer
    mlayer = keras.layers.concatenate([lstm, aux_inputs])
    
    DenseA = keras.layers.Dense(50, activation='tanh', name="DenseA")(mlayer)

    dpt = keras.layers.Dropout(rate=0.1)(DenseA)

    DenseB = keras.layers.Dense(10, activation='tanh', name="DenseB")(dpt)
    # Hint:
    #   try not to use relu just before the softmax layer
    # FC = keras.layers.Dense(10, activation='relu', name="Dense")(dpt)
    # 

    output = keras.layers.Dense(4, activation='softmax', name="jet_class")(DenseB)

    log.info('Will create model...')

    model = keras.models.Model(inputs=[part_inputs, aux_inputs], outputs=output)

    log.info('Will compile model...')
    
    #-----------------------------------------------------------------------------------
    # Compile and fit model
    #
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    model.summary()
   
    #-----------------------------------------------------------------------------------
    # save the diagram of the model structure for testing if need
    #   
    if options.plot_model: 
        keras.utils.plot_model(model, to_file='model.png', show_shapes=True)

    csv_logger = keras.callbacks.CSVLogger(getOutName('train_RNN_loss.csv'))

    log.info('Will fit model...')

    fhist = model.fit([train_data, aux_train_vars],
                      train_labels_bin,
                      epochs=40,
                      batch_size=1024,
                      callbacks=[csv_logger])

    log.info(str(fhist))

    model.summary()

    log.info('trainRNN - all done in %.1f seconds' %(time.time()-timeStart))

    return model

#======================================================================================================        
def saveRNNPrediction(model, rnn_data, jets, outkey = "RNN_"):

    if outkey == None:
        return

    fname_pred = getOutName('%s_predictions.csv' %outkey)
   
    jets["RNN_u"], jets["RNN_c"], jets["RNN_b"], jets["RNN_g"] = zip(*model.predict(rnn_data))
    
    jets.to_csv(fname_pred, index=False)


#======================================================================================================        
def saveModel(model, outkey="first_DNN"):

    if outkey == None:
        return

    fname_model  = getOutName('%s_model.h5'        %outkey)
    fname_weight = getOutName('%s_weights.h5'      %outkey)
    fname_arch   = getOutName('%s_arch.json'       %outkey)

    if model == None:
        #   
        # Just check that output files do not already exist
        #   
        for f in [fname_model, fname_weight, fname_arch]:
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


#======================================================================================================        
def main_trainRNN():

    if len(args) != 1:
        log.warning('Wrong number of command line arguments: %s' %len(args))
        return

    fname = args[0]

    if not os.path.isfile(fname):
        log.warning('Input file does not exist: %s' %fname)
        return    
    
    if not options.inputjet:
        return
    

    train_file = pd.read_csv(fname)
    jet_file   = pd.read_csv(options.inputjet).sort_values(by='jet_id')

    #
    # 1. Prepare training data for RNN 
    #
    train_file['particle_charge'] = train_file['particle_category'].apply(particle_charge)

    # get theta and phi
    part_vect = train_file[['particle_px', 'particle_py', 'particle_pz']].values
    train_file['particle_theta_x'] = get_theta(part_vect, axis='x')
    train_file['particle_phi_x']   = np.arctan(train_file['particle_pz'] / train_file['particle_py'])

    jet_vect = jet_file[['jet_px', 'jet_py', 'jet_pz']].values
    jet_file['jet_theta_x'] = get_theta(jet_vect, axis='x')
    jet_file['jet_phi_x']   = np.arctan(jet_file['jet_pz'] / jet_file['jet_py'])

    #
    # 2. Select input variables
    #
    input_var_names = ['particle_category', 'particle_px', 'particle_py', 'particle_pz', 'particle_energy', 'particle_mass', 
                       'particle_charge', 'particle_theta_x', 'particle_phi_x']

    input_jetvar_names = ['number_of_particles_in_this_jet', 'jet_px', 'jet_py', 'jet_pz', 'jet_energy', 'jet_mass',
                          'jet_theta_x', 'jet_phi_x']

    #
    # 3. Prepare numpy array inputs to the keras
    #
    id_list = train_file.drop_duplicates(subset=['jet_id'])['jet_id']
    nevt    = len(id_list)

    print("number of training jets = ", nevt)

    out_train_data  = np.zeros((nevt, options.nparticle, len(input_var_names)))
    out_train_label = np.zeros((nevt))

    i = 0
    timePrev = time.time()
    
    train_jets = train_file.groupby('jet_id', sort=True)

    if len(jet_file) != nevt:
        print("INFO - number of jets in jet file = %d, number of jet in particle file = %d"%(len(jet_file), nevt))
        jet_file = jet_file.head(nevt)

    for evt_id, subset in train_jets:
        subset = subset.sort_values(by='particle_category')

        subset_data = subset[input_var_names].values
        nparticle   = subset_data.shape[0]

        for j in range(nparticle):
            if j < options.nparticle:
                out_train_data[i][j] = subset_data[j]
        i += 1

        if i % 2000 == 0:
            log.info('main_trainRNN_fast - Processing event #%7d/%7d, delta t=%.2fs' %(i, nevt, time.time() - timePrev))
            timePrev = time.time()

    pd_jet_vars = jet_file[input_jetvar_names].values

    if options.model:
        # load the given RNN model
        model = load_model(options.model)
    else:
        out_train_label = jet_file['label'].apply(sub_oneHotLabelFast)
        
        model = trainRNN(out_train_data, pd_jet_vars, out_train_label)

        saveModel(model, "Di_RNN_event_vars" )
    
    # 
    # Save the output to the jet file 
    # 
    saveRNNPrediction(model, [out_train_data, pd_jet_vars], jet_file, outkey="jet_with_RNN")    
        
#======================================================================================================        
if __name__ == '__main__':

    timeStart = time.time()

    log.info('Start job at %s:%s' %(socket.gethostname(), os.getcwd()))
    log.info('Current time: %s' %(time.asctime(time.localtime())))
   
    main_trainRNN()

    log.info('Local time: %s' %(time.asctime(time.localtime())))
    log.info('Total time: %.1fs' %(time.time()-timeStart))
