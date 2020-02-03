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

import tensorflow as tf

from optparse import OptionParser
p = OptionParser()

p.add_option('-o', '--outdir',   type='string',                      default='models/')
p.add_option('-t', '--testfile', type='string',                      default=None)
p.add_option('-w', '--weight',   type='string',                      default=None)
p.add_option('--njet',           type='int',                         default=10)
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
    one_hot_dict   = {1:[1, 0, 0, 0],
                      4:[0, 1, 0, 0],
                      5:[0, 0, 1, 0],
                      21:[0, 0, 0, 1]
                     }

    for i in var_labels:

        if len(one_hot_labels) == 0:
            one_hot_labels = np.array(one_hot)
        else:
            one_hot_labels = np.vstack((one_hot_labels, np.array(one_hot_dict[i])))

    return one_hot_labels


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
# Event variable processing
#======================================================================================================
def event_mass(df):
    e  = df['jet_energy'].sum()
    px = df['jet_px'].sum()
    py = df['jet_py'].sum()
    pz = df['jet_pz'].sum()

    mass2 = (e**2 - (px**2+py**2+pz**2))

    if mass2 > 0:
        return mass2**0.5
    else:
        return -(-mass2)**0.5

#======================================================================================================
def event_njet(df):
    return len(df['jet_id'])

#======================================================================================================
def event_p3(df):
    px = df['jet_px'].sum()
    py = df['jet_py'].sum()
    pz = df['jet_pz'].sum()
    return (px**2 + py**2 + pz**2)**0.5

#======================================================================================================
def event_theta_x(df):
    px = df['jet_px'].sum()
    p3 = event_p3(df)
    
    if p3 < 1e-15:
        return 0

    return math.acos(px/p3)

#======================================================================================================
def getEventsVars(grouped_evt):
    pd_event_vars = pd.DataFrame()
    pd_event_vars['event_p3']      = grouped_evt.apply(event_p3)
    #pd_event_vars['event_mass']    = grouped_evt.apply(event_mass)
    pd_event_vars['event_njet']    = grouped_evt.apply(event_njet)
    pd_event_vars['event_theta_x'] = grouped_evt.apply(event_theta_x)
   
    return pd_event_vars 
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
    jet_inputs = keras.layers.Input(shape=(timeSteps, nFeatures), name="jets_inputs")

    masked_input = keras.layers.Masking()(jet_inputs)

    log.info('jet_inputs shape=%s' %jet_inputs.shape)
    
    lstm = keras.layers.Bidirectional(LSTM(10, return_sequences=False, name='LSTM'), merge_mode='concat', weights=None)(masked_input)

    nAuxFeatures = aux_train_vars.shape[1]
    
    aux_inputs = keras.layers.Input(shape=(nAuxFeatures, ), name="aux_inputs")

    mlayer = keras.layers.concatenate([lstm, aux_inputs])

    dpt = keras.layers.Dropout(rate=0.1)(mlayer)

    FC = keras.layers.Dense(10, activation='tanh', name="Dense")(dpt)
    # Hint:
    #   try not to use relu just before the softmax layer
    # FC = keras.layers.Dense(10, activation='relu', name="Dense")(dpt)
    # 

    output = keras.layers.Dense(4, activation='softmax', name="jet_class")(FC)

    log.info('Will create model...')

    model = keras.models.Model(inputs=[jet_inputs, aux_inputs], outputs=output)

    log.info('Will compile model...')
    
    # save the diagram of the model
    keras.utils.plot_model(model, to_file=getOutName('model.png'), show_shapes=True)

    #-----------------------------------------------------------------------------------
    # Compile and fit model
    #
    model.compile(loss=mycrossentropy, optimizer='adam', metrics=['acc'])
    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

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
def trainMLP(train_data, train_labels, title):

    '''Train simple MLP algorithm'''
    train_labels_bin = keras.utils.to_categorical(oneHotLabelFast(train_labels))

    log.info('trainMLP - with title=%s' %(title))
    log.info('   train_data   len=%s, shape=%s'     %(len(train_data),       train_data  .shape))
    log.info('   train_labels len=%s, shape=%s'     %(len(train_labels),     train_labels.shape))    
    log.info('   train_labels_bin len=%s, shape=%s' %(len(train_labels_bin), train_labels_bin.shape))    

    log.info('Will configure model...')
    
    ndim = len(train_data.keys()) 
    
    model = Sequential()
    model.add(Dense(64, input_dim=ndim, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(10, activation='sigmoid'))
    model.add(Dense(4, activation='softmax'))

    log.info('Will compile model...')
    
    csv_logger = keras.callbacks.CSVLogger(getOutName('train_DNN_loss.csv'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    log.info('Will fit model...')
    
    model.fit(train_data, train_labels_bin,
              validation_split=0.1,
              epochs=80,
              batch_size=10000,
              callbacks=[csv_logger])

    return model


#======================================================================================================        
def savePrediction(model, test_data=[], lfile = None):

    outkey = "first_DNN"

    if outkey == None:
        return

    fname_pred   = getOutName('%s_predictions.csv' %outkey)

    if model == None:
        #   
        # Just check that output files do not already exist
        #   
        for f in [fname_model, fname_weight, fname_arch, fname_pred]:
            if os.path.isfile(f):
                raise Exception('saveModel - file already exists: \"%s\"' %f) 

        return 

    log.info('saveModel - save model predictions for test and training data')

    pdata = model.predict(test_data)

    # WITH PANDAS
    #  -- clear and easy implement
    sub=pd.DataFrame() # Data format of Pandas
    sub['id']    = lfile['jet_id']
    sub['label'] = list(pdata.argmax(axis=1))
    sub['label'] = sub['label'].apply(sub_process)
    
    sub.to_csv(fname_pred, index=False) 
     
    # SAVE PREDICT TO TXT
    #    np.savetxt(fname_pred, pdata, fmt='%.18e',delimiter=None)
    #

    # IF NO PANDAS
    #    out_dict = {0:1,
    #                1:4,
    #                2:5,
    #                3:21
    #                }
    #
    #    with open(fname_pred, 'w') as f:
    #         for pdict in pdata:
    #             max_index = np.argmax(pdict) 
    #             f.write('%d\n'%out_dict[max_index])


#======================================================================================================        
def saveRNNPrediction(model, id_list, out_test_data, evtid_jetid_dict):

    outkey = "first_RNN"

    if outkey == None:
        return

    fname_pred   = getOutName('%s_predictions.csv' %outkey)

    pdata = model.predict(out_test_data).argmax(axis=1)
    
    if len(pdata) != len(id_list):
        log.error('saveRNNPrediction - Wrong number of pridictions')
        return

    with open(fname_pred, 'w') as f:
        f.write('id,label\n')

        for i in range(len(id_list)):
            evt_id = id_list.values[i]

            label = sub_process(pdata[i])

            for jet_id in evtid_jetid_dict[evt_id]:
                f.write('%s,%d\n'%(jet_id, label))

#======================================================================================================        
def saveRNNPrediction_fast(model, out_test_data, test_events):

    outkey = "first_RNN"

    if outkey == None:
        return

    fname_pred   = getOutName('%s_predictions.csv' %outkey)
    
    pdata = model.predict(out_test_data).argmax(axis=1)

    i = 0 
    sub = pd.DataFrame()

    for evt_id, subset in test_events:
        label = sub_process(pdata[i])
        i += 1
        
        sub_out = pd.DataFrame()
        
        sub_out['id']    = subset['jet_id']
        sub_out['label'] = np.ones(len(sub_out['id']))*label

        sub = pd.concat([sub, sub_out])
    
    sub.to_csv(fname_pred, index=False)


#======================================================================================================        
def saveModel(model, train_data=[], outkey="first_DNN"):

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
def main_trainRNN_fast():

    if len(args) != 1:
        log.warning('Wrong number of command line arguments: %s' %len(args))
        return

    fname = args[0]

    if not os.path.isfile(fname):
        log.warning('Input file does not exist: %s' %fname)
        return    
    
    train_file = pd.read_csv(fname)
    
    # prepare training data for RNN
    input_var_names = ['number_of_particles_in_this_jet', 'jet_px', 'jet_py', 'jet_pz', 'jet_energy', 'jet_mass', 'jet_theta_x', 'jet_pt_x']
    id_list = train_file.drop_duplicates(subset=['event_id'])['event_id']
    nevt    = len(id_list)

    print("number of events = ", nevt)

    out_train_data  = np.zeros((nevt, options.njet, len(input_var_names)))
    out_train_label = np.zeros((nevt))

    i = 0
    timePrev = time.time()
    
    # Using groupby function will speed those code by factor ~2!
    #   -- Much faster than expect when run on large statistics: factor > 80 in full statistics
    train_events = train_file.groupby('event_id')

    for evt_id, subset in train_events:
        out_train_label[i] = subset['label'].apply(sub_oneHotLabelFast).values[0]

        subset_data = subset[input_var_names].values
        njet        = subset_data.shape[0]

        for j in range(njet):
            if j < options.njet:
                out_train_data[i][j] = subset_data[j]
        i += 1

        if i % 2000 == 0:
            log.info('main_trainRNN_fast - Processing event #%7d/%7d, delta t=%.2fs' %(i, nevt, time.time() - timePrev))
            timePrev = time.time()

    pd_event_vars = getEventsVars(train_events)

    model = trainRNN(out_train_data, pd_event_vars.values, out_train_label)

    saveModel(model, out_train_data, "Di_RNN_event_vars" )
    
    #model = keras.models.load_model(options.weight)

    if options.testfile and  os.path.isfile(options.testfile):
        test_file = pd.read_csv(options.testfile)

        id_list = test_file.drop_duplicates(subset=['event_id'])['event_id']
        nevt    = len(id_list)

        print("number of test events = ", nevt)

        out_test_data  = np.zeros((nevt, options.njet, len(input_var_names)))

        i = 0
        timePrev = time.time()
        
        # Use groupby
        test_events = test_file.groupby('event_id')

        for evt_id, subset in test_events:
            subset_data = subset[input_var_names].values
            njet        = subset_data.shape[0]

            for j in range(njet):
                if j < options.njet:
                    out_test_data[i][j] = subset_data[j]
            i += 1

            if i % 2000 == 0:
                log.info('main_testRNN_fast - Processing event #%7d/%7d, delta t=%.2fs' %(i, nevt, time.time() - timePrev))
                timePrev = time.time()

        test_event_vars = getEventsVars(test_events)

        saveRNNPrediction_fast(model, [out_test_data, test_event_vars], test_events)
 

#======================================================================================================        
def main_trainRNN():

    if len(args) != 1:
        log.warning('Wrong number of command line arguments: %s' %len(args))
        return

    fname = args[0]

    if not os.path.isfile(fname):
        log.warning('Input file does not exist: %s' %fname)
        return    
    
    train_file = pd.read_csv(fname)
    
    # prepare training data for RNN
    input_var_names = ['number_of_particles_in_this_jet', 'jet_px', 'jet_py', 'jet_pz', 'jet_energy', 'jet_mass', 'jet_theta_x', 'jet_pt_x']
    id_list = train_file.drop_duplicates(subset=['event_id'])['event_id']
    nevt    = len(id_list)

    print("number of events = ", nevt)

    out_train_data  = np.zeros((nevt, options.njet, len(input_var_names)))
    out_train_label = np.zeros((nevt))
     
    timePrev = time.time()

    for i in range(nevt):
        evt_id = id_list.values[i]
        is_same_evtid = train_file['event_id'] == evt_id
        
        subset_train = train_file[is_same_evtid][input_var_names].values
        subset_label = train_file[is_same_evtid]['label'].values[0]
        
        njet = subset_train.shape[0]

        out_train_label[i] = subset_label

        for j in range(njet):
            if j < options.njet:
                out_train_data[i][j] = subset_train[j]

        if i % 2000 == 0:
                log.info('main_trainRNN - Processing event #%7d/%7d, delta t=%.2fs' %(i, nevt, time.time() - timePrev))
                timePrev = time.time()


    model = trainRNN(out_train_data, out_train_label)

    saveModel(model, out_train_data)

    if options.testfile and  os.path.isfile(options.testfile):
        test_file = pd.read_csv(options.testfile)

        id_list = test_file.drop_duplicates(subset=['event_id'])['event_id']
        nevt    = len(id_list)

        print("number of test events = ", nevt)

        out_test_data  = np.zeros((nevt, options.njet, len(input_var_names)))

        evtid_data_dict  = {}
        evtid_jetid_dict = {}

        for i in range(nevt):
            evt_id = id_list.values[i]
            is_same_evtid = test_file['event_id'] == evt_id
            
            subset_test      = test_file[is_same_evtid][input_var_names].values

            njet = subset_test.shape[0]

            for j in range(njet):
                if j <= options.njet:
                    out_test_data[i][j] = subset_test[j]
            
            evtid_data_dict[evt_id]  = out_test_data[i]
            evtid_jetid_dict[evt_id] = test_file[is_same_evtid]['jet_id'].values

        saveRNNPrediction(model, id_list, out_test_data, evtid_jetid_dict)
 

#======================================================================================================        
def main_trainMLP():

    if len(args) != 1:
        log.warning('Wrong number of command line arguments: %s' %len(args))
        return

    fname = args[0]

    if not os.path.isfile(fname):
        log.warning('Input file does not exist: %s' %fname)
        return    
  
    #===================================================================================================
    # IF NO PANDAS IMPORTED: 
    #   You can get the csv input file:
    #    
    #    train_file = np.genfromtxt(fname, dtype=float, delimiter=',', names=True)
    #
    #    all_data = []
    #
    #    for var_name in ['number_of_particles_in_this_jet', 'jet_px', 'jet_py', 'jet_pz', 'jet_energy', 'jet_mass']:
    #        if len(all_data) == 0:
    #            all_data = train_file[var_name]
    #        else:
    #            all_data = np.vstack((all_data, train_file[var_name]))
    #
    #    all_data = all_data.T
    #===================================================================================================
    
    train_file = pd.read_csv(fname)
    input_var_names = ['number_of_particles_in_this_jet', 'jet_pt_x', 'jet_theta_x', 'jet_p3', 'jet_energy', 'jet_mass']
    #input_var_names = ['number_of_particles_in_this_jet', 'jet_pt', 'jet_sin_theta', 'jet_tan_phi', 'jet_pz', 'jet_energy', 'jet_mass']
    #input_var_names = ['number_of_particles_in_this_jet', 'jet_px', 'jet_py', 'jet_pz', 'jet_energy', 'jet_mass']

    all_data   = train_file[input_var_names]

    all_labels = train_file['label']

    train_data   = all_data[:]    
    train_labels = all_labels[:]  

    if len(train_data) != len(train_labels):
        raise Exception('Length of test data and labels do not match')
    
    log.info('Select indices for test events...')

    model = trainMLP(train_data, train_labels, 'MLP')  
    saveModel(model, train_data)

    if options.testfile and  os.path.isfile(options.testfile):
        test_file = pd.read_csv(options.testfile)
        test_data = test_file[input_var_names]
        savePrediction(model, test_data, test_file)
 
        
#======================================================================================================        
if __name__ == '__main__':

    timeStart = time.time()

    log.info('Start job at %s:%s' %(socket.gethostname(), os.getcwd()))
    log.info('Current time: %s' %(time.asctime(time.localtime())))
    
    main_trainRNN_fast()
    #main_trainRNN()

    log.info('Local time: %s' %(time.asctime(time.localtime())))
    log.info('Total time: %.1fs' %(time.time()-timeStart))
