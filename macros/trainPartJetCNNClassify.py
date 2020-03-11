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
from joblib import Parallel,delayed

import bisect 
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
# Helper class: vector
#======================================================================================================
class Vector:

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        
        self.l = (self.x**2 + self.y**2 + self.z**2)**0.5
        
    def getTheta(self, vect):
        if self.l < 10e-10 or vect.l < 10e-10:
            return 0

        cos_theta = (self.x*vect.x + self.y*vect.y + self.z*vect.z)/self.l/vect.l

        if not abs(cos_theta) <= 1:
            print("getTheta - error - find cos_theta = %s"%cos_theta)

            if cos_theta > 0:
                return 0
            else:
                return math.pi

        return math.acos(cos_theta)

    def dot(self, vect):
        return (self.x*vect.x + self.y*vect.y + self.z*vect.z)

    def cross(self, vect):
        out_x =   self.y*vect.z - self.z*vect.y
        out_y = -(self.x*vect.z - self.z*vect.x)
        out_z =   self.x*vect.y - self.y*vect.x

        return Vector(out_x, out_y, out_z)

    def getStr(self):
        return '(x, y, z, l) = (%.2f, %.2f, %.2f, %.2f)'%(self.x, self.y, self.z, self.l)

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
def old_particle_theta_phi_bin(df, sum_theta, sum_phi):
  
    # FOR STUDY

    # calculate delta theta bin
    particle_theta = df['particle_theta_x']

    theta = particle_theta - sum_theta
    
    theta_bin_edges = [-0.4, -0.3, -0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2, 0.3, 0.4]

    theta_bin = bisect.bisect_left(theta_bin_edges, theta)
    
    # calculate delta phi bin: note that there is a approximate transformation.
    particle_phi = df['particle_phi_x']

    phi_origin = particle_phi - sum_phi

#    if phi_origin > 3.142:
#        phi_origin = phi_origin - 3.142
#    elif phi_origin < -3.142:
#        phi_origin = phi_origin + 3.142

    # sin(phi/2) = sin(phi_origin/2) * sin(sum_theta) 
    # phi = phi_origin*math.sin(sum_theta) # only be true when phi_origin is very small, but it should be fine here
    phi = 2*math.asin(math.sin(phi_origin/2)*math.sin(sum_theta) )

    phi_bin_edges = [-0.4, -0.3, -0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2, 0.3, 0.4]

    phi_bin = bisect.bisect_left(phi_bin_edges, phi)

    if phi_bin == 0 or phi_bin == 11:
        print("particle_theta = ", particle_theta, ", sum_theta = ", sum_theta, ", particle_phi = ", particle_phi, ", sum_phi = ", sum_phi)

        print("theta_bin = ", theta_bin, ", theta = ", theta, ", sum_theta = ", sum_theta)
        print("phi_bin = ", phi_bin, ", phi = ", phi, ", phi_origin = ", phi_origin)
        

    return (theta_bin, phi_bin)


#======================================================================================================
def getJetAxisXYBins(df, x_bin_edges, y_bin_edges):
    # 
    # get JetAxis x and y 
    #
    theta = df['jet_axis_theta']
    phi   = df['jet_axis_phi']

    x = math.sin(theta)*math.cos(phi)
    y = math.sin(theta)*math.sin(phi)

    #
    # Since most of theta < 0.4, x or y should < sin(0.4), we normalize x and y
    # 
    x_bin = bisect.bisect_left(x_bin_edges, x/0.3894 + 0.5)
    y_bin = bisect.bisect_left(y_bin_edges, y/0.3894 + 0.5)
    
    return (x_bin, y_bin)

#======================================================================================================
def jet_axis_theta(df, sum_part_vect):
    # 
    # calculate delta theta
    #
    particle_vect = Vector(df['particle_px'], df['particle_py'], df['particle_pz'])
    theta = particle_vect.getTheta(sum_part_vect)

    return theta


#======================================================================================================
def jet_axis_phi(df, sum_part_vect):
    # 
    # calculate delta phi bin: consider the angle between the (sum_part_vect + x-axis) plane and the (particle_vect + sum_part_vect) plane.
    #
    particle_vect = Vector(df['particle_px'], df['particle_py'], df['particle_pz'])
    x_axis = Vector(1, 0, 0)

    # normal vector of (sum_part_vect + x-axis) plane
    n_jet_xaxis = sum_part_vect.cross(x_axis)

    # normal vector of (sum_part_vect + x-axis) plane
    n_jet_part  = sum_part_vect.cross(particle_vect)

    plane_theta = n_jet_xaxis.getTheta(n_jet_part) # in [0, pi]

    #
    # consider the direction of the plane_theta
    #
    plane_sign_cos = n_jet_xaxis.cross(n_jet_part).dot(sum_part_vect)

    if plane_sign_cos >= 0:
        phi = plane_theta
    else:
        phi = plane_theta + np.pi

    return phi


#======================================================================================================
def particle_phi_bin(df, sum_part_vect, phi_bin_edges):
    # 
    # calculate delta phi bin: consider the angle between the (sum_part_vect + x-axis) plane and the (particle_vect + sum_part_vect) plane.
    #
    particle_vect = Vector(df['particle_px'], df['particle_py'], df['particle_pz'])
    x_axis = Vector(1, 0, 0)

    # normal vector of (sum_part_vect + x-axis) plane
    n_jet_xaxis = sum_part_vect.cross(x_axis)

    # normal vector of (sum_part_vect + x-axis) plane
    n_jet_part  = sum_part_vect.cross(particle_vect)

    plane_theta = n_jet_xaxis.getTheta(n_jet_part) # in [0, pi]

    #
    # consider the direction of the plane_theta
    #
    plane_sign_cos = n_jet_xaxis.cross(n_jet_part).dot(sum_part_vect)

    if plane_sign_cos >= 0:
        phi = plane_theta
    else:
        phi = plane_theta + np.pi
     
    phi_bin   = bisect.bisect_left(phi_bin_edges, phi/2/np.pi)

    return phi_bin


#======================================================================================================
def particle_phi_bin_x(phi, phi_bin_edges):
    # 
    # calculate delta phi bin
    #
     
    phi_bin   = bisect.bisect_left(phi_bin_edges, phi/2/np.pi)

    return phi_bin

#======================================================================================================
def trainCNN(train_data, aux_train_vars, train_labels):

    '''Train CNN algorithm'''

    timeStart = time.time()

    train_labels_bin = keras.utils.to_categorical(train_labels)
    #train_labels_bin = keras.utils.to_categorical(oneHotLabelFast(train_labels))

    log.info('trainCNN - start')
    log.info('   train_data       len=%s, shape=%s, dtype=%s' %(len(train_data),       train_data      .shape, train_data      .dtype))
    log.info('   aux_train_vars   len=%s, shape=%s, dtype=%s' %(len(aux_train_vars),   aux_train_vars  .shape, aux_train_vars  .dtype))
    log.info('   train_labels     len=%s, shape=%s, dtype=%s' %(len(train_labels),     train_labels    .shape, train_labels    .dtype))
    log.info('   train_labels_bin len=%s, shape=%s, dtype=%s' %(len(train_labels_bin), train_labels_bin.shape, train_labels_bin.dtype))

    log.info('Will configure model...')

    n_theta_bin = train_data.shape[1] # number of theta bins
    n_phi_bin   = train_data.shape[2] # number of phi bins
    n_color_bin = train_data.shape[3] # number of phi bins

    log.info('trainCNN - number of N theta bins: %d' %n_theta_bin)
    log.info('trainCNN - number of N phi bins:   %d' %n_phi_bin)

    #-----------------------------------------------------------------------------------
    # Create and connect graphs
    #
    # 1. Prepare conv2D sequential 
    hits_inputs = keras.layers.Input(shape=(n_theta_bin, n_phi_bin, n_color_bin), name="hits_image")

    #reshape_inputs = keras.layers.Reshape(target_shape=(n_theta_bin, n_phi_bin, 2), input_shape=(n_theta_bin, n_phi_bin, 2))(hits_inputs) #TODO not nec

    conv2d_1  = keras.layers.Conv2D(filters=4, kernel_size=(4, 4), padding='same', activation='relu')(hits_inputs)
    conv2d_2  = keras.layers.Conv2D(filters=4, kernel_size=(3, 3), padding='same', activation='relu')(conv2d_1)
    maxpool_1 = keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2,2))(conv2d_2)
    #maxpool_1 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2))(conv2d_2)
    conv2d_3  = keras.layers.Conv2D(filters=4, kernel_size=(4, 4), padding='same', activation='relu')(maxpool_1)
    
    out_cnn   = keras.layers.Flatten()(conv2d_3) 
    
    # 2. Prepare extra input layer of lepton input variables
    nAuxFeatures = aux_train_vars.shape[1]
    
    aux_inputs = keras.layers.Input(shape=(nAuxFeatures, ), name="aux_inputs")

    # 3. Combine CNN output layer and extra input layer
    mlayer = keras.layers.concatenate([out_cnn, aux_inputs])
    
    DenseC = keras.layers.Dense(100, activation='tanh', name="DenseC")(mlayer)
    
    DenseA = keras.layers.Dense(50, activation='tanh', name="DenseA")(DenseC)

    dpt = keras.layers.Dropout(rate=0.1)(DenseA)

    DenseB = keras.layers.Dense(10, activation='tanh', name="DenseB")(dpt)

    output = keras.layers.Dense(4, activation='softmax', name="jet_class")(DenseB)

    log.info('Will create model...')

    model = keras.models.Model(inputs=[hits_inputs, aux_inputs], outputs=output)

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

    csv_logger = keras.callbacks.CSVLogger(getOutName('train_CNN_loss.csv'))

    log.info('Will fit model...')

    fhist = model.fit([train_data, aux_train_vars],
                      train_labels_bin,
                      epochs=40,
                      batch_size=1024,
                      callbacks=[csv_logger])

    log.info(str(fhist))

    log.info('trainCNN - all done in %.1f seconds' %(time.time()-timeStart))

    return model

#======================================================================================================        
def saveCNNPrediction(model, rnn_data, jets, outkey = "CNN_"):

    if outkey == None:
        return

    fname_pred = getOutName('%s_predictions.csv' %outkey)
   
    jets["CNN_u"], jets["CNN_c"], jets["CNN_b"], jets["CNN_g"] = zip(*model.predict(rnn_data))
    
    jets[['jet_id', 'CNN_u', 'CNN_c', 'CNN_b', 'CNN_g']].to_csv(fname_pred, index=False)


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
def joblib_process_particle_groups(subset, x_bin_edges, y_bin_edges):
    #
    # Use sum particle direction as jet direction (they should very close)
    #
    sum_e, sum_px, sum_py, sum_pz = subset[['particle_energy', 'particle_px', 'particle_py', 'particle_pz']].sum().values
    
    sum_p3        = (sum_px**2 + sum_py**2 + sum_pz**2)**0.5
    sum_theta     = math.acos(sum_px/sum_p3)
   
    #
    # Case that the sum_theta_x is very small, the jet axis close to the x axis
    #
    if sum_theta < 10e-10:
        subset['jet_axis_theta'] = subset['particle_theta_x']
        subset['jet_axis_phi']   = subset['particle_phi_x']

    else:
        sum_part_vect = Vector(sum_px, sum_py, sum_pz)

        #
        # calculate delta theta bin: Try numpy here in order to make it faster
        #
        subset['jet_axis_theta'] = subset.apply(jet_axis_theta, axis = 1, args=(sum_part_vect, ))
        subset['jet_axis_phi']   = subset.apply(jet_axis_phi,   axis = 1, args=(sum_part_vect, ))
    
    #
    # Prepare theta bins
    #
    subset['x_bin'], subset['y_bin'] = zip(*subset.apply(getJetAxisXYBins, axis = 1, args=(x_bin_edges, y_bin_edges, )))

    out_sub_data  = np.zeros((1, len(x_bin_edges)+1, len(y_bin_edges)+1, 3))
    
    for (j, k), group in subset.groupby(['x_bin', 'y_bin']):
        sum_cell_e  = group['particle_energy'].sum()
        sum_cell_p3 = group['particle_p3'].sum()
        sum_cell_c  = group['particle_charge'].sum()
        
        out_sub_data[0, j, k][0] = sum_cell_e/sum_e
        out_sub_data[0, j, k][1] = sum_cell_p3/sum_p3
        out_sub_data[0, j, k][2] = sum_cell_c

    return out_sub_data


#======================================================================================================        
def main_trainCNN():

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
    # 1. Prepare training data for CNN 
    #
    train_file['particle_charge'] = train_file['particle_category'].apply(particle_charge)

    # get theta and phi
    part_vect = train_file[['particle_px', 'particle_py', 'particle_pz']].values
    train_file['particle_theta_x'] = get_theta(part_vect, axis='x')
    train_file['particle_phi_x']   = np.arctan2(-train_file['particle_pz'], -train_file['particle_py']) + np.pi
    train_file['particle_p3']      = np.sqrt((part_vect*part_vect).sum(1)) 

    jet_vect = jet_file[['jet_px', 'jet_py', 'jet_pz']].values
    jet_file['jet_theta_x'] = get_theta(jet_vect, axis='x')
    jet_file['jet_phi_x']   = np.arctan(jet_file['jet_pz'] / jet_file['jet_py'])
    
    #
    # 2. Select input variables
    #
    input_jetvar_names = ['number_of_particles_in_this_jet', 'jet_px', 'jet_py', 'jet_pz', 'jet_energy', 'jet_mass',
                          'jet_theta_x', 'jet_phi_x']

    #
    # 3. Prepare numpy array inputs to the keras
    #
    # most of theta/0.4 should with in this edges, but it is possible we have theta/0.4 > 1
    # impossible for the phi/2pi > 1.0 nor < 0

    # TODO: can try more bins like:
    x_bin_edges = [0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0]
    y_bin_edges = [0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0]
 
    id_list = train_file.drop_duplicates(subset=['jet_id'])['jet_id']
    nevt    = len(id_list)

    print("number of training jets = ", nevt)

    out_train_data  = np.zeros((nevt, len(x_bin_edges)+1, len(y_bin_edges)+1, 3))
    out_train_label = np.zeros((nevt))

    train_jets = train_file.groupby('jet_id', sort=True)

    if len(jet_file) != nevt:
        print("INFO - number of jets in jet file = %d, number of jet in particle file = %d"%(len(jet_file), nevt))
        jet_file = jet_file.head(nevt)

    #
    # Use joblib to speed up "for" loop
    #
    timePrev = time.time()

    out_sub_datas = Parallel(n_jobs=40)(delayed(joblib_process_particle_groups)(subset, x_bin_edges, y_bin_edges) for evt_id, subset in train_jets)

    # concatenate the out_sub_datas, len(out_sub_datas) == nevt 
    np.concatenate(out_sub_datas, axis=0, out=out_train_data)

    log.info('Processing delta t =  t=%.2fs' %(time.time() - timePrev))

    pd_jet_vars = jet_file[input_jetvar_names].values

    #
    # Training/Evaluation
    #
    if options.model:
        # load the given CNN model
        model = load_model(options.model)
    else:
        out_train_label = jet_file['label'].apply(sub_oneHotLabelFast)
        
        model = trainCNN(out_train_data, pd_jet_vars, out_train_label)

        saveModel(model, "Di_CNN_event_vars" )
    
    # 
    # Save the output to the jet file 
    # 
    saveCNNPrediction(model, [out_train_data, pd_jet_vars], jet_file, outkey="jet_with_CNN")    
        
#======================================================================================================        
if __name__ == '__main__':

    timeStart = time.time()

    log.info('Start job at %s:%s' %(socket.gethostname(), os.getcwd()))
    log.info('Current time: %s' %(time.asctime(time.localtime())))
   
    main_trainCNN()

    log.info('Local time: %s' %(time.asctime(time.localtime())))
    log.info('Total time: %.1fs' %(time.time()-timeStart))


