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
import pandas as pd

import matplotlib

from optparse import OptionParser
p = OptionParser()

p.add_option('--outdir', '-o',   type='string',        default=None)
p.add_option('--outname',        type='string',        default='new_train_jet.csv')

p.add_option('--debug', '-d',    action='store_true',  default=False)
p.add_option('--do-add',         action='store_true',  default=False)

(options,args) = p.parse_args()

matplotlib.use('Agg')

import matplotlib.pyplot as plt # import pyplot AFTER setting batch backend

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
def saveFig(plt, name):
    
    outname = getOutName(name)

    if not outname:
        return

    log.info('saveFig - %s' %outname)
    
    plt.savefig(outname, format='pdf')
    
#======================================================================================================
def getOutName(name):
    
    if not options.outdir:
        return None
        
    outdir = '%s/' %(options.outdir.rstrip('/'))
        
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    return '%s/%s' %(outdir.rstrip('/'), name)

#======================================================================================================
def getClassName(iclass):
    class_names = {1: 'd',
                   4: 'c',
                   5: 'b',
                   21:'g'
                   }
    
    return class_names[iclass]


#======================================================================================================        
def getClassColor(class_name):

    colors = {'d': 'blue',
              'c': 'red',
              'b': 'magenta',
              'g': 'green',
              }

    return colors[class_name]

#======================================================================================================        
def getVarBinsMinMax(var_name):

    var_ranges = {'number_of_particles_in_this_jet' :(60,     0.0,  60),
                  'jet_px'                          :(100,   -500,  500),
                  'jet_py'                          :(100,   -80,   80),
                  'jet_pz'                          :(100,   -80,   80),
                  'jet_energy'                      :(100,   0.0,  550),
                  'jet_mass'                        :(100,   0.0,   60),
                  }

    result = var_ranges[var_name]

    return result

#======================================================================================================        
def getDataFrameRange(df_vars):
    
    var_max = None
    var_min = None

    for df_var in df_vars:
        if not var_max or df_var.max() > var_max:
            var_max = df_var.max()

        if not var_min or df_var.min() < var_min:
            var_min = df_var.min()

    return (var_max, var_min)

#======================================================================================================
# make plots
#======================================================================================================
def plotInputVars(fname):

    '''Plot accuracy and loss as function of epoch'''

    #
    # Plot 1d histogram with variable values
    #
    ifile  = np.genfromtxt(fname, dtype=float, delimiter=',', names=True)
    labels = ifile['label']

    for var_name in ['number_of_particles_in_this_jet', 'jet_px', 'jet_py', 'jet_pz', 'jet_energy', 'jet_mass']:

        plt.clf()

        nbin, xmin, xmax = getVarBinsMinMax(var_name)      
 
        for label in [1, 4, 5, 21]:
            class_name = getClassName(label)
            select_data = ifile[labels == label][var_name]
            
            color = getClassColor(class_name)
            phd = plt.hist(select_data.flatten(), range=(xmin, xmax), bins=nbin, histtype='step', color=color, label=class_name, density=True)

        plt.autoscale(enable=True, axis='y', tight=None)
        plt.legend(loc='upper right')
        plt.xlabel(var_name)
        plt.ylabel('Density')
        plt.grid(True)
        saveFig(plt, '%s.pdf' %(var_name))

#======================================================================================================
def plotInterestVars(fname):

    ''' Plot interesting variables '''

    #
    # Pandas is a useful tool !
    #
    ifile  = pd.read_csv(fname)
    labels = ifile['label']

    plt.clf()
    phd = plt.hist(labels, range=(0, 22), bins=22, histtype='step', color='red', label='label', density=True)
    plt.grid(True)

    saveFig(plt, '%s.pdf' %('label_all'))

    #
    # get 4 class indexes refs
    #
    class_d = ifile['label'] == 1
    class_c = ifile['label'] == 4
    class_b = ifile['label'] == 5
    class_g = ifile['label'] == 21
    
    class_dict = {1:class_d,
                  4:class_c,
                  5:class_b,
                  21:class_g
                  }
    # 
    # Plot pT 
    # 
    for var_name in ['jet_p3', 'jet_theta_x', 'jet_theta_y', 'jet_theta_z']:
    #for var_name in ['jet_pt_x', 'jet_pt_y', 'jet_phi_x', 'jet_phi_y', 'jet_phi_z']:
    #for var_name in ['jet_pt', 'jet_sin_theta', 'jet_tan_phi']:
        plt.clf()
        
        var_class_4 = []

        for label in [1, 4, 5, 21]:
            var_class_4 += [ifile[class_dict[label] ][var_name] ]

        var_max, var_min = getDataFrameRange(var_class_4)

        for label in [1, 4, 5, 21]: 
            class_name  = getClassName(label) 
            class_color = getClassColor(class_name) 
            select_data = ifile[class_dict[label] ][var_name]
            
            phd = plt.hist(select_data.values, range=(var_min, var_max), bins=100, histtype='step', color=class_color, label=class_name, density=True)
 
        plt.autoscale(enable=True, axis='y', tight=None)
        plt.legend(loc='upper right')
        plt.xlabel(var_name)
        plt.ylabel('Density')
        plt.grid(True)
        saveFig(plt, '%s.pdf' %(var_name))
      
    # Save the new DataFrame
    ifile.to_csv('new_train_jet.csv', index=False)


#======================================================================================================
def checkSameEventID(fname):

    ''' Plot interesting variables '''

    ifile  = pd.read_csv(fname)
    
    id_sort   = ifile.sort_values(by='event_id').drop_duplicates(subset=['event_id'])['event_id']
    id_labels = ifile[['event_id','label']]
    
    for evt_id in id_sort.values:
        init_label = -1

        subset_same_evtid = id_labels[id_labels['event_id'] == evt_id]['label'].values

        for label in subset_same_evtid:
            if init_label < 0:
                init_label = label
            elif init_label != label:
                print('ERROR: find the event id = %s with different label jet.'%evt_id)

        print('INFO: all jet in the event id = %s with same label = %d'%(evt_id, init_label) )
    

#======================================================================================================
# variable processing
#======================================================================================================
def jet_pt(df):
    px = df['jet_px']
    py = df['jet_py']

    return (px**2+py**2)**0.5

#======================================================================================================
def jet_pt_y(df):
    px = df['jet_px']
    pz = df['jet_pz']

    return (px**2+pz**2)**0.5

#======================================================================================================
def jet_pt_x(df):
    py = df['jet_py']
    pz = df['jet_pz']

    return (py**2+pz**2)**0.5

#======================================================================================================
def jet_energy(df):
    px = df['jet_px']
    py = df['jet_py']
    pz = df['jet_pz']
    m  = df['jet_mass']
    return (px**2 + py**2 + pz**2 + m**2)**0.5

#======================================================================================================
def jet_sin_theta(df):
    # cos theta 
    px = df['jet_px']
    py = df['jet_py']
    pz = df['jet_pz']
    return pz/(px**2 + py**2 + pz**2)**0.5

#======================================================================================================
def jet_p3(df):
    # cos theta 
    px = df['jet_px']
    py = df['jet_py']
    pz = df['jet_pz']
    return (px**2 + py**2 + pz**2)**0.5


#======================================================================================================
def jet_theta_x(df):
    px = df['jet_px']
    p3 = df['jet_p3']

    return math.acos(px/p3)

#======================================================================================================
def jet_theta_y(df):
    py = df['jet_py']
    p3 = df['jet_p3']

    return math.acos(py/p3)

#======================================================================================================
def jet_theta_z(df):
    pz = df['jet_pz']
    p3 = df['jet_p3']

    return math.acos(pz/p3)

#======================================================================================================
def jet_tan_phi(df):
    px = df['jet_px']
    py = df['jet_py']
    return py/px

#======================================================================================================
def jet_phi_z(df):
    px = df['jet_px']
    py = df['jet_py']
    return math.atan(py/px)

#======================================================================================================
def jet_phi_y(df):
    px = df['jet_px']
    pz = df['jet_pz']
    return math.atan(pz/px)

#======================================================================================================
def jet_phi_x(df):
    py = df['jet_py']
    pz = df['jet_pz']
    return math.atan(pz/py)

#======================================================================================================
def AddNewVarsToFile(fname):

    ''' Add interesting variables '''

    # 
    # Add pT 
    #
    ifile  = pd.read_csv(fname)

   # ifile['jet_pt_y'] = ifile.apply(jet_pt_y,     axis = 1)  # axis = 1 mean process on column; = 0 is on row
    ifile['jet_pt_x'] = ifile.apply(jet_pt_x,     axis = 1)  # axis = 1 mean process on column; = 0 is on row

   # ifile['jet_phi_x'] = ifile.apply(jet_phi_x,    axis = 1)  
   # ifile['jet_phi_y'] = ifile.apply(jet_phi_y,    axis = 1)  
   # ifile['jet_phi_z'] = ifile.apply(jet_phi_z,    axis = 1)  

    ifile['jet_p3'] = ifile.apply(jet_p3, axis = 1) # must add before theta calculation 
   
    ifile['jet_theta_x'] = ifile.apply(jet_theta_x, axis = 1)  
    #ifile['jet_theta_y'] = ifile.apply(jet_theta_y, axis = 1)  
    #ifile['jet_theta_z'] = ifile.apply(jet_theta_z, axis = 1)  
    
    # ifile['jet_energy2'] = ifile.apply(jet_energy, axis = 1)  
    #   -- find that the jet_energy2 == jet_energy
      
    # Save the new DataFrame
    out_name = getOutName(options.outname)
    ifile.to_csv(out_name, index=False)

#======================================================================================================
def main():

    if len(args) != 1:
        log.warning('Wrong number of command line arguments: %s' %len(args))
        return

    fname = args[0]

    if not os.path.isfile(fname):
        log.warning('Input file does not exist: %s' %fname)
        return    

    #plotInputVars(fname)      

    if options.do_add:
        AddNewVarsToFile(fname)
    
    else:
        checkSameEventID(fname)
        #plotInterestVars(fname)

#======================================================================================================
if __name__ == '__main__':

    timeStart = time.time()

    log.info('Start job at %s:%s' %(socket.gethostname(), os.getcwd()))
    log.info('Current time: %s' %(time.asctime(time.localtime())))

    main()

    log.info('Local time: %s' %(time.asctime(time.localtime())))
    log.info('Total time: %.1fs' %(time.time()-timeStart))
