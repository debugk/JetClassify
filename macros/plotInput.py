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

from optparse import OptionParser
p = OptionParser()

p.add_option('--outdir', '-o',   type='string',        default=None)
p.add_option('--key',    '-k',   type='string',        default=None)
p.add_option('--min-nevent',     type='int',           default=400)
p.add_option('--nclass',         type='int',           default=3)

p.add_option('--debug', '-d',    action='store_true',  default=False)
p.add_option('--batch', '-b',    action='store_true',  default=False)
p.add_option('--show',  '-s',    action='store_true',  default=False)
p.add_option('--show-extra',     action='store_true',  default=False)
p.add_option('--predict',        action='store_true',  default=False)

p.add_option('--do-all-rocs',    action='store_true',  default=False)
p.add_option('--no-roc',         action='store_true',  default=False)
p.add_option('--no-acc',         action='store_true',  default=False)
p.add_option('--no-rnn',         action='store_true',  default=False)

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

    if options.show:
        plt.show()


#======================================================================================================
def main():

    if len(args) != 1:
        log.warning('Wrong number of command line arguments: %s' %len(args))
        return

    fname = args[0]

    if not os.path.isfile(fname):
        log.warning('Input file does not exist: %s' %fname)
        return    

    plotInputVars(fname)      
  

#======================================================================================================
if __name__ == '__main__':

    timeStart = time.time()

    log.info('Start job at %s:%s' %(socket.gethostname(), os.getcwd()))
    log.info('Current time: %s' %(time.asctime(time.localtime())))

    main()

    log.info('Local time: %s' %(time.asctime(time.localtime())))
    log.info('Total time: %.1fs' %(time.time()-timeStart))
