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
p.add_option('--inputjet',       type='string',        default=None)

p.add_option('--debug', '-d',    action='store_true',  default=False)
p.add_option('--do-add',         action='store_true',  default=False)
p.add_option('--do-event',       action='store_true',  default=False)
p.add_option('--do-part',        action='store_true',  default=False)

(options,args) = p.parse_args()

matplotlib.use('Agg')

jetid_lables_dict = {}

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
class Vector:

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        
        self.l = (self.x**2 + self.y**2 + self.z**2)**0.5

        self.theta = math.acos(self.x/self.l)
        self.phi   = math.atan(self.z/self.y)

        self.eta   = - math.log(math.tan(self.theta/2) )

    def getDR(self, vect):
        delta_eta = self.eta - vect.eta 
        delta_phi = self.phi - vect.phi 
        return (delta_phi**2 + delta_eta**2)**0.5

    def getTheta(self, vect):
        cos_theta = (self.x*vect.x + self.y*vect.y + self.z*vect.z)/self.l/vect.l

        if not abs(cos_theta) <= 1:
            print("getTheta - error - find cos_theta = %s"%cos_theta)

            if cos_theta > 0:
                return 0
            else:
                return math.pi

        return math.acos(cos_theta)

    def getTransversal(self):
        return (self.z**2 + self.y**2)**0.5

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
    ifile['jet_eta_x'] = ifile.apply(jet_eta_x,     axis = 1)  # axis = 1 mean process on column; = 0 is on row
    for var_name in ['jet_eta_x']:
    #for var_name in ['jet_p3', 'jet_theta_x', 'jet_theta_y', 'jet_theta_z']:
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
def plotEventVars(fname):

    ''' Plot interesting variables '''
    
    ifile  = pd.read_csv(fname)   
    events = ifile.groupby('event_id')

    pd_event_vars = pd.DataFrame()
    
#    pd_event_vars['event_mass'] = events.apply(event_mass)
#    pd_event_vars['event_njet'] = events.apply(event_njet)
#    pd_event_vars['event_pt_x'] = events.apply(event_pt_x)
#    pd_event_vars['event_pt_y'] = events.apply(event_pt_y)
#    pd_event_vars['event_pt_z'] = events.apply(event_pt_z)

#    pd_event_vars['event_p3']      = events.apply(event_p3)
#    pd_event_vars['event_theta_x'] = events.apply(event_theta_x)
#    pd_event_vars['event_theta_y'] = events.apply(event_theta_y)
#    pd_event_vars['event_theta_z'] = events.apply(event_theta_z)

    pd_event_vars['event_px'] = events.apply(event_px)
    pd_event_vars['event_py'] = events.apply(event_py)
    pd_event_vars['event_pz'] = events.apply(event_pz)

    pd_event_vars['label'] = events.apply(event_label)
    
    plt.clf()
    phd = plt.hist(pd_event_vars['label'], range=(0, 22), bins=22, histtype='step', color='red', label='event_label', density=True)
    plt.grid(True)

    saveFig(plt, '%s.pdf' %('label_event'))
    
    #
    # get 4 class indexes refs
    #
    class_d = pd_event_vars['label'] == 1
    class_c = pd_event_vars['label'] == 4
    class_b = pd_event_vars['label'] == 5
    class_g = pd_event_vars['label'] == 21
    
    class_dict = {1:class_d,
                  4:class_c,
                  5:class_b,
                  21:class_g
                  }
    # 
    # Plot pT 
    # 
    for var_name in ['event_px', 'event_py', 'event_pz']:
    #for var_name in ['event_theta_x', 'event_theta_y', 'event_theta_z']:
    #for var_name in ['event_pt_x', 'event_pt_y', 'event_pt_z']:
    #for var_name in ['event_mass', 'event_njet']:
        plt.clf()
        
        var_class_4 = []

        for label in [1, 4, 5, 21]:
            var_class_4 += [pd_event_vars[class_dict[label] ][var_name] ]

        var_max, var_min = getDataFrameRange(var_class_4)

        for label in [1, 4, 5, 21]: 
            class_name  = getClassName(label) 
            class_color = getClassColor(class_name) 
            select_data = pd_event_vars[class_dict[label] ][var_name]
            
            phd = plt.hist(select_data.values, range=(var_min, var_max), bins=100, histtype='step', color=class_color, label=class_name, density=True)
 
        plt.autoscale(enable=True, axis='y', tight=None)
        plt.legend(loc='upper right')
        plt.xlabel(var_name)
        plt.ylabel('Density')
        plt.grid(True)
        saveFig(plt, '%s.pdf' %(var_name))
      
    # Save the new DataFrame
    pd_event_vars.to_csv('new_event_jet.csv', index=False)


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
def jet_eta_x(df):
    px = df['jet_px']
    p3 = df['jet_p3']
    theta = math.acos(px/p3)

    return - math.log(math.tan(theta/2) )

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
def event_label(df):
    label = df['label'].sum()/len(df['label'])
    return label

#======================================================================================================
def event_njet(df):
    return len(df['label'])

#======================================================================================================
def event_px(df):
    px = df['jet_px'].sum()
    return px

#======================================================================================================
def event_py(df):
    py = df['jet_py'].sum()
    return py

#======================================================================================================
def event_pz(df):
    pz = df['jet_pz'].sum()
    return pz

#======================================================================================================
def event_pt_x(df):
    py = df['jet_py'].sum()
    pz = df['jet_pz'].sum()
    return (py**2 + pz**2)**0.5

#======================================================================================================
def event_pt_y(df):
    px = df['jet_px'].sum()
    pz = df['jet_pz'].sum()
    return (px**2 + pz**2)**0.5

#======================================================================================================
def event_pt_z(df):
    px = df['jet_px'].sum()
    py = df['jet_py'].sum()
    return (px**2 + py**2)**0.5

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
    #p3 = df['event_p3']
    if p3 < 1e-15:
        return 0    

    return math.acos(px/p3)

#======================================================================================================
def event_theta_y(df):
    py = df['jet_py'].sum()
    p3 = event_p3(df)
    #p3 = df['event_p3']
    if p3 < 1e-15:
        return 0

    return math.acos(py/p3)

#======================================================================================================
def event_theta_z(df):
    pz = df['jet_pz'].sum()
    p3 = event_p3(df)
    #p3 = df['event_p3']
    if p3 < 1e-15:
        return 0
    
    return math.acos(pz/p3)

#======================================================================================================
# Particle variable processing     
#======================================================================================================
def part_jet_e(df):
    e = df['particle_energy'].sum()
    return e

#======================================================================================================
def part_jet_cone_frac_dr(df):
    jet_px = df['particle_px'].sum()
    jet_py = df['particle_py'].sum()
    jet_pz = df['particle_pz'].sum()

    jet_vector = Vector(jet_px, jet_py, jet_pz)
    
    jet_e  = df['particle_energy'].sum() 
    jet_pt = jet_vector.getTransversal()

    e_dr01 = 0
    e_dr02 = 0
    e_dr03 = 0
    e_dr04 = 0

    pt_dr01 = 0
    pt_dr02 = 0
    pt_dr03 = 0
    pt_dr04 = 0

    for i in range(len(df)):
        part_vector = Vector(df.iloc[i]['particle_px'], df.iloc[i]['particle_py'], df.iloc[i]['particle_pz'])
        dr = part_vector.getDR(jet_vector)

        if dr < 1:
            e_dr01  += df.iloc[i]['particle_energy']
            pt_dr01 += part_vector.getTransversal()
    
        if dr < 2:
            e_dr02  += df.iloc[i]['particle_energy']
            pt_dr02 += part_vector.getTransversal()
    
        if dr < 3:
            e_dr03  += df.iloc[i]['particle_energy']
            pt_dr03 += part_vector.getTransversal()
    
        if dr < 4:
            e_dr04  += df.iloc[i]['particle_energy']
            pt_dr04 += part_vector.getTransversal()

    return (e_dr01/jet_e,   e_dr02/jet_e,   e_dr03/jet_e,   e_dr04/jet_e,
            pt_dr01/jet_pt, pt_dr02/jet_pt, pt_dr03/jet_pt, pt_dr04/jet_pt)

#======================================================================================================
def part_jet_n11(df):
    p_11 = (df['particle_category_abs'] == 11).sum()

    return p_11

#======================================================================================================
def part_jet_n13(df):
    p_13 = (df['particle_category_abs'] == 13).sum()

    return p_13

#======================================================================================================
def part_jet_n321(df):
    p_321 = (df['particle_category_abs'] == 321).sum()

    return p_321

#======================================================================================================
def part_jet_n130(df):
    p_130 = (df['particle_category_abs'] == 130).sum()

    return p_130

#======================================================================================================
def part_jet_n211(df):
    p_211 = (df['particle_category_abs'] == 211).sum()

    return p_211

#======================================================================================================
def part_jet_n321_and_n211(df):
    is_321_id = (df['particle_category_abs'] == 321).sum()
    is_211_id = (df['particle_category_abs'] == 211).sum()

    if is_321_id > 0 and is_211_id > 0:
        return is_321_id + is_211_id
    else:
        return -(is_321_id + is_211_id)

#======================================================================================================
def part_jet_mass_11(df):
    is_same_id = df['particle_category_abs'] == 11

    if len(is_same_id) == 0:
        return -10

    e  = df[is_same_id]['particle_energy'].sum()
    px = df[is_same_id]['particle_px'].sum()
    py = df[is_same_id]['particle_py'].sum()
    pz = df[is_same_id]['particle_pz'].sum()
    
    return abs(e**2 - px**2 - py**2 -pz**2)**0.5

#======================================================================================================
def part_jet_mass_13(df):
    is_same_id = df['particle_category_abs'] == 13

    if len(is_same_id) == 0:
        return -10

    e  = df[is_same_id]['particle_energy'].sum()
    px = df[is_same_id]['particle_px'].sum()
    py = df[is_same_id]['particle_py'].sum()
    pz = df[is_same_id]['particle_pz'].sum()
    
    return abs(e**2 - px**2 - py**2 -pz**2)**0.5

#======================================================================================================
def part_jet_D_mass(df):
    is_321_id = df['particle_category_abs'] == 321
    is_211_id = df['particle_category_abs'] == 211

    if len(is_321_id) == 0:
        return -10

    e  = df[is_321_id | is_211_id]['particle_energy'].sum()
    px = df[is_321_id | is_211_id]['particle_px'].sum()
    py = df[is_321_id | is_211_id]['particle_py'].sum()
    pz = df[is_321_id | is_211_id]['particle_pz'].sum()
    
    mass = abs(e**2 - px**2 - py**2 -pz**2)**0.5

    return mass

#======================================================================================================
def part_jet_D_mass_v2(df):
    is_321_id = df['particle_category'] == 321
    is_211_id = df['particle_category'] == -211

    kaon_df = df[is_321_id]
    pion_df = df[is_211_id]

    for index_k in range(len(kaon_df)):
        kaon = kaon_df.iloc[index_k]

        for index_pi in range(len(pion_df)):
            pion = pion_df.iloc[index_pi]
            
            e  = kaon['particle_energy'] + pion['particle_energy']
            px = kaon['particle_px'] + pion['particle_px'] 
            py = kaon['particle_py'] + pion['particle_py']
            pz = kaon['particle_pz'] + pion['particle_pz'] 
    
            mass = abs(e**2 - px**2 - py**2 -pz**2)**0.5

            if mass < 2.1 and mass > 1.7:
                return mass

    return 0 

#======================================================================================================
def part_jet_D_mass_v3(df):
    is_321_id = df['particle_category'] == -321
    is_211_id = df['particle_category'] == 211

    kaon_df = df[is_321_id]
    pion_df = df[is_211_id]

    for index_k in range(len(kaon_df)):
        kaon = kaon_df.iloc[index_k]

        for index_pi in range(len(pion_df)):
            pion = pion_df.iloc[index_pi]
            
            e  = kaon['particle_energy'] + pion['particle_energy']
            px = kaon['particle_px'] + pion['particle_px'] 
            py = kaon['particle_py'] + pion['particle_py']
            pz = kaon['particle_pz'] + pion['particle_pz'] 
    
            mass = abs(e**2 - px**2 - py**2 -pz**2)**0.5

            if mass < 2.1 and mass > 1.7:
                return mass

    return 0 

#======================================================================================================
def part_jet_count_D_mass_p(df):
    is_321_id = df['particle_category'] == -321
    is_211_id = df['particle_category'] == 211

    kaon_df = df[is_321_id]
    pion_df = df[is_211_id]

    count = 0

    for index_k in range(len(kaon_df)):
        kaon = kaon_df.iloc[index_k]

        for index_pi in range(len(pion_df)):
            pion = pion_df.iloc[index_pi]
            
            e  = kaon['particle_energy'] + pion['particle_energy']
            px = kaon['particle_px'] + pion['particle_px'] 
            py = kaon['particle_py'] + pion['particle_py']
            pz = kaon['particle_pz'] + pion['particle_pz'] 
    
            mass = abs(e**2 - px**2 - py**2 -pz**2)**0.5

            if mass < 1.87 and mass > 1.86:
                count += 1

    return count

#======================================================================================================
def part_jet_count_D_mass_n(df):
    is_321_id = df['particle_category'] == 321
    is_211_id = df['particle_category'] == -211

    kaon_df = df[is_321_id]
    pion_df = df[is_211_id]

    count = 0

    for index_k in range(len(kaon_df)):
        kaon = kaon_df.iloc[index_k]

        for index_pi in range(len(pion_df)):
            pion = pion_df.iloc[index_pi]
            
            e  = kaon['particle_energy'] + pion['particle_energy']
            px = kaon['particle_px'] + pion['particle_px'] 
            py = kaon['particle_py'] + pion['particle_py']
            pz = kaon['particle_pz'] + pion['particle_pz'] 
    
            mass = abs(e**2 - px**2 - py**2 -pz**2)**0.5

            if mass < 1.87 and mass > 1.86:
                count += 1

    return count


#======================================================================================================
def part_jet_count_j_mass_e(df):
    p_e = df['particle_category'] == 11
    n_e = df['particle_category'] == -11

    pe_df = df[p_e]
    ne_df = df[n_e]

    count = 0

    for index_pe in range(len(pe_df)):
        pe = pe_df.iloc[index_pe]

        for index_ne in range(len(ne_df)):
            ne = ne_df.iloc[index_ne]
            
            e  = pe['particle_energy'] + ne['particle_energy']
            px = pe['particle_px'] + ne['particle_px'] 
            py = pe['particle_py'] + ne['particle_py']
            pz = pe['particle_pz'] + ne['particle_pz'] 
    
            mass = abs(e**2 - px**2 - py**2 -pz**2)**0.5

            if mass < 3.099 and mass > 3.095:
                count += 1

    return count

#======================================================================================================
def part_jet_count_j_mass_m(df):
    p_m = df['particle_category'] == 13
    n_m = df['particle_category'] == -13

    pm_df = df[p_m]
    nm_df = df[n_m]

    count = 0

    for index_pm in range(len(pm_df)):
        pm = pm_df.iloc[index_pm]

        for index_nm in range(len(nm_df)):
            nm = nm_df.iloc[index_nm]
            
            e  = pm['particle_energy'] + nm['particle_energy']
            px = pm['particle_px'] + nm['particle_px'] 
            py = pm['particle_py'] + nm['particle_py']
            pz = pm['particle_pz'] + nm['particle_pz'] 
    
            mass = abs(e**2 - px**2 - py**2 -pz**2)**0.5

            if mass < 3.099 and mass > 3.095:
                count += 1

    return count


#======================================================================================================
def part_jet_cone_frac_theta(df):
    jet_px = df['particle_px'].sum()
    jet_py = df['particle_py'].sum()
    jet_pz = df['particle_pz'].sum()

    jet_vector = Vector(jet_px, jet_py, jet_pz)
    
    jet_e  = df['particle_energy'].sum() 
    jet_pt = jet_vector.getTransversal()

    e_theta01 = 0
    e_theta02 = 0
    e_theta03 = 0
    e_theta04 = 0

    pt_theta01 = 0
    pt_theta02 = 0
    pt_theta03 = 0
    pt_theta04 = 0

    for i in range(len(df)):
        part_vector = Vector(df.iloc[i]['particle_px'], df.iloc[i]['particle_py'], df.iloc[i]['particle_pz'])
        theta = part_vector.getTheta(jet_vector)

        if theta < 0.1:
            e_theta01  += df.iloc[i]['particle_energy']
            pt_theta01 += part_vector.getTransversal()
    
        if theta < 0.2:
            e_theta02  += df.iloc[i]['particle_energy']
            pt_theta02 += part_vector.getTransversal()
    
        if theta < 0.3:
            e_theta03  += df.iloc[i]['particle_energy']
            pt_theta03 += part_vector.getTransversal()
    
        if theta < 0.4:
            e_theta04  += df.iloc[i]['particle_energy']
            pt_theta04 += part_vector.getTransversal()

    return (e_theta01/jet_e,   e_theta02/jet_e,   e_theta03/jet_e,   e_theta04/jet_e, 
            pt_theta01/jet_pt, pt_theta02/jet_pt, pt_theta03/jet_pt, pt_theta04/jet_pt)


#======================================================================================================
def AddNewVarsToFile(fname):

    ''' Add interesting variables '''

    # 
    # Add pT 
    #
    ifile  = pd.read_csv(fname)

   # ifile['jet_pt_y'] = ifile.apply(jet_pt_y,     axis = 1)  # axis = 1 mean process on column; = 0 is on row
    ifile['jet_pt_x'] = ifile.apply(jet_pt_x,     axis = 1)  

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
# Particle variable processing
#======================================================================================================
def plotJetWithParticleVars(fname):
    ''' Plot particle variables '''

    ifile  = pd.read_csv(fname)
    ifile['particle_category_abs'] = ifile['particle_category'].apply(abs)

    jets   = ifile.groupby('jet_id')

    pd_jets_vars = pd.DataFrame()

    #pd_jets_vars['part_jet_e'] = jets.apply(part_jet_e)
    #pd_jets_vars['part_jet_n11'] = jets.apply(part_jet_n11)
    #pd_jets_vars['part_jet_n13'] = jets.apply(part_jet_n13)
#    pd_jets_vars['part_jet_n321'] = jets.apply(part_jet_n321)
#    pd_jets_vars['part_jet_n130'] = jets.apply(part_jet_n130)
#    pd_jets_vars['part_jet_n211'] = jets.apply(part_jet_n211)
#    pd_jets_vars['part_jet_D_mass'] = jets.apply(part_jet_D_mass)
#    pd_jets_vars['part_jet_n321_and_n211'] = jets.apply(part_jet_n321_and_n211)
#    pd_jets_vars['part_jet_mass_11'] = jets.apply(part_jet_mass_11)
#    pd_jets_vars['part_jet_mass_13'] = jets.apply(part_jet_mass_13)
    pd_jets_vars['part_jet_count_D_mass_p'] = jets.apply(part_jet_count_D_mass_p)
    pd_jets_vars['part_jet_count_D_mass_n'] = jets.apply(part_jet_count_D_mass_n)
    pd_jets_vars['part_jet_count_j_mass_e'] = jets.apply(part_jet_count_j_mass_e)
    pd_jets_vars['part_jet_count_j_mass_m'] = jets.apply(part_jet_count_j_mass_m)

    #(pd_jets_vars['part_jet_e_frac_dr01'], 
    # pd_jets_vars['part_jet_e_frac_dr02'],
    # pd_jets_vars['part_jet_e_frac_dr03'],
    # pd_jets_vars['part_jet_e_frac_dr04'],
    # pd_jets_vars['part_jet_pt_frac_dr01'], 
    # pd_jets_vars['part_jet_pt_frac_dr02'],
    # pd_jets_vars['part_jet_pt_frac_dr03'],
    # pd_jets_vars['part_jet_pt_frac_dr04']
    # ) = zip(*jets.apply(part_jet_cone_frac_dr))

    #(pd_jets_vars['part_jet_e_frac_theta01'], 
    # pd_jets_vars['part_jet_e_frac_theta02'],
    # pd_jets_vars['part_jet_e_frac_theta03'],
    # pd_jets_vars['part_jet_e_frac_theta04'],
    # pd_jets_vars['part_jet_pt_frac_theta01'], 
    # pd_jets_vars['part_jet_pt_frac_theta02'],
    # pd_jets_vars['part_jet_pt_frac_theta03'],
    # pd_jets_vars['part_jet_pt_frac_theta04']
    # ) = zip(*jets.apply(part_jet_cone_frac_theta))

    if not options.inputjet:
        return 

    jet_file = pd.read_csv(options.inputjet)
    
    merge_result = pd.merge(pd_jets_vars, jet_file, on=['jet_id'], how='inner')

    #merge_result['diff_e'] = merge_result.apply(lambda x: x['jet_energy'] - x['part_jet_e'], axis = 1)
    #==================================================================
    # MERGE is useful, otherwise you can do as below, very ... ugly:
    #    - if options.inputjet:
    #    -     jet_file = pd.read_csv(options.inputjet)
    #    - 
    #    -     global jetid_lables_dict 
    #    -     jetid_lables_dict = dict(jet_file[['jet_id', 'label']].values) 
    #    - 
    #    - pd_jets_vars['label'] = pd_jets_vars['jet_id'].apply(lambda x: jetid_lables_dict[x])
    #==================================================================

    #
    # get 4 class indexes refs
    #
    class_d = merge_result['label'] == 1
    class_c = merge_result['label'] == 4
    class_b = merge_result['label'] == 5
    class_g = merge_result['label'] == 21
    
    class_dict = {1:class_d,
                  4:class_c,
                  5:class_b,
                  21:class_g
                  }
    # 
    # Plot 
    #
   # plots  = ['part_jet_pt_frac_dr01', 'part_jet_pt_frac_dr02', 'part_jet_pt_frac_dr03', 'part_jet_pt_frac_dr04']
   # plots += ['part_jet_pt_frac_theta01', 'part_jet_pt_frac_theta02', 'part_jet_pt_frac_theta03', 'part_jet_pt_frac_theta04']
   # plots += ['part_jet_e_frac_dr01', 'part_jet_e_frac_dr02', 'part_jet_e_frac_dr03', 'part_jet_e_frac_dr04']
   # plots += ['part_jet_e_frac_theta01', 'part_jet_e_frac_theta02', 'part_jet_e_frac_theta03', 'part_jet_e_frac_theta04']

    plots = ['part_jet_count_D_mass_p', 'part_jet_count_D_mass_n', 'part_jet_count_j_mass_e', 'part_jet_count_j_mass_m']
    #plots = ['part_jet_n321', 'part_jet_n130', 'part_jet_n211', 'part_jet_D_mass', 'part_jet_n321_and_n211']
    #plots = ['part_jet_n11', 'part_jet_n13', 'part_jet_mass_13', 'part_jet_mass_11']

    for var_name in plots:
        plt.clf()
        
        var_class_4 = []

        for label in [1, 4, 5, 21]:
            var_class_4 += [merge_result[class_dict[label] ][var_name] ]

        var_max, var_min = getDataFrameRange(var_class_4)

        for label in [1, 4, 5, 21]: 
            class_name  = getClassName(label) 
            class_color = getClassColor(class_name) 
            select_data = merge_result[class_dict[label] ][var_name]
            
            phd = plt.hist(select_data.values, range=(var_min, var_max), bins=100, histtype='step', color=class_color, label=class_name, density=True)
 
        plt.autoscale(enable=True, axis='y', tight=None)
        plt.legend(loc='upper right')
        plt.xlabel(var_name)
        plt.ylabel('Density')
        plt.grid(True)
        saveFig(plt, '%s.pdf' %(var_name))

        plt.yscale('log')
        saveFig(plt, '%s_logy.pdf' %(var_name))


    out_name = getOutName(options.outname)
    merge_result.to_csv(out_name, index=False)

    return
    

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
    
    elif options.do_event:
        plotEventVars(fname)

    elif options.do_part:
        plotJetWithParticleVars(fname)

    else:
        #checkSameEventID(fname)
        plotInterestVars(fname)

#======================================================================================================
if __name__ == '__main__':

    timeStart = time.time()

    log.info('Start job at %s:%s' %(socket.gethostname(), os.getcwd()))
    log.info('Current time: %s' %(time.asctime(time.localtime())))

    main()

    log.info('Local time: %s' %(time.asctime(time.localtime())))
    log.info('Total time: %.1fs' %(time.time()-timeStart))
