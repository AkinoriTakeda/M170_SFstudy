#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Within-trial comparisons (sensor data)

@author: Akinori Takeda
"""

import mne
import numpy as np
import time
import os
locVal = locals()


#--- set data path & get datafiles' name ---#
filedir = ''

direcname = 'Gradiometer'
ExpID = 'SF'

#- make directory for saving results -#
os.chdir(filedir+'/GrandAverage/Datafiles')
if not os.path.exists('./StatisticalData'):
    os.makedirs('./StatisticalData/SensorData/'+ExpID+'/'+direcname)
os.chdir('./StatisticalData/SensorData/'+ExpID+'/'+direcname)
savedir = os.getcwd()

if not os.path.exists('./WithinTrial'):
    os.mkdir('./WithinTrial')
os.chdir('./WithinTrial')
savedir += '/WithinTrial'


#- parameter setting -#
Tmin = 0
Tmax = 0.35
Nperm = 1000
tail = 0
njobs = 4


# for TFCE
thresh_tfce = dict(start=0, step=0.2)


#- get file names -#
datadir = filedir+'/GrandAverage/Datafiles/forERFanalysis'
files = os.listdir(datadir+'/'+ExpID+'/'+direcname)
filenames = [i for i in files if 'target' not in i]


#--- Do Cluster-based permutation test ---#
print('\n< Statistical analysis: within-condition comparison >')
startT_all = time.time()
for filename in filenames:
    condname = filename.split('-')[0].split('_')
    cond = condname[1] + '_' + condname[2]
    
    # load data
    os.chdir(datadir+'/'+ExpID+'/'+direcname)
    print('[%s data]' % cond.replace('_', '-'))
    Data = mne.read_epochs(filename, preload=True)
    
    # make dataset used
    times = Data.times
    
    sfreq = Data.info['sfreq']
    timemask_pre = np.where(((-Tmax-1/sfreq) <= times)&(times <= (Tmin-1/sfreq)))[0]
    timemask_post = np.where((Tmin <= times)&(times <= Tmax))[0]
    
    data_pre = Data.copy().get_data()[:,:,timemask_pre]
    data_post = Data.copy().get_data()[:,:,timemask_post]
    X = data_post.transpose([0,2,1]) - data_pre.transpose([0,2,1])
    
    # connectivity setting
    connectivity, ch_names = mne.channels.find_ch_connectivity(Data.info, ch_type='grad')
    
    # do cluster-based permutation test (using 1-sample t-test)
    startT = time.time()
    t_obs, _, p_val, H0 = mne.stats.permutation_cluster_1samp_test(X, threshold=thresh_tfce, n_permutations=Nperm, tail=tail, 
                                                                   connectivity=connectivity, n_jobs=njobs, buffer_size=None)
    elapsed_time = (time.time() - startT)/60
    p_val = p_val.reshape(t_obs.shape)
    
    # save results
    os.chdir(savedir)
    if not os.path.exists('./'+cond):
        os.mkdir('./'+cond)
    os.chdir('./'+cond)
    
    np.save('ObservedTstatistic.npy', t_obs)
    np.save('Pvalues.npy', p_val)
    np.save('ObservedClusterLevelStats.npy', H0)
    
    print('  -> Finished.')
    print('     Elapsed_time: {0}'.format(elapsed_time)+" [min]\n")
    del condname, cond, Data, times, timemask_post, data_post, X, t_obs, p_val, H0, startT, elapsed_time
del filename, filenames

elapsed_time_all = (time.time() - startT_all)/3600
print('\n  ==> All processes took {0}'.format(elapsed_time_all)+" hours.\n")


