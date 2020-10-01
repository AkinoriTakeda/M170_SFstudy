#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make Grand-average Data of Surface Source Estimate

@author: Akinori Takeda
"""

import mne
import numpy as np
import os
locVal = locals()


#--- set data path ---#
filedir = ''
ExpID = 'SF'
useFsaveModel = 'oct6'


# make subject list
SubjDir = [i for i in os.listdir(filedir) if 'Subject' in i and 'Test' not in i and '.' not in i]
rmvSubjList = []

SubjList = ['Subject%d' % (n+1) for n in np.arange(len(SubjDir)) if 'Subject%d' % (n+1) not in rmvSubjList and os.path.exists(filedir+'/Subject%d/SF' % (n+1))]
SubjN = len(SubjList)
del SubjDir


# setting some parameters
os.chdir(filedir+'/'+SubjList[0]+'/'+ExpID+'/Datafiles/EpochData')
Epochs = mne.read_epochs('ProperEpochData-epo.fif', preload=True)
epochs = Epochs.copy().pick_types(meg=True)

conditions = list(epochs.event_id.keys())
conditions2 = [i for i in conditions if i != 'target']
times = epochs.times
del Epochs, epochs


# directory manipulation
os.chdir(filedir+'/GrandAverage/Datafiles')
if not os.path.exists('./forSurfaceSrcEstAnalysis'):
    os.mkdir('./forSurfaceSrcEstAnalysis')
os.chdir('./forSurfaceSrcEstAnalysis')

srcdir1 = 'SurfSrcEst_dSPM_forEvoked'
if not os.path.exists('./'+srcdir1):
    os.mkdir('./'+srcdir1)
os.chdir('./'+srcdir1)

if useFsaveModel == 'ico5':
    dirname = '%sexp_N%d' % (ExpID, SubjN)
else:
    dirname = '%sexp_N%d_fsaverage%s' % (ExpID, SubjN, useFsaveModel.capitalize())
if not os.path.exists('./'+dirname):
    os.mkdir('./'+dirname)
os.chdir('./'+dirname)
savedir = os.getcwd()


#--- make grand-average data ---#
print('\n<< Make Grand-average Datasets (%s Exp, N=%d) >>' % (ExpID, SubjN))
for i, SubjID in enumerate(SubjList):
    print('< %s data >' % SubjID)
    if useFsaveModel == 'ico5':
        os.chdir(filedir+'/'+SubjID+'/'+ExpID+'/Datafiles/SourceEstimate/'+srcdir1+'/Fsaverage')
    else:
        os.chdir(filedir+'/'+SubjID+'/'+ExpID+'/Datafiles/SourceEstimate/'+srcdir1+'/Fsaverage_%s' % useFsaveModel)
    sourcedir = os.getcwd()
    
    # make datasets
    print('[make datasets]')
    for ii, cond in enumerate(conditions2):
        print(' > processing %s data...' % cond)
        # [Processing Surface source data]
        # make container if not exists
        if i == 0:
            exec('SurfData_%s = []' % cond)
        
        # load data
        if useFsaveModel == 'ico5':
            surfdataname = 'SrcEst_MeanTC_%s_fsaverage' % cond
        else:
            surfdataname = 'SrcEst_MeanTC_%s_fsaverage%s' % (cond, useFsaveModel.capitalize())
        SurfData = mne.read_source_estimate(sourcedir+'/'+surfdataname)
        
        # add to container
        locVal['SurfData_%s' % cond].append(SurfData.data)
        del surfdataname, SurfData
    del ii, cond, sourcedir
    print(' => Finished.\n')
del i, SubjID


# convert to numpy array data & save datasets
print('<< Saving datasets >>')
os.chdir(savedir)
for cond in conditions2:
    print(' > processing %s data...' % cond)
    locVal['SurfData_%s' % cond] = np.array(locVal['SurfData_%s' % cond])
    np.save('SurfData_%s.npy' % cond, locVal['SurfData_%s' % cond])
del cond
print(' => Finished.')
