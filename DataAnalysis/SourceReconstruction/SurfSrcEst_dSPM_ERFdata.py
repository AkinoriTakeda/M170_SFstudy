#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Surface Source estimation of evoked data (ERF) by dSPM

@author: Akinori Takeda
"""

import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse
import time
import os
locVal = locals()


#--- set data path & load MEG data ---#
filedir = ''
SubjID = 'Subject18'
ExpID = 'SF'


# load epoch data
os.chdir(filedir+'/'+SubjID+'/'+ExpID+'/Datafiles/EpochData')
Epochs = mne.read_epochs('ProperEpochData-epo.fif', preload=True)
epochs = Epochs.copy().pick_types(meg=True)
del Epochs

conditions = list(epochs.event_id.keys())


#- directory manipulation -#
os.chdir(filedir+'/'+SubjID+'/'+ExpID+'/Datafiles')
if not os.path.exists('./SourceEstimate'):
    os.mkdir('./SourceEstimate')
os.chdir('./SourceEstimate')

direcname = 'SurfSrcEst_dSPM_forEvoked'
if not os.path.exists('./'+direcname):
    os.mkdir('./'+direcname)
os.chdir('./'+direcname)
savedir = os.getcwd()
del direcname


#%%
#--- get covariance data ---#
# load covariance data
os.chdir(filedir+'/'+SubjID+'/'+ExpID)
NoiseCov = mne.read_cov('NoiseCov_fromParticipantNoise-cov.fif')


#--- get source space & forward solution data ---#
os.chdir(filedir+'/'+SubjID)

# load leadfield matrix
fwdfile = [i for i in os.listdir(os.getcwd()) if 'forSrcEst-fwd.fif' in i and 'SurfaceSourceSpace' in i][0]
fwd = mne.read_forward_solution(fwdfile)
fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=False)


#--- get MRI data ---#
# get MRI subject name
MRIsubject = ''

# set directory name of MRI data
subjects_dir = ''

# load SourceMorph instance (fsaverage ico5)
morph = mne.read_source_morph(filedir+'/'+SubjID+'/SrcMorph_%stoFsaverage-morph.h5' % SubjID)


#%%
#--- Source estimation by dSPM ---#
os.chdir(savedir)
for name in ['IndivBrain', 'Fsaverage']:
    if not os.path.exists('./'+name):
        os.mkdir('./'+name)
del name

preTmin = -0.1
preTmax = 0
conditions2 = [i for i in conditions if i != 'target']
epochs_basecorred = epochs[conditions2].copy().apply_baseline(baseline=(preTmin, preTmax))


# calculate filter weights
# make inverse operators:
print('[Make inverse operator]')
InvOperator = make_inverse_operator(epochs.info, forward=fwd, noise_cov=NoiseCov, 
                                    loose=0.2, depth=0.8, fixed=False, limit_depth_chs=False, 
                                    verbose=None) 

snr = 3.0
lambda2 = 1.0 / snr ** 2
method = "dSPM"  # use MNE method (could also be dSPM or sLORETA)
pick_ori = None # None: pooling the orientations


# apply filter weights to data & calculate source time courses
print('\n<< Estimating source time courses using surface source space model & dSPM >>')
nRep = 0
startT = time.time()
for cond in conditions2:
    print('< Processing %s condition data >' % (cond.replace('_','-')))
    
    # apply inverse operator
    stc = apply_inverse(epochs_basecorred[cond].copy().average(), InvOperator, 
                        lambda2, method=method, pick_ori=pick_ori)
    
    # morph data to fsaverage
    stc_morphed = morph.apply(stc)
    
    # save SourceEstimate instance
    print('\n> saving data...')
    os.chdir(savedir+'/IndivBrain')
    stc.save('SrcEst_MeanTC_%s' % cond)
    
    os.chdir(savedir+'/Fsaverage')
    stc_morphed.save('SrcEst_MeanTC_%s_fsaverage' % cond)
    
    print('   --> finished.\n')
    del stc, stc_morphed
    
    nRep += 1
    print('  => finished (%d/%d done).\n' % (nRep, len(conditions2)))
print('\n   ==> All processings were done!')
del cond, nRep

elapsed_time = time.time() - startT
print('       Elapsed_time: {0}'.format(elapsed_time)+" [sec]")


