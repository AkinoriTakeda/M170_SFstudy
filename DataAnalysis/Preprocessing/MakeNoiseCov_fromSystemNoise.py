#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make noise covariance datafile from system noise data for ICA in preprocessing
(*MNE-Python ver.0.17 was used)

@author: Akinori Takeda
"""

import mne
import numpy as np
import os

#--- setting data path ---#
filedir = ''
SubjID = 'Subject18'
ExpID = 'SF'
CovRankEst = True

os.chdir(filedir+'/'+SubjID+'/'+ExpID)
rootdir = os.getcwd()

EmpRoom = [i for i in os.listdir(rootdir+'/Datafiles/OTPprocessed') if 'SystemNoise' in i and 'tsss' not in i][0]
EmpRoomData = mne.io.read_raw_fif(rootdir+'/Datafiles/OTPprocessed/'+EmpRoom, preload=True)
sfreq=EmpRoomData.info['sfreq']

EmpRoomData.filter(1.0, 200.0)
EmpRoomData.notch_filter(np.arange(60.0, 301.0, 60))
picks = mne.pick_types(EmpRoomData.info, meg=True, eeg=False, eog=False, stim=False, exclude='bads')


# Covariance calculation:
tmax = ((EmpRoomData.last_samp-EmpRoomData.first_samp+1)/sfreq)-1
tmin = tmax-120
print('\ntime range used for calculation of covariance: %d ~ %d sec' % (tmin, tmax))

if CovRankEst:
    rank = None
else:
    rank = 'full'

NoiseCov = mne.compute_raw_covariance(EmpRoomData, tmin=tmin, tmax=tmax, picks=picks, 
                                      method='shrunk', rank=rank)


# save data files
os.chdir(rootdir)
newfile = 'NoiseCov_fromEmptyRoom-cov.fif'
mne.write_cov(newfile, NoiseCov)
