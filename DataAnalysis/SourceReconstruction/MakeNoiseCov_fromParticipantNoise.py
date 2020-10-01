#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make noise covariance datafile from preprocessed participant noise data

(*MNE-Python: ver. 0.17)

@author: Akinori Takeda
"""

import mne
import os

#--- setting data path ---#
filedir = ''
SubjID = 'Subject18'
ExpID = 'SF'
CovRankEst = True

os.chdir(filedir+'/'+SubjID+'/'+ExpID)
rootdir = os.getcwd()

PartNoise = 'ParticipantNoise_Preprocessed_trans_raw_tsss.fif'
PartNoiseData = mne.io.read_raw_fif(rootdir+'/Datafiles/ICAforConcatenatedRaws/'+PartNoise, preload=True)

sfreq=PartNoiseData.info['sfreq']
picks = mne.pick_types(PartNoiseData.info, meg=True, eeg=False, eog=False, stim=False, exclude='bads')


# Covariance calculation:
print('\n<< Calculating Covariance from Participant Noise Data >>')

if CovRankEst:
    rank = None
    newfile = 'NoiseCov_fromParticipantNoise-cov.fif'
else:
    rank = 'full'
    newfile = 'NoiseCov_fromParticipantNoise_FullRank-cov.fif'

NoiseCov = mne.compute_raw_covariance(PartNoiseData, picks=picks, method='shrunk',
                                      rank=rank)

# save data files
os.chdir(rootdir)
mne.write_cov(newfile, NoiseCov)
