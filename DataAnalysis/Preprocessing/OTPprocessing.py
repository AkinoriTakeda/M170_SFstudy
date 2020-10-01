#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raw data processing by oversampled temporal projection (OTP)

@author: Akinori Takeda
"""

import mne 
from mne.preprocessing import oversampled_temporal_projection
import time
import shutil
import os

# change working directory
filedir = ''
SubjID = 'Subject18'
ExpID = 'SF'
os.chdir(filedir+'/'+SubjID+'/'+ExpID)

if os.path.exists('./Datafiles'):
    filelist = [i for i in os.listdir(os.getcwd()+'/Datafiles/OriginalData') if '_raw.fif' in i]
else:
    filelist = [i for i in os.listdir(os.getcwd()) if '_raw.fif' in i]
filelist.sort()

# file management
if not os.path.exists('./Datafiles'):
    os.mkdir('./Datafiles')
rootdir = os.getcwd()+'/Datafiles'

os.chdir(rootdir)
if not os.path.exists('./OriginalData'):
    os.mkdir('./OriginalData')
    
    os.chdir(filedir+'/'+SubjID+'/'+ExpID)
    for fname in filelist:
        shutil.move(fname, rootdir+'/OriginalData')
    
os.chdir(rootdir)
if not os.path.exists('./OTPprocessed'):
    os.mkdir('./OTPprocessed')


# processing Raw data by OTP
StartT = time.time()
for data in filelist:
    os.chdir(rootdir+'/OriginalData')
    rawdata = mne.io.read_raw_fif(data, preload=True)
    picks = mne.pick_types(rawdata.info, meg=True)
    
    startT = time.time()
    rawdata_processed = oversampled_temporal_projection(rawdata, picks=picks)
    elapsed_time = time.time() - startT
    print('\nElapsed time: {0}'.format(elapsed_time)+" [sec]\n")
    
    # save OTP-processed data
    os.chdir(rootdir+'/OTPprocessed')
    
    newfile = data.split('_')[0] + '_OTPprocessed_raw.fif'
    rawdata_processed.save(newfile, overwrite=True)
    del rawdata, picks, startT, rawdata_processed, elapsed_time, newfile
del data

Elapsed_time = (time.time() - StartT)/60
print('\n   ---> All processes took {0} minutes.'.format(Elapsed_time))



