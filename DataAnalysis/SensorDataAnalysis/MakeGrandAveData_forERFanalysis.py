#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Making Grand-average datasets of ERF waveforms

@author: Akinori Takeda
"""

import mne
import numpy as np
import os
locVal = locals()


#--- set data path & get datafiles' name ---#
filedir = ''
os.chdir(filedir)


# directory manipulation
if not os.path.exists('./GrandAverage'):
    os.makedirs('./GrandAverage/Datafiles/forERFanalysis')
os.chdir('./GrandAverage/Datafiles/forERFanalysis')
rootdir = os.getcwd()

if not os.path.exists('./SF'):
    for dtype in ['Gradiometer', 'Magnetometer']:
        os.makedirs(rootdir+'/SF/'+dtype)
    del dtype


#--- make grand-average dataset ---#
# get individual evoked response data
Subjectlist = [i for i in os.listdir(filedir) if 'Subject' in i and '.py' not in i]
SubjectList = ['Subject%d' % (i+1) for i in np.arange(len(Subjectlist))]

rmvSubj = []
Subjects = [i for i in SubjectList if i not in rmvSubj]


# make dataset
print('\n<< Make grand-average dataset (n=%d) >>' % len(Subjects))
for SubjID in Subjects:
    print('< %s >' % SubjID)
    os.chdir(filedir+'/'+SubjID+'/SF/Datafiles/EpochData')
    
    # load -epo.fif file
    EpochData = mne.read_epochs('ProperEpochData-epo.fif', preload=True)
    
    #- baseline correction & making each condition dataset -#
    if SubjID == Subjects[0]:
        eveIDinfo = list(EpochData.event_id.keys())
        magInfo = EpochData.copy().pick_types(meg='mag').info
        gradInfo = EpochData.copy().pick_types(meg='grad').info
        tmin = EpochData.tmin
        BLmin = -0.1
        BLmax = 0
        print('\nBaseline: {0} ~ {1} ms'.format(BLmin*1000,BLmax*1000))
        DataFileList = []
    
    for condition in eveIDinfo:
        if SubjID == Subjects[0]:
            exec('MagData_'+condition+' = []')
            exec('GradData_'+condition+' = []')
        
        # baseline correction
        data = EpochData[condition].copy().apply_baseline(baseline=(BLmin, BLmax))
        
        # get data of each sensor type
        magdata = data.copy().pick_types(meg='mag').average()
        graddata = data.copy().pick_types(meg='grad').average()
        
        exec('MagData_'+condition+'.append(magdata.data)')
        exec('GradData_'+condition+'.append(graddata.data)')
        
        if SubjID == Subjects[0]:
            DataFileList.append('MagData_'+condition)
            DataFileList.append('GradData_'+condition)
        
        del data, magdata, graddata
    print('\n')
    del EpochData, condition
del SubjID


# make EpochArray instance for grand-average
print('\n[Saving datasets]')
for condition in eveIDinfo:
    datafiles = [i for i in DataFileList if condition in i]
    
    # convert list to np.array data, and then make EpochArray instance
    for data in datafiles:
        print('Processing %s...' % data)
        
        if 'Mag' in data:
            exec(data+'_GA = mne.EpochsArray(np.array('+data+'), magInfo, tmin=tmin)')
            os.chdir(rootdir+'/SF/Magnetometer')
        else:
            exec(data+'_GA = mne.EpochsArray(np.array('+data+'), gradInfo, tmin=tmin)')
            os.chdir(rootdir+'/SF/Gradiometer')
        
        filename = data + '-epo.fif'
        exec(data+'_GA.save(filename)')
        print('\n')
        
        del locVal[data+'_GA'], filename
    del datafiles, data
del condition

print('  ---> All processes were done!')

