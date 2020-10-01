#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visual inspection of epoch data

@author: Akinori Takeda
"""

import mne
import pickle
import numpy as np
import seaborn as sns
import os
sns.set(style="white")


# change working directory
filedir = ''
SubjID = 'Subject18'
ExpID = 'SF'

# load epoch data file
os.chdir(filedir+'/'+SubjID+'/'+ExpID+'/Datafiles/EpochData')
epochdata = mne.read_epochs('ICAprocessed_EpochData2-epo.fif', preload=True)
print('\n')
print(epochdata)

# baseline correction
sfreq = epochdata.info['sfreq']
BLmin = -0.1 - (1/sfreq)
BLmax = -(1/sfreq)
EpochData = epochdata.copy().apply_baseline(baseline=(BLmin, BLmax))
print('Baseline: {0} ~ {1} ms'.format(EpochData.baseline[0]*sfreq,EpochData.baseline[1]*sfreq))


# variance setting
events = epochdata.events
properEvents = epochdata.events
eventsIdx = np.arange(events.shape[0])
properEventsIdx = np.arange(events.shape[0])
eveID = dict(FearF_BSF=1, FearF_LSF=2, FearF_HSF=4, FearF_Equ=8,
             NeutF_BSF=3, NeutF_LSF=5, NeutF_HSF=6, NeutF_Equ=9,
             House_BSF=10, House_LSF=12, House_HSF=16, House_Equ=17, target=18)
eveIDinfo = np.array(list(eveID.items()), dtype='object')
BadTrialIdcs = []

print('\n<< Results of Visual Inspection: %s-%s experiment >>' % (SubjID, ExpID))


#%%
'''
Visual inspection of epoch data
'''
#- preparation for plotting -#
condIdx = 13  # 1~13

condition = eveIDinfo[(condIdx-1), 0]
trigN = eveIDinfo[(condIdx-1), -1]
thisEveIdx = eventsIdx[events[:,-1]==trigN]

# get time course data of each sensor type
epochs = EpochData[condition].copy()
epochs.plot()


#%%
# remove indices of bad trials
badtrials = []

if len(badtrials) == 0:
    print('\n[%s condition (trigger No.: %d)] There were no bad trials.' % (condition, trigN))
else:
    badtrials = np.array(badtrials)-1
    badtrials = badtrials.tolist()
    badtrialIdx = thisEveIdx[badtrials]
    if len(badtrials) == 1:
        print('\n[%s condition (trigger No.: %d)] There was 1 bad trial.' % (condition, trigN))
        print('  > Index of the bad trial:')
        print(badtrialIdx)
    else:
        print('\n[%s condition (trigger No.: %d)] There were %d bad trials.' % (condition, trigN, len(badtrials)))
        print('  > Indices of the bad trials:')
        print(badtrialIdx)
    
    for n in badtrialIdx:
        properEventsIdx = properEventsIdx[properEventsIdx[:]!=n]
    del n, badtrialIdx

exec('BadTrialIdcs.append(dict('+condition+'=badtrials))')

#  ---> next condition processing...


#%%
# After checking all trials ...
# Save indices of improper trials
os.chdir(filedir+'/'+SubjID+'/'+ExpID+'/Datafiles/EpochData')
with open('BadTrialIdcs_byVisInspection.txt', 'wb') as fp:
    pickle.dump(BadTrialIdcs, fp)



'''
Random selection of trials
'''
#- seed setting -#
GrandSeed = 100 + int(SubjID[7:])*len(ExpID)
np.random.seed(GrandSeed)
SeedsList = np.random.permutation(np.arange(10, len(eveID)*10+1, 10))


#- check the number of trials in each condition -#
properEvents = properEvents[properEventsIdx,:]
nTrialsList = [properEventsIdx[properEvents[:,-1]==i[1]].shape[0] for i in eveIDinfo if i[0]!='target']

if np.min(nTrialsList) >= 100:
    nTrial = 100  # the number of trials used
else:
    nTrial = np.min(nTrialsList)

print('\nN of trials in each condition:')
print(nTrialsList)
print('N of trials which will be used: %d' % nTrial)


#- random selection of trials -#
IdxList = []
for i,eveinfo in enumerate(eveIDinfo):
    eveTrigN = eveinfo[1]
    thisEveIdx = properEventsIdx[properEvents[:,-1]==eveTrigN]
    nIdx = thisEveIdx.shape[0]
    
    if eveinfo[0]!='target':
        # random selection
        np.random.seed(SeedsList[i])
        trialIdx = np.random.choice(np.arange(nIdx), nTrial, replace=False)
        trialIdx.sort()
    
        # add to list
        IdxList.append(thisEveIdx[trialIdx])
        del trialIdx
    else:
        IdxList.append(thisEveIdx)
    
    del eveTrigN, thisEveIdx, nIdx
del i, eveinfo


#- concatenate indices & make an index data -#
useTrialIdx = np.concatenate(IdxList)
useTrialIdx.sort()


#- extract epochs -#
EpochDataAvail = epochdata[useTrialIdx].copy()
print('\n')
print(EpochDataAvail)



'''
Save Epoch data
'''
os.chdir(filedir+'/'+SubjID+'/'+ExpID+'/Datafiles/EpochData')
EpochDataAvail.save('ProperEpochData-epo.fif')



