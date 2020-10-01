#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Analysis of Participants' task performance

@author: Akinori Takeda
"""

import mne
import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os
sns.set(style="whitegrid")

# change working directory
filedir = ''
SubjID = 'Subject18'
ExpID = 'SF'
os.chdir(filedir+'/'+SubjID+'/'+ExpID)

# get name of fif file
files = os.listdir(os.getcwd())
if os.path.exists('./Datafiles'):
    datafiles = [ii for ii in os.listdir(filedir+'/'+SubjID+'/'+ExpID+'/Datafiles/OriginalData') if 'run' in ii and '.fif' in ii]
else:
    datafiles = [ii for ii in files if 'run' in ii and '.fif' in ii]
datafiles.sort()
del files
print('\n')
print(datafiles)


#%%
# Also check whether subject correctly responded to target by specified button
# from PsychoPy output
os.chdir(filedir+'/'+SubjID+'/'+ExpID)
Psyfiles = os.listdir(os.getcwd())
Psydatafiles = [ii for ii in Psyfiles if 'MainExp' in ii and '.csv' in ii]
filename = Psydatafiles[0]
del Psyfiles, Psydatafiles

# load data file
Psydata = pd.read_csv(filename)

# target info
target = Psydata[Psydata['Images']=='target']
targetL = target[target['correctSide']=='L']
targetR = target[target['correctSide']=='R']

badtargLIdx = []
badtargRIdx = []
nbadtargL = np.zeros(9)
nbadtargR = np.zeros(9)

for n in np.arange(9):
    targL = targetL[targetL['sessionN.thisN']==n]
    targR = targetR[targetR['sessionN.thisN']==n]
    nbadL = 0
    nbadR = 0
    badLIdx = []
    badRIdx = []
    
    # check 1: specified to left but responded by right button
    for m in np.arange(targL.shape[0]):
        if targL['mouse.leftButton'].values[m] == 0  and targL['mouse.rightButton'].values[m] == 1:
            nbadL += 1
            badLIdx.append(m)
    del m
    badLIdx = np.array(badLIdx)
    
    # check 2: specified to right but responded by left button
    for m in np.arange(targR.shape[0]):
        if targR['mouse.leftButton'].values[m] == 1  and targR['mouse.rightButton'].values[m] == 0:
            nbadR += 1
            badRIdx.append(m)
    del m
    badRIdx = np.array(badRIdx)
    
    # fix index value
    if target['Right first'].values[0]:
        badLIdx += 24
    else:
        badRIdx += 24
        
    
    badtargLIdx.append(badLIdx)
    badtargRIdx.append(badRIdx)
    nbadtargL[n] = nbadL
    nbadtargR[n] = nbadR
    
    del targL, targR, nbadL, nbadR, badLIdx, badRIdx
del n, Psydata
    

# show detail
print('\n<< Results of Behavior Analysis: %s-%s experiment >>' % (SubjID, ExpID))
print('< Improper trials 1: Target with responses by incorrect-side button >')
print('N of trials specified to left but responded by right button: %d/216 images (%0.2f %%)' % (nbadtargL.sum(), ((nbadtargL.sum()*100)/216.)))
for i, n in enumerate(nbadtargL):
    if n == 1:
        print('  > %d bad trial in Run %d' % (n, (i+1)))
    elif n > 1:
        print('  > %d bad trials in Run %d' % (n, (i+1)))
del i, n
print('N of trials specified to right but responded by left button: %d/216 images (%0.2f %%)' % (nbadtargR.sum(), ((nbadtargR.sum()*100)/216.)))
for i, n in enumerate(nbadtargR):
    if n == 1:
        print('  > %d bad trial in Run %d' % (n, (i+1)))
    elif n > 1:
        print('  > %d bad trials in Run %d' % (n, (i+1)))
del i, n

if nbadtargL.sum() == 0 and nbadtargR.sum() == 0:
    print('  --> Subject responded to target correctly.')
else:
    print('  --> N of Targets with incorrect-side responses: %d/432 images (%0.2f %%)' % ((nbadtargL.sum()+nbadtargR.sum()), (((nbadtargL.sum()+nbadtargR.sum())*100)/432.)))


#%%
# < load trigger data & check whether there are improper trials > 
EventsID = [1,2,3,4,5,6,8,9,10,12,16,17,18,20]
Events=[]
TrigTCs=[]
RTs=[]

ImpropStimIdcs=[]
PropStimIdcs=[]
nImpropStimIdcs=np.zeros(len(datafiles))
nPropStimIdcs=np.zeros(len(datafiles))

ImpropTargetIdcs=[]
PropTargetIdcs=[]
nImpropTargetIdcs=np.zeros(len(datafiles))
nPropTargetIdcs=np.zeros(len(datafiles))


for i,name in enumerate(datafiles):
    # 1. load trigger data
    if os.path.exists('./Datafiles'):
        rawdata = mne.io.read_raw_fif(filedir+'/'+SubjID+'/'+ExpID+'/Datafiles/OriginalData/'+name, preload=True)
    else:
        rawdata = mne.io.read_raw_fif(name, preload=True)
    picks_trig = mne.pick_types(rawdata.info, meg=False, eeg=False, stim=True)
    sfreq = rawdata.info['sfreq']
    
    
    # 2. get event data 
    events = mne.find_events(rawdata, stim_channel='STI101', min_duration=(2 / sfreq))
    eventsIdx = np.arange(events.shape[0])
    
    TargetIdx=[]
    StimuliIdx=[]
    for n in eventsIdx:
        if events[n, -1] == 18:
            TargetIdx.append(n)
        if events[n, -1] in [1,2,3,4,5,6,8,9,10,12,16,17]:
            StimuliIdx.append(n)
    del n
    TargetIdx = np.array(TargetIdx)
    StimuliIdx = np.array(StimuliIdx)
  
    # get trigger time course data
    trigts=rawdata.copy().pick_channels(['STI101'])
    trigTC=trigts.get_data()[0]
    TrigTCs.append(trigTC)
    del trigts, trigTC
    
    
    # 3. Check whether there are improper trials
    #  [type 1] Stimuli with response
    #  [type 2] Target without response
    
    if events[-1, -1] in [1,2,3,4,5,6,8,9,10,12,16,17]:
        pattern = 1 # end with stimuli
        print('Run %d: Pattern 1 (ends with stimulus)' % (i+1))
    elif events[-1, -1] == 18:
        pattern = 2 # end with target
        print('Run %d: Pattern 2 (ends with target)' % (i+1))
    else:
        pattern = 3 # end with response
        print('Run %d: Pattern 3 (ends with response)' % (i+1))
    
    
    # Check 1: Stimuli with response
    impropStimIdx=[]
    propStimIdx=[]
    if pattern == 1:
        for n in StimuliIdx[:-1]:
            if events[n+1, -1] == 20:
                impropStimIdx.append(n)
            else:
                propStimIdx.append(n)
        del n
        propStimIdx.append(StimuliIdx[-1])
    else:
        for n in StimuliIdx:
            if events[n+1, -1] == 20:
                impropStimIdx.append(n)
            else:
                propStimIdx.append(n)
        del n
    
    
    # Check 2: Target without response
    impropTargetIdx=[]
    propTargetIdx=[]
    ResponseIdx=[]
    if pattern == 1:
        if events[-2,-1] == 18:
            for n in TargetIdx[:-1]:
                if events[n+1, -1] == 20:
                    propTargetIdx.append(n)
                    ResponseIdx.append((n+1))
                elif events[n+1, -1] != 18 and events[n+2, -1] == 20:
                    propTargetIdx.append(n)
                    ResponseIdx.append((n+2))
                else:
                    impropTargetIdx.append(n)
            del n
            impropTargetIdx.append(TargetIdx[-1])
        else:
            for n in TargetIdx:
                if events[n+1, -1] == 20:
                    propTargetIdx.append(n)
                    ResponseIdx.append((n+1))
                elif events[n+1, -1] != 18 and events[n+2, -1] == 20:
                    propTargetIdx.append(n)
                    ResponseIdx.append((n+2))
                else:
                    impropTargetIdx.append(n)
            del n
    elif pattern == 2:
        for n in TargetIdx[:-1]:
            if events[n+1, -1] == 20:
                propTargetIdx.append(n)
                ResponseIdx.append((n+1))
            elif events[n+1, -1] != 18 and events[n+2, -1] == 20:
                propTargetIdx.append(n)
                ResponseIdx.append((n+2))
            else:
                impropTargetIdx.append(n)
        del n
        impropTargetIdx.append(TargetIdx[-1])
    else:
        for n in TargetIdx:
            if events[n+1, -1] == 20:
                propTargetIdx.append(n)
                ResponseIdx.append((n+1))
            elif events[n+1, -1] != 18 and events[n+2, -1] == 20:
                propTargetIdx.append(n)
                ResponseIdx.append((n+2))
            else:
                impropTargetIdx.append(n)
        del n
    
    
    # remove indices in proper target indices which are included
    # in the indices of target trials with incorrect-side response
    if nbadtargL[i] != 0:
        propTargetIdx2 = []
        ResponseIdx2 = []
        for l, n in enumerate(propTargetIdx):
            if n not in TargetIdx[badtargLIdx[i]]:
                propTargetIdx2.append(n)
                ResponseIdx2.append(ResponseIdx[l])
        del l, n
        propTargetIdx = propTargetIdx2
        ResponseIdx = ResponseIdx2
        del propTargetIdx2, ResponseIdx2
    
    if nbadtargR[i] != 0:
        propTargetIdx2 = []
        ResponseIdx2 = []
        for l, n in enumerate(propTargetIdx):
            if n not in TargetIdx[badtargRIdx[i]]:
                propTargetIdx2.append(n)
                ResponseIdx2.append(ResponseIdx[l])
        del l, n
        propTargetIdx = propTargetIdx2
        ResponseIdx = ResponseIdx2
        del propTargetIdx2, ResponseIdx2
    
    
    # make dataset for RT calculation
    RT = events[ResponseIdx,0]-events[propTargetIdx,0]
    
    
    # save indices
    Events.append(events)
    RTs.append(RT)
    
    ImpropStimIdcs.append(np.array(impropStimIdx))
    PropStimIdcs.append(np.array(propStimIdx))
    ImpropTargetIdcs.append(np.array(impropTargetIdx))
    PropTargetIdcs.append(np.array(propTargetIdx))
    
    nImpropStimIdcs[i]=len(impropStimIdx)
    nPropStimIdcs[i]=len(propStimIdx)
    nImpropTargetIdcs[i]=len(impropTargetIdx)
    nPropTargetIdcs[i]=len(propTargetIdx)
    
    del impropStimIdx, propStimIdx, impropTargetIdx, propTargetIdx, pattern, events, eventsIdx, rawdata, RT
del i, name


#%%
os.chdir(filedir+'/'+SubjID+'/'+ExpID)

# plot the number of Target trials with incorrect-side response
plt.figure()
plt.plot(np.arange(9)+1, nbadtargL, '-o', color='orange', label='N of Target specified L but responded by R')
plt.plot(np.arange(9)+1, nbadtargR, '-o', color='darkblue', label='N of Target specified R but responded by L')
plt.xlim(0.5, 9.5)
plt.ylim(0, np.max([np.max(nbadtargL), np.max(nbadtargR)])+5)
plt.xlabel('Run No.')
plt.ylabel('N of Improper Trials')
plt.title('%s: %s experiment' % (SubjID, ExpID))
plt.legend(frameon=True, framealpha=1., fontsize=9.)

# save figure
plt.savefig('ImproperTarget.png')


# plot the number of improper trials
plt.figure()
plt.plot(np.arange(9)+1, nImpropStimIdcs, '-o', label='N of Stimuli with response')
plt.plot(np.arange(9)+1, nImpropTargetIdcs, '-o', label='N of Target without response')
plt.xlim(0.5, 9.5)
plt.ylim(0, np.max([np.max(nImpropStimIdcs), np.max(nImpropTargetIdcs)])+3)
plt.xlabel('Run No.')
plt.ylabel('N of Improper Trials')
plt.title('%s: %s experiment' % (SubjID, ExpID))
plt.legend(frameon=True, framealpha=1., fontsize=9.)

# save figure
plt.savefig('ImproperTrials.png')


# plot Reaction Times
meanRTs = [i.mean() for i in RTs]
sdRTs = [ii.std() for ii in RTs]

plt.figure()
plt.plot(np.arange(9)+1, meanRTs, 'r-o', linewidth=2.5, label='Mean Reaction Time (ms)')
plt.errorbar(np.arange(9)+1, meanRTs, yerr=sdRTs, fmt='r-o', elinewidth=1.5)
plt.xlim(0.5, 9.5)
plt.ylim(0, 1000)
plt.xlabel('Run No.')
plt.ylabel('Mean Reaction Time (ms)')
plt.title('%s: %s experiment' % (SubjID, ExpID))
plt.legend(frameon=True, framealpha=1., fontsize=9.)

# save figure
plt.savefig('MeanReactionTime.png')


# also calculate mean RT & SD of all proper trials
for i, RT in enumerate(RTs):
    if i == 0:
        allRT = RT
    else:
        allRT = np.concatenate((allRT, RT))
del i, RT


    
# show detail
print('\n< Improper trials 2 >')
print('[Case 1]')
print('N of Stimuli with response: %d/%d images (%.2f %%)' % (nImpropStimIdcs.sum(), (nImpropStimIdcs.sum()+nPropStimIdcs.sum()), (nImpropStimIdcs.sum()*100./(nImpropStimIdcs.sum()+nPropStimIdcs.sum()))))
for i, n in enumerate(nImpropStimIdcs):
    if n == 1:
        print('  > %d bad trial in Run %d' % (n, (i+1)))
    elif n > 1:
        print('  > %d bad trials in Run %d' % (n, (i+1)))
del i, n
print('[Case 2]')
print('N of Target without response: %d/%d images (%.2f %%)' % (nImpropTargetIdcs.sum(), (nImpropTargetIdcs.sum()+nPropTargetIdcs.sum()+nbadtargL.sum()+nbadtargR.sum()), (nImpropTargetIdcs.sum()*100./(nImpropTargetIdcs.sum()+nPropTargetIdcs.sum()+nbadtargL.sum()+nbadtargR.sum()))))
for i, n in enumerate(nImpropTargetIdcs):
    if n == 1:
        print('  > %d bad trial in Run %d' % (n, (i+1)))
    elif n > 1:
        print('  > %d bad trials in Run %d' % (n, (i+1)))
del i, n
print('  --> N of improper Target trials: %d/%d images (%0.2f %%)' % ((nImpropTargetIdcs.sum()+nbadtargL.sum()+nbadtargR.sum()), (nImpropTargetIdcs.sum()+nPropTargetIdcs.sum()+nbadtargL.sum()+nbadtargR.sum()), ((nImpropTargetIdcs.sum()+nbadtargL.sum()+nbadtargR.sum())*100./(nImpropTargetIdcs.sum()+nPropTargetIdcs.sum()+nbadtargL.sum()+nbadtargR.sum()))))
print('      N of all improper trials: %d/%d images (%0.2f %%)' % ((nImpropStimIdcs.sum()+nImpropTargetIdcs.sum()+nbadtargL.sum()+nbadtargR.sum()), (nImpropStimIdcs.sum()+nPropStimIdcs.sum()+nImpropTargetIdcs.sum()+nPropTargetIdcs.sum()+nbadtargL.sum()+nbadtargR.sum()), ((nImpropStimIdcs.sum()+nImpropTargetIdcs.sum()+nbadtargL.sum()+nbadtargR.sum())*100./(nImpropStimIdcs.sum()+nPropStimIdcs.sum()+nImpropTargetIdcs.sum()+nPropTargetIdcs.sum()+nbadtargL.sum()+nbadtargR.sum()))))

print('\n< Reaction Time >')
print('mean RT of 9 runs: %.2f +/- %.2f[SD] ms' % (np.array(meanRTs).mean(), np.array(meanRTs).std()))
print('mean RT of all proper trials (%d trials): %.2f +/- %.2f[SD] ms' % (nPropTargetIdcs.sum(), allRT.mean(), allRT.std()))


#%%
# save proper trials' information
with open('ProperStimIdcs.txt', 'wb') as fp:
    pickle.dump(PropStimIdcs, fp)
with open('ProperTargetIdcs.txt', 'wb') as fp:
    pickle.dump(PropTargetIdcs, fp)

