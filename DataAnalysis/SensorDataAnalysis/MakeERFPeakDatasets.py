#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sensor-of-interest (SOI) analysis 
with statistical results (Within-condition comparison)

@author: Akinori Takeda
"""

import mne
from mne.viz import plot_topomap
import numpy as np
from scipy import stats as stats
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import seaborn as sns
import os
locVal = locals()


#--- set data path & get datafiles' name ---#
filedir = ''
ExpID = 'SF'
os.chdir(filedir)

# load data
Subjectlist = [i for i in os.listdir(filedir) if 'Subject' in i and '.py' not in i and 'Test' not in i]
SubjectList = ['Subject%d' % (i+1) for i in np.arange(len(Subjectlist))]

rmvSubj = []
Subjectlist = [i for i in SubjectList if i not in rmvSubj]

Subjects = []
for SubjID in Subjectlist:
    os.chdir(filedir+'/'+SubjID)
    if 'SF' in os.listdir(os.getcwd()) and 'Subliminal' in os.listdir(os.getcwd()):
        Subjects.append(SubjID)
del SubjID


# make dataset
print('\n<< Make datasets (N=%d) >>' % len(Subjects))
for SubjID in Subjects:
    print('< %s >' % SubjID)
    os.chdir(filedir+'/'+SubjID+'/'+ExpID+'/Datafiles/EpochData')
    
    # load -epo.fif file
    EpochData = mne.read_epochs('ProperEpochData-epo.fif', preload=True)
    
    #- baseline correction & making each condition dataset -#y
    if SubjID == Subjects[0]:
        eveIDinfo = list(EpochData.event_id.keys())
        conditionlist = [i for i in eveIDinfo if 'target' not in i]
        magInfo = EpochData.copy().pick_types(meg='mag').info
        gradInfo = EpochData.copy().pick_types(meg='grad').info
        tmin = EpochData.tmin
        BLmin = -0.1
        BLmax = 0
        print('\nBaseline: {0} ~ {1} ms'.format(BLmin*1000,BLmax*1000))
        
        times = EpochData.times
        timemask = np.where((-0.1 <= times)&(times <= 0.35))[0]
        
        Chlayout = mne.channels.layout.find_layout(gradInfo)
        ch_names = mne.utils._clean_names(gradInfo['ch_names'])
        
        SOIchs = ['MEG1333','MEG1342','MEG2412','MEG2422','MEG2432','MEG2512','MEG2522','MEG2523','MEG2532','MEG2632','MEG2643']
        TW_BSF = [137, 213]
        TW_LSF = [137, 208]
        TW_HSF = [177, 221]
        TW_Equ = [199, 265]
        SOIchsIdx = np.array([i for i, n in enumerate(ch_names) if n in SOIchs])
    
    
    for condition in conditionlist:
        # baseline correction
        data = EpochData[condition].copy().apply_baseline(baseline=(BLmin, BLmax))
        
        # get data of each sensor type
        data2 = data.copy().pick_types(meg='grad').get_data()
        
        # make dataset
        graddata = data2[:, :, timemask]
        exec(SubjID+'_'+condition+' = graddata[:,SOIchsIdx,:]')
        locVal[SubjID+'_'+condition] *= 1e13
        
        del data, data2, graddata
    print('\n')
    del EpochData, condition
del SubjID


#%%
# setting directory for saving figs
os.chdir(filedir+'/GrandAverage/DataPlots/ERFdataPlots')
if not os.path.exists('./ERF_SOIplots'):
    os.mkdir('./ERF_SOIplots')
os.chdir('./ERF_SOIplots')
savedir = os.getcwd()


MagLayout = mne.channels.layout.find_layout(magInfo)
MagPos = MagLayout.pos[:,:2]

# color scale setting for Ch pos plot
cmap = cm.RdBu_r
my_cmap = cmap(np.arange(cmap.N))
my_cmap[:,-1] = 0
my_cmap = ListedColormap(my_cmap)


# plot sensors included in SOI
mask = np.zeros((len(ch_names),1), dtype=bool)
mask[SOIchsIdx, :] = True
maskparams = dict(marker='o', markerfacecolor='b', markeredgecolor='b', markersize=7)

plt.figure(figsize=(7,7))
ax = plt.gcf().gca()
plt.suptitle('Sensors included in SOI ($N_{sensors}$ = %d)' % SOIchsIdx.shape[0], fontsize=15)
im, _ = plot_topomap(np.zeros(len(ch_names)), Chlayout.pos, mask=mask, cmap=my_cmap, res=600,
                     sensors='k.', axes=ax, mask_params=maskparams, vmin=-6, vmax=6)

fig = plt.gcf()
fig.savefig('SOIsensors.png')
plt.close(fig)
del ax, fig, im


# plot SOI location
iter_ch = [(x, y) for x, y in enumerate(Chlayout.names) if y in ch_names]
ch_pairs = np.array(iter_ch, dtype='object').reshape((len(iter_ch) // 2, -1))
ChlocIdx = np.array([i for i, n in enumerate(ch_pairs) if n[1] in SOIchs or n[3] in SOIchs])

mask2 = np.zeros((ch_pairs.shape[0],1), dtype=bool)
mask2[ChlocIdx, :] = True
maskparams2 = dict(marker='o', markerfacecolor='m', markeredgecolor='m', markersize=10)

plt.figure(figsize=(7,7))
ax2 = plt.gcf().gca()
plt.suptitle('SOI location ($N_{Chloc}$ = %d)' % ChlocIdx.shape[0], fontsize=15)
im2, _ = plot_topomap(np.zeros(ch_pairs.shape[0]), MagPos, mask=mask2, cmap=my_cmap, res=600,
                      sensors='k.', axes=ax2, mask_params=maskparams2, vmin=-6, vmax=6)

fig2 = plt.gcf()
fig2.savefig('SOIsensorlocation.png')
plt.close(fig2)
del ax2, fig2, im2


#%%
# make grand-average datasets
for cond in conditionlist:
    exec(cond+'_GA = []')
    
    for SubjID in Subjects:
        data = locVal[SubjID+'_'+cond].mean(1)
        meanTC = data.mean(0)
        locVal[cond+'_GA'].append(meanTC)
        del data, meanTC
    del SubjID
    
    locVal[cond+'_GA'] = np.array(locVal[cond+'_GA'])
del cond

# calculate 95% confidence intervals
MinVs = []
MaxVs = []

for cond in conditionlist:
    data = locVal[cond+'_GA']
    
    # calculate 95% confidence interval
    cis = mne.stats.bootstrap_confidence_interval(data, ci=0.95, n_bootstraps=5000, stat_fun='mean')
    Btm = cis[0,:]
    UP = cis[1,:]
    
    # add to container
    MinVs.append(Btm.min())
    MaxVs.append(UP.max())
    del data, cis, Btm, UP
del cond

Vmin = np.min(MinVs)*1.1
Vmax = np.max(MaxVs)*1.1
del MinVs, MaxVs

pltMaxV = round(Vmax)
if pltMaxV < Vmax:
    pltMaxV = round(Vmax+1)
pltMinV = round(Vmin)
if Vmin < pltMinV:
    pltMinV = round(Vmin-1)

CategoryList = ['NeutF', 'FearF', 'House']
SFNameList = ['BSF', 'LSF', 'HSF', 'Equ']


# plot data #
os.chdir(savedir)
if not os.path.exists('./GrandAveragePlots'):
    os.mkdir('./GrandAveragePlots')
os.chdir('./GrandAveragePlots')

# [1] across SF
print('\n< Plot grand-average data (across SF) >')
if not os.path.exists('./acrossSF'):
    os.mkdir('./acrossSF')
os.chdir('./acrossSF')

for cond in CategoryList:
    print('[%s data]' % cond)
    if cond == 'FearF':
        title = 'Fearful face'
    elif cond == 'NeutF':
        title = 'Neutral face'
    else:
        title = cond
    
    # plot
    plt.figure(figsize=(8.5,5))
    gs = gridspec.GridSpec(1,10)
    ax1 = plt.gcf().add_subplot(gs[:, :8])
    
    plt.gcf().suptitle(title, fontsize=15)
    ax1.vlines(0, pltMinV, pltMaxV, linewidth=1)
    ax1.set_ylim(pltMinV, pltMaxV)
    ax1.hlines(0, pltTimes[0], pltTimes[-1], linewidth=1)
    
    peakLatencies = []
    for sf in SFNameList:
        data = locVal[cond+'_'+sf+'_GA']
        meanTC = data.mean(0)
        
        MinValIdx = np.where(meanTC == meanTC.min())[0][0]
        peakLatencies.append(pltTimes[MinValIdx])
        
        if sf == 'Equ':
            label = 'EQU'
        else:
            label = sf
        
        # 95% confidence interval
        cis = mne.stats.bootstrap_confidence_interval(data, ci=0.95, n_bootstraps=5000, stat_fun='mean')
        Btm = cis[0,:]
        Up = cis[1,:]
        
        ax1.plot(pltTimes, meanTC, linewidth=2, color=colorList[sf], label=label)
        ax1.fill_between(pltTimes, Btm, Up, alpha=0.3, color=colorList[sf])
        del data, meanTC, cis, Btm, Up, MinValIdx
    del sf
        
    ax1.set_xlim(pltTimes[0], pltTimes[-1])
    ax1.spines['right'].set_color('none')
    ax1.spines['top'].set_color('none')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Amplitude (fT/cm)')
    
    legends, labels = ax1.get_legend_handles_labels()
    plt.gcf().legend(legends, labels, loc=(0.775, 0.75), prop={'size': 12}, frameon=True, framealpha=1.)
    plt.gcf().subplots_adjust(top=0.92, bottom=0.12)
    
    axes = plt.axes([0.775, 0.1, 0.2, 0.4])
    text = '< Peak Latency >'
    for i, sf in enumerate(SFNameList):
        text += '\n    %s: %d ms' % (sf, peakLatencies[i])
    axes.text(0.1, 0.3, text, fontsize=12)
    axes.set_axis_off()
    
    # save figure
    fig = plt.gcf()
    fig.savefig('Waveform_%s.png' % cond)
    plt.close(fig)
    del fig, title, gs, ax1, legends, labels, peakLatencies
del cond

print(' --> done.')


# [2] across Category
print('\n< Plot grand-average data (across category) >')
os.chdir(savedir+'/GrandAveragePlots')
if not os.path.exists('./acrossCategory'):
    os.mkdir('./acrossCategory')
os.chdir('./acrossCategory')

lineTypeList = dict(FearF='--', NeutF='-', House=':')
for sf in SFNameList:
    print('[%s data]' % sf)
    if sf == 'Equ':
        title = 'EQU'
    else:
        title = sf
    
    # plot
    plt.figure(figsize=(10,6.5))
    gs = gridspec.GridSpec(1,10)
    ax1 = plt.gcf().add_subplot(gs[:, :8])
    
    plt.gcf().suptitle(title, fontsize=24)
    ax1.vlines(0, pltMinV, pltMaxV, linewidth=1)
    ax1.set_ylim(pltMinV, pltMaxV)
    ax1.hlines(0, pltTimes[0], pltTimes[-1], linewidth=1)
    
    peakLatencies = []
    for cond in CategoryList:
        data = locVal[cond+'_'+sf+'_GA']
        meanTC = data.mean(0)
        
        MinValIdx = np.where(meanTC == meanTC.min())[0][0]
        peakLatencies.append(pltTimes[MinValIdx])
        
        if cond == 'FearF':
            label = 'Fearful face'
        elif cond == 'NeutF':
            label = 'Neutral face'
        else:
            label = cond
        
        ax1.plot(pltTimes, meanTC, linewidth=2, color=colorList[sf], 
                 label=label, linestyle=lineTypeList[cond])
        del data, meanTC, MinValIdx
    del cond
    
    # plot significant time interval indicator
    yval = pltMinV + 2
    width = 1
    shadeTs = np.arange(locVal['TW_'+sf][0], locVal['TW_'+sf][1]+1)
    ax1.hlines(yval, 0, locVal['TW_'+sf][0], linewidth=1)
    ax1.fill_between(shadeTs, np.ones(shadeTs.shape)*(yval-width), np.ones(shadeTs.shape)*(yval+width), 
                     alpha=1, color='k')
    ax1.hlines(yval, locVal['TW_'+sf][1], pltTimes[-1], linewidth=1)
    
    ax1.set_xlim(pltTimes[0], pltTimes[-1])
    ax1.spines['right'].set_color('none')
    ax1.spines['top'].set_color('none')
    ax1.spines['left'].set_linewidth(2)
    ax1.spines['bottom'].set_linewidth(2)
    ax1.set_xlabel('Time (ms)', fontsize=22, labelpad=10)
    ax1.set_ylabel('Amplitude (fT/cm)', fontsize=22, labelpad=10)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    
    legends, labels = ax1.get_legend_handles_labels()
    plt.gcf().legend(legends, labels, loc=(0.75, 0.785), prop={'size': 18}, frameon=True, framealpha=1.)
    plt.gcf().legends[0].get_frame().set_linewidth(1)
    plt.gcf().legends[0].get_frame().set_edgecolor('k')
    
    plt.gcf().subplots_adjust(top=0.93, bottom=0.15)

    # save figure
    fig = plt.gcf()
    fig.savefig('Waveform_%s.png' % sf)
    plt.close(fig)
    del fig, title, gs, ax1, legends, labels, peakLatencies
del sf

print(' --> done.')


# also make summary plots
plt.figure(figsize=(10.5,7))
gs = gridspec.GridSpec(1,10)
ax1 = plt.gcf().add_subplot(gs[:, :8])
plt.gcf().suptitle('Neutral face', fontsize=24)

width = 1
margin = 1
yval1 = pltMinV + 2
yval2 = yval1-width*2-margin
yval3 = yval2-width*2-margin
yval4 = yval3-width*2-margin
pltVmin = yval4-width-margin

ax1.vlines(0, pltVmin, pltMaxV, linewidth=1)
ax1.set_ylim(pltVmin, pltMaxV)
ax1.hlines(0, pltTimes[0], pltTimes[-1], linewidth=1)
plt.axvspan(137, 265, pltVmin, pltMaxV, alpha=0.25, color='yellow')

for sf in SFNameList:
    if sf == 'Equ':
        label = 'EQU'
    else:
        label = sf
    data = locVal['NeutF_'+sf+'_GA']
    meanTC = data.mean(0)
    
    ax1.plot(pltTimes, meanTC, linewidth=2, color=colorList[sf], label=label)
    del data, meanTC
del sf

# plot significant time interval indicator
for i, sf in enumerate(SFNameList):
    shadeTs = np.arange(locVal['TW_'+sf][0], locVal['TW_'+sf][1]+1)
    if sf == 'Equ':
        label = 'EQU'
    else:
        label = sf

    ax1.hlines(locVal['yval%d' % (i+1)], 0, locVal['TW_'+sf][0], linewidth=1)
    ax1.fill_between(shadeTs, np.ones(shadeTs.shape)*(locVal['yval%d' % (i+1)]-width),
                     np.ones(shadeTs.shape)*(locVal['yval%d' % (i+1)]+width), 
                     alpha=1, color='k')
    ax1.hlines(locVal['yval%d' % (i+1)], locVal['TW_'+sf][1], pltTimes[-1], linewidth=1)
    ax1.text(355, locVal['yval%d' % (i+1)]-width*1.5, label, horizontalalignment='left', 
                         verticalalignment='bottom', fontsize=16)
del i, sf

ax1.set_xlim(pltTimes[0], pltTimes[-1])
ax1.spines['right'].set_color('none')
ax1.spines['top'].set_color('none')
ax1.spines['left'].set_linewidth(2)
ax1.spines['bottom'].set_linewidth(2)
ax1.set_xlabel('Time (ms)', fontsize=22, labelpad=10)
ax1.set_ylabel('Amplitude (fT/cm)', fontsize=22, labelpad=10)
ax1.set_yticks([-20, -10, 0, 10, 20])
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

legends, labels = ax1.get_legend_handles_labels()
plt.gcf().legend(legends, labels, loc=(0.75, 0.75), prop={'size': 18}, frameon=True, framealpha=1.)
plt.gcf().legends[0].get_frame().set_linewidth(1)
plt.gcf().legends[0].get_frame().set_edgecolor('k')

plt.gcf().subplots_adjust(top=0.92, bottom=0.14)

# save figure
fig = plt.gcf()
fig.savefig('Waveform_AllConditions.png')
plt.close(fig)


#%%
#- make peak amplitude & latency datasets -#
conditionlist2 = ['NeutF_BSF', 'NeutF_LSF', 'NeutF_HSF', 'NeutF_Equ', 'FearF_BSF', 'FearF_LSF', 'FearF_HSF', 'FearF_Equ', 'House_BSF', 'House_LSF', 'House_HSF', 'House_Equ']
Dataset_Amp = np.zeros((len(Subjects), len(conditionlist2)))
Dataset_Lat = np.zeros((len(Subjects), len(conditionlist2)))

# make directory for saving data
os.chdir(filedir+'/GrandAverage/Datafiles')
if not os.path.exists('./forERF_SOIanalysis'):
    os.mkdir('./forERF_SOIanalysis')
os.chdir('./forERF_SOIanalysis')

direcnames = [ExpID, 'Gradiometer']
for name in direcnames:
    if not os.path.exists('./'+name):
        os.mkdir('./'+name)
    os.chdir('./'+name)
del name
datasavedir = os.getcwd()

print('\n< Make datasets >')
for i, cond in enumerate(conditionlist2):
    for ii, SubjID in enumerate(Subjects):
        data = locVal[SubjID+'_'+cond].mean(1)
        meanTC = data.mean(0)
        
        TWmask = np.where((locVal['TW_'+cond.split('_')[-1]][0] <= pltTimes)&(pltTimes <= locVal['TW_'+cond.split('_')[-1]][1]))[0]
        MinValIdx = np.where(meanTC == meanTC[TWmask].min())[0][0]
        Dataset_Amp[ii, i] = meanTC[MinValIdx]
        Dataset_Lat[ii, i] = pltTimes[MinValIdx]
        
        del data, meanTC, MinValIdx, TWmask
    del SubjID, ii
del cond, i

# save datasets
np.savetxt(datasavedir+'/Dataset_PeakAmplitude.csv', Dataset_Amp, delimiter=',')
np.savetxt(datasavedir+'/Dataset_PeakLatency.csv', Dataset_Lat, delimiter=',')
print(' --> done.')


#%%
#- plot peak amplitude & latency data -#
Dataset_Amp = np.loadtxt(datasavedir+'/Data_RestrictedToEachTW/Dataset_PeakAmplitude.csv', delimiter=',')
Dataset_Lat = np.loadtxt(datasavedir+'/Data_RestrictedToEachTW/Dataset_PeakLatency.csv', delimiter=',')

pal = ['darkgray', 'dodgerblue', 'tomato', 'springgreen', 'darkgray', 'dodgerblue', 'tomato', 'springgreen', 'darkgray', 'dodgerblue', 'tomato', 'springgreen']
innerplot = None # None or 'point'

os.chdir(savedir)
if not os.path.exists('./HemiViolinPlots'):
    os.mkdir('./HemiViolinPlots')
os.chdir('./HemiViolinPlots')


# [1] Peak Amplitude
# make data for plot
import pandas as pd
df_NeutF = pd.DataFrame(dict(BSF=Dataset_Amp[:,0], LSF=Dataset_Amp[:,1], HSF=Dataset_Amp[:,2], EQU=Dataset_Amp[:,3]))
df_FearF = pd.DataFrame(dict(BSF=Dataset_Amp[:,4], LSF=Dataset_Amp[:,5], HSF=Dataset_Amp[:,6], EQU=Dataset_Amp[:,7]))
df_House = pd.DataFrame(dict(BSF=Dataset_Amp[:,8], LSF=Dataset_Amp[:,9], HSF=Dataset_Amp[:,10], EQU=Dataset_Amp[:,11]))

df_NeutF_melt = pd.melt(df_NeutF)
df_NeutF_melt['Category'] = 'Neutral face'
df_FearF_melt = pd.melt(df_FearF)
df_FearF_melt['Category'] = 'Fearful face'
df_House_melt = pd.melt(df_House)
df_House_melt['Category'] = 'House'

df = pd.concat([df_NeutF_melt, df_FearF_melt, df_House_melt], axis=0)
pltYmin = -65
pltYmax = 25

# plot
plt.figure(figsize=(14.5,8))
plt.gcf().suptitle('Individual Peak Amplitude', fontsize=20)
v = sns.violinplot(x='Category', y='value', data=df, hue='variable', palette=pal, scale='count', inner=innerplot, dodge=True, jitter=True, bw=0.4)

prop = v.properties()
if innerplot == 'point':
    propdata = prop['children'][:24] # for the case 'inner' arg == 'point'
    propdata = propdata[::2]
else:
    propdata = prop['children'][:12] # for the case 'inner' arg == 'point'

for n in propdata:
    m = np.mean(n.get_paths()[0].vertices[:,0])
    n.get_paths()[0].vertices[:,0] = np.clip(n.get_paths()[0].vertices[:,0], m, np.inf)

# also plot something
for yval in np.arange(pltYmin+5, pltYmax-4, 10):
    plt.hlines(yval, -1, 3, color='gray', alpha=0.3)

# plot 95% confidence interval
cis = mne.stats.bootstrap_confidence_interval(Dataset_Amp, ci=0.95, n_bootstraps=5000, stat_fun='mean')
Btm = cis[0,:]
Up = cis[1,:]

pos = np.array([-0.3, -0.1, 0.1, 0.3, 0.7, 0.9, 1.1, 1.3, 1.7, 1.9, 2.1, 2.3])
for i, p in enumerate(pos):
    plt.vlines(p+0.025, Dataset_Amp.mean(0)[i], Up[i], color='k')
    plt.vlines(p+0.025, Btm[i], Dataset_Amp.mean(0)[i], color='k')
plt.errorbar(pos+0.025, Dataset_Amp.mean(0), yerr=np.zeros(Dataset_Amp.shape[1]), fmt='o', color='w', markeredgecolor='k', ecolor='k')
del cis, Btm, Up, i, p

for col in np.arange(Dataset_Amp.shape[1]):
    plt.scatter(np.ones(Dataset_Amp.shape[0])*(pos[col]-0.03), Dataset_Amp[:,col], c=pal[col], marker='o', edgecolor='k')

plt.xlim([-0.5, 2.5])
plt.ylim([pltYmin, pltYmax])
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.xlabel('Category', fontsize=20, labelpad=10)
plt.ylabel('Amplitude (fT/cm)', fontsize=20, labelpad=10)
plt.legend(bbox_to_anchor=(1,1.01), loc='upper left', prop={'size': 17}, frameon=True, framealpha=1., edgecolor='k')
plt.gcf().subplots_adjust(top=0.93, bottom=0.11, left=0.09, right=0.82)

# save figure
plt.gcf().savefig('HemiViolinPlot_PeakAmplitude.png')
plt.close(plt.gcf())

del df_NeutF, df_FearF, df_House, df_NeutF_melt, df_FearF_melt, df_House_melt, df, pltYmin, pltYmax, v, prop, propdata, n, m, yval, pos, col


# [2] Peak Latency
# make data for plot
df_NeutF = pd.DataFrame(dict(BSF=Dataset_Lat[:,0], LSF=Dataset_Lat[:,1], HSF=Dataset_Lat[:,2], EQU=Dataset_Lat[:,3]))
df_FearF = pd.DataFrame(dict(BSF=Dataset_Lat[:,4], LSF=Dataset_Lat[:,5], HSF=Dataset_Lat[:,6], EQU=Dataset_Lat[:,7]))
df_House = pd.DataFrame(dict(BSF=Dataset_Lat[:,8], LSF=Dataset_Lat[:,9], HSF=Dataset_Lat[:,10], EQU=Dataset_Lat[:,11]))

df_NeutF_melt = pd.melt(df_NeutF)
df_NeutF_melt['Category'] = 'Neutral face'
df_FearF_melt = pd.melt(df_FearF)
df_FearF_melt['Category'] = 'Fearful face'
df_House_melt = pd.melt(df_House)
df_House_melt['Category'] = 'House'

df = pd.concat([df_NeutF_melt, df_FearF_melt, df_House_melt], axis=0)
pltYmin = 75
pltYmax = 325

# plot
plt.figure(figsize=(14.5,8))
plt.gcf().suptitle('Individual Peak Latency', fontsize=20)
v = sns.violinplot(x='Category', y='value', data=df, hue='variable', palette=pal, scale='count', inner=innerplot, dodge=True, jitter=True, bw=0.4)

prop = v.properties()
if innerplot == 'point':
    propdata = prop['children'][:24] # for the case 'inner' arg == 'point'
    propdata = propdata[::2]
else:
    propdata = prop['children'][:12] # for the case 'inner' arg == 'point'

for n in propdata:
    m = np.mean(n.get_paths()[0].vertices[:,0])
    n.get_paths()[0].vertices[:,0] = np.clip(n.get_paths()[0].vertices[:,0], m, np.inf)

# also plot something
for yval in np.arange(pltYmin, pltYmax+1, 25):
    plt.hlines(yval, -1, 3, color='gray', alpha=0.3)

# plot 95% confidence interval
cis = mne.stats.bootstrap_confidence_interval(Dataset_Lat, ci=0.95, n_bootstraps=5000, stat_fun='mean')
Btm = cis[0,:]
Up = cis[1,:]

pos = np.array([-0.3, -0.1, 0.1, 0.3, 0.7, 0.9, 1.1, 1.3, 1.7, 1.9, 2.1, 2.3])
for i, p in enumerate(pos):
    plt.vlines(p+0.025, Dataset_Lat.mean(0)[i], Up[i], color='k')
    plt.vlines(p+0.025, Btm[i], Dataset_Lat.mean(0)[i], color='k')
plt.errorbar(pos+0.025, Dataset_Lat.mean(0), yerr=np.zeros(Dataset_Lat.shape[1]), fmt='o', color='w', markeredgecolor='k', ecolor='k')
del cis, Btm, Up, i, p

for col in np.arange(Dataset_Lat.shape[1]):
    plt.scatter(np.ones(Dataset_Lat.shape[0])*(pos[col]-0.03), Dataset_Lat[:,col], c=pal[col], marker='o', edgecolor='k')

plt.xlim([-0.5, 2.5])
plt.ylim([pltYmin, pltYmax])
plt.xticks(fontsize=17)
plt.yticks(np.arange(75, 326, 25), np.arange(75, 326, 25), fontsize=17)
plt.xlabel('Category', fontsize=20, labelpad=10)
plt.ylabel('Latency (ms)', fontsize=20, labelpad=10)
plt.legend(bbox_to_anchor=(1,1.01), loc='upper left', prop={'size': 17}, frameon=True, framealpha=1., edgecolor='k')
plt.gcf().subplots_adjust(top=0.93, bottom=0.11, left=0.09, right=0.82)

# save figure
plt.gcf().savefig('HemiViolinPlot_PeakLatency.png')
plt.close(plt.gcf())

del df_NeutF, df_FearF, df_House, df_NeutF_melt, df_FearF_melt, df_House_melt, df, pltYmin, pltYmax, v, prop, propdata, n, m, yval, pos, col

