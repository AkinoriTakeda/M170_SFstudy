#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot Results of statistical analysis
(within-trial comparison/sensor data)

@author: Akinori Takeda
"""

import mne
from mne.viz import plot_topomap
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import seaborn as sns
import os
locVal = locals()
sns.set_style('ticks')


#--- set data path & get datafiles' name ---#
filedir = ''

Chtype = 'Gradiometer'
ExpID = 'SF'

#- load data -#
datadir = filedir+'/GrandAverage/Datafiles/forERFanalysis'
files = os.listdir(datadir+'/'+ExpID+'/'+Chtype)
filenames = [i for i in files if 'target' not in i]

os.chdir(datadir+'/'+ExpID+'/'+Chtype)
conditions = []
for i, filename in enumerate(filenames):
    # get condition name
    condname = filename.split('-')[0].split('_')
    cond = condname[1] + '_' + condname[2]
    conditions.append(cond)
    
    # load data
    exec(cond+' = mne.read_epochs(filename, preload=True)')
    if i == 0:
        times = locVal[cond].copy().times
        sfreq = locVal[cond].info['sfreq']
    
    # make RSS data
    data = locVal[cond].copy().get_data()
    exec(cond+'_RSS = np.zeros((data.shape[0], int(data.shape[1]/2), data.shape[2]))')
    for n in np.arange(data.shape[0]):
        wave = data[n, :, :]
        wave = wave.reshape((len(wave) // 2, 2, -1))
        wave = np.sqrt(np.sum(wave ** 2, axis=1))
        #wave = np.sqrt(np.sum(waveform ** 2, axis=1) / 2)  # RMS
        locVal[cond+'_RSS'][n, :, :] = wave
        del wave
    del n, data, condname, cond
del i, filename


# load statistical data
dirname = 'WithinTrial'
statdatadir = filedir+'/GrandAverage/Datafiles/StatisticalData/SensorData/'+ExpID+'/'+Chtype+'/'+dirname

for cond in conditions:
    os.chdir(statdatadir+'/'+cond)
    exec('Tobs_'+cond+' = np.load(\'ObservedTstatistic.npy\')')
    exec('Pval_'+cond+' = np.load(\'Pvalues.npy\')')
    exec('H0_'+cond+' = np.load(\'ObservedClusterLevelStats.npy\')')
del cond


# get magnetometer layout data
print('\n< get information of magnetometer ch layout >')
os.chdir(datadir+'/'+ExpID+'/Magnetometer')
magInfo = mne.io.read_info(os.listdir(os.getcwd())[0])

MagLayout = mne.channels.layout.find_layout(magInfo)
MagPos = MagLayout.pos[:,:2]


# get channel information
Chlayout = mne.channels.layout.find_layout(locVal['FearF_BSF'].info)
ch_names = mne.utils._clean_names(locVal['FearF_BSF'].info['ch_names'])
iter_ch = [(x, y) for x, y in enumerate(Chlayout.names) if y in ch_names]
ch_pairs = np.array(iter_ch, dtype='object').reshape((len(iter_ch) // 2, -1))


# for plotting additional Ch pos markers
Chpos=np.array(Chlayout.pos, float)[:,:2]
center = 0.5*(Chpos.max(axis=0)+Chpos.min(axis=0))
Chpos -= center
scale = 0.85/(Chpos.max(axis=0)-Chpos.min(axis=0))
Chpos*=scale

pos_x, pos_y = Chpos.T


# color scale setting for Ch pos plot
cmap = cm.RdBu_r
my_cmap = cmap(np.arange(cmap.N))
my_cmap[:,-1] = 0
my_cmap = ListedColormap(my_cmap)


#%%
'''
Preparation for plotting
'''
#- setting Exp & condition which will be plotted -#
pltType = 'acrossSF'
#pltType = 'acrossCategory'

'''
[memo] pltType
 'acrossSF' ---> plot all SF conditions data within selected category
 'acrossCategory' ---> plot selected SF condition data within all category
'''

pltTopoV = 'RMS'  # 'RMS', 'planar' or 'Tval'


#- parameter setting -#
pltTmin = -0.1
pltTmax = 0.35
timemask = np.where((pltTmin <= times) & (times <= pltTmax))[0]

# set time range for topomap
topoTime = np.concatenate((np.arange(0,101,20), np.arange(110, 201, 10), np.arange(220, 261, 20)))/1000.
topoTimemask = np.array([i for i, t in enumerate(times) if t in topoTime])


CategoryList = ['FearF', 'NeutF', 'House']
SFNameList = ['BSF', 'LSF', 'HSF', 'Equ']

colorList = dict(BSF='black',LSF='blue',HSF='red',Equ='green')
lineTypeList = dict(FearF='-', NeutF='--', House=':')


# dataset setting
if pltType == 'acrossSF':
    condType1 = CategoryList
    condType2 = SFNameList
elif pltType == 'acrossCategory':
    condType1 = SFNameList
    condType2 = CategoryList



#- directory manipulation -#
# make directory for data plot if not exist
dirlist = ['StatisticalData', 'SensorData', ExpID, Chtype, dirname, pltType]
os.chdir(filedir+'/GrandAverage/DataPlots')
for dname in dirlist:
    if not os.path.exists('./'+dname):
        os.mkdir('./'+dname)
    os.chdir('./'+dname)
currDir = os.getcwd()

os.chdir(currDir)
picType = ['topomap', 'indivCh']
for i in picType:
    if i not in os.listdir(os.getcwd()):
        os.mkdir('./'+i)
del i


#%%
#- value range setting -#
# [1] for individual sensor plot
# get max & min values
MinVs = []
MaxVs = []

for cond in conditions:
    data = locVal[cond].copy().crop(tmin=pltTmin, tmax=pltTmax).get_data()
    data *= 1e13
    
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

pltMaxV = round(Vmax, -1)
pltMinV = round(Vmin, -1)


# [2] for topomap
topoMaxVs = []
for cond in conditions:
    if pltTopoV != 'Tval':
        data = locVal[cond].copy().get_data()[:,:,timemask]
        
        if pltTopoV == 'RMS':
            wave = data.mean(0)
            wave = wave.reshape((len(wave) // 2, 2, -1))
            meanTC = np.sqrt(np.sum(wave ** 2, axis=1) / 2)
            #meanTC = np.sqrt(np.sum(wave ** 2, axis=1))  # RSS
            topoMaxVs.append(meanTC.max())
        
        if pltTopoV == 'planar':
            wave = data.reshape((data.shape[0], -1, 2, data.shape[-1])).mean(2)
            meanTC = wave.mean(0)
            topoMaxVs.append(np.max([np.abs(meanTC.max()), np.abs(meanTC.min())]))
        
        del data, wave, meanTC
    else:
        data = locVal['Tobs_'+cond].T
        meanTC = data.reshape((102, 2, data.shape[-1])).mean(1)
        topoMaxVs.append(np.max([np.abs(meanTC.max()), np.abs(meanTC.min())]))
        del data, meanTC
del cond

if pltTopoV != 'Tval':
    topoMaxV = np.max(topoMaxVs)*1e13
    topoylabel = 'Amplitude (fT/cm)'
else:
    topoMaxV = np.max(topoMaxVs)
    topoylabel = 'Adjusted $\it{t}$ values'
del topoMaxVs

ylabel = 'Amplitude (fT/cm)'
Topomaskparams = dict(marker='o', markerfacecolor='k', markeredgecolor='k', markersize=1)


#%%
'''
Make topomaps
'''
useMinPval = True
if not useMinPval:
    alphaP = 0.005

for i in condType1:
    # title setting
    if pltType == 'acrossSF':
        if i == 'FearF':
            condname = 'Fearful face'
        elif i == 'NeutF':
            condname = 'Neutral face'
        else:
            condname = i
        suptitle = 'ERF waveforms (Gradiometer RSS): %s' % condname
    else:
        if i == 'Equ':
            condname = 'Equiluminant'
        else:
            condname = i
        suptitle = 'ERF waveforms (Gradiometer RSS): %s condition' % condname
    savename = 'Topomap_%s.png' % i
    

    # make topomap
    sns.set(style="ticks")
    fig, ax = plt.subplots(len(condType2), topoTime.shape[0]+2, figsize=(22,4))
    fig.suptitle(suptitle, fontsize=18)
    
    for m, n in enumerate(condType2):
        if pltType == 'acrossSF':
            data = locVal[i+'_'+n+'_RSS'][:,:,topoTimemask]
            
            if useMinPval:
                Pval = locVal['Pval_'+i+'_'+n].reshape(locVal['Clusters'].shape) == locVal['Pval_'+i+'_'+n].min()
            else:
                Pval = locVal['Pval_'+i+'_'+n].reshape(locVal['Clusters'].shape) < alphaP
        else:
            data = locVal[n+'_'+i+'_RSS'][:,:,topoTimemask]
            
            if useMinPval:
                Pval = locVal['Pval_'+n+'_'+i].reshape(locVal['Clusters'].shape) == locVal['Pval_'+n+'_'+i].min()
            else:
                Pval = locVal['Pval_'+n+'_'+i].reshape(locVal['Clusters'].shape) < alphaP
        
        avedata = data.mean(0)*1e13
        timeIdx = np.unique(np.where(Pval)[0])
        spaceIdx = [np.where(Pval[t,:])[0] for t in timeIdx]
        
        spaceIdxTopo = []
        for l, time in enumerate(topoTime):
            if time*1000 in timeIdx:
                sIdx = spaceIdx[np.where(timeIdx == time*1000)[0][0]]
                
                sensors = np.zeros(204, dtype=bool)
                sensors[sIdx] = True
                sensorIdx = np.zeros((102,1), dtype=bool)
                for s in np.arange(sensorIdx.shape[0]):
                    si = sensors.reshape(102,-1)[s,:]
                    if True in si:
                        sensorIdx[s,:] = True
                    del si
                del s
                
                spaceIdxTopo.append(sensorIdx)
                del sIdx, sensors, sensorIdx
            else:
                spaceIdxTopo.append(None)
        del l, time
        
        
        # plot
        for l, time in enumerate(topoTime):
            if time*1000 in timeIdx:
                im, _ = plot_topomap(avedata[:,l], magInfo, vmin=-topoMaxV, vmax=topoMaxV, res=600,
                                     cmap='jet', axes=ax[m,l+1], show=False, contours=0,
                                     mask=spaceIdxTopo[l], mask_params=Topomaskparams)
            else:
                im, _ = plot_topomap(avedata[:,l], magInfo, vmin=-topoMaxV, vmax=topoMaxV, res=600,
                                     cmap='jet', axes=ax[m,l+1], show=False, contours=0)
            if m == 0:
                ax[m, l+1].set_title('%d ms' % (time*1000), fontsize=10)
        
        if pltType == 'acrossSF':
            if n == 'Equ':
                ax[m,0].text(-0.9, 0.5, 'Equiluminant', fontsize=12)
            else:
                ax[m,0].text(-0.2, 0.5, n, fontsize=12)
        else:
            if n == 'FearF':
                ax[m,0].text(-0.8, 0.5, 'Fearful face', fontsize=12)
            elif n == 'NeutF':
                ax[m,0].text(-0.8, 0.5, 'Neutral face', fontsize=12)
            else:
                ax[m,0].text(-0.7, 0.5, 'House', fontsize=12)
        ax[m,0].set_axis_off()
        ax[m,-1].set_axis_off()
        
        del data, avedata, l, time
    del m, n
    
    # add colorbar
    axes = plt.axes([0.27, 0.05, 0.73, 0.8])
    clb = fig.colorbar(im, ax=axes)
    clb.set_label(topoylabel, labelpad=20, rotation=270, fontsize=12)
    clb.ax.tick_params(labelsize=10)
    axes.set_axis_off()
    
    fig.subplots_adjust(top=0.85, bottom=0.02)
    
    # save figure
    os.chdir(currDir+'/topomap')
    fig.savefig(savename)
    plt.close(fig)
    
    del condname, suptitle, savename, fig, ax, im, axes, clb
del i


#%%
'''
Make individual sensor plot
'''

print('\n<< Make individual sensor plots >>')
for cond in conditions:
    os.chdir(currDir+'/indivCh')
    if not os.path.exists('./'+cond):
        os.mkdir('./'+cond)
    os.chdir('./'+cond)
    print('> Processing %s data...' % cond.replace('_','-'))
    
    #-- preparation for plotting --#
    if useMinPval:
        Pval = locVal['Pval_'+cond] == locVal['Pval_'+cond].min()
    else:
        Pval = locVal['Pval_'+cond] < alphaP
    Pval_liberal = locVal['Pval_'+cond] < 0.05
    
    #- detect time points & sensors with siginificant statistical values -#
    timeIdx = np.where(Pval)[0]
    timeIdx = np.unique(timeIdx)
    spaceIdx = [np.where(Pval[t,:])[0] for t in timeIdx]
    
    timeIdx_lib = np.unique(np.where(Pval_liberal)[0])
    spaceIdx_lib = [np.where(Pval_liberal[t,:])[0] for t in timeIdx_lib]
    
    #- make data for plot -#
    pltTimes = locVal[cond].copy().crop(tmin=-0.1, tmax=0.35).times*1000
    data = locVal[cond].copy().crop(tmin=-0.1, tmax=0.35).get_data()
    data *= 1e13
    sfreq = locVal[cond].info['sfreq']
    
    #- detect peak values -#
    TWmask = np.array([i for i, t in enumerate(pltTimes) if t in timeIdx])
    Chmask = np.unique(np.concatenate([np.where(Pval[t,:])[0] for t in timeIdx]))
    PeakValues = data.mean(0)[Chmask, :][:, TWmask]
    MaxV = PeakValues.min()
    data2 = locVal[cond].copy().crop(tmin=0, tmax=0.35).get_data()
    data2 *= 1e13
    
    MaxVch, MaxVt = np.where(data2.mean(0) == MaxV)
    
    
    # < Plot summary > #
    spcMask = np.unique(np.concatenate(spaceIdx))
    mask = np.zeros((Pval.shape[1],1), dtype=bool)
    mask[spcMask, :] = True
    maskparams = dict(marker='+', markerfacecolor='r', markeredgecolor='r', markersize=7)
    
    PeakTval = locVal['Tobs_'+cond][MaxVt[0], MaxVch[0]]
    PeakPval = locVal['Pval_'+cond][MaxVt[0], MaxVch[0]]
    
    # plot #
    plt.figure(figsize=(7,7))
    ax = plt.gcf().gca()
    plt.suptitle('Condition: %s\nPeak Latency: %d ms (Ch: %s)' % (cond.replace('_','-'), MaxVt[0], ch_names[MaxVch[0]]) + '\n(Adjusted $\it{0}$ value = {1}, $\it{2}$ = {3})'.format('t', round(PeakTval, 2), 'p', PeakPval))
    im, _ = plot_topomap(np.zeros(Pval.shape[1]), Chlayout.pos, mask=mask, cmap=my_cmap, res=600,
                         sensors='k,', axes=ax, mask_params=maskparams, vmin=-6, vmax=6)
    
    spcMask_lib = np.unique(np.concatenate(spaceIdx_lib))
    ax.scatter(pos_x[spcMask_lib], pos_y[spcMask_lib], c='k', marker='+', linewidth=0.9)
    
    spcMask_notSigni = np.array([i for i in np.arange(Pval.shape[1]) if i not in spcMask_lib])
    ax.scatter(pos_x[spcMask_notSigni], pos_y[spcMask_notSigni], s=10, c='k', marker='o', edgecolors='black')
    del spcMask_notSigni
    
    for n in np.arange(10):
        ax.scatter(pos_x[MaxVch[0]], pos_y[MaxVch[0]], s=160, c='w', marker='o', edgecolors='black', alpha=1)
        
    if not useMinPval:
        axes = plt.axes([0.2, 0, 0.6, 0.12])
        axes.text(0.4, 0.8, r'*$\alpha$ < {0}'.format(alphaP), fontsize=14)
        axes.set_axis_off()
        del axes
    
    # save figure
    os.chdir(currDir+'/indivCh/'+cond)
    fig = plt.gcf()
    fig.savefig('SensorsWithSigniVal_%s.png' % cond)
    plt.close(fig)
    del fig, ax, im
    
    
    # < Plot individual data > #
    for pltChidx in np.arange(locVal['Tobs_'+cond].shape[-1]):
        #-- [plot 1] plot waveform --#
        #- identify temporal clusters -#
        signi_tp = np.where(Pval[:,pltChidx])[0]
        identified_timeclus = []
        if signi_tp.shape[0] == 1:
            identified_timeclus.append(signi_tp.tolist())
        elif signi_tp.shape[0] >= 2:
            cluster_tp = []
            for i, tp in enumerate(signi_tp[:-1]):
                cluster_tp.append(tp)
                
                if (signi_tp[(i+1)] - tp) > (1/sfreq)*1000:
                    identified_timeclus.append(cluster_tp)
                    del cluster_tp
                    cluster_tp = []
                
                if i == (len(signi_tp[:-1])-1):
                    identified_timeclus.append(cluster_tp)
            del i, tp, cluster_tp
            
            if identified_timeclus[-1]==[] or signi_tp[-1] == (identified_timeclus[-1][-1]+(1/sfreq)*1000):
                identified_timeclus[-1].append(signi_tp[-1])
            
        # also identify temporal clusters in more liberal data
        signi_tp2 = np.where(Pval_liberal[:,pltChidx])[0]
        identified_timeclus2 = []
        if signi_tp2.shape[0] == 1:
            identified_timeclus2.append(signi_tp2.tolist())
        elif signi_tp2.shape[0] >= 2:
            cluster_tp2 = []
            for i, tp in enumerate(signi_tp2[:-1]):
                cluster_tp2.append(tp)
                
                if (signi_tp2[(i+1)] - tp) > (1/sfreq)*1000:
                    identified_timeclus2.append(cluster_tp2)
                    del cluster_tp2
                    cluster_tp2 = []
                
                if i == (len(signi_tp2[:-1])-1):
                    identified_timeclus2.append(cluster_tp2)
            del i, tp, cluster_tp2
            
            if identified_timeclus2[-1]==[] or signi_tp2[-1] == (identified_timeclus2[-1][-1]+(1/sfreq)*1000):
                identified_timeclus2[-1].append(signi_tp2[-1])
        
        #- make mean time course -#
        meanTC = data[:,pltChidx,:].mean(0)
        
        # 95% confidence interval
        cis = mne.stats.bootstrap_confidence_interval(data[:,pltChidx,:], ci=0.95, n_bootstraps=5000, stat_fun='mean')
        Btm = cis[0,:]
        Up = cis[1,:]
        
        #- plot -#
        plt.figure(figsize=(6.5,5))
        plt.suptitle('%s (Ch: %s)' % (cond.replace('_','-'), ch_names[pltChidx]))
        plt.vlines(0, pltMinV-9, pltMaxV, linewidth=1)
        plt.ylim(pltMinV-9, pltMaxV)
        
        plt.hlines(0, pltTimes[0], pltTimes[-1], linewidth=1)
        plt.plot(pltTimes, meanTC, linewidth=2, color=colorList[cond.split('_')[-1]])
        plt.fill_between(pltTimes, Btm, Up, alpha=0.3, color=colorList[cond.split('_')[-1]])
        plt.xlim(pltTimes[0], pltTimes[-1])
        plt.gcf().gca().spines['right'].set_color('none')
        plt.gcf().gca().spines['top'].set_color('none')
        plt.gcf().gca().spines['left'].set_linewidth(2)
        plt.gcf().gca().spines['bottom'].set_linewidth(2)
        plt.xlabel('Time (ms)',fontsize=17, labelpad=10)
        plt.ylabel(ylabel,fontsize=17, labelpad=10)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        
        #- shading significant time windows -#
        yval = pltMinV-3.5

        # using liberal alpha (0.05)
        if signi_tp2.shape[0] >= 1:
            color = 'gray'
            alpha = 0.8
            
            for c in identified_timeclus2:
                if len(c) == 1:
                    plt.vlines(c[0], pltMinV-5.5, pltMinV-1.5, alpha=alpha, color=color, linewidth=1)
                elif len(c) >= 2:
                    plt.fill_between(c, np.ones(len(c))*(pltMinV-5.5), np.ones(len(c))*(pltMinV-1.5), 
                                     alpha=alpha, color=color)
            del c, color, alpha
            
            # plot horizontal line
            for i, c in enumerate(identified_timeclus2):
                if i == 0:
                    plt.hlines(yval, 0, c[0], linewidth=1)
                    prevEnd = c[-1]
                else:
                    plt.hlines(yval, prevEnd, c[0], linewidth=1)
                    prevEnd = c[-1]
                
                if i == (len(identified_timeclus2)-1) and c[-1] != pltTimes[-1]:
                    plt.hlines(yval, c[-1], pltTimes[-1], linewidth=1)
            del i, c, prevEnd, yval
        else:
            plt.hlines(yval, 0, pltTimes[-1], linewidth=1)
        
        # using more rigolous alpha
        if signi_tp.shape[0] >= 1:
            color = 'k'
            alpha = 1
            
            for c in identified_timeclus:
                if len(c) == 1:
                    plt.vlines(c[0], pltMinV-5.5, pltMinV-1.5, alpha=alpha, color=color, linewidth=1)
                elif len(c) >= 2:
                    plt.fill_between(c, np.ones(len(c))*(pltMinV-5.5), np.ones(len(c))*(pltMinV-1.5), 
                                     alpha=alpha, color=color)
            del c, color, alpha
        
        plt.gcf().subplots_adjust(top=0.92, bottom=0.15, left=0.16, right=0.96)
        
        # save figure
        os.chdir(currDir+'/indivCh/'+cond)
        if not os.path.exists('./Waveform'):
            os.mkdir('./Waveform')
        os.chdir('./Waveform')
        
        fig = plt.gcf()
        fig.savefig('Waveform_%s_%s.png' % (cond, ch_names[pltChidx]))
        plt.close(fig)
        del fig
        
        
        #-- [plot 2] plot sensor location --#
        #- [Version 1] plot sensors at peak latency -#
        if signi_tp.shape[0] != 0:
            if pltChidx == MaxVch[0]:
                t = MaxVt[0]
            else:
                if signi_tp.shape[0] == 1:
                    t = signi_tp[0]
                elif signi_tp.shape[0] >= 2:
                    thisMaxV = data2.mean(0)[pltChidx,signi_tp].min()
                    t = np.where(data2.mean(0)[pltChidx,:] == thisMaxV)[0][0]
            
            mask = np.zeros((Pval.shape[1],1), dtype=bool)
            mask[spaceIdx[np.where(timeIdx == t)[0][0]], :] = True
            maskparams = dict(marker='+', markerfacecolor='r', markeredgecolor='r', markersize=7)
            
            PeakTval = locVal['Tobs_'+cond][t, pltChidx]
            PeakPval = locVal['Pval_'+cond][t, pltChidx]
            
            # plot #
            plt.figure(figsize=(7,7))
            ax = plt.gcf().gca()
            plt.suptitle('Peak Latency: %d ms' % t + '\n(Adjusted $\it{0}$ value = {1}, $\it{2}$ = {3})'.format('t', round(PeakTval, 2), 'p', PeakPval))
            im, _ = plot_topomap(np.zeros(Pval.shape[1]), Chlayout.pos, mask=mask, cmap=my_cmap, res=600,
                                 sensors='k,', axes=ax, mask_params=maskparams, vmin=-6, vmax=6)
            
            Chidx = spaceIdx_lib[np.where(timeIdx_lib == t)[0][0]]
            ax.scatter(pos_x[Chidx], pos_y[Chidx], c='k', marker='+', linewidth=0.9)
            
            spcMask_notSigni = np.array([i for i in np.arange(Pval.shape[1]) if i not in Chidx])
            ax.scatter(pos_x[spcMask_notSigni], pos_y[spcMask_notSigni], s=5, c='k', marker='o', edgecolors='black')
            del spcMask_notSigni
            
            for n in np.arange(10):
                ax.scatter(pos_x[pltChidx], pos_y[pltChidx], s=160, c='w', marker='o', edgecolors='black', alpha=1)
            
            # save figure
            os.chdir(currDir+'/indivCh/'+cond)
            if not os.path.exists('./SensorLoc_PeakLatency'):
                os.mkdir('./SensorLoc_PeakLatency')
            os.chdir('./SensorLoc_PeakLatency')
            
            fig = plt.gcf()
            fig.savefig('SensorLoc1_%s_%s.png' % (cond, ch_names[pltChidx]))
            plt.close(fig)
            del fig, ax, im

        
            #- [Version 2] plot sensors with significant values in time window of temporal clusters -#
            if pltChidx == MaxVch[0]:
                t = MaxVt[0]
                tempclus = [c for c in identified_timeclus if t in c]
            else:
                if signi_tp.shape[0] == 1:
                    t = signi_tp[0]
                    tempclus = identified_timeclus
                elif signi_tp.shape[0] >= 2:
                    thisMaxV = data2.mean(0)[pltChidx,signi_tp].min()
                    t = np.where(data2.mean(0)[pltChidx,:] == thisMaxV)[0][0]
                    tempclus = [c for c in identified_timeclus if t in c]
                    
            if tempclus == []:
                tw = np.concatenate(identified_timeclus)
                thisMaxV = data2.mean(0)[pltChidx,tw].min()
                t = np.where(data2.mean(0)[pltChidx,:] == thisMaxV)[0][0]
                tempclus = [c for c in identified_timeclus if t in c]
            tempclus = tempclus[0]
            
            spatclus = np.unique(np.concatenate([np.where(Pval[i,:])[0] for i in tempclus]))
            
            PeakTvals = locVal['Tobs_'+cond][tempclus, pltChidx]
            PeakPvals = locVal['Pval_'+cond][tempclus, pltChidx]
            Tvalues = np.unique(PeakTvals)
            Pvalues = np.unique(PeakPvals)
            
            mask = np.zeros((Pval.shape[1],1), dtype=bool)
            mask[spatclus, :] = True
            maskparams = dict(marker='+', markerfacecolor='r', markeredgecolor='r', markersize=7)
            
            # title setting
            if len(tempclus) == 1:
                title = 'Peak Latency: %d ms (*only 1 t.p. was significant)' % t + '\n(Adjusted $\it{0}$ value = {1}, $\it{2}$ = {3})'.format('t', round(Tvalues[0], 2), 'p', Pvalues[0])
            elif len(tempclus) >= 2:
                title1 = 'Time Interval: %d-%d ms' % (tempclus[0], tempclus[-1])
                if useMinPval:
                    if Tvalues.shape[0] > 1:
                        title2 = '\n(Adjusted $\it{0}$ values = {1}-{2}, $\it{3}$ = {4})'.format('t', round(PeakTvals.min(), 2), round(PeakTvals.max(), 2), 'p', locVal['Pval_'+cond].min())
                    else:
                        title2 = '\n(Adjusted $\it{0}$ value = {1}, $\it{2}$ = {3})'.format('t', round(Tvalues[0], 2), 'p', locVal['Pval_'+cond].min())
                else:
                    if Tvalues.shape[0] > 1:
                        if Pvalues.shape[0] > 1:
                            title2 = '\n(Adjusted $\it{0}$ values = {1}-{2}, $\it{3}$ $\leqq$ {4})'.format('t', round(PeakTvals.min(), 2), round(PeakTvals.max(), 2), 'p', Pvalues.max())
                        else:
                            title2 = '\n(Adjusted $\it{0}$ values = {1}-{2}, $\it{3}$ = {4})'.format('t', round(PeakTvals.min(), 2), round(PeakTvals.max(), 2), 'p', Pvalues[0])
                    else:
                        if Pvalues.shape[0] > 1:
                            title2 = '\n(Adjusted $\it{0}$ value = {1}, $\it{2}$ $\leqq$ {3})'.format('t', round(Tvalues[0], 2), 'p', Pvalues.max())
                        else:
                            title2 = '\n(Adjusted $\it{0}$ value = {1}, $\it{2}$ = {3})'.format('t', round(Tvalues[0], 2), 'p', Pvalues[0])
                title = title1 + title2
            
            # plot
            plt.figure(figsize=(7,7))
            ax = plt.gcf().gca()
            plt.suptitle(title)
            im, _ = plot_topomap(np.zeros(Pval.shape[1]), Chlayout.pos, mask=mask, cmap=my_cmap, res=600,
                                 sensors='k,', axes=ax, mask_params=maskparams, vmin=-6, vmax=6)
            
            if identified_timeclus2 != []:
                spatclus2 = np.unique(np.concatenate([np.where(Pval_liberal[i,:])[0] for i in tempclus]))
            
            ax.scatter(pos_x[spatclus2], pos_y[spatclus2], c='k', marker='+', linewidth=0.9)
            
            spcMask_notSigni = np.array([i for i in np.arange(Pval.shape[1]) if i not in spatclus2])
            ax.scatter(pos_x[spcMask_notSigni], pos_y[spcMask_notSigni], s=5, c='k', marker='o', edgecolors='black')
            del spcMask_notSigni
            
            for n in np.arange(10):
                ax.scatter(pos_x[pltChidx], pos_y[pltChidx], s=160, c='w', marker='o', edgecolors='black', alpha=1)
            
            # plot additional information
            axes = plt.axes([0.2, 0, 0.6, 0.12])
            if not useMinPval:
                axes.text(0.4, 0.8, r'*$\alpha$ < {0}'.format(alphaP), fontsize=14)
            
            if identified_timeclus2 != []:
                tempclus2 = [c for c in identified_timeclus2 if t in c][0]
                
                PeakTvals2 = locVal['Tobs_'+cond][tempclus2, pltChidx]
                PeakPvals2 = locVal['Pval_'+cond][tempclus2, pltChidx]
                Tvalues2 = np.unique(PeakTvals2)
                Pvalues2 = np.unique(PeakPvals2)
                
                text1 = r'[If set $\alpha$ < 0.05] Time Interval: %d-%d ms' % (tempclus2[0], tempclus2[-1])
                if Tvalues2.shape[0] > 1:
                    if Pvalues2.shape[0] > 1:
                        text2 = '\n(Adjusted $\it{0}$ values = {1}-{2}, $\it{3}$ $\leqq$ {4})'.format('t', round(PeakTvals2.min(), 2), round(PeakTvals2.max(), 2), 'p', Pvalues2.max())
                    else:
                        text2 = '\n(Adjusted $\it{0}$ values = {1}-{2}, $\it{3}$ = {4})'.format('t', round(PeakTvals2.min(), 2), round(PeakTvals2.max(), 2), 'p', Pvalues2[0])
                else:
                    if Pvalues2.shape[0] > 1:
                        text2 = '\n(Adjusted $\it{0}$ value = {1}, $\it{2}$ $\leqq$ {3})'.format('t', round(Tvalues2[0], 2), 'p', Pvalues2.max())
                    else:
                        text2 = '\n(Adjusted $\it{0}$ value = {1}, $\it{2}$ = {3})'.format('t', round(Tvalues2[0], 2), 'p', Pvalues2[0])
                
                axes.text(0.05, 0.2, text1+text2)
            axes.set_axis_off()
            
            # save figure
            os.chdir(currDir+'/indivCh/'+cond)
            if not os.path.exists('./SensorLoc_TemporalClus'):
                os.mkdir('./SensorLoc_TemporalClus')
            os.chdir('./SensorLoc_TemporalClus')
            
            fig = plt.gcf()
            fig.savefig('SensorLoc2_%s_%s.png' % (cond, ch_names[pltChidx]))
            plt.close(fig)
            del fig, ax, im, axes

print('\n  ==> Finished.')


