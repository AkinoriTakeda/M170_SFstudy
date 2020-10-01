#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Plotting grand-average ERF waveforms

@author: Akinori Takeda
"""

import mne
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
locVal = locals()


#--- set data path & get datafiles' name ---#
filedir = ''
os.chdir(filedir+'/GrandAverage/Datafiles/forERFanalysis')
datadir = os.getcwd()


pltDataType = 'Grad' # 'Mag', 'Grad' or 'GradRMS'

# get information about magnetometer 
if pltDataType == 'Grad':
    print('\n< get information of magnetometer ch layout >')
    os.chdir(datadir+'/SF/Magnetometer')
    magData = mne.read_epochs(os.listdir(os.getcwd())[0], preload=True)
    magInfo = magData.info
    del magData


# load grand-average datasets
if pltDataType == 'Grad':
    direcname = 'Gradiometer'
else:
    direcname = 'Magnetometer'

DataNameList = []
print('\n< load grand-average data >')
os.chdir(datadir+'/SF/'+direcname)
filenames = os.listdir(os.getcwd())
for filename in filenames:
    # load data
    print('load %s data' % filename.split('-')[0].replace('_', '-'))
    exec(filename.split('-')[0]+'_SF = mne.read_epochs(\''+filename+'\', preload=True)')
    DataNameList.append(filename.split('-')[0]+'_SF')
    
    if pltDataType == 'Grad':
        #- make gradiometer RMS dataset -#
        print('  Make RMS data...')
        waveform = locVal[filename.split('-')[0]+'_SF'].get_data().mean(0)
        tmin = locVal[filename.split('-')[0]+'_SF'].tmin
        
        # calculate root mean square (RMS) waveform
        waveform = waveform.reshape((len(waveform) // 2, 2, -1))
        waveform = np.sqrt(np.sum(waveform ** 2, axis=1) / 2)
        exec(filename.split('-')[0][:4]+'RMS'+filename.split('-')[0][4:]+'_SF = mne.EvokedArray(waveform, magInfo, tmin=tmin)')
        DataNameList.append(filename.split('-')[0][:4]+'RMS'+filename.split('-')[0][4:]+'_SF')
        del waveform, tmin
    print('\n')
del filename, filenames
print('   ---> All data were loaded.')


if pltDataType == 'Mag':
    # get layout information
    Chlayout = mne.channels.layout.find_layout(locVal[DataNameList[0]].info)
else:
    GradDataName = [ii for ii in DataNameList if 'GradData' in ii]
    GradRMSDataName = [iii for iii in DataNameList if 'GradRMSData' in iii]
    del ii, iii
    
    # get layout information
    Chlayout = mne.channels.layout.find_layout(locVal[GradDataName[0]].info)
    del DataNameList, magInfo


#%%
'''
Preparation for plotting
(mainly for topoplot)
'''
#- setting conditions which will be plotted -#
pltType = 'acrossSF'
#pltType = 'acrossCategory'

'''
[memo] pltType
 'acrossSF' -> plot all SF conditions data within selected category 
 'acrossCategory' -> plot selected SF condition data within all category 
'''


#- parameter setting -#
pltTmin = -0.1
pltTmax = 0.4
times = np.arange(pltTmin*1e3, pltTmax*1e3+1)

# set time range for topomap
topoTime = np.concatenate((np.arange(0,101,20), np.arange(110, 201, 10), np.arange(220, 261, 20)))/1000.


CategoryList = ['FearF', 'NeutF', 'House']
SFNameList = ['BSF', 'LSF', 'HSF', 'Equ']

colorList = dict(BSF='black',LSF='blue',HSF='red',Equ='green')
lineTypeList = dict(FearF='-', NeutF='--', House=':')


if pltDataType == 'Mag':
    sensorType = 'Magnetometer'
    DataName2 = DataNameList
elif pltDataType == 'Grad':
    sensorType = 'Gradiometer'
    DataName2 = GradDataName
else:
    sensorType = 'Gradiometer RMS'
    DataName2 = GradRMSDataName
DataName = [i for i in DataName2 if 'target' not in i]
del i

# dataset setting
if pltType == 'acrossSF':
    condType1 = CategoryList
    condType2 = SFNameList
elif pltType == 'acrossCategory':
    condType1 = SFNameList
    condType2 = CategoryList



#- directory manipulation -#
# make directory for data plot if not exist
os.chdir(filedir+'/GrandAverage')
if not os.path.exists('./DataPlots'):
    os.mkdir('./DataPlots')
os.chdir('./DataPlots')

# make directory for sensor data plot if not exist
if not os.path.exists('./ERFdataPlots'):
    os.mkdir('./ERFdataPlots')
os.chdir('./ERFdataPlots')

# make directory for selected sensor-type data plot if not exist
if not os.path.exists('./'+sensorType.replace(' ', '_')):
    os.mkdir('./'+sensorType.replace(' ', '_'))
os.chdir('./'+sensorType.replace(' ', '_'))
currdir=os.getcwd()

# make directory if not exist & change current directory
currDir = currdir+'/'+pltType
os.chdir(currdir)
if pltType not in os.listdir(os.getcwd()):
    os.mkdir(currDir)

os.chdir(currDir)
if pltDataType == 'GradRMS':
    picType = ['topoplot', 'butterflyPlot']
else:
    picType = ['topoplot', 'topomap', 'butterflyPlot']
for i in picType:
    if i not in os.listdir(os.getcwd()):
        os.mkdir('./'+i)
del i



#- value range setting -#
# get max & min values
MinVs = []
MaxVs = []
for i in DataName2:
    data = locVal[i].copy().crop(tmin=pltTmin, tmax=pltTmax)
    if pltDataType == 'GradRMS':
        meanV = data.data
    else:
        meanV = data.get_data().mean(0)
    
    MinVs.append(np.min(meanV))
    MaxVs.append(np.max(meanV))
            
    del data, meanV
del i

vmin = np.min(MinVs)
vmax = np.max(MaxVs)
del MinVs, MaxVs


if pltDataType == 'GradRMS':
    Vmin = 0
else:
    Vmin = vmin*1.1
Vmax = vmax*1.1


# value range setting for topomap
if pltDataType == 'Mag':
    topoMaxV = np.max([np.abs(Vmin), np.abs(Vmax)])*1e15
    ylim=dict(mag=(Vmin*1e15, Vmax*1e15))
    ylabel = 'Amplitude [fT]'
else:
    topoMaxVs = [np.max(locVal[n].copy().crop(tmin=pltTmin, tmax=pltTmax).data) for n in GradRMSDataName]
    topoMaxV = np.max(topoMaxVs)*1e13
    del topoMaxVs
    ylim=dict(grad=(Vmin*1e13, Vmax*1e13))
    ylabel = 'Amplitude [fT/cm]'


# value range setting for selected sensor plot
if pltDataType != 'GradRMS':
    if pltType == 'acrossSF':
        withShadeCh = 'SEM' # 'SD', 'SEM' or None
    else:
        withShadeCh = None
    
    MinVs2 = []
    MaxVs2 = []
    for n in DataName2:
        data = locVal[n].copy().crop(tmin=pltTmin, tmax=pltTmax)
        meanV = data.get_data().mean(0)
    
        if withShadeCh == 'SD':
            stdV = data.get_data().std(0)
            MinVs2.append(np.min(meanV-stdV))
            MaxVs2.append(np.max(meanV+stdV))
            del stdV
        elif withShadeCh == 'SEM':
            semV = data.get_data().std(0)/np.sqrt(data.get_data().shape[0])
            MinVs2.append(np.min(meanV-semV))
            MaxVs2.append(np.max(meanV+semV))
            del semV
        else:
            MinVs2.append(np.min(meanV))
            MaxVs2.append(np.max(meanV))
                
        del data, meanV
    del n
    MinV = np.min(MinVs2)
    MaxV = np.max(MaxVs2)
    del MinVs2, MaxVs2
    
    if pltDataType == 'Mag':
        MinV *= 1.1e15
        MaxV *= 1.1e15
    else:
        MinV *= 1.1e13
        MaxV *= 1.1e13
    
    
    # also get the max value of RMS data
    if pltDataType == 'Grad':
        MaxVs2 = []
        
        for n in GradRMSDataName:
            data = locVal[n].copy().crop(tmin=pltTmin, tmax=pltTmax)
            MaxVs2.append(np.max(data.data))
            del data
        del n
        
        RMSMaxV = np.max(MaxVs2)
        RMSMaxV *= 1.1e13
        del MaxVs2



#- layout setting (coordinate alignment) -#
marginW = 0.05
marginH = 0.065

if pltDataType == 'Grad':
    pos = locVal['Chlayout'].pos.copy()
    pos2 = locVal['Chlayout'].pos.copy()
    Chidx_midline = 28*2 # Ch29 locates on midline
    
    # adjustment of plot positions
    pos[:,0] = ((pos[:,0]-pos[:,0].min())/(pos[:,0].max()-pos[:,0].min()))*(1-marginW*2)+marginW
    pos[:,1] = ((pos[:,1]-pos[:,1].min())/(pos[:,1].max()-pos[:,1].min()))*(1-marginH*2)+marginH
    APpos_error = (pos[Chidx_midline, 0]+(pos[Chidx_midline, 2]/2.)) - 0.5
    pos[:,0] = pos[:,0] - APpos_error
    pos[:,1] = pos[:,1] - APpos_error
    
    # adjustment of the size of each plot
    pos[:,2] = pos[:,2]*0.95
    pos[:,3] = pos[:,3]*0.87
    pos2[:,2:] = pos2[:,2:]*0.95

    # get channel information
    ch_names = mne.utils._clean_names(locVal[GradDataName[0]].info['ch_names'])
    iter_ch = [(x, y) for x, y in enumerate(locVal['Chlayout'].names) if y in ch_names]
    ch_pairs = np.array(iter_ch, dtype='object').reshape((len(iter_ch) // 2, -1))
else:
    pos = locVal['Chlayout'].pos.copy()
    Chidx_midline = 28 # Ch29 locates on midline
    
    # adjustment of plot positions
    pos[:,0] = ((pos[:,0]-pos[:,0].min())/(pos[:,0].max()-pos[:,0].min()))*(1-marginW*2)+marginW
    pos[:,1] = ((pos[:,1]-pos[:,1].min())/(pos[:,1].max()-pos[:,1].min()))*(1-marginH*2)+marginH
    APpos_error = (pos[Chidx_midline, 0]+(pos[Chidx_midline, 2]/2.)) - 0.5
    pos[:,0] = pos[:,0] - APpos_error
    pos[:,1] = pos[:,1] - APpos_error
    
    # adjustment of the size of each plot
    pos[:,2:] = pos[:,2:]*0.95

    # get channel information
    if pltDataType == 'Mag':
        ch_names = mne.utils._clean_names(locVal[DataNameList[0]].info['ch_names'])
    else:
        ch_names = mne.utils._clean_names(locVal[GradRMSDataName[0]].info['ch_names'])
    iter_ch = [(x, y) for x, y in enumerate(locVal['Chlayout'].names) if y in ch_names]


#%%
'''
Plot data
'''

# make plot
print('\n<< Start plotting & saving figures >>')
print('[sensor type: \'%s\', plot type: \'%s\']' % (sensorType, pltType))
expname = 'Supraliminal'
expAbbr = 'SF'

nRep = 1
for i in condType1:
    # make set of the names of data which will be plotted
    DataNames = [l for l in DataName if i in l and l.split('_')[-1] == expAbbr]
    
    # title setting
    if pltType == 'acrossSF':
        if i == 'FearF':
            condname = 'Fearful face'
        elif i == 'NeutF':
            condname = 'Neutral face'
        else:
            condname = i
        suptitle = 'ERF waveforms: %s - %s Exp - %s' % (condname, expname, sensorType)
    else:
        suptitle = 'ERF waveforms: %s condition - %s Exp - %s' % (i, expname, sensorType)
    savename = 'Topoplot_SFExp_%s_%s.png' % (i, pltDataType)
    
    suptitletopo = suptitle.replace('waveforms', 'topomaps')
    suptitlebutterfly = suptitle.replace('waveforms', 'butterfly plots')
    savenametopo = savename.replace('Topoplot', 'Topomap')
    savenamebutterfly = savename.replace('Topoplot', 'ButterflyPlot')
    
    
    #- make figures -#
    # [1] ERF topoplot 
    sns.set(style="white")
    fig = plt.figure(figsize=(23,25))
    fig.suptitle(suptitle, fontsize=25)
    for idx, name in iter_ch:
        ch_idx = ch_names.index(name)
        if idx != ch_idx:
            print('Error: index %d does not match ch_idx' % idx)
        ax = plt.axes(pos[idx])
        
        # plot
        for n in np.arange(len(condType2)):
            name = [k for k in DataNames if condType2[n] in k]
            
            data = locVal[name[0]].copy().crop(tmin=pltTmin, tmax=pltTmax)
            if pltDataType == 'GradRMS':
                meanV = data.data
            else:
                meanV = data.get_data().mean(0)
            
            # plot waveform
            if pltType == 'acrossSF':
                if condType2[n] == 'Equ':
                    labelname = 'Equiluminant'
                else:
                    labelname = condType2[n]
                
                ax.plot(times, meanV[ch_idx,:], color=colorList[condType2[n]],
                        label=labelname, linewidth=1)
            else:
                if condType2[n] == 'FearF':
                    labelname = 'Fearful face'
                elif condType2[n] == 'NeutF':
                    labelname = 'Neutral face'
                else:
                    labelname = condType2[n]
                ax.plot(times, meanV[ch_idx,:], color=colorList[i], label=labelname, 
                            linestyle=lineTypeList[condType2[n]], linewidth=1)
            del name, data, meanV
        del n
        ax.set_xlim([times[0], times[-1]])
        ax.set_ylim([Vmin, Vmax])
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.spines['bottom'].set_position(('data',0))
        ax.spines['left'].set_position(('data',0))    
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        legends, labels = ax.get_legend_handles_labels()
    fig.legend(legends, labels, loc=(0.8, 0.85), prop={'size': 22})
    
    # scale indicator
    if pltDataType == 'Grad':
        axs = plt.axes(np.array([0.1,0.03,pos2[0,2]*1.2,pos2[0,3]*1.2]))
    else:
        axs = plt.axes(np.array([0.1,0.03,pos[0,2]*1.2,pos[0,3]*1.2]))
    axs.set_xlim([times[0], times[-1]])
    axs.set_ylim([Vmin, Vmax])
    axs.vlines(0,Vmin,Vmax)
    axs.hlines(0,times[0],times[-1])
    axs.spines['right'].set_color('none')
    axs.spines['top'].set_color('none')
    axs.xaxis.set_ticks_position('bottom')
    axs.yaxis.set_ticks_position('left')
    axs.spines['bottom'].set_position(('data',0))
    axs.spines['left'].set_position(('data',0))    
    axs.set_xticklabels([])
    axs.set_yticklabels([])
    if pltDataType == 'Mag':
        axs.text(0, Vmax*1.2, '%d fT' % (Vmax*1e15), va='bottom', ha='center', fontsize=22)
        axs.text(0, Vmin*1.2, '%d fT' % (Vmin*1e15), va='top', ha='center', fontsize=22)
    elif pltDataType == 'Grad':
        axs.text(0, Vmax*1.2, '%d fT/cm' % (Vmax*1e13), va='bottom', ha='center', fontsize=22)
        axs.text(0, Vmin*1.2, '%d fT/cm' % (Vmin*1e13), va='top', ha='center', fontsize=22)
    else:
        axs.text(0, Vmax*1.1, '%d fT/cm' % (Vmax*1e13), va='bottom', ha='center', fontsize=22)
        axs.text(0, -1e-12, '0', va='top', ha='center', fontsize=22)
    axs.text(times[0]-50, 0, '%d ms' % times[0], va='center', ha='right', fontsize=22)
    axs.text(times[-1]+50, 0, '%d ms' % times[-1], va='center', ha='left', fontsize=22)
    
    # save figure
    os.chdir(currDir+'/'+picType[0])
    fig.savefig(savename)
    plt.close(fig)
    
    
    
    # [2] ERF topomap (for 'Mag' & 'Grad')
    if pltDataType != 'GradRMS':
        # plot
        sns.set(style="ticks")
        fig2, ax2 = plt.subplots(len(condType2), topoTime.shape[0]+2, figsize=(22,4))
        if pltDataType == 'Grad':
            fig2.suptitle(suptitletopo+' RMS', fontsize=18)
        else:
            fig2.suptitle(suptitletopo, fontsize=18)
        
        for n in np.arange(len(condType2)):
            name = [k for k in DataNames if condType2[n] in k]
            
            data = locVal[name[0]].copy().crop(tmin=pltTmin, tmax=pltTmax)
            avedata = data.copy().average()
            scale = None
            
            # plot topomaps
            for m, time in enumerate(topoTime):
                if n==0:
                    if m==(topoTime.shape[0]-1):
                        colorbar=True
                    else:
                        colorbar=False
                    time_format = '%01d ms'
                else:
                    colorbar=False
                    time_format = ''
                
                avedata.plot_topomap(times=time, time_format=time_format, cmap='jet', 
                                     colorbar=colorbar, vmin=-topoMaxV, vmax=topoMaxV,
                                     axes=ax2[n,m+1], units=ylabel[10:], scalings=scale,
                                     show=False, time_unit='ms')
            del m
            
            if pltType == 'acrossSF':
                if condType2[n] == 'Equ':
                    labelname = 'Equiluminant'
                else:
                    labelname = condType2[n]
                ax2[n,0].text(-0.9, 0.5, labelname)
            else:
                if condType2[n] == 'FearF':
                    labelname = 'Fearful face'
                elif condType2[n] == 'NeutF':
                    labelname = 'Neutral face'
                else:
                    labelname = condType2[n]
                ax2[n,0].text(-0.8, 0.5, labelname)
            ax2[n,0].set_axis_off()
        del n
        fig2.subplots_adjust(top=0.85, bottom=0.02)
        
        # save figure
        os.chdir(currDir+'/'+picType[1])
        fig2.savefig(savenametopo)
        plt.close(fig2)
    
    
    
    # [3] ERF butterfly plot 
    # plot
    sns.set(style="whitegrid")
    fig3, ax3 = plt.subplots(len(condType2), 1, figsize=(8,10))
    fig3.suptitle(suptitlebutterfly, fontsize=18)
    for n in np.arange(len(condType2)):
        name = [k for k in DataNames if condType2[n] in k]
        
        data = locVal[name[0]].copy().crop(tmin=pltTmin, tmax=pltTmax)
        if pltDataType == 'Mag':
            avedata = data.average().data
            avedata *= 1e15
        elif pltDataType == 'Grad':
            avedata = data.average().data
            avedata *= 1e13
        else:
            avedata = data.data
            avedata *= 1e13
        
        # plot waveforms
        for l in np.arange(avedata.shape[0]):
            ax3[n].plot(times, avedata[l,:], 'k', linewidth=0.5)
        
        ax3[n].set_xlim([times[0], times[-1]])
        if pltDataType == 'Mag':
            ax3[n].vlines(0, ylim['mag'][0], ylim['mag'][-1], linestyles='dashed')
            ax3[n].set_ylim(ylim['mag'][0], ylim['mag'][-1])
        else:
            ax3[n].vlines(0, ylim['grad'][0], ylim['grad'][-1], linestyles='dashed')
            ax3[n].set_ylim(ylim['grad'][0], ylim['grad'][-1])
        
        if pltType == 'acrossSF':
            if condType2[n] == 'Equ':
                labelname = 'Equiluminant'
            else:
                labelname = condType2[n]
        else:
            if condType2[n] == 'FearF':
                labelname = 'Fearful face'
            elif condType2[n] == 'NeutF':
                labelname = 'Neutral face'
            else:
                labelname = condType2[n]
        ax3[n].set_title(labelname)
        if n == (len(condType2)-1):
            ax3[n].set_xlabel('Time [ms]')
            ax3[n].set_ylabel(ylabel)
    del n
    plt.tight_layout()
    fig3.subplots_adjust(top=0.93)

    # save figure
    os.chdir(currDir+'/'+picType[-1])
    fig3.savefig(savenamebutterfly)
    plt.close(fig3)

    
    
    #- [4] plot ERF time course in selected Ch -#
    if pltDataType != 'GradRMS':
        os.chdir(currDir+'/'+picType[0])
        if 'IndivSensorPlots' not in os.listdir(os.getcwd()):
            os.mkdir('IndivSensorPlots')
        os.chdir('./IndivSensorPlots')

        for ChlocIdx in np.arange(102):
            # title setting
            if pltType == 'acrossSF':
                suptitleCh = 'ERF waveforms in Ch pos No.%d: %s - %s Exp - %s' % ((ChlocIdx+1), condname, expname, sensorType)
            else:
                suptitleCh = 'ERF waveforms in Ch pos No.%d: %s condition - %s Exp - %s' % ((ChlocIdx+1), i, expname, sensorType)
            savenameCh = 'Chplot_posNo%d_SFExp_%s_%s.png' % ((ChlocIdx+1), i, pltDataType)
                
            # plot
            if pltDataType == 'Mag':
                sns.set(style="whitegrid")
                figCh = plt.figure(figsize=(9,7))
                figCh.suptitle(suptitleCh, fontsize=16)
                for n in np.arange(len(condType2)):
                    name = [k for k in DataNames if condType2[n] in k]
                    data = locVal[name[0]].copy().crop(tmin=pltTmin, tmax=pltTmax)
                    
                    data = data.get_data()
                    data *= 1e15
                    meanV = data.mean(0)
                    
                    if withShadeCh == 'SD':
                        shadeV = data.std(0)
                    elif withShadeCh == 'SEM':
                        shadeV = data.std(0)/np.sqrt(data.shape[0])
                        
                    # plot waveform (& SD or SEM)
                    if pltType == 'acrossSF':
                        if condType2[n] == 'Equ':
                            labelname = 'Equiluminant'
                        else:
                            labelname = condType2[n]
                        
                        plt.plot(times, meanV[ChlocIdx,:], color=colorList[condType2[n]], 
                                 label=labelname, linewidth=1)
                        if withShadeCh != None:
                            plt.fill_between(times, meanV[ChlocIdx]-shadeV[ChlocIdx], meanV[ChlocIdx]+shadeV[ChlocIdx], 
                                             alpha=0.3, color=colorList[condType2[n]])
                            del shadeV
                    else:
                        if condType2[n] == 'FearF':
                            labelname = 'Fearful face'
                        elif condType2[n] == 'NeutF':
                            labelname = 'Neutral face'
                        else:
                            labelname = condType2[n]
                        plt.plot(times, meanV[ChlocIdx,:], color=colorList[i], label=labelname, 
                                 linestyle=lineTypeList[condType2[n]], linewidth=1)
                        if withShadeCh != None:
                            plt.fill_between(times, meanV[ChlocIdx]-shadeV[ChlocIdx], meanV[ChlocIdx]+shadeV[ChlocIdx], 
                                             alpha=0.3, color=colorList[i], linestyle=lineTypeList[condType2[n]])
                            del shadeV
                    
                    del name, data, meanV
                del n
                plt.vlines(0, MinV, MaxV)
                plt.xlim([times[0], times[-1]])
                plt.ylim([MinV, MaxV])
                plt.xlabel('Time (ms)', fontsize=15)
                plt.ylabel(ylabel, fontsize=15)
                plt.title(iter_ch[ChlocIdx][1], fontsize=15)
                plt.legend(frameon=True, framealpha=1., prop={'size': 14})
            else:
                sns.set(style="whitegrid")
                figCh = plt.figure(figsize=(18,14))
                figCh.suptitle(suptitleCh, fontsize=25)
                
                gs = gridspec.GridSpec(18,2)
                ax1 = figCh.add_subplot(gs[:8, 0])  # for gradiometers along latitude
                ax2 = figCh.add_subplot(gs[10:, 0]) # for gradiometers along longitude
                ax3 = figCh.add_subplot(gs[5:13, 1]) # for gradiometer RMS
                
                # check y-coordinates
                chpair = ch_pairs[ChlocIdx,:]
                grad1 = chpair[:2]
                grad2 = chpair[2:]
                
                if locVal['Chlayout'].pos[grad1[0]][1] > locVal['Chlayout'].pos[grad2[0]][1]:
                    latiGrad = grad1
                    longiGrad = grad2
                else:
                    latiGrad = grad2
                    longiGrad = grad1
                
                # plot
                for n in np.arange(len(condType2)):
                    name = [k for k in DataNames if condType2[n] in k]
                    data = locVal[name[0]].copy().crop(tmin=pltTmin, tmax=pltTmax)
            
                    data = data.get_data()
                    data *= 1e13
                    meanV = data.mean(0)
                    
                    if withShadeCh == 'SD':
                        shadeV = data.std(0)
                    elif withShadeCh == 'SEM':
                        shadeV = data.std(0)/np.sqrt(data.shape[0])
                    
                    # also get RMS data
                    nameRMS = [k for k in GradRMSDataName if name[0][4:] in k]
                    dataRMS = locVal[nameRMS[0]].copy().crop(tmin=pltTmin, tmax=pltTmax)
                    meanVRMS = dataRMS.data*1e13
                    
                    # plot waveform (& SD or SEM)
                    if pltType == 'acrossSF':
                        if condType2[n] == 'Equ':
                            labelname = 'Equiluminant'
                        else:
                            labelname = condType2[n]
                        
                        ax1.plot(times, meanV[latiGrad[0],:], color=colorList[condType2[n]], 
                                 label=labelname, linewidth=1)
                        ax2.plot(times, meanV[longiGrad[0],:], color=colorList[condType2[n]], 
                                 label=labelname, linewidth=1)
                        ax3.plot(times, meanVRMS[ChlocIdx,:], color=colorList[condType2[n]], 
                                 label=labelname, linewidth=1)
                        
                        if withShadeCh != None:
                            ax1.fill_between(times, meanV[latiGrad[0]]-shadeV[latiGrad[0]], meanV[latiGrad[0]]+shadeV[latiGrad[0]], 
                                             alpha=0.3, color=colorList[condType2[n]])
                            ax2.fill_between(times, meanV[longiGrad[0]]-shadeV[longiGrad[0]], meanV[longiGrad[0]]+shadeV[longiGrad[0]], 
                                             alpha=0.3, color=colorList[condType2[n]])
                            del shadeV
                    else:
                        if condType2[n] == 'FearF':
                            labelname = 'Fearful face'
                        elif condType2[n] == 'NeutF':
                            labelname = 'Neutral face'
                        else:
                            labelname = condType2[n]
                        
                        ax1.plot(times, meanV[latiGrad[0],:], color=colorList[i], label=labelname, 
                                 linestyle=lineTypeList[condType2[n]], linewidth=1)
                        ax2.plot(times, meanV[longiGrad[0],:], color=colorList[i], label=labelname, 
                                 linestyle=lineTypeList[condType2[n]], linewidth=1)
                        ax3.plot(times, meanVRMS[ChlocIdx,:], color=colorList[i], label=labelname, 
                                 linestyle=lineTypeList[condType2[n]], linewidth=1)
                        
                    del name, data, meanV, nameRMS, dataRMS, meanVRMS
                del n
                ax1.vlines(0, MinV, MaxV)
                ax2.vlines(0, MinV, MaxV)
                ax3.vlines(0, 0, RMSMaxV)
                ax1.set_xlim([times[0], times[-1]])
                ax2.set_xlim([times[0], times[-1]])
                ax3.set_xlim([times[0], times[-1]])
                ax1.set_ylim([MinV, MaxV])
                ax2.set_ylim([MinV, MaxV])
                ax3.set_ylim([0, RMSMaxV])
                ax1.set_xlabel('Time (ms)', fontsize=15)
                ax2.set_xlabel('Time (ms)', fontsize=15)
                ax3.set_xlabel('Time (ms)', fontsize=15)
                ax1.set_ylabel(ylabel, fontsize=15)
                ax2.set_ylabel(ylabel, fontsize=15)
                ax3.set_ylabel(ylabel, fontsize=15)
                ax1.set_title('%s (derivative along latitude)' % latiGrad[1], fontsize=18)
                ax2.set_title('%s (derivative along longitude)' % longiGrad[1], fontsize=18)
                ax3.set_title('%sx RMS waveforms' % latiGrad[1][:-1], fontsize=18)
                legends, labels = ax3.get_legend_handles_labels()
                figCh.legend(legends, labels, loc=(0.75, 0.75), prop={'size': 20})
            
            # save figure
            figCh.savefig(savenameCh)
            plt.close(figCh)
    
    
    print(' Finish %d/%d processes (%.1f %%)' % (nRep, len(condType1), (nRep*100.)/(len(condType1))))
    nRep += 1
        
print('   ---> All processings finished!')


