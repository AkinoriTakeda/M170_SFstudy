#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot statistical results on brain surface (post hoc paired tests)

@author: Akinori Takeda
"""

import mne
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os
locVal = locals()
sns.set_style('ticks')


#--- set data path & get datafiles' name ---#
filedir = ''
ExpID = 'SF'
useFsaveModel = 'oct6'

srcdir1 = 'SurfSrcEst_dSPM_forEvoked'


# make subject list
SubjDir = [i for i in os.listdir(filedir) if 'Subject' in i and 'Test' not in i and '.' not in i]
rmvSubjList = []

SubjList = ['Subject%d' % (n+1) for n in np.arange(len(SubjDir)) if 'Subject%d' % (n+1) not in rmvSubjList and os.path.exists(filedir+'/Subject%d/SF' % (n+1))]
SubjN = len(SubjList)
del SubjDir


#- setting some parameters -#
os.chdir(filedir+'/'+SubjList[0]+'/'+ExpID+'/Datafiles/EpochData')
Epochs = mne.read_epochs('ProperEpochData-epo.fif', preload=True)
epochs = Epochs.copy().pick_types(meg=True)

conditions = list(epochs.event_id.keys())
conditions2 = [i for i in conditions if i != 'target']
times = epochs.times
del Epochs, epochs


MRIsubject = 'fsaverage'
subjects_dir = ''

src = mne.read_source_spaces(subjects_dir+'/'+MRIsubject+'/bem/%s-%s-src.fif' % (MRIsubject, useFsaveModel))
nuseVerts = src[-1]['nuse']


# load labels of interest
HCPlabels = mne.read_labels_from_annot(MRIsubject, parc='HCPMMP1', hemi='both', 
                                       surf_name='inflated', subjects_dir=subjects_dir)

HCPlabellist = []
ROIname = ['VVC','PIT','FFC','V8','V4_']
for roi in ROIname:
    for r in [i for i in HCPlabels if roi in i.name and i.hemi=='rh']:
        HCPlabellist.append(r)

CombinedLabel = HCPlabellist[0] + HCPlabellist[1] + HCPlabellist[2] + HCPlabellist[3] + HCPlabellist[4]


# for making SourceEstimate instance of fsaverage
dirname = filedir+'/Subject1/'+ExpID+'/Datafiles/SourceEstimate/'+srcdir1
if useFsaveModel == 'ico5':
    templateSTC = mne.read_source_estimate(dirname+'/Fsaverage/SrcEst_MeanTC_FearF_BSF_fsaverage')
else:
    templateSTC = mne.read_source_estimate(dirname+'/Fsaverage_%s/SrcEst_MeanTC_FearF_BSF_fsaverage%s' % (useFsaveModel, useFsaveModel.capitalize()))


for signiROI in HCPlabellist:
    labelname = signiROI.name.split('_')[1]
    if useFsaveModel == 'ico5':
        exec('vertIdx_'+labelname+' = signiROI.get_vertices_used()')
    else:
        labelSTC = templateSTC.in_label(signiROI)
        if useFsaveModel == 'oct6':
            exec('vertIdx_'+labelname+' = np.array([i for i, n in enumerate(templateSTC.rh_vertno) if n in labelSTC.rh_vertno])')
        else:
            exec('vertIdx_'+labelname+' = labelSTC.rh_vertno')
        del labelSTC


# directory setting of statistical data 
TOImin = 137
TOImax = 265
timemask = np.where((TOImin/1000. <= times) & (times <= TOImax/1000.))[0]

dirname = 'PostAnalyses_Fsave%s_%dto%dms' % (useFsaveModel.capitalize(), TOImin, TOImax)


#- load data -#
print('\n< load source timecourse data >')
if useFsaveModel == 'ico5':
    os.chdir(filedir+'/GrandAverage/Datafiles/forSurfaceSrcEstAnalysis/'+srcdir1+'/%sexp_N%d' % (ExpID, SubjN))
else:
    os.chdir(filedir+'/GrandAverage/Datafiles/forSurfaceSrcEstAnalysis/'+srcdir1+'/%sexp_N%d_fsaverage%s' % (ExpID, SubjN, useFsaveModel.capitalize()))
for cond in conditions2:
    # load surface source data
    print(' > Loading %s data...' % cond.replace('_', '-'))
    exec('SurfData_%s = np.load(\'SurfData_%s.npy\')' % (cond, cond))
    locVal['SurfData_'+cond] = locVal['SurfData_'+cond][:,nuseVerts:,:]
del cond


#- directory manipulation -#
statdatadir = filedir+'/GrandAverage/Datafiles/StatisticalData/SourceData/'+ExpID+'/'+srcdir1+'/'+dirname

# make directory for data plot if not exist
dirlist = ['StatisticalData', 'SourceData', ExpID, srcdir1]
os.chdir(filedir+'/GrandAverage/DataPlots')
for dname in dirlist:
    if not os.path.exists('./'+dname):
        os.mkdir('./'+dname)
    os.chdir('./'+dname)
currDir = os.getcwd()


CategoryList = ['NeutF', 'FearF', 'House']
SFNameList = ['BSF', 'LSF', 'HSF', 'Equ']

alphaP = 0.05


#%%
'''
Make figures
'''

Combs = [('Category', 'BSF'), ('Category', 'LSF'), ('Category', 'HSF'), ('Category', 'Equ'),
         ('SF', 'NeutF'), ('SF', 'FearF'), ('SF', 'House')]

os.chdir(currDir+'/'+dirname)
if not os.path.exists('./PostAnalysis_SignificantVerts'):
    os.mkdir('./PostAnalysis_SignificantVerts')
os.chdir('./PostAnalysis_SignificantVerts')
savedir = os.getcwd()


# selecting a condition and a cluster
ef = Combs[6]
genTW = [186, 265]

useSmooth = True

tiltView = True
if tiltView:
    elev = -125
    vdis = 400
else:
    elev = -150
    vdis = 460


# load data
os.chdir(statdatadir+'/%s_at%s/SignificantMasks' % ef)
signiMask = np.load('SigniMask_%dto%dms.npy' % (genTW[0], genTW[1]))
Vmask = np.array([True in signiMask[:,vert] for vert in np.arange(signiMask.shape[1])])


# make functional label
binarydata = np.zeros(Vmask.shape)
binarydata[Vmask] = 1

binarydata2 = np.concatenate((np.zeros(Vmask.shape), binarydata))

stc = np.zeros((binarydata2.shape[0],2))
stc[:,0] = binarydata2

SrcEst = mne.SourceEstimate(stc, vertices=templateSTC.vertices, tmin=0, 
                            tstep=templateSTC.tstep, subject=MRIsubject)

_, RHfunclabel = mne.stc_to_label(SrcEst, src=src, smooth=True, connected=True,
                                  subjects_dir=subjects_dir)

# check the number of labels
if len(RHfunclabel) != 1:
    print('\n\n <!> several functional labels were made!')
    del RHfunclabel

del binarydata, binarydata2, stc, SrcEst


# plot vertices with significant values
print('\n<< Plot vertices within significant cluster >>')
meanTC = np.ones(templateSTC.shape)
meanTC[0,:] = 0
SrcEst = mne.SourceEstimate(meanTC, vertices=templateSTC.vertices, tmin=templateSTC.tmin, 
                            tstep=templateSTC.tstep, subject=templateSTC.subject)
kwargs = dict(colormap='mne', clim=dict(kind='value', pos_lims=[4.5, 5.5, 6.5]))


SrcPlot = SrcEst.plot(subject=MRIsubject, surface='inflated', hemi='rh',
                      subjects_dir=subjects_dir, time_unit='ms', views='ven', 
                      initial_time=0, time_label=None, colorbar=False, background='w', size=(900,1200), 
                      **kwargs, smoothing_steps=5)
SrcPlot.show_view(view=dict(azimuth=-270, elevation=elev, roll=0), distance=vdis)

for l in HCPlabellist:
    if '_VVC_' in l.name:
        color = 'c'
    elif '_FFC_' in l.name:
        color = 'lightpink'
    elif '_PIT_' in l.name:
        color = 'navy'
    elif 'V8' in l.name:
        color = 'forestgreen'
    else:
        color = 'm'
    
    SrcPlot.add_label(l, borders=True, color=color)
    del color
del l

SrcPlot.add_label(RHfunclabel[0], borders=True, color='k')

signiVerts = np.where(Vmask)[0]
if useFsaveModel == 'oct6':
    SrcPlot.add_foci(templateSTC.rh_vertno[signiVerts], coords_as_verts=True, scale_factor=0.22, color='y', hemi='rh')
else:
    SrcPlot.add_foci(signiVerts, coords_as_verts=True, scale_factor=0.22, color='y', hemi='rh')


# screenshot source plot
screenshot = SrcPlot.screenshot()
SrcPlot.close()

# preparation for plotting screenshot
nonwhitePix = (screenshot != 255).any(-1)
nonwhite_row = nonwhitePix.any(1)
nonwhite_col = nonwhitePix.any(0)
cropped_screenshot = screenshot[nonwhite_row][:, nonwhite_col]

# plot figure
plt.figure(figsize=(9,9.91))
plt.imshow(cropped_screenshot)
if tiltView:
    plt.ylim(cropped_screenshot.shape[0], 170)
else:
    plt.ylim(cropped_screenshot.shape[0], 370)
plt.axis('off')
plt.tight_layout()

# save figure
os.chdir(savedir)
if not os.path.exists('./%s_at%s' % ef):
    os.mkdir('./%s_at%s' % ef)
os.chdir('./%s_at%s' % ef)

plt.gcf().savefig('SigniVerts_%dto%dms.png' % (genTW[0], genTW[1]))
plt.close(plt.gcf())


#- also plot raw value -#
# value range setting
if ef[0] == 'Category':
    CondList = CategoryList
elif ef[0] == 'SF':
    CondList = SFNameList

MaxVs = []
MinVs = []
for cond in CondList:
    if ef[0] == 'Category':
        data = locVal['SurfData_%s_%s' % (cond, ef[-1])][:,:,timemask]
    else:
        data = locVal['SurfData_%s_%s' % (ef[-1], cond)][:,:,timemask]
    Data = data.mean(0).T
    
    MeanData = np.zeros(Vmask.shape)
    for vert in np.arange(signiMask.shape[-1]):
        tdata = Data[signiMask[:, vert], vert]
    
        if tdata.shape[0] != 0:
            MeanData[vert] = tdata.mean(0)
        del tdata
    del vert
    
    MaxVs.append(MeanData.max())
    MinVs.append(np.unique(MeanData)[1])
    del data, Data, MeanData
del cond

MaxV = np.max(MaxVs)
MinV = np.min(MinVs)
del MaxVs, MinVs

Vmin = MinV - (MaxV-MinV)/4.
if Vmin < 0:
    Vmin = 0
Vmid = (MaxV+Vmin)/2


if ef[0] == 'Category':
    fig = plt.figure(figsize=(20,8))
    for i, cond in enumerate(CategoryList):
        data = locVal['SurfData_%s_%s' % (cond, ef[-1])][:,:,timemask]
        Data = data.mean(0).T
        
        MeanData = np.zeros(Vmask.shape)
        for vert in np.arange(signiMask.shape[-1]):
            tdata = Data[signiMask[:, vert], vert]
        
            if tdata.shape[0] != 0:
                MeanData[vert] = tdata.mean(0)
            del tdata
        del vert
        
        MeanData2 = np.concatenate((np.zeros(Vmask.shape), MeanData))
        SrcData = np.zeros((MeanData2.shape[0],2))
        SrcData[:,0] = MeanData2
        del data, Data, MeanData, MeanData2
            
        
        SrcEst = mne.SourceEstimate(SrcData, vertices=templateSTC.vertices, tmin=0, 
                                    tstep=1, subject=templateSTC.subject)
        kwargs = dict(colormap='hot', clim=dict(kind='value', lims=[Vmin, Vmid, MaxV]))
        
        # plot
        if useSmooth:
            SrcPlot = SrcEst.plot(subject=MRIsubject, surface='inflated', hemi='rh',
                                  subjects_dir=subjects_dir, time_unit='ms', views='ven', 
                                  initial_time=0, time_label=None, colorbar=False, background='w', size=(900,1200), 
                                  **kwargs)
        else:
            SrcPlot = SrcEst.plot(subject=MRIsubject, surface='inflated', hemi='rh',
                                  subjects_dir=subjects_dir, time_unit='ms', views='ven', 
                                  initial_time=0, time_label=None, colorbar=False, background='w', size=(900,1200), 
                                  **kwargs, smoothing_steps=1)
        SrcPlot.show_view(view=dict(azimuth=-270, elevation=elev, roll=0), distance=vdis)
        
        for l in HCPlabellist:
            if '_VVC_' in l.name:
                color = 'c'
            elif '_FFC_' in l.name:
                color = 'lightpink'
            elif '_PIT_' in l.name:
                color = 'navy'
            elif 'V8' in l.name:
                color = 'forestgreen'
            else:
                color = 'm'
            
            SrcPlot.add_label(l, borders=True, color=color)
            del color
        del l
        
        SrcPlot.add_label(RHfunclabel[0], borders=True, color='k')
        
        # screenshot source plot
        screenshot = SrcPlot.screenshot()
        SrcPlot.close()
        
        # preparation for plotting screenshot
        nonwhitePix = (screenshot != 255).any(-1)
        nonwhite_row = nonwhitePix.any(1)
        nonwhite_col = nonwhitePix.any(0)
        cropped_screenshot = screenshot[nonwhite_row][:, nonwhite_col]
        
        plt.subplot(1, 3, (i+1))
        plt.imshow(cropped_screenshot)
        if tiltView:
            plt.ylim(cropped_screenshot.shape[0], 170)
        else:
            plt.ylim(cropped_screenshot.shape[0], 370)
        plt.axis('off')
        
        if cond == 'NeutF':
            plt.title('Neutral face', fontsize=25)
        elif cond == 'FearF':
            plt.title('Fearful face', fontsize=25)
        else:
            plt.title('House', fontsize=25)
        del SrcPlot
    del cond
    
    plt.subplots_adjust(left=0.025, right=0.85, top=0.9, bottom=0.05)
    
    colormap='hot'
    clim=dict(kind='value', lims=[round(Vmin, 2), round(Vmid, 2), round(MaxV, 2)])
    
    cax = plt.axes([0.9, 0.05, 0.02, 0.25])
    cbar = mne.viz.plot_brain_colorbar(cax, clim, colormap)
    cbar.set_label('dSPM value',labelpad=40, rotation=270, fontsize=15)
    cbar.ax.tick_params(labelsize=14)
    cbar.outline.axes.spines['left'].set_visible(True)
    cbar.outline.axes.spines['bottom'].set_visible(True)
    cbar.outline.axes.spines['top'].set_visible(True)
    
    # save figure
    os.chdir(savedir)
    if not os.path.exists('./%s_at%s' % ef):
        os.mkdir('./%s_at%s' % ef)
    os.chdir('./%s_at%s' % ef)
    
    if useSmooth:
        fig.savefig('MeanAmpAcrossSigniTime_%dto%dms.png' % (genTW[0], genTW[1]))
    else:
        fig.savefig('MeanAmpAcrossSigniTime_%dto%dms_noSmoothing.png' % (genTW[0], genTW[1]))
    plt.close(fig)
    
    
    # [additional] also make fearful-neutral face plot
    if ef[1] == 'BSF' and genTW[0] == 151:
        data1 = locVal['SurfData_NeutF_BSF'][:,:,timemask]
        data2 = locVal['SurfData_FearF_BSF'][:,:,timemask]
        Data1 = data1.mean(0).T
        Data2 = data2.mean(0).T
        Data = Data2 - Data1
        
        MeanData = np.zeros(Vmask.shape)
        for vert in np.arange(signiMask.shape[-1]):
            tdata = Data[signiMask[:, vert], vert]
        
            if tdata.shape[0] != 0:
                MeanData[vert] = tdata.mean(0)
            del tdata
        del vert
        
        MeanData2 = np.concatenate((np.zeros(Vmask.shape), MeanData))
        SrcData = np.zeros((MeanData2.shape[0],2))
        SrcData[:,0] = MeanData2
        
        MaxV2 = MeanData[Vmask].max()
        MinV2 = 0
        del data1, data2, Data1, Data2, Data, MeanData, MeanData2
            
        SrcEst = mne.SourceEstimate(SrcData, vertices=templateSTC.vertices, tmin=0, 
                                    tstep=1, subject=templateSTC.subject)
        kwargs = dict(colormap='hot', clim=dict(kind='value', lims=[MinV2, (MaxV2+MinV2)/2, MaxV2]))
        
        # plot
        if useSmooth:
            SrcPlot = SrcEst.plot(subject=MRIsubject, surface='inflated', hemi='rh',
                                  subjects_dir=subjects_dir, time_unit='ms', views='ven', 
                                  initial_time=0, time_label=None, colorbar=False, background='w', size=(900,1200), 
                                  **kwargs)
        else:
            SrcPlot = SrcEst.plot(subject=MRIsubject, surface='inflated', hemi='rh',
                                  subjects_dir=subjects_dir, time_unit='ms', views='ven', 
                                  initial_time=0, time_label=None, colorbar=False, background='w', size=(900,1200), 
                                  **kwargs, smoothing_steps=1)        #SrcPlot.show_view(view=dict(azimuth=-270, elevation=-125, roll=0), distance=400)
        SrcPlot.show_view(view=dict(azimuth=-270, elevation=elev, roll=0), distance=vdis)
        
        for l in HCPlabellist:
            if '_VVC_' in l.name:
                color = 'c'
            elif '_FFC_' in l.name:
                color = 'lightpink'
            elif '_PIT_' in l.name:
                color = 'navy'
            elif 'V8' in l.name:
                color = 'forestgreen'
            else:
                color = 'm'
            
            SrcPlot.add_label(l, borders=True, color=color)
            del color
        del l
        
        SrcPlot.add_label(RHfunclabel[0], borders=True, color='k')
        
        # screenshot source plot
        screenshot = SrcPlot.screenshot()
        SrcPlot.close()
        del SrcPlot
        
        # preparation for plotting screenshot
        nonwhitePix = (screenshot != 255).any(-1)
        nonwhite_row = nonwhitePix.any(1)
        nonwhite_col = nonwhitePix.any(0)
        cropped_screenshot = screenshot[nonwhite_row][:, nonwhite_col]
        
        # plot
        plt.figure(figsize=(12,10))
        plt.imshow(cropped_screenshot)
        if tiltView:
            plt.ylim(cropped_screenshot.shape[0], 170)
        else:
            plt.ylim(cropped_screenshot.shape[0], 370)
        plt.axis('off')
        plt.title('Fearful face - Neutral face', fontsize=25)
        
        plt.subplots_adjust(left=0.01, right=0.85, top=0.925, bottom=0.05)
        
        colormap='hot'
        clim=dict(kind='value', lims=[0, round((MaxV2+MinV2)/2, 2), round(MaxV2, 2)])
        
        cax = plt.axes([0.75, 0.05, 0.04, 0.25])
        cbar = mne.viz.plot_brain_colorbar(cax, clim, colormap)
        cbar.set_label('dSPM value',labelpad=40, rotation=270, fontsize=18)
        cbar.ax.set_yticklabels(clim['lims'])
        cbar.ax.tick_params(labelsize=18)
        cbar.outline.axes.spines['left'].set_visible(True)
        cbar.outline.axes.spines['bottom'].set_visible(True)
        cbar.outline.axes.spines['top'].set_visible(True)
        
        # save figure
        os.chdir(savedir)
        if not os.path.exists('./%s_at%s' % ef):
            os.mkdir('./%s_at%s' % ef)
        os.chdir('./%s_at%s' % ef)
        
        if useSmooth:
            plt.gcf().savefig('AmpDiff_FearFandNeutF_%dto%dms.png' % (genTW[0], genTW[1]))
        else:
            plt.gcf().savefig('AmpDiff_FearFandNeutF_%dto%dms_noSmoothing.png' % (genTW[0], genTW[1]))
        plt.close(plt.gcf())
    
else:
    fig = plt.figure(figsize=(27,10))
    for i, cond in enumerate(SFNameList):
        data = locVal['SurfData_%s_%s' % (ef[-1], cond)][:,:,timemask]
        Data = data.mean(0).T
        
        MeanData = np.zeros(Vmask.shape)
        for vert in np.arange(signiMask.shape[-1]):
            tdata = Data[signiMask[:, vert], vert]
        
            if tdata.shape[0] != 0:
                MeanData[vert] = tdata.mean(0)
            del tdata
        del vert
        
        MeanData2 = np.concatenate((np.zeros(Vmask.shape), MeanData))
        SrcData = np.zeros((MeanData2.shape[0],2))
        SrcData[:,0] = MeanData2
        del data, Data, MeanData, MeanData2
        
        SrcEst = mne.SourceEstimate(SrcData, vertices=templateSTC.vertices, tmin=0, 
                                    tstep=1, subject=templateSTC.subject)
        kwargs = dict(colormap='hot', clim=dict(kind='value', lims=[Vmin, Vmid, MaxV]))
        
        # plot
        if useSmooth:
            SrcPlot = SrcEst.plot(subject=MRIsubject, surface='inflated', hemi='rh',
                                  subjects_dir=subjects_dir, time_unit='ms', views='ven', 
                                  initial_time=0, time_label=None, colorbar=False, background='w', size=(900,1200), 
                                  **kwargs)
        else:
            SrcPlot = SrcEst.plot(subject=MRIsubject, surface='inflated', hemi='rh',
                                  subjects_dir=subjects_dir, time_unit='ms', views='ven', 
                                  initial_time=0, time_label=None, colorbar=False, background='w', size=(900,1200), 
                                  **kwargs, smoothing_steps=1)
        SrcPlot.show_view(view=dict(azimuth=-270, elevation=elev, roll=0), distance=vdis)
        
        for l in HCPlabellist:
            if '_VVC_' in l.name:
                color = 'c'
            elif '_FFC_' in l.name:
                color = 'lightpink'
            elif '_PIT_' in l.name:
                color = 'navy'
            elif 'V8' in l.name:
                color = 'forestgreen'
            else:
                color = 'm'
            
            SrcPlot.add_label(l, borders=True, color=color)
            del color
        del l
        
        SrcPlot.add_label(RHfunclabel[0], borders=True, color='k')
        
        # screenshot source plot
        screenshot = SrcPlot.screenshot()
        SrcPlot.close()
        
        # preparation for plotting screenshot
        nonwhitePix = (screenshot != 255).any(-1)
        nonwhite_row = nonwhitePix.any(1)
        nonwhite_col = nonwhitePix.any(0)
        cropped_screenshot = screenshot[nonwhite_row][:, nonwhite_col]
        
        plt.subplot(1, 4, (i+1))
        plt.imshow(cropped_screenshot)
        if tiltView:
            plt.ylim(cropped_screenshot.shape[0], 170)
        else:
            plt.ylim(cropped_screenshot.shape[0], 370)
        plt.axis('off')
        
        if cond == 'Equ':
            plt.title('Equiluminant', fontsize=25)
        else:
            plt.title(cond, fontsize=25)
        del SrcPlot
    del cond
    
    plt.subplots_adjust(left=0.025, right=0.875, top=0.93, bottom=0.025)
    
    colormap='hot'
    clim=dict(kind='value', lims=[round(Vmin, 2), round(Vmid, 2), round(MaxV, 2)])
    
    cax = plt.axes([0.91, 0.1, 0.0175, 0.25])
    cbar = mne.viz.plot_brain_colorbar(cax, clim, colormap)
    cbar.set_label('dSPM value',labelpad=40, rotation=270, fontsize=17)
    cbar.ax.tick_params(labelsize=16)
    cbar.outline.axes.spines['left'].set_visible(True)
    cbar.outline.axes.spines['bottom'].set_visible(True)
    cbar.outline.axes.spines['top'].set_visible(True)
    
    # save figure
    os.chdir(savedir)
    if not os.path.exists('./%s_at%s' % ef):
        os.mkdir('./%s_at%s' % ef)
    os.chdir('./%s_at%s' % ef)
    
    if useSmooth:
        fig.savefig('MeanAmpAcrossSigniTime_%dto%dms.png' % (genTW[0], genTW[1]))
    else:
        fig.savefig('MeanAmpAcrossSigniTime_%dto%dms_noSmoothing.png' % (genTW[0], genTW[1]))
    plt.close(fig)


#%%

#%%
'''
Plotting significant vertices at each time point
'''

Combs = [('Category', 'BSF'), ('Category', 'LSF'), ('Category', 'HSF'), ('Category', 'Equ'),
         ('SF', 'NeutF'), ('SF', 'FearF'), ('SF', 'House')]

os.chdir(currDir+'/'+dirname)
if not os.path.exists('./PostAnalysis_SignificantVerts_eachTP'):
    os.mkdir('./PostAnalysis_SignificantVerts_eachTP')
os.chdir('./PostAnalysis_SignificantVerts_eachTP')
savedir = os.getcwd()


# selecting a condition and a cluster
ef = Combs[6]
genTW = [186, 265]

useSmooth = True

timeIdx = np.where((genTW[0]/1000. <= times[timemask]) & (times[timemask] <= genTW[1]/1000.))[0]
times2 = np.arange(genTW[0], genTW[1]+1)


# load data
os.chdir(statdatadir+'/%s_at%s/SignificantMasks' % ef)
signiMask = np.load('SigniMask_%dto%dms.npy' % (genTW[0], genTW[1]))

print('\n<< Plot vertices within significant cluster >>')
for i, tIdx in enumerate(timeIdx):
    print('[plot vertices in %d ms]' % times2[i])
    Vmask = signiMask[tIdx, :]
    
    # make functional label
    binarydata = np.zeros(Vmask.shape)
    binarydata[Vmask] = 1
    
    binarydata2 = np.concatenate((np.zeros(Vmask.shape), binarydata))
    
    stc = np.zeros((binarydata2.shape[0],2))
    stc[:,0] = binarydata2
    
    SrcEst = mne.SourceEstimate(stc, vertices=templateSTC.vertices, tmin=0, 
                                tstep=templateSTC.tstep, subject=MRIsubject)
    
    _, RHfunclabel = mne.stc_to_label(SrcEst, src=src, smooth=True, connected=True,
                                      subjects_dir=subjects_dir)
    del binarydata, binarydata2, stc, SrcEst
    
    
    # plot vertices with significant values
    meanTC = np.ones(templateSTC.shape)
    meanTC[0,:] = 0
    SrcEst = mne.SourceEstimate(meanTC, vertices=templateSTC.vertices, tmin=templateSTC.tmin, 
                                tstep=templateSTC.tstep, subject=templateSTC.subject)
    kwargs = dict(colormap='mne', clim=dict(kind='value', pos_lims=[4.5, 5.5, 6.5]))
    
    SrcPlot = SrcEst.plot(subject=MRIsubject, surface='inflated', hemi='rh',
                          subjects_dir=subjects_dir, time_unit='ms', views='ven', 
                          initial_time=0, time_label=None, colorbar=False, size=(600,800), 
                          **kwargs, smoothing_steps=5)
    SrcPlot.show_view(view=dict(azimuth=-270, elevation=-150, roll=0), distance=460)
    
    for l in HCPlabellist:
        if '_VVC_' in l.name:
            color = 'c'
        elif '_FFC_' in l.name:
            color = 'lightpink'
        elif '_PIT_' in l.name:
            color = 'navy'
        elif 'V8' in l.name:
            color = 'forestgreen'
        else:
            color = 'm'
        
        SrcPlot.add_label(l, borders=True, color=color)
        del color
    del l
    
    for rhlabel in RHfunclabel:
        SrcPlot.add_label(rhlabel, borders=True, color='k')
    
    signiVerts = np.where(Vmask)[0]
    if useFsaveModel == 'oct6':
        SrcPlot.add_foci(templateSTC.rh_vertno[signiVerts], coords_as_verts=True, scale_factor=0.22, color='y', hemi='rh')
    else:
        SrcPlot.add_foci(signiVerts, coords_as_verts=True, scale_factor=0.22, color='y', hemi='rh')
    
    
    # save figure
    os.chdir(savedir)
    if not os.path.exists('./%s_at%s' % ef):
        os.mkdir('./%s_at%s' % ef)
    os.chdir('./%s_at%s' % ef)
    
    if not os.path.exists('./SigniCluster_%dto%dms' % (genTW[0], genTW[1])):
        os.mkdir('./SigniCluster_%dto%dms' % (genTW[0], genTW[1]))
    os.chdir('./SigniCluster_%dto%dms' % (genTW[0], genTW[1]))
    
    SrcPlot.save_single_image('SigniVerts_%dms.png' % times2[i])
    SrcPlot.close()


#%%
#- also plot some brains in a figure -#
Combs = [('Category', 'BSF'), ('Category', 'LSF'), ('Category', 'HSF'), ('Category', 'Equ'),
         ('SF', 'NeutF'), ('SF', 'FearF'), ('SF', 'House')]

os.chdir(currDir+'/'+dirname)
if not os.path.exists('./PostAnalysis_SignificantVerts_eachTP'):
    os.mkdir('./PostAnalysis_SignificantVerts_eachTP')
os.chdir('./PostAnalysis_SignificantVerts_eachTP')
savedir = os.getcwd()


ef = Combs[0]
genTW = [151, 183]

useSmooth = True

os.chdir(statdatadir+'/%s_at%s/SignificantMasks' % ef)
signiMask = np.load('SigniMask_%dto%dms.npy' % (genTW[0], genTW[1]))

timeIdx = np.array([i for i, t in enumerate(times[timemask]) if t in np.arange(genTW[0],genTW[1],3)/1000. or t == 0.183])

fig = plt.figure(figsize=(24,14))
for i, tIdx in enumerate(timeIdx):
    Vmask = signiMask[tIdx, :]
    
    # make functional label
    binarydata = np.zeros(Vmask.shape)
    binarydata[Vmask] = 1
    
    binarydata2 = np.concatenate((np.zeros(Vmask.shape), binarydata))
    
    stc = np.zeros((binarydata2.shape[0],2))
    stc[:,0] = binarydata2
    
    SrcEst = mne.SourceEstimate(stc, vertices=templateSTC.vertices, tmin=0, 
                                tstep=templateSTC.tstep, subject=MRIsubject)
    
    _, RHfunclabel = mne.stc_to_label(SrcEst, src=src, smooth=True, connected=True,
                                      subjects_dir=subjects_dir)
    del binarydata, binarydata2, stc, SrcEst
    
    
    # plot vertices with significant values
    meanTC = np.ones(templateSTC.shape)
    meanTC[0,:] = 0
    SrcEst = mne.SourceEstimate(meanTC, vertices=templateSTC.vertices, tmin=templateSTC.tmin, 
                                tstep=templateSTC.tstep, subject=templateSTC.subject)
    kwargs = dict(colormap='mne', clim=dict(kind='value', pos_lims=[4.5, 5.5, 6.5]))
    
    SrcPlot = SrcEst.plot(subject=MRIsubject, surface='inflated', hemi='rh',
                          subjects_dir=subjects_dir, time_unit='ms', views='ven', 
                          initial_time=0, time_label=None, colorbar=False, background='w', size=(900,1200), 
                          **kwargs)
    SrcPlot.show_view(view=dict(azimuth=-270, elevation=-150, roll=0), distance=460)
    
    for l in HCPlabellist:
        if '_VVC_' in l.name:
            color = 'c'
        elif '_FFC_' in l.name:
            color = 'lightpink'
        elif '_PIT_' in l.name:
            color = 'navy'
        elif 'V8' in l.name:
            color = 'forestgreen'
        else:
            color = 'm'
        
        SrcPlot.add_label(l, borders=True, color=color)
        del color
    del l
    
    for rhlabel in RHfunclabel:
        SrcPlot.add_label(rhlabel, borders=True, color='k')
    
    signiVerts = np.where(Vmask)[0]
    if useFsaveModel == 'oct6':
        SrcPlot.add_foci(templateSTC.rh_vertno[signiVerts], coords_as_verts=True, scale_factor=0.22, color='y', hemi='rh')
    else:
        SrcPlot.add_foci(signiVerts, coords_as_verts=True, scale_factor=0.22, color='y', hemi='rh')
    
    # screenshot source plot
    screenshot = SrcPlot.screenshot()
    SrcPlot.close()
    
    # preparation for plotting screenshot
    nonwhitePix = (screenshot != 255).any(-1)
    nonwhite_row = nonwhitePix.any(1)
    nonwhite_col = nonwhitePix.any(0)
    cropped_screenshot = screenshot[nonwhite_row][:, nonwhite_col]
    
    plt.subplot(2, 6, (i+1))
    plt.imshow(cropped_screenshot)
    plt.ylim(cropped_screenshot.shape[0], 370)
    plt.axis('off')
    plt.title('%d ms' % (times[timemask][tIdx]*1000), fontsize=22)
    del SrcPlot
del i, tIdx

plt.tight_layout()

# save figure
os.chdir(savedir)
if not os.path.exists('./%s_at%s' % ef):
    os.mkdir('./%s_at%s' % ef)
os.chdir('./%s_at%s' % ef)

if not os.path.exists('./SigniCluster_%dto%dms' % (genTW[0], genTW[1])):
    os.mkdir('./SigniCluster_%dto%dms' % (genTW[0], genTW[1]))
os.chdir('./SigniCluster_%dto%dms' % (genTW[0], genTW[1]))

fig.savefig('SigniClusters_%sAt%s_%dto%dms.png' % (ef[0], ef[1], genTW[0], genTW[1]))
plt.close(fig)


