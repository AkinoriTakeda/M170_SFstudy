#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot mean amplitude distributions on brain surface
with statistical results of within-trial comparisons

@author: Akinori Takeda
"""

import mne
import numpy as np
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
    del r
del roi

CombinedLabel = HCPlabellist[0] + HCPlabellist[1] + HCPlabellist[2] + HCPlabellist[3] + HCPlabellist[4]

dirname = filedir+'/Subject1/'+ExpID+'/Datafiles/SourceEstimate/'+srcdir1
if useFsaveModel == 'ico5':
    templateSTC = mne.read_source_estimate(dirname+'/Fsaverage/SrcEst_MeanTC_FearF_BSF_fsaverage')
else:
    templateSTC = mne.read_source_estimate(dirname+'/Fsaverage_%s/SrcEst_MeanTC_FearF_BSF_fsaverage%s' % (useFsaveModel, useFsaveModel.capitalize()))
del dirname


#- load data -#
print('\n< load source timecourse data >')
if useFsaveModel == 'ico5':
    os.chdir(filedir+'/GrandAverage/Datafiles/forSurfaceSrcEstAnalysis/'+srcdir1+'/%sexp_N%d' % (ExpID, SubjN))
else:
    os.chdir(filedir+'/GrandAverage/Datafiles/forSurfaceSrcEstAnalysis/'+srcdir1+'/%sexp_N%d_fsaverage%s' % (ExpID, SubjN, useFsaveModel.capitalize()))

tmin = 0
tmax = 0.3
timemask = np.where((tmin <= times) & (times <= tmax))[0]

for cond in conditions2:
    # load surface source data
    print(' > Loading %s data...' % cond.replace('_', '-'))
    exec('SurfData_%s = np.load(\'SurfData_%s.npy\')' % (cond, cond))
    locVal['SurfData_'+cond] = locVal['SurfData_'+cond][:,:,timemask]
del cond


# load statistical data
dirname = 'WithinTrialComparisons'
statdatadir = filedir+'/GrandAverage/Datafiles/StatisticalData/SourceData/'+ExpID+'/'+srcdir1+'/'+dirname
for cond in conditions2:
    os.chdir(statdatadir+'/'+cond)
    exec('Pval_'+cond+' = np.load(\'Pvalues.npy\')')
del cond

useMinVal = False
if not useMinVal:
    alphaP = 0.005


#- value range setting -#
MinVs = []
MaxVs = []

for cond in conditions2:
    data = locVal['SurfData_'+cond][:,nuseVerts:,:]
    MeanData = data.mean(0).T
    
    if useMinVal:
        Pmask = locVal['Pval_'+cond] == locVal['Pval_'+cond].min()
    else:
        Pmask = locVal['Pval_'+cond] < alphaP
    
    MinVs.append(MeanData[Pmask].min())
    MaxVs.append(MeanData[Pmask].max())
    del data, MeanData, Pmask
del cond

MinV = np.min(MinVs)
MaxV = np.max(MaxVs)
del MinVs, MaxVs


#- directory manipulation -#
# make directory for data plot if not exist
dirlist = ['StatisticalData', 'SourceData', ExpID, srcdir1, dirname, 'SrcSpacePlots']

os.chdir(filedir+'/GrandAverage/DataPlots')
for dname in dirlist:
    if not os.path.exists('./'+dname):
        os.mkdir('./'+dname)
    os.chdir('./'+dname)
currDir = os.getcwd()


#%%
#-- plot data with statistical results --#

genTW = [80, 265]

timeIdx = np.where((genTW[0]/1000. <= times[timemask]) & (times[timemask] <= genTW[1]/1000.))[0]
times2 = np.arange(genTW[0], genTW[1]+1)

print('\n\n<< Plot amplitude distributions on brain surface with statistical results >>')
for cond in conditions2:
    print('< %s data >' % cond.replace('_', '-'))
    
    data = locVal['SurfData_'+cond]
    MeanData = data[:,:,timeIdx].mean(0)
    
    if useMinVal:
        Pmask = locVal['Pval_'+cond][timeIdx,:] == locVal['Pval_'+cond][timeIdx,:].min()
    else:
        Pmask = locVal['Pval_'+cond][timeIdx,:] < alphaP
    
    RHdata = MeanData[nuseVerts:, :]
    RHdata2 = np.zeros(RHdata.shape)
    
    for t in np.arange(Pmask.shape[0]):
        RHdata2[Pmask[t,:],t] = RHdata[Pmask[t,:], t]
    del t
    
    MeanData2 = np.zeros(MeanData.shape)
    MeanData2[nuseVerts:, :] = RHdata2
    
    SrcEst = mne.SourceEstimate(MeanData2, vertices=templateSTC.vertices, tmin=genTW[0]/1000., 
                                tstep=templateSTC.tstep, subject=MRIsubject)
    
    if '_Equ' in cond:
        kwargs = dict(colormap='hot', clim=dict(kind='value', lims=[MinV, (4+MinV)/2, 4]))
    else:
        kwargs = dict(colormap='hot', clim=dict(kind='value', lims=[MinV, (6.6+MinV)/2, 6.6]))
    
    os.chdir(currDir)
    if not os.path.exists('./'+cond):
        os.mkdir('./'+cond)
    os.chdir('./'+cond)
    
    
    for i, t in enumerate(times2):
        print('[plot data at %d ms]' % t)
        
        # make functional label
        Vmask = Pmask[i, :]
        binarydata = np.zeros(Vmask.shape)
        binarydata[Vmask] = 1
        
        binarydata2 = np.concatenate((np.zeros(Vmask.shape), binarydata))
        
        stc = np.zeros((binarydata2.shape[0],2))
        stc[:,0] = binarydata2
        
        SrcEst_label = mne.SourceEstimate(stc, vertices=templateSTC.vertices, tmin=0, 
                                          tstep=templateSTC.tstep, subject=MRIsubject)
        
        _, RHfunclabel = mne.stc_to_label(SrcEst_label, src=src, smooth=True, connected=True,
                                          subjects_dir=subjects_dir)
        del binarydata, binarydata2, stc, SrcEst_label
        
        # make surface plot
        SrcPlot = SrcEst.plot(subject=MRIsubject, surface='inflated', hemi='rh',
                              subjects_dir=subjects_dir, time_unit='ms', views='ven', 
                              initial_time=round(t), time_label=None, colorbar=True, size=(600,800), 
                              **kwargs)
        #SrcPlot.show_view(view=dict(azimuth=-270, elevation=-125, roll=0), distance=400)
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
        
        if RHfunclabel != []:
            for rhlabel in RHfunclabel:
                SrcPlot.add_label(rhlabel, borders=True, color='k')
            del rhlabel
        
        # save figure
        SrcPlot.save_single_image('SrcActivity_%dms.png' % t)
        SrcPlot.close()
        
        del Vmask, RHfunclabel, SrcPlot
    print('  --> done.\n')
    del i, t, data, MeanData, Pmask, SrcEst
del cond


#%%

#%%
'''
make figures for publication
'''
import matplotlib.pyplot as plt

cond = 'NeutF_Equ'
genTW = [165, 224]

timeIdx = np.array([i for i, t in enumerate(times[timemask]) if t*1000 in genTW])


# value range setting (unified scale)
MinVs = []
MaxVs = []
for con in ['NeutF_Equ', 'FearF_Equ', 'House_Equ']:
    data = locVal['SurfData_'+con]
    MeanData = data[:,:,timeIdx].mean(0)
    
    if useMinVal:
        Pmask = locVal['Pval_'+con][timeIdx,:] == locVal['Pval_'+con][timeIdx,:].min()
    else:
        Pmask = locVal['Pval_'+con][timeIdx,:] < alphaP
    
    RHdata = MeanData[nuseVerts:, :]
    
    MinVs.append(RHdata.T[Pmask].min())
    MaxVs.append(RHdata.T[Pmask].max())
    del data, MeanData, Pmask, RHdata
del con

MaxV = np.max(MaxVs)
MinV = np.min(MinVs)
del MaxVs, MinVs

Vmin = MinV - (MaxV-MinV)/9.
if Vmin < 0:
    Vmin = 0
Vmid = (MaxV+Vmin)/2


print('< %s data >' % cond.replace('_', '-'))
data = locVal['SurfData_'+cond]
MeanData = data[:,:,timeIdx].mean(0)

if useMinVal:
    Pmask = locVal['Pval_'+cond][timeIdx,:] == locVal['Pval_'+cond][timeIdx,:].min()
else:
    Pmask = locVal['Pval_'+cond][timeIdx,:] < alphaP

RHdata = MeanData[nuseVerts:, :]
RHdata2 = np.zeros(RHdata.shape)

for t in np.arange(Pmask.shape[0]):
    RHdata2[Pmask[t,:],t] = RHdata[Pmask[t,:], t]
del t

MeanData2 = np.zeros(MeanData.shape)
MeanData2[nuseVerts:, :] = RHdata2

SrcEst = mne.SourceEstimate(MeanData2, vertices=templateSTC.vertices, tmin=genTW[0]/1000., 
                            tstep=(genTW[1]-genTW[0])/1000, subject=MRIsubject)

if '_Equ' in cond:
    kwargs = dict(colormap='hot', clim=dict(kind='value', lims=[Vmin, Vmid, MaxV]))
else:
    kwargs = dict(colormap='hot', clim=dict(kind='value', lims=[5.8, 6.2, 6.6]))


fig = plt.figure(figsize=(13,8))
for i, t in enumerate(genTW):
    print('[plot data at %d ms]' % t)
    
    # make functional label
    idx = np.where(times[timemask][timeIdx] == t/1000)[0][0]
    Vmask = Pmask[idx, :]
    binarydata = np.zeros(Vmask.shape)
    binarydata[Vmask] = 1
    
    binarydata2 = np.concatenate((np.zeros(Vmask.shape), binarydata))
    
    stc = np.zeros((binarydata2.shape[0],2))
    stc[:,0] = binarydata2
    
    SrcEst_label = mne.SourceEstimate(stc, vertices=templateSTC.vertices, tmin=0, 
                                      tstep=templateSTC.tstep, subject=MRIsubject)
    
    _, RHfunclabel = mne.stc_to_label(SrcEst_label, src=src, smooth=True, connected=True,
                                      subjects_dir=subjects_dir)
    
    del binarydata, binarydata2, stc, SrcEst_label, idx
    
    
    # make surface plot
    SrcPlot = SrcEst.plot(subject=MRIsubject, surface='inflated', hemi='rh',
                          subjects_dir=subjects_dir, time_unit='ms', views='ven', 
                          initial_time=round(t), time_label=None, colorbar=False, background='w', size=(900,1200), 
                          **kwargs)
    #SrcPlot.show_view(view=dict(azimuth=-270, elevation=-125, roll=0), distance=400)
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
    
    if RHfunclabel != []:
        for rhlabel in RHfunclabel:
            SrcPlot.add_label(rhlabel, borders=True, color='k')
        del rhlabel
    
    # screenshot source plot
    screenshot = SrcPlot.screenshot()
    SrcPlot.close()
    
    # preparation for plotting screenshot
    nonwhitePix = (screenshot != 255).any(-1)
    nonwhite_row = nonwhitePix.any(1)
    nonwhite_col = nonwhitePix.any(0)
    cropped_screenshot = screenshot[nonwhite_row][:, nonwhite_col]
    
    plt.subplot(1, 2, (i+1))
    plt.imshow(cropped_screenshot)
    plt.ylim(cropped_screenshot.shape[0], 370)
    plt.axis('off')
    plt.title('%d ms' % t, fontsize=25)

plt.subplots_adjust(left=0.02, right=0.8, top=0.9, bottom=0.05)

colormap='hot'
if '_Equ' in cond:
    clim=dict(kind='value', lims=[round(Vmin, 2), round(Vmid, 2), round(MaxV, 2)])
else:
    clim=dict(kind='value', lims=[round(MinV, 2), round((6.6+MinV)/2, 2), 6.6])

cax = plt.axes([0.85, 0.05, 0.025, 0.25])
cbar = mne.viz.plot_brain_colorbar(cax, clim, colormap)
cbar.set_label('dSPM value',labelpad=40, rotation=270, fontsize=15)
cbar.ax.tick_params(labelsize=14)
cbar.outline.axes.spines['left'].set_visible(True)
cbar.outline.axes.spines['bottom'].set_visible(True)
cbar.outline.axes.spines['top'].set_visible(True)

# save figure
os.chdir(currDir)
fig.savefig('SrcDist_%dand%dms_%s.png' % (genTW[0], genTW[1], cond))
plt.close(fig)

