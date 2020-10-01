#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot statistical results on brain surface (1-way ANOVA)

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

MRIsubject = 'fsaverage'
subjects_dir = ''

src = mne.read_source_spaces(subjects_dir+'/'+MRIsubject+'/bem/%s-%s-src.fif' % (MRIsubject, useFsaveModel))

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


# directory setting of statistical data 
TOImin = 137
TOImax = 265

dirname = 'PostAnalyses_Fsave%s_%dto%dms' % (useFsaveModel.capitalize(), TOImin, TOImax)

ANOVAdirname = 'ANOVA_Fsave%s_%dto%dms' % (useFsaveModel.capitalize(), TOImin, TOImax)
ANOVAdatadir = filedir+'/GrandAverage/Datafiles/StatisticalData/SourceData/'+ExpID+'/'+srcdir1+'/'+ANOVAdirname+'/Interaction'
os.chdir(ANOVAdatadir)
Pval_Interaction = np.load('Pvalues.npy')
del ANOVAdirname


#- directory manipulation -#
statdatadir = filedir+'/GrandAverage/Datafiles/StatisticalData/SourceData/'+ExpID+'/'+srcdir1+'/'+dirname

# make directory for data plot if not exist
dirlist = ['StatisticalData', 'SourceData', ExpID, srcdir1, dirname, 'SrcSpacePlots']

os.chdir(filedir+'/GrandAverage/DataPlots')
for dname in dirlist:
    if not os.path.exists('./'+dname):
        os.mkdir('./'+dname)
    os.chdir('./'+dname)
currDir = os.getcwd()


#%%
Combs = [('Category', 'BSF'), ('Category', 'LSF'), ('Category', 'HSF'), ('Category', 'Equ'),
         ('SF', 'NeutF'), ('SF', 'FearF'), ('SF', 'House')]

pltTimes = np.arange(137, 266, 1)
alphaP = 0.05

tiltView = True
if tiltView:
    elev = -125
    vdis = 400
else:
    elev = -150
    vdis = 460


print('\n<< Plot TFCE scores on brain surface >>')
for comb in Combs:
    factor = comb[0]
    cond = comb[1]
    print('< %s at %s >' % (factor, cond))
    
    # load data
    os.chdir(statdatadir+'/%s_at%s' % (factor, cond))
    scores_1wayANOVA = np.load('TFCEscores.npy')
    Pval_1wayANOVA = np.load('Pvalues.npy')
    
    # make dataset
    Pv_1wayANOVA = Pval_1wayANOVA < alphaP
    
    # make SourceEstimate instance
    lefthemidata = np.zeros(scores_1wayANOVA.T.shape)
    scores = np.concatenate((lefthemidata, scores_1wayANOVA.T))
    SrcEst = mne.SourceEstimate(scores, vertices=templateSTC.vertices, tmin=TOImin/1000, 
                                tstep=templateSTC.tstep, subject=MRIsubject)
    
    # colormap setting
    if True in Pv_1wayANOVA:
        kwargs = dict(colormap='hot', clim=dict(kind='value', lims=[0, scores_1wayANOVA[Pv_1wayANOVA].min(), scores_1wayANOVA.max()]))
    else:
        kwargs = dict(colormap='hot', clim=dict(kind='value', lims=[0, scores_1wayANOVA.max()/2, scores_1wayANOVA.max()]))
    
    # plot
    for t in pltTimes:
        SrcPlot = SrcEst.plot(subject=MRIsubject, surface='inflated', hemi='rh',
                              subjects_dir=subjects_dir, time_unit='ms', views='ven', 
                              initial_time=t, time_label='%d ms', colorbar=True, size=(600,800), 
                              **kwargs, smoothing_steps=5)
        SrcPlot.show_view(view=dict(azimuth=-270, elevation=elev, roll=0), distance=vdis)
        SrcPlot.add_label(CombinedLabel, borders=True)
        
        # make directory for saving figures
        os.chdir(currDir)
        if not os.path.exists('./%s_at%s' % (factor, cond)):
            os.mkdir('./%s_at%s' % (factor, cond))
        os.chdir('./%s_at%s' % (factor, cond))
        
        # save figure
        SrcPlot.save_single_image('Fvalues_%dms.png' % t)
        SrcPlot.close()
        del SrcPlot
    del t, lefthemidata, scores, SrcEst, kwargs
del comb    


# plot each vertex
print('\n<< Plot vertices with significant p values >>')
meanTC = np.ones(templateSTC.shape)
meanTC[0,:] = 0
SrcEst = mne.SourceEstimate(meanTC, vertices=templateSTC.vertices, tmin=templateSTC.tmin, 
                            tstep=templateSTC.tstep, subject=templateSTC.subject)
kwargs = dict(colormap='mne', clim=dict(kind='value', pos_lims=[4.5, 5.5, 6.5]))

if useFsaveModel == 'ico5':
    VertsInLabel = CombinedLabel.get_vertices_used()
else:
    labelSTC = templateSTC.in_label(CombinedLabel)
    if useFsaveModel == 'oct6':
        VertsInLabel = np.array([i for i, n in enumerate(templateSTC.rh_vertno) if n in labelSTC.rh_vertno])
    else:
        VertsInLabel = labelSTC.rh_vertno
    del labelSTC

Pv_Interaction = Pval_Interaction < alphaP
for comb in Combs:
    factor = comb[0]
    cond = comb[1]
    print('< %s at %s >' % (factor, cond))
    
    # load p value data
    os.chdir(statdatadir+'/%s_at%s' % (factor, cond))
    Pval_1wayANOVA = np.load('Pvalues.npy')
    
    # make dataset
    Pv_1wayANOVA = Pval_1wayANOVA < alphaP
    
    # plot
    for t in pltTimes:
        SrcPlot = SrcEst.plot(subject=MRIsubject, surface='inflated', hemi='rh',
                              subjects_dir=subjects_dir, time_unit='ms', views='ven', 
                              initial_time=t, time_label=None, colorbar=False, size=(600,800), 
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
        
        # show location
        tidx = t - TOImin
        signiVerts = VertsInLabel[Pv_1wayANOVA[tidx,VertsInLabel]]
        signiVerts_Interaction = VertsInLabel[Pv_Interaction[tidx,VertsInLabel]]
        signiVerts_Interaction2 = np.array([i for i in signiVerts_Interaction if i not in signiVerts])
        
        if signiVerts.shape[0] != 0:
            if useFsaveModel == 'oct6':
                SrcPlot.add_foci(templateSTC.rh_vertno[signiVerts], coords_as_verts=True, scale_factor=0.22, color='y', hemi='rh')
            else:
                SrcPlot.add_foci(signiVerts, coords_as_verts=True, scale_factor=0.22, color='y', hemi='rh')
        if signiVerts_Interaction2.shape[0] != 0:
            if useFsaveModel == 'oct6':
                SrcPlot.add_foci(templateSTC.rh_vertno[signiVerts_Interaction2], coords_as_verts=True, scale_factor=0.22, color='k', hemi='rh')
            else:
                SrcPlot.add_foci(signiVerts_Interaction2, coords_as_verts=True, scale_factor=0.22, color='k', hemi='rh')
        
        # save figure
        os.chdir(currDir+'/%s_at%s' % (factor, cond))
        SrcPlot.save_single_image('SignificantVerts_%dms.png' % t)
        SrcPlot.close()
        del SrcPlot, tidx, signiVerts, signiVerts_Interaction, signiVerts_Interaction2
    del t, Pv_1wayANOVA
del comb


# plot significant vertex clusters
print('\n<< Plot significant vertex clusters >>')
meanTC = np.ones(templateSTC.shape)
meanTC[0,:] = 0
SrcEst = mne.SourceEstimate(meanTC, vertices=templateSTC.vertices, tmin=templateSTC.tmin, 
                            tstep=templateSTC.tstep, subject=templateSTC.subject)
kwargs = dict(colormap='mne', clim=dict(kind='value', pos_lims=[4.5, 5.5, 6.5]))

if useFsaveModel == 'ico5':
    VertsInLabel = CombinedLabel.get_vertices_used()
else:
    labelSTC = templateSTC.in_label(CombinedLabel)
    if useFsaveModel == 'oct6':
        VertsInLabel = np.array([i for i, n in enumerate(templateSTC.rh_vertno) if n in labelSTC.rh_vertno])
    else:
        VertsInLabel = labelSTC.rh_vertno
    del labelSTC

Pv_Interaction = Pval_Interaction < alphaP
for comb in Combs:
    factor = comb[0]
    cond = comb[1]
    print('< %s at %s >' % (factor, cond))
    
    # load p value data
    os.chdir(statdatadir+'/%s_at%s' % (factor, cond))
    Pval_1wayANOVA = np.load('Pvalues.npy')
    
    # make dataset
    Pv_1wayANOVA = Pval_1wayANOVA < alphaP
    
    # plot
    for t in pltTimes:
        SrcPlot = SrcEst.plot(subject=MRIsubject, surface='inflated', hemi='rh',
                              subjects_dir=subjects_dir, time_unit='ms', views='ven', 
                              initial_time=t, time_label=None, colorbar=False, size=(600,800), 
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
        
        # show location
        tidx = t - TOImin
        signiVerts = VertsInLabel[Pv_1wayANOVA[tidx,VertsInLabel]]
        
        if signiVerts.shape[0] != 0:
            if useFsaveModel == 'oct6':
                SrcPlot.add_foci(templateSTC.rh_vertno[signiVerts], coords_as_verts=True, scale_factor=0.22, color='y', hemi='rh')
            else:
                SrcPlot.add_foci(signiVerts, coords_as_verts=True, scale_factor=0.22, color='y', hemi='rh')
            
            # plot outer edges of significant clusters
            Vmask = Pv_1wayANOVA[tidx, :]
            
            # make functional label
            binarydata = np.zeros(Vmask.shape)
            binarydata[Vmask] = 1
            
            binarydata2 = np.concatenate((np.zeros(Vmask.shape), binarydata))
            
            stc = np.zeros((binarydata2.shape[0],2))
            stc[:,0] = binarydata2
            
            SrcEst2 = mne.SourceEstimate(stc, vertices=templateSTC.vertices, tmin=0, 
                                         tstep=templateSTC.tstep, subject=MRIsubject)
            
            _, RHfunclabel = mne.stc_to_label(SrcEst2, src=src, smooth=True, connected=True,
                                              subjects_dir=subjects_dir)
            
            for rhlabel in RHfunclabel:
                SrcPlot.add_label(rhlabel, borders=True, color='k')
            del Vmask, binarydata, binarydata2, stc, SrcEst2, RHfunclabel, rhlabel
        
        # save figure
        os.chdir(currDir+'/%s_at%s' % (factor, cond))
        SrcPlot.save_single_image('SigniVertClus_%dms.png' % t)
        SrcPlot.close()
        del SrcPlot, tidx, signiVerts
    del t, Pv_1wayANOVA
del comb

print('\n   ==> Finished!')

