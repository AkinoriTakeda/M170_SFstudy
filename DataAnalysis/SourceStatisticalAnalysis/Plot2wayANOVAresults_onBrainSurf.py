#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot statistical results on brain surface (2-way ANOVA)

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

MRIsubject = 'fsaverage'
subjects_dir = ''


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
srcdir1 = 'SurfSrcEst_dSPM_forEvoked'
dirname = filedir+'/Subject1/'+ExpID+'/Datafiles/SourceEstimate/'+srcdir1
if useFsaveModel == 'ico5':
    templateSTC = mne.read_source_estimate(dirname+'/Fsaverage/SrcEst_MeanTC_FearF_BSF_fsaverage')
else:
    templateSTC = mne.read_source_estimate(dirname+'/Fsaverage_%s/SrcEst_MeanTC_FearF_BSF_fsaverage%s' % (useFsaveModel, useFsaveModel.capitalize()))
del dirname


#- load data -#
# directory setting of statistical data 
TOImin = 137
TOImax = 265
dirname = 'ANOVA_Fsave%s_%dto%dms' % (useFsaveModel.capitalize(), TOImin, TOImax)

# load statistical data
statdatadir = filedir+'/GrandAverage/Datafiles/StatisticalData/SourceData/'+ExpID+'/'+srcdir1+'/'+dirname
effects = ['MainEffect_Category','MainEffect_SF','Interaction']
for cond in effects:
    os.chdir(statdatadir+'/'+cond)
    exec('scores_'+cond+' = np.load(\'TFCEscores.npy\')')
    exec('Pval_'+cond+' = np.load(\'Pvalues.npy\')')
del cond


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
pltTimes = np.arange(137, 266, 1)
alphaP = 0.05


# plot TFCE scores on brain surface
for ef in effects:
    # make SourceEstimate instance
    lefthemidata = np.zeros(locVal['scores_'+ef].T.shape)
    Sval = np.concatenate((lefthemidata, locVal['scores_'+ef].T))
    SrcEst = mne.SourceEstimate(Sval, vertices=templateSTC.vertices, tmin=TOImin/1000, 
                                tstep=templateSTC.tstep, subject=MRIsubject)
    
    # colormap setting
    Pv = locVal['Pval_'+ef] < alphaP
    kwargs = dict(colormap='hot', clim=dict(kind='value', lims=[0, locVal['scores_'+ef][Pv].min(), locVal['Fobs_'+ef].max()]))
    # [Note added in 2021/07/30] 
    # My colleague pointed out that the name of a variable in line 92 was wrong 
    # (when I uploaded this script to Github, I thought I had changed all the places where this variable name appeared, but I must have left that out).
    # When referring to this script, please replace 'Fobs_' with 'scores_.'
    
    # plot
    for t in pltTimes:
        SrcPlot = SrcEst.plot(subject=MRIsubject, surface='inflated', hemi='rh',
                              subjects_dir=subjects_dir, time_unit='ms', views='ven', 
                              initial_time=t, time_label='%d ms', colorbar=True, size=(600,800), 
                              **kwargs, smoothing_steps=5)
        SrcPlot.show_view(view=dict(azimuth=-270, elevation=-125, roll=0), distance=400)
#        SrcPlot.show_view(view=dict(azimuth=-270, elevation=-150, roll=0), distance=460)
        SrcPlot.add_label(CombinedLabel, borders=True)
        
        # save figure
        os.chdir(currDir)
        if not os.path.exists('./'+ef):
            os.mkdir('./'+ef)
        os.chdir('./'+ef)
        
        SrcPlot.save_single_image('TFCEscores_%dms.png' % t)
        SrcPlot.close()
        del SrcPlot
    del t, lefthemidata, Sval, SrcEst, Pv, kwargs
del ef


# plot each vertex
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

for ef in effects:
    Pv = locVal['Pval_'+ef] < alphaP
    
    # plot
    for t in pltTimes:
        SrcPlot = SrcEst.plot(subject=MRIsubject, surface='inflated', hemi='rh',
                              subjects_dir=subjects_dir, time_unit='ms', views='ven', 
                              initial_time=t, time_label=None, colorbar=False, size=(600,800), 
                              **kwargs, smoothing_steps=5)
        SrcPlot.show_view(view=dict(azimuth=-270, elevation=-125, roll=0), distance=400)
#        SrcPlot.show_view(view=dict(azimuth=-270, elevation=-150, roll=0), distance=460)
        
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
        signiVerts = VertsInLabel[Pv[tidx,VertsInLabel]]
        notsigniVerts = np.array([i for i in VertsInLabel if i not in signiVerts])
        
        if signiVerts.shape[0] != 0:
            if useFsaveModel == 'oct6':
                SrcPlot.add_foci(templateSTC.rh_vertno[signiVerts], coords_as_verts=True, scale_factor=0.22, color='y', hemi='rh')
            else:
                SrcPlot.add_foci(signiVerts, coords_as_verts=True, scale_factor=0.22, color='y', hemi='rh')
        if notsigniVerts.shape[0] != 0:
            if useFsaveModel == 'oct6':
                SrcPlot.add_foci(templateSTC.rh_vertno[notsigniVerts], coords_as_verts=True, scale_factor=0.22, color='k', hemi='rh')
            else:
                SrcPlot.add_foci(notsigniVerts, coords_as_verts=True, scale_factor=0.22, color='k', hemi='rh')
        
        # save figure
        os.chdir(currDir+'/'+ef)
        SrcPlot.save_single_image('SignificantVerts_%dms.png' % t)
        SrcPlot.close()
        del SrcPlot, tidx, signiVerts, notsigniVerts
    del t, Pv
del ef
