#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make mask data of significant spatiotemporal clusters

*Prior to running this program, we checked outputs from programs for plotting 
 ANOVA results, and isolated vertices were visually inspected and identified.

@author: Akinori Takeda
"""

import mne
import numpy as np
import os
locVal = locals()


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

# load labels of interest
HCPlabels = mne.read_labels_from_annot(MRIsubject, parc='HCPMMP1', hemi='both', 
                                       surf_name='inflated', subjects_dir=subjects_dir)

HCPlabellist = []
ROIname = ['VVC','PIT','FFC','V8','V4_']
for roi in ROIname:
    for r in [i for i in HCPlabels if roi in i.name and i.hemi=='rh']:
        HCPlabellist.append(r)

CombinedLabel = HCPlabellist[0] + HCPlabellist[1] + HCPlabellist[2] + HCPlabellist[3] + HCPlabellist[4]

if useFsaveModel != 'ico5':
    # for making SourceEstimate instance of fsaverage
    dirname = filedir+'/Subject1/'+ExpID+'/Datafiles/SourceEstimate/'+srcdir1+'/Fsaverage_%s' % useFsaveModel
    templateSTC = mne.read_source_estimate(dirname+'/SrcEst_MeanTC_FearF_BSF_fsaverage%s' % useFsaveModel.capitalize())
    del dirname

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
alphaP = 0.05

dirname = 'PostAnalyses_Fsave%s_%dto%dms' % (useFsaveModel.capitalize(), TOImin, TOImax)
statdatadir = filedir+'/GrandAverage/Datafiles/StatisticalData/SourceData/'+ExpID+'/'+srcdir1+'/'+dirname

ANOVAdirname = 'ANOVA_Fsave%s_%dto%dms' % (useFsaveModel.capitalize(), TOImin, TOImax)
ANOVAdatadir = filedir+'/GrandAverage/Datafiles/StatisticalData/SourceData/'+ExpID+'/'+srcdir1+'/'+ANOVAdirname

timemask_TOI = np.where((TOImin/1000 <= times) & (times <= TOImax/1000))[0]


#- load statistical data -#
effects = ['MainEffect_Category','MainEffect_SF']
for cond in effects:
    os.chdir(ANOVAdatadir+'/'+cond)
    exec('Pval_'+cond+' = np.load(\'Pvalues.npy\')')
del cond

CategoryList = ['NeutF', 'FearF', 'House']
SFNameList = ['BSF', 'LSF', 'HSF', 'Equ']


#%%
'''
Analysis 1: Main effect of Category (2-way ANOVA)
'''

Pval = locVal['Pval_MainEffect_Category']

# time window setting
genTW = [204, 265] # general time window
timemask_TW = np.where((genTW[0]/1000 <= times[timemask_TOI]) & (times[timemask_TOI] <= genTW[1]/1000))[0]

if genTW[0] == TOImin:
    timemask_excludeTW = np.where(genTW[1]/1000 < times[timemask_TOI])[0]
elif genTW[1] == TOImax:
    timemask_excludeTW = np.where(times[timemask_TOI] < genTW[0]/1000)[0]
else:
    timemask_excludeTW = np.where((times[timemask_TOI] < genTW[0]/1000) | (genTW[1]/1000 < times[timemask_TOI]))[0]

# make significant spatiotemporal mask
signiMask = np.zeros(Pval.shape, dtype=bool)

signiMask[:,locVal['vertIdx_FFC']] = Pval[:,locVal['vertIdx_FFC']] < alphaP
signiMask[:,locVal['vertIdx_VVC']] = Pval[:,locVal['vertIdx_VVC']] < alphaP
signiMask[:,locVal['vertIdx_PIT']] = Pval[:,locVal['vertIdx_PIT']] < alphaP
signiMask[:,locVal['vertIdx_V4']] = Pval[:,locVal['vertIdx_V4']] < alphaP
signiMask[:,locVal['vertIdx_V8']] = Pval[:,locVal['vertIdx_V8']] < alphaP

signiMask[timemask_excludeTW, :] = False

# additional process
if genTW[0] == 158:
    signiMask2 = np.zeros(Pval.shape, dtype=bool)
    timemask_excludeTW2 = np.where((203/1000) < times[timemask_TOI])[0]
    signiMask2[:, locVal['vertIdx_PIT']] = Pval[:,locVal['vertIdx_PIT']] < alphaP
    
    signiMask2[timemask_excludeTW2, :] = False
    signiMask[:, locVal['vertIdx_PIT']] = signiMask2[:, locVal['vertIdx_PIT']]
    
elif genTW[0] == 204:
    signiMask2 = np.zeros(Pval.shape, dtype=bool)
    timemask_excludeTW2 = np.where(times[timemask_TOI] < (207/1000))[0]
    signiMask2[:, locVal['vertIdx_FFC']] = Pval[:,locVal['vertIdx_FFC']] < alphaP
    signiMask2[:, locVal['vertIdx_VVC']] = Pval[:,locVal['vertIdx_VVC']] < alphaP
    
    signiMask2[timemask_excludeTW2, :] = False
    signiMask[:, locVal['vertIdx_FFC']] = signiMask2[:, locVal['vertIdx_FFC']]
    signiMask[:, locVal['vertIdx_VVC']] = signiMask2[:, locVal['vertIdx_VVC']]

del signiMask2, timemask_excludeTW2


# save mask
os.chdir(ANOVAdatadir+'/MainEffect_Category')
if not os.path.exists('./SignificantMasks'):
    os.mkdir('./SignificantMasks')
os.chdir('./SignificantMasks')

np.save('SigniMask_%dto%dms.npy' % (genTW[0], genTW[1]), signiMask)


#%%
'''
Analysis 2: Simple effects of Category at each SF level (1-way ANOVA)
'''

Combs = [('Category', 'BSF'), ('Category', 'LSF'), ('Category', 'HSF')]

comb = Combs[1]
os.chdir(statdatadir+'/%s_at%s' % comb)
Pval = np.load('Pvalues.npy')


# time window setting
genTW = [144, 201] # general time window
timemask_TW = np.where((genTW[0]/1000 <= times[timemask_TOI]) & (times[timemask_TOI] <= genTW[1]/1000))[0]

if genTW[0] == TOImin:
    timemask_excludeTW = np.where(genTW[1]/1000 < times[timemask_TOI])[0]
elif genTW[1] == TOImax:
    timemask_excludeTW = np.where(times[timemask_TOI] < genTW[0]/1000)[0]
else:
    timemask_excludeTW = np.where((times[timemask_TOI] < genTW[0]/1000) | (genTW[1]/1000 < times[timemask_TOI]))[0]

# make significant spatiotemporal mask
signiMask = np.zeros(Pval.shape, dtype=bool)

if comb[-1] == 'BSF' and genTW[0] == 151:
    signiMask[:,locVal['vertIdx_FFC']] = Pval[:,locVal['vertIdx_FFC']] < alphaP
    signiMask[:,locVal['vertIdx_VVC']] = Pval[:,locVal['vertIdx_VVC']] < alphaP
    signiMask[:,locVal['vertIdx_V8']] = Pval[:,locVal['vertIdx_V8']] < alphaP
    signiMask[timemask_excludeTW, :] = False
    
    # exclude data points from 177 ms ~ in V8
    signiMask2 = np.zeros(Pval.shape, dtype=bool)
    timemask_excludeTW2 = np.where((176/1000) < times[timemask_TOI])[0]
    signiMask2[:, locVal['vertIdx_V8']] = Pval[:,locVal['vertIdx_V8']] < alphaP
    signiMask2[timemask_excludeTW2, :] = False
    signiMask[:, locVal['vertIdx_V8']] = signiMask2[:, locVal['vertIdx_V8']]
    del signiMask2, timemask_excludeTW2
    
elif comb[-1] == 'BSF' and genTW[0] == 176:
    signiMask[:,locVal['vertIdx_FFC']] = Pval[:,locVal['vertIdx_FFC']] < alphaP
    signiMask[:,locVal['vertIdx_VVC']] = Pval[:,locVal['vertIdx_VVC']] < alphaP
    signiMask[:,locVal['vertIdx_PIT']] = Pval[:,locVal['vertIdx_PIT']] < alphaP
    signiMask[:,locVal['vertIdx_V4']] = Pval[:,locVal['vertIdx_V4']] < alphaP
    signiMask[:,locVal['vertIdx_V8']] = Pval[:,locVal['vertIdx_V8']] < alphaP
    signiMask[timemask_excludeTW, :] = False
    
    # exclude data points ~182 ms in lateral FG
    signiMask2 = np.zeros(Pval.shape, dtype=bool)
    signiMask2[:, locVal['vertIdx_FFC']] = Pval[:,locVal['vertIdx_FFC']] < alphaP
    timemask_excludeTW2 = np.where(times[timemask_TOI] < (183/1000))[0]
    signiMask2[timemask_excludeTW2, :] = False
    signiMask[:, locVal['vertIdx_FFC']] = signiMask2[:, locVal['vertIdx_FFC']]
    del signiMask2, timemask_excludeTW2
    
    # exclude data points ~183 ms in medial FG
    signiMask2 = np.zeros(Pval.shape, dtype=bool)
    signiMask2[:, locVal['vertIdx_VVC']] = Pval[:,locVal['vertIdx_VVC']] < alphaP
    timemask_excludeTW2 = np.where(times[timemask_TOI] < (184/1000))[0]
    signiMask2[timemask_excludeTW2, :] = False
    signiMask[:, locVal['vertIdx_VVC']] = signiMask2[:, locVal['vertIdx_VVC']]
    del signiMask2, timemask_excludeTW2
    
    # exclude data points ~174 ms in V8
    signiMask2 = np.zeros(Pval.shape, dtype=bool)
    signiMask2[:, locVal['vertIdx_V8']] = Pval[:,locVal['vertIdx_V8']] < alphaP
    timemask_excludeTW2 = np.where(times[timemask_TOI] < (175/1000))[0]
    signiMask2[timemask_excludeTW2, :] = False
    signiMask[:, locVal['vertIdx_V8']] = signiMask2[:, locVal['vertIdx_V8']]
    del signiMask2, timemask_excludeTW2
    
elif comb[-1] == 'LSF' and genTW[0] == 144:
    signiMask[:,locVal['vertIdx_FFC']] = Pval[:,locVal['vertIdx_FFC']] < alphaP
    signiMask[:,locVal['vertIdx_VVC']] = Pval[:,locVal['vertIdx_VVC']] < alphaP
    signiMask[:,locVal['vertIdx_V8']] = Pval[:,locVal['vertIdx_V8']] < alphaP
    signiMask[timemask_excludeTW, :] = False
    
    # exclude data point in 144 ms in medial FG
    signiMask2 = np.zeros(Pval.shape, dtype=bool)
    timemask_excludeTW2 = np.where((times[timemask_TOI] < (145/1000)) | ((201/1000) < times[timemask_TOI]))[0]
    signiMask2[:, locVal['vertIdx_VVC']] = Pval[:,locVal['vertIdx_VVC']] < alphaP
    signiMask2[timemask_excludeTW2, :] = False
    signiMask[:, locVal['vertIdx_VVC']] = signiMask2[:, locVal['vertIdx_VVC']]
    del signiMask2, timemask_excludeTW2
    
elif comb[-1] == 'LSF' and genTW[0] == 146:
    signiMask[:,locVal['vertIdx_PIT']] = Pval[:,locVal['vertIdx_PIT']] < alphaP
    signiMask[timemask_excludeTW, :] = False
    
elif comb[-1] == 'LSF' and genTW[0] == 222:
    signiMask[:,locVal['vertIdx_FFC']] = Pval[:,locVal['vertIdx_FFC']] < alphaP
    signiMask[:,locVal['vertIdx_VVC']] = Pval[:,locVal['vertIdx_VVC']] < alphaP
    signiMask[:,locVal['vertIdx_PIT']] = Pval[:,locVal['vertIdx_PIT']] < alphaP
    signiMask[:,locVal['vertIdx_V4']] = Pval[:,locVal['vertIdx_V4']] < alphaP
    signiMask[:,locVal['vertIdx_V8']] = Pval[:,locVal['vertIdx_V8']] < alphaP
    signiMask[timemask_excludeTW, :] = False
    
    # exclude isolated data point in PIT vertex No.2
    signiMask[:,locVal['vertIdx_PIT'][1]] = False
    
    # exclude data points in V4 vertex No.7,9,32
    signiMask2 = np.zeros(Pval.shape, dtype=bool)
    timemask_excludeTW2 = np.where(((239/1000) <= times[timemask_TOI]) & (times[timemask_TOI] <= (244/1000)))[0]
    signiMask2[:, locVal['vertIdx_V4'][6]] = Pval[:,locVal['vertIdx_V4'][6]] < alphaP
    signiMask2[:, locVal['vertIdx_V4'][8]] = Pval[:,locVal['vertIdx_V4'][8]] < alphaP
    signiMask2[:, locVal['vertIdx_V4'][31]] = Pval[:,locVal['vertIdx_V4'][31]] < alphaP
    signiMask2[timemask_excludeTW2, :] = False
    signiMask[:, locVal['vertIdx_V4'][6]] = signiMask2[:, locVal['vertIdx_V4'][6]]
    signiMask[:, locVal['vertIdx_V4'][8]] = signiMask2[:, locVal['vertIdx_V4'][8]]
    signiMask[:, locVal['vertIdx_V4'][31]] = signiMask2[:, locVal['vertIdx_V4'][31]]
    del signiMask2, timemask_excludeTW2
    
elif comb[-1] == 'HSF' and genTW[0] == 137:
    signiMask[:,locVal['vertIdx_PIT']] = Pval[:,locVal['vertIdx_PIT']] < alphaP
    signiMask[:,locVal['vertIdx_V4']] = Pval[:,locVal['vertIdx_V4']] < alphaP
    signiMask[timemask_excludeTW, :] = False
    
    # exclude data points in V4 vertex No.11,20,22,36
    signiMask[:,locVal['vertIdx_V4'][10]] = False
    signiMask[:,locVal['vertIdx_V4'][19]] = False
    signiMask[:,locVal['vertIdx_V4'][21]] = False
    signiMask[:,locVal['vertIdx_V4'][35]] = False
    
elif comb[-1] == 'HSF' and genTW[0] == 164:
    signiMask[:,locVal['vertIdx_FFC']] = Pval[:,locVal['vertIdx_FFC']] < alphaP
    signiMask[:,locVal['vertIdx_VVC']] = Pval[:,locVal['vertIdx_VVC']] < alphaP
    signiMask[:,locVal['vertIdx_PIT']] = Pval[:,locVal['vertIdx_PIT']] < alphaP
    signiMask[:,locVal['vertIdx_V8']] = Pval[:,locVal['vertIdx_V8']] < alphaP
    signiMask[timemask_excludeTW, :] = False
    
    # exclude data points from 193 ms ~ in V8
    signiMask2 = np.zeros(Pval.shape, dtype=bool)
    timemask_excludeTW2 = np.where((192/1000) < times[timemask_TOI])[0]
    signiMask2[:, locVal['vertIdx_V8']] = Pval[:,locVal['vertIdx_V8']] < alphaP
    signiMask2[timemask_excludeTW2, :] = False
    signiMask[:, locVal['vertIdx_V8']] = signiMask2[:, locVal['vertIdx_V8']]
    del signiMask2, timemask_excludeTW2
    
elif comb[-1] == 'HSF' and genTW[0] == 214:
    signiMask[:,locVal['vertIdx_FFC']] = Pval[:,locVal['vertIdx_FFC']] < alphaP
    signiMask[:,locVal['vertIdx_VVC']] = Pval[:,locVal['vertIdx_VVC']] < alphaP
    signiMask[:,locVal['vertIdx_PIT']] = Pval[:,locVal['vertIdx_PIT']] < alphaP
    signiMask[:,locVal['vertIdx_V4']] = Pval[:,locVal['vertIdx_V4']] < alphaP
    signiMask[:,locVal['vertIdx_V8']] = Pval[:,locVal['vertIdx_V8']] < alphaP
    signiMask[timemask_excludeTW, :] = False
    
    # exclude data points ~216 ms in lateral & medial FG
    signiMask2 = np.zeros(Pval.shape, dtype=bool)
    timemask_excludeTW2 = np.where(times[timemask_TOI] < (217/1000))[0]
    signiMask2[:, locVal['vertIdx_FFC']] = Pval[:,locVal['vertIdx_FFC']] < alphaP
    signiMask2[:, locVal['vertIdx_VVC']] = Pval[:,locVal['vertIdx_VVC']] < alphaP
    signiMask2[timemask_excludeTW2, :] = False
    signiMask[:, locVal['vertIdx_FFC']] = signiMask2[:, locVal['vertIdx_FFC']]
    signiMask[:, locVal['vertIdx_VVC']] = signiMask2[:, locVal['vertIdx_VVC']]
    del signiMask2, timemask_excludeTW2


# save mask
os.chdir(statdatadir+'/%s_at%s' % comb)
if not os.path.exists('./SignificantMasks'):
    os.mkdir('./SignificantMasks')
os.chdir('./SignificantMasks')

np.save('SigniMask_%dto%dms.npy' % (genTW[0], genTW[1]), signiMask)


#%%
'''
Analysis 3: Simple effects of SF at each Category level (1-way ANOVA)
'''

Combs = [('SF', 'NeutF'), ('SF', 'FearF'), ('SF', 'House')]

comb = Combs[2]
os.chdir(statdatadir+'/%s_at%s' % comb)
Pval = np.load('Pvalues.npy')


# time window setting
genTW = [186, 265] # general time window
timemask_TW = np.where((genTW[0]/1000 <= times[timemask_TOI]) & (times[timemask_TOI] <= genTW[1]/1000))[0]

if genTW[0] == TOImin:
    timemask_excludeTW = np.where(genTW[1]/1000 < times[timemask_TOI])[0]
elif genTW[1] == TOImax:
    timemask_excludeTW = np.where(times[timemask_TOI] < genTW[0]/1000)[0]
else:
    timemask_excludeTW = np.where((times[timemask_TOI] < genTW[0]/1000) | (genTW[1]/1000 < times[timemask_TOI]))[0]

# make significant spatiotemporal mask
signiMask = np.zeros(Pval.shape, dtype=bool)

if comb[-1] == 'NeutF' and genTW[0] == 137:
    signiMask[:,locVal['vertIdx_FFC']] = Pval[:,locVal['vertIdx_FFC']] < alphaP
    signiMask[:,locVal['vertIdx_VVC']] = Pval[:,locVal['vertIdx_VVC']] < alphaP
    signiMask[:,locVal['vertIdx_PIT']] = Pval[:,locVal['vertIdx_PIT']] < alphaP
    signiMask[:,locVal['vertIdx_V4']] = Pval[:,locVal['vertIdx_V4']] < alphaP
    signiMask[:,locVal['vertIdx_V8']] = Pval[:,locVal['vertIdx_V8']] < alphaP
    signiMask[timemask_excludeTW, :] = False
    
    # exclude data points from 202 ms ~ in V4 & V8
    signiMask2 = np.zeros(Pval.shape, dtype=bool)
    timemask_excludeTW2 = np.where((202/1000) < times[timemask_TOI])[0]
    signiMask2[:, locVal['vertIdx_V4']] = Pval[:,locVal['vertIdx_V4']] < alphaP
    signiMask2[:, locVal['vertIdx_V8']] = Pval[:,locVal['vertIdx_V8']] < alphaP
    signiMask2[timemask_excludeTW2, :] = False
    signiMask[:, locVal['vertIdx_V4']] = signiMask2[:, locVal['vertIdx_V4']]
    signiMask[:, locVal['vertIdx_V8']] = signiMask2[:, locVal['vertIdx_V8']]
    del signiMask2, timemask_excludeTW2
    
elif comb[-1] == 'NeutF' and genTW[0] == 212:
    signiMask[:,locVal['vertIdx_FFC']] = Pval[:,locVal['vertIdx_FFC']] < alphaP
    signiMask[:,locVal['vertIdx_VVC']] = Pval[:,locVal['vertIdx_VVC']] < alphaP
    signiMask[:,locVal['vertIdx_PIT']] = Pval[:,locVal['vertIdx_PIT']] < alphaP
    signiMask[:,locVal['vertIdx_V4']] = Pval[:,locVal['vertIdx_V4']] < alphaP
    signiMask[:,locVal['vertIdx_V8']] = Pval[:,locVal['vertIdx_V8']] < alphaP
    signiMask[timemask_excludeTW, :] = False
    
    # exclude data points ~241 ms in medial FG
    signiMask2 = np.zeros(Pval.shape, dtype=bool)
    timemask_excludeTW2 = np.where(times[timemask_TOI] < (242/1000))[0]
    signiMask2[:, locVal['vertIdx_VVC']] = Pval[:,locVal['vertIdx_VVC']] < alphaP
    signiMask2[timemask_excludeTW2, :] = False
    signiMask[:, locVal['vertIdx_VVC']] = signiMask2[:, locVal['vertIdx_VVC']]
    del signiMask2, timemask_excludeTW2
    
    # exclude data points ~215 ms in lateral FG
    signiMask2 = np.zeros(Pval.shape, dtype=bool)
    timemask_excludeTW2 = np.where(times[timemask_TOI] < (216/1000))[0]
    signiMask2[:, locVal['vertIdx_FFC']] = Pval[:,locVal['vertIdx_FFC']] < alphaP
    signiMask2[timemask_excludeTW2, :] = False
    signiMask[:, locVal['vertIdx_FFC']] = signiMask2[:, locVal['vertIdx_FFC']]
    del signiMask2, timemask_excludeTW2
    
    # also exclude isolated data points in lateral FG vertex No.23
    signiMask[:,locVal['vertIdx_FFC'][22]] = False
    
else:
    signiMask[:,locVal['vertIdx_FFC']] = Pval[:,locVal['vertIdx_FFC']] < alphaP
    signiMask[:,locVal['vertIdx_VVC']] = Pval[:,locVal['vertIdx_VVC']] < alphaP
    signiMask[:,locVal['vertIdx_PIT']] = Pval[:,locVal['vertIdx_PIT']] < alphaP
    signiMask[:,locVal['vertIdx_V4']] = Pval[:,locVal['vertIdx_V4']] < alphaP
    signiMask[:,locVal['vertIdx_V8']] = Pval[:,locVal['vertIdx_V8']] < alphaP
    signiMask[timemask_excludeTW, :] = False


# save mask
os.chdir(statdatadir+'/%s_at%s' % comb)
if not os.path.exists('./SignificantMasks'):
    os.mkdir('./SignificantMasks')
os.chdir('./SignificantMasks')

np.save('SigniMask_%dto%dms.npy' % (genTW[0], genTW[1]), signiMask)


