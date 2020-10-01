#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post hoc paired tests & plotting their results

@author: Akinori Takeda
"""

import mne
import numpy as np
from scipy import stats as stats
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
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


# load statistical data
os.chdir(ANOVAdatadir+'/Interaction')
Scores = np.load('TFCEscores.npy')
Pval = np.load('Pvalues.npy')

CategoryList = ['NeutF', 'FearF', 'House']
SFNameList = ['BSF', 'LSF', 'HSF', 'Equ']


def DetectTemporalClus(Pv):
    signiTP = np.where(Pv)[0]
    identified_timeclus = []
    if signiTP.shape[0] == 1:
        identified_timeclus.append(signiTP.tolist())
    elif signiTP.shape[0] >= 2:
        cluster_tp = []
        for i, tp in enumerate(signiTP[:-1]):
            cluster_tp.append(tp)
            
            if (signiTP[(i+1)] - tp) > 1:
                identified_timeclus.append(cluster_tp)
                del cluster_tp
                cluster_tp = []
            
            if i == (len(signiTP[:-1])-1):
                identified_timeclus.append(cluster_tp)
        del i, tp, cluster_tp
        
        if identified_timeclus[-1]==[] or signiTP[-1] == (identified_timeclus[-1][-1]+1):
            identified_timeclus[-1].append(signiTP[-1])
    
    return signiTP, identified_timeclus


#%%
'''
Analysis 1: post hoc comparisons for the simple effects of category at each SF
'''

Combs = [('Category', 'BSF'), ('Category', 'LSF'), ('Category', 'HSF')]

comb = Combs[2] # select a condition

# time window setting
genTW = [214, 265] # general time window. change them to select a cluster

os.chdir(statdatadir+'/%s_at%s/SignificantMasks' % comb)
signiMask = np.load('SigniMask_%dto%dms.npy' % (genTW[0], genTW[1]))

# use the above with commenting by '#' if restricted to individual ROIs
signiMask[:,locVal['vertIdx_FFC']] = False
signiMask[:,locVal['vertIdx_VVC']] = False
signiMask[:,locVal['vertIdx_PIT']] = False
signiMask[:,locVal['vertIdx_V4']] = False
#signiMask[:,locVal['vertIdx_V8']] = False


# make dataset
data_NeutF = locVal['SurfData_NeutF_%s' % comb[-1]][:,:,timemask_TOI]
data_FearF = locVal['SurfData_FearF_%s' % comb[-1]][:,:,timemask_TOI]
data_House = locVal['SurfData_House_%s' % comb[-1]][:,:,timemask_TOI]

Data_NeutF = np.array([data_NeutF[n,:,:].T[signiMask].mean(0) for n in np.arange(SubjN)])
Data_FearF = np.array([data_FearF[n,:,:].T[signiMask].mean(0) for n in np.arange(SubjN)])
Data_House = np.array([data_House[n,:,:].T[signiMask].mean(0) for n in np.arange(SubjN)])

del data_NeutF, data_FearF, data_House


# paired tests
t1, p1 = stats.wilcoxon(Data_NeutF, Data_FearF)
t2, p2 = stats.wilcoxon(Data_NeutF, Data_House)
t3, p3 = stats.wilcoxon(Data_FearF, Data_House)

# apply FDR correction for multiple comparison correction
_, pval_corred = mne.stats.fdr_correction([p1, p2, p3], alpha=alphaP, method='indep')


#- print results -#
Pv = np.unique(Pval[signiMask])
Sv = np.unique(Scores[signiMask])
if Sv.shape == 1:
    if Pv.shape == 1:
        statText = '       (TFCE scores = {0}, p value = {1})'.format(Sv.max(), Pv.min())
    else:
        statText = '       (TFCE scores = {0}, p value = {1}-{2})'.format(Sv.max(), Pv.min(), Pv.max())
else:
    if Pv.shape == 1:
        statText = '       (TFCE scores = {0}-{1}, p value = {2})'.format(Sv.min(), Sv.max(), Pv.min())
    else:
        statText = '       (TFCE scores = {0}-{1}, p value = {2}-{3})'.format(Sv.min(), Sv.max(), Pv.min(), Pv.max())

Vmask = np.array([True in signiMask[:,vert] for vert in np.arange(signiMask.shape[1])])
Tmask = np.array([True in signiMask[t,:] for t in np.arange(signiMask.shape[0])])

roiText = '  ROI: '
nROI = 0
for signiROI in HCPlabellist:
    labelname = signiROI.name.split('_')[1]
    if list(set(np.where(Vmask)[0])&set(locVal['vertIdx_'+labelname])) != []:
        if nROI == 0:
            roiText += labelname 
        else:
            roiText += ', %s' % labelname 
        nROI += 1
    del labelname
del signiROI, nROI

Tmask = np.array([True in signiMask[t,:] for t in np.arange(signiMask.shape[0])])
timeText = '  Time window: '
_, timeclus = DetectTemporalClus(Tmask)
for i, n in enumerate(timeclus):
    tclu = times[timemask_TOI][n]
    tclu *= 1000
    
    if tclu.shape[0] == 1:
        timeText += '%d' % tclu[0]
    elif tclu.shape[0] >=2:
        timeText += '%d-%d' % (tclu[0], tclu[-1])
    
    if i != (len(timeclus)-1):
        timeText += ', '
    del tclu
del i, n
timeText += ' ms'


print('\n< Results of paired t test >')
print('  Data: 1-way ANOVA for 2-way ANOVA interaction (%s at %s)' % comb)
print(statText)
print(roiText)
print(timeText)

print('\n[Mean & SD]')
print('NeutF: {0} +/- {1}'.format(Data_NeutF.mean(), Data_NeutF.std()))
print('FearF: {0} +/- {1}'.format(Data_FearF.mean(), Data_FearF.std()))
print('House: {0} +/- {1}'.format(Data_House.mean(), Data_House.std()))

print('\n[T value]')
print('NeutF vs. FearF: {0}'.format(t1))
print('NeutF vs. House: {0}'.format(t2))
print('FearF vs. House: {0}'.format(t3))

print('\n[p value]')
print('NeutF vs. FearF: {0}'.format(p1))
print('NeutF vs. House: {0}'.format(p2))
print('FearF vs. House: {0}'.format(p3))

print('\n[FDR-corrected p value]')
print('NeutF vs. FearF: {0}'.format(pval_corred[0]))
print('NeutF vs. House: {0}'.format(pval_corred[1]))
print('FearF vs. House: {0}'.format(pval_corred[2]))



# plot individual samples
df = pd.DataFrame({'Neutral face':Data_NeutF, 'Fearful face':Data_FearF, 'House':Data_House})
colorList = dict(BSF='darkgray', LSF='dodgerblue', HSF='tomato', Equ='springgreen')
pal = [colorList[comb[-1]], colorList[comb[-1]], colorList[comb[-1]]]
innerplot = None # None or 'point'
pltYmin = 0
pltYmax = 10

plt.figure(figsize=(7,7))
plt.gcf().suptitle('Mean Amplitudes within significant cluster (%d-%d ms)' % (genTW[0], genTW[1]), fontsize=15)
v = sns.violinplot(data=df, palette=pal, scale='count', inner=innerplot, dodge=True, jitter=True, bw=0.4)

prop = v.properties()
if innerplot == 'point':
    propdata = prop['children'][:6]
    propdata = propdata[::2]
else:
    propdata = prop['children'][:3]

for n in propdata:
    m = np.mean(n.get_paths()[0].vertices[:,0])
    n.get_paths()[0].vertices[:,0] = np.clip(n.get_paths()[0].vertices[:,0], m, np.inf)

# also plot holizontal lines
for yval in np.arange(pltYmin, pltYmax+20, 5):
    plt.hlines(yval, -1, 3, color='gray', alpha=0.3)

# plot 95% confidence interval
cis_NeutF = mne.stats.bootstrap_confidence_interval(Data_NeutF, ci=0.95, n_bootstraps=5000, stat_fun='mean')
cis_FearF = mne.stats.bootstrap_confidence_interval(Data_FearF, ci=0.95, n_bootstraps=5000, stat_fun='mean')
cis_House = mne.stats.bootstrap_confidence_interval(Data_House, ci=0.95, n_bootstraps=5000, stat_fun='mean')

plt.vlines(0.075, Data_NeutF.mean(), cis_NeutF[1], color='k')
plt.vlines(0.075, cis_NeutF[0], Data_NeutF.mean(), color='k')
plt.errorbar(0.075, Data_NeutF.mean(), yerr=0, fmt='o', color='w', markeredgecolor='k', ecolor='k', ms=7)
plt.vlines(1.075, Data_FearF.mean(), cis_FearF[1], color='k')
plt.vlines(1.075, cis_FearF[0], Data_FearF.mean(), color='k')
plt.errorbar(1.075, Data_FearF.mean(), yerr=0, fmt='o', color='w', markeredgecolor='k', ecolor='k', ms=7)
plt.vlines(2.075, Data_House.mean(), cis_House[1], color='k')
plt.vlines(2.075, cis_House[0], Data_House.mean(), color='k')
plt.errorbar(2.075, Data_House.mean(), yerr=0, fmt='o', color='w', markeredgecolor='k', ecolor='k', ms=7)

# plot individual data
plt.scatter(np.ones(Data_NeutF.shape[0])*(-0.075), Data_NeutF, c=pal[0], marker='o', edgecolor='k', s=100)
plt.scatter(np.ones(Data_FearF.shape[0])*(1-0.075), Data_FearF, c=pal[1], marker='o', edgecolor='k', s=100)
plt.scatter(np.ones(Data_House.shape[0])*(2-0.075), Data_House, c=pal[2], marker='o', edgecolor='k', s=100)

# parameter setting
plt.xlim([-0.5, 3.5])
plt.ylim([pltYmin-3, pltYmax+20])
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.xlabel('Category', fontsize=20, labelpad=10)
plt.ylabel('Amplitude ($\it{a.u.}$)', fontsize=20, labelpad=10)
plt.gcf().subplots_adjust(top=0.9, bottom=0.12, left=0.14, right=0.95)

# save figure
dirlist = ['StatisticalData', 'SourceData', ExpID, srcdir1, dirname, 'PosthocTestResults']

os.chdir(filedir+'/GrandAverage/DataPlots')
for dname in dirlist:
    if not os.path.exists('./'+dname):
        os.mkdir('./'+dname)
    os.chdir('./'+dname)
currDir = os.getcwd()

if not os.path.exists(currDir+'/%s_at%s' % comb):
    os.mkdir(currDir+'/%s_at%s' % comb)
os.chdir(currDir+'/%s_at%s' % comb)

plt.gcf().savefig('SigniCluster_%dto%dms.png' % (genTW[0], genTW[1]))
plt.close(plt.gcf())


#%%
'''
Analysis 2: post hoc comparisons for the simple effects of SF at each category
'''

Combs = [('SF', 'NeutF'), ('SF', 'FearF'), ('SF', 'House')]

comb = Combs[0] # select a condition

# time window setting
genTW = [137, 215] # general time window. change them to select a cluster

os.chdir(statdatadir+'/%s_at%s/SignificantMasks' % comb)
signiMask = np.load('SigniMask_%dto%dms.npy' % (genTW[0], genTW[1]))

# use the above with commenting by '#' if restricted to individual ROIs
signiMask[:,locVal['vertIdx_FFC']] = False
signiMask[:,locVal['vertIdx_VVC']] = False
signiMask[:,locVal['vertIdx_PIT']] = False
signiMask[:,locVal['vertIdx_V4']] = False
#signiMask[:,locVal['vertIdx_V8']] = False


# make dataset
data_BSF = locVal['SurfData_%s_BSF' % comb[-1]][:,:,timemask_TOI]
data_LSF = locVal['SurfData_%s_LSF' % comb[-1]][:,:,timemask_TOI]
data_HSF = locVal['SurfData_%s_HSF' % comb[-1]][:,:,timemask_TOI]
data_Equ = locVal['SurfData_%s_Equ' % comb[-1]][:,:,timemask_TOI]

Data_BSF = np.array([data_BSF[n,:,:].T[signiMask].mean(0) for n in np.arange(SubjN)])
Data_LSF = np.array([data_LSF[n,:,:].T[signiMask].mean(0) for n in np.arange(SubjN)])
Data_HSF = np.array([data_HSF[n,:,:].T[signiMask].mean(0) for n in np.arange(SubjN)])
Data_Equ = np.array([data_Equ[n,:,:].T[signiMask].mean(0) for n in np.arange(SubjN)])

del data_BSF, data_LSF, data_HSF, data_Equ


# paired tests
t1, p1 = stats.wilcoxon(Data_BSF, Data_LSF)
t2, p2 = stats.wilcoxon(Data_BSF, Data_HSF)
t3, p3 = stats.wilcoxon(Data_BSF, Data_Equ)
t4, p4 = stats.wilcoxon(Data_LSF, Data_HSF)
t5, p5 = stats.wilcoxon(Data_LSF, Data_Equ)
t6, p6 = stats.wilcoxon(Data_HSF, Data_Equ)

# apply FDR correction for multiple comparison correction
_, pval_corred = mne.stats.fdr_correction([p1, p2, p3, p4, p5, p6], alpha=alphaP, method='indep')


#- print results -#
Pv = np.unique(Pval[signiMask])
Sv = np.unique(Scores[signiMask])
if Sv.shape[0] == 1:
    if Pv.shape[0] == 1:
        statText = '       (TFCE scores = {0}, p value = {1})'.format(Sv.max(), Pv.min())
    else:
        statText = '       (TFCE scores = {0}, p value = {1}-{2})'.format(Sv.max(), Pv.min(), Pv.max())
else:
    if Pv.shape[0] == 1:
        statText = '       (TFCE scores = {0}-{1}, p value = {2})'.format(Sv.min(), Sv.max(), Pv.min())
    else:
        statText = '       (TFCE scores = {0}-{1}, p value = {2}-{3})'.format(Sv.min(), Sv.max(), Pv.min(), Pv.max())


Vmask = np.array([True in signiMask[:,vert] for vert in np.arange(signiMask.shape[1])])
Tmask = np.array([True in signiMask[t,:] for t in np.arange(signiMask.shape[0])])

roiText = '  ROI: '
nROI = 0
for signiROI in HCPlabellist:
    labelname = signiROI.name.split('_')[1]
    if list(set(np.where(Vmask)[0])&set(locVal['vertIdx_'+labelname])) != []:
        if nROI == 0:
            roiText += labelname 
        else:
            roiText += ', %s' % labelname 
        nROI += 1
    del labelname
del signiROI, nROI

timeText = '  Time window: '
_, timeclus = DetectTemporalClus(Tmask)
for i, n in enumerate(timeclus):
    tclu = times[timemask_TOI][n]
    tclu *= 1000
    
    if tclu.shape[0] == 1:
        timeText += '%d' % tclu[0]
    elif tclu.shape[0] >=2:
        timeText += '%d-%d' % (tclu[0], tclu[-1])
    
    if i != (len(timeclus)-1):
        timeText += ', '
    del tclu
del i, n
timeText += ' ms'


print('\n< Results of paired t test >')
print('  Data: 1-way ANOVA for 2-way ANOVA interaction (%s at %s)' % comb)
print(statText)
print(roiText)
print(timeText)

print('\n[Mean & SD]')
print('BSF: {0} +/- {1}'.format(Data_BSF.mean(), Data_BSF.std()))
print('LSF: {0} +/- {1}'.format(Data_LSF.mean(), Data_LSF.std()))
print('HSF: {0} +/- {1}'.format(Data_HSF.mean(), Data_HSF.std()))
print('Equiluminant: {0} +/- {1}'.format(Data_Equ.mean(), Data_Equ.std()))

print('\n[T value]')
print('BSF vs. LSF: {0}'.format(t1))
print('BSF vs. HSF: {0}'.format(t2))
print('BSF vs. Equ: {0}'.format(t3))
print('LSF vs. HSF: {0}'.format(t4))
print('LSF vs. Equ: {0}'.format(t5))
print('HSF vs. Equ: {0}'.format(t6))

print('\n[p value]')
print('BSF vs. LSF: {0}'.format(p1))
print('BSF vs. HSF: {0}'.format(p2))
print('BSF vs. Equ: {0}'.format(p3))
print('LSF vs. HSF: {0}'.format(p4))
print('LSF vs. Equ: {0}'.format(p5))
print('HSF vs. Equ: {0}'.format(p6))

print('\n[FDR-corrected p value]')
print('BSF vs. LSF: {0}'.format(pval_corred[0]))
print('BSF vs. HSF: {0}'.format(pval_corred[1]))
print('BSF vs. Equ: {0}'.format(pval_corred[2]))
print('LSF vs. HSF: {0}'.format(pval_corred[3]))
print('LSF vs. Equ: {0}'.format(pval_corred[4]))
print('HSF vs. Equ: {0}'.format(pval_corred[5]))



# plot individual samples
df = pd.DataFrame({'BSF':Data_BSF, 'LSF':Data_LSF, 'HSF':Data_HSF, 'EQU':Data_Equ})
pal = ['darkgray', 'dodgerblue', 'tomato', 'springgreen']
innerplot = None # None or 'point'
pltYmin = 0
pltYmax = 10

plt.figure(figsize=(8,7))
plt.gcf().suptitle('Mean Amplitudes within significant cluster (%d-%d ms)' % (genTW[0], genTW[1]), fontsize=15)
v = sns.violinplot(data=df, palette=pal, scale='count', inner=innerplot, dodge=True, jitter=True, bw=0.4)

prop = v.properties()
if innerplot == 'point':
    propdata = prop['children'][:8]
    propdata = propdata[::2]
else:
    propdata = prop['children'][:4]

for n in propdata:
    m = np.mean(n.get_paths()[0].vertices[:,0])
    n.get_paths()[0].vertices[:,0] = np.clip(n.get_paths()[0].vertices[:,0], m, np.inf)

# also plot holizontal lines
for yval in np.arange(pltYmin, pltYmax+17, 5):
    plt.hlines(yval, -1, 4, color='gray', alpha=0.3)

# plot 95% confidence interval
cis_BSF = mne.stats.bootstrap_confidence_interval(Data_BSF, ci=0.95, n_bootstraps=5000, stat_fun='mean')
cis_LSF = mne.stats.bootstrap_confidence_interval(Data_LSF, ci=0.95, n_bootstraps=5000, stat_fun='mean')
cis_HSF = mne.stats.bootstrap_confidence_interval(Data_HSF, ci=0.95, n_bootstraps=5000, stat_fun='mean')
cis_Equ = mne.stats.bootstrap_confidence_interval(Data_Equ, ci=0.95, n_bootstraps=5000, stat_fun='mean')

plt.vlines(0.075, Data_BSF.mean(), cis_BSF[1], color='k')
plt.vlines(0.075, cis_BSF[0], Data_BSF.mean(), color='k')
plt.errorbar(0.075, Data_BSF.mean(), yerr=0, fmt='o', color='w', markeredgecolor='k', ecolor='k', ms=7)
plt.vlines(1.075, Data_LSF.mean(), cis_LSF[1], color='k')
plt.vlines(1.075, cis_LSF[0], Data_LSF.mean(), color='k')
plt.errorbar(1.075, Data_LSF.mean(), yerr=0, fmt='o', color='w', markeredgecolor='k', ecolor='k', ms=7)
plt.vlines(2.075, Data_HSF.mean(), cis_HSF[1], color='k')
plt.vlines(2.075, cis_HSF[0], Data_HSF.mean(), color='k')
plt.errorbar(2.075, Data_HSF.mean(), yerr=0, fmt='o', color='w', markeredgecolor='k', ecolor='k', ms=7)
plt.vlines(3.075, Data_Equ.mean(), cis_Equ[1], color='k')
plt.vlines(3.075, cis_Equ[0], Data_Equ.mean(), color='k')
plt.errorbar(3.075, Data_Equ.mean(), yerr=0, fmt='o', color='w', markeredgecolor='k', ecolor='k', ms=7)

# plot individual data
plt.scatter(np.ones(Data_BSF.shape[0])*(-0.075), Data_BSF, c=pal[0], marker='o', edgecolor='k', s=100)
plt.scatter(np.ones(Data_LSF.shape[0])*(1-0.075), Data_LSF, c=pal[1], marker='o', edgecolor='k', s=100)
plt.scatter(np.ones(Data_HSF.shape[0])*(2-0.075), Data_HSF, c=pal[2], marker='o', edgecolor='k', s=100)
plt.scatter(np.ones(Data_Equ.shape[0])*(3-0.075), Data_Equ, c=pal[3], marker='o', edgecolor='k', s=100)

# parameter setting
plt.xlim([-0.5, 3.5])
plt.ylim([pltYmin-2, pltYmax+17])
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.xlabel('SF', fontsize=20, labelpad=10)
plt.ylabel('Amplitude ($\it{a.u.}$)', fontsize=20, labelpad=10)
plt.gcf().subplots_adjust(top=0.9, bottom=0.12, left=0.12, right=0.95)


# save figure
dirlist = ['StatisticalData', 'SourceData', ExpID, srcdir1, dirname, 'PosthocTestResults']

os.chdir(filedir+'/GrandAverage/DataPlots')
for dname in dirlist:
    if not os.path.exists('./'+dname):
        os.mkdir('./'+dname)
    os.chdir('./'+dname)
currDir = os.getcwd()

if not os.path.exists(currDir+'/%s_at%s' % comb):
    os.mkdir(currDir+'/%s_at%s' % comb)
os.chdir(currDir+'/%s_at%s' % comb)

plt.gcf().savefig('SigniCluster_%dto%dms.png' % (genTW[0], genTW[1]))
plt.close(plt.gcf())


#%%

#%%
'''
make summary plot of significant time points
'''

Combs = [('Category', 'BSF'), ('Category', 'LSF'), ('Category', 'HSF'), ('Category', 'Equ'), 
         ('SF', 'NeutF'), ('SF', 'FearF'), ('SF', 'House')]

SigniTimes = dict(Category_atBSF=[[151, 183], [176, 265]],
                  Category_atLSF=[[144, 201], [222, 265]],
                  Category_atHSF=[[137, 152], [164, 216], [214, 265]],
                  Category_atEqu=[],
                  SF_atNeutF=[[137, 215], [212, 265]],
                  SF_atFearF=[[137, 216]],
                  SF_atHouse=[[137, 161], [186, 265]])

width = 0.3
margin = 0.4

yval1 = width+margin
yval2 = round(yval1+width*2+margin, 1)
yval3 = round(yval2+width*2+margin, 1)
yval4 = round(yval3+width*2+margin, 1)
yval5 = round(yval4+width*2+margin, 1)
yval6 = round(yval5+width*2+margin, 1)
yval7 = round(yval6+width*2+margin, 1)
pltMaxV = round(yval7+width*2+margin, 1)


# plotting a figure
plt.figure(figsize=(11,5.5))
gs = gridspec.GridSpec(2,10)
ax1 = plt.gcf().add_subplot(gs[:, :7])
ax2 = plt.gcf().add_subplot(gs[0, 7:])
ax3 = plt.gcf().add_subplot(gs[1, 7:])
plt.gcf().suptitle('Time intervals significant clusters spanned', fontsize=15)
kwgs = dict(edgecolor='k', linewidth=1, alpha=1)
for i, comb in enumerate(Combs):
    signiTs = SigniTimes['%s_at%s' % comb]
    
    ax1.hlines(locVal['yval%d' % (7-i)], 100, TOImin, linewidth=2, color='gray')
    ax1.hlines(locVal['yval%d' % (7-i)], TOImax, 300, linewidth=2, color='gray')
    
    if comb[0] == 'Category':
        if comb[-1] == 'BSF':
            Tclus1 = signiTs[0]
            Tclus2 = signiTs[-1]
            overlap = np.arange(Tclus2[0], (Tclus1[-1]+1))
            
            pltT1 = np.arange(Tclus1[0], (Tclus1[-1]+1))
            pltT1_short = np.arange(Tclus1[0], Tclus2[0])
            pltT2 = np.arange(Tclus2[0], (Tclus2[-1]+1))
            pltT2_short = np.arange((Tclus1[-1]+1), (Tclus2[-1]+1))
            
            area1_high = np.ones(pltT1.shape[0])*(yval7+width)
            area1_low = np.concatenate((np.ones(pltT1_short.shape[0])*(yval7-width), np.linspace((yval7-width), (yval7+width), num=overlap.shape[0])))
            area2_high = np.concatenate((np.linspace((yval7-width), (yval7+width), num=overlap.shape[0]), np.ones(pltT2_short.shape[0])*(yval7+width)))
            area2_low = np.ones(pltT2.shape[0])*(yval7-width)
            
            ax1.fill_between(pltT1, area1_low, area1_high, facecolor='magenta', label='Face > House', **kwgs)
            ax1.fill_between(pltT2, area2_low, area2_high, facecolor='cyan', label='House > Face', **kwgs)
            ax1.hlines(yval7, TOImin, Tclus1[0], linewidth=2, color='k')
            del Tclus1, Tclus2, overlap, pltT1, pltT1_short, pltT2, pltT2_short, area1_high, area1_low, area2_high, area2_low
            
        elif comb[-1] == 'LSF':
            for ii, tw in enumerate(signiTs):
                pltT = np.arange(tw[0], (tw[1]+1))
    
                if ii == 0 and tw[0] != TOImin:
                    ax1.hlines(locVal['yval%d' % (7-i)], TOImin, tw[0], linewidth=2, color='k')
    
                if tw[0] == 144:
                    ax1.fill_between(pltT, np.ones(pltT.shape[0])*(locVal['yval%d' % (7-i)]-width), 
                                     np.ones(pltT.shape[0])*(locVal['yval%d' % (7-i)]+width), 
                                     facecolor='magenta', **kwgs)
                else:
                    ax1.fill_between(pltT, np.ones(pltT.shape[0])*(locVal['yval%d' % (7-i)]-width), 
                                     np.ones(pltT.shape[0])*(locVal['yval%d' % (7-i)]+width), 
                                     facecolor='cyan', **kwgs)
                
                if ii != (len(signiTs)-1):
                    ax1.hlines(locVal['yval%d' % (7-i)], tw[1], signiTs[ii+1][0], linewidth=2, color='k')
                del pltT
            del ii, tw
            
            if signiTs[-1][-1] != TOImax:
                ax1.hlines(locVal['yval%d' % (7-i)], signiTs[-1][-1], TOImax, linewidth=2, color='k')
            
        elif comb[-1] == 'HSF':
            pltT = np.arange(signiTs[0][0], (signiTs[0][1]+1))
            ax1.fill_between(pltT, np.ones(pltT.shape[0])*(locVal['yval%d' % (7-i)]-width), 
                             np.ones(pltT.shape[0])*(locVal['yval%d' % (7-i)]+width), 
                             facecolor='cyan', **kwgs)
            ax1.hlines(locVal['yval%d' % (7-i)], signiTs[0][-1], signiTs[1][0], linewidth=2, color='k')
            del pltT
            
            Tclus1 = signiTs[1]
            Tclus2 = signiTs[2]
            overlap = np.arange(Tclus2[0], (Tclus1[-1]+1))
            
            pltT1 = np.arange(Tclus1[0], (Tclus1[-1]+1))
            pltT1_short = np.arange(Tclus1[0], Tclus2[0])
            pltT2 = np.arange(Tclus2[0], (Tclus2[-1]+1))
            pltT2_short = np.arange((Tclus1[-1]+1), (Tclus2[-1]+1))
            
            area1_high = np.ones(pltT1.shape[0])*(locVal['yval%d' % (7-i)]+width)
            area1_low = np.concatenate((np.ones(pltT1_short.shape[0])*(locVal['yval%d' % (7-i)]-width), np.linspace((locVal['yval%d' % (7-i)]-width), (locVal['yval%d' % (7-i)]+width), num=overlap.shape[0])))
            area2_high = np.concatenate((np.linspace((locVal['yval%d' % (7-i)]-width), (locVal['yval%d' % (7-i)]+width), num=overlap.shape[0]), np.ones(pltT2_short.shape[0])*(locVal['yval%d' % (7-i)]+width)))
            area2_low = np.ones(pltT2.shape[0])*(locVal['yval%d' % (7-i)]-width)
            
            ax1.fill_between(pltT1, area1_low, area1_high, facecolor='magenta', **kwgs)
            ax1.fill_between(pltT2, area2_low, area2_high, facecolor='cyan', **kwgs)
            del Tclus1, Tclus2, overlap, pltT1, pltT1_short, pltT2, pltT2_short, area1_high, area1_low, area2_high, area2_low
            
        else:
            ax1.hlines(locVal['yval%d' % (7-i)], TOImin, TOImax, linewidth=2, color='k')
            
    else:
        if comb[-1] == 'NeutF':
            Tclus1 = signiTs[0]
            Tclus2 = signiTs[1]
            overlap = np.arange(Tclus2[0], (Tclus1[-1]+1))
            
            pltT1 = np.arange(Tclus1[0], (Tclus1[-1]+1))
            pltT1_short = np.arange(Tclus1[0], Tclus2[0])
            pltT2 = np.arange(Tclus2[0], (Tclus2[-1]+1))
            pltT2_short = np.arange((Tclus1[-1]+1), (Tclus2[-1]+1))
            
            area1_high = np.ones(pltT1.shape[0])*(locVal['yval%d' % (7-i)]+width)
            area1_low = np.concatenate((np.ones(pltT1_short.shape[0])*(locVal['yval%d' % (7-i)]-width), np.linspace((locVal['yval%d' % (7-i)]-width), (locVal['yval%d' % (7-i)]+width), num=overlap.shape[0])))
            area2_high = np.concatenate((np.linspace((locVal['yval%d' % (7-i)]-width), (locVal['yval%d' % (7-i)]+width), num=overlap.shape[0]), np.ones(pltT2_short.shape[0])*(locVal['yval%d' % (7-i)]+width)))
            area2_low = np.ones(pltT2.shape[0])*(locVal['yval%d' % (7-i)]-width)
            
            ax1.fill_between(pltT1, area1_low, area1_high, facecolor='springgreen', label='LSF $\geq$ BSF, HSF > EQU', **kwgs)
            ax1.fill_between(pltT2, area2_low, area2_high, facecolor='slateblue', label='BSF, LSF $\geq$ HSF $\geq$ EQU', **kwgs)
            del Tclus1, Tclus2, overlap, pltT1, pltT1_short, pltT2, pltT2_short, area1_high, area1_low, area2_high, area2_low
            
        elif comb[-1] == 'FearF':
            pltT = np.arange(signiTs[0][0], (signiTs[0][1]+1))
            ax1.fill_between(pltT, np.ones(pltT.shape[0])*(locVal['yval%d' % (7-i)]-width), 
                             np.ones(pltT.shape[0])*(locVal['yval%d' % (7-i)]+width), 
                             facecolor='springgreen', **kwgs)
            ax1.hlines(locVal['yval%d' % (7-i)], signiTs[0][-1], TOImax, linewidth=2, color='k')
            del pltT
            
        else:
            pltT = np.arange(signiTs[0][0], (signiTs[0][1]+1)) # cluster 1
            ax1.fill_between(pltT, np.ones(pltT.shape[0])*(locVal['yval%d' % (7-i)]-width), 
                             np.ones(pltT.shape[0])*(locVal['yval%d' % (7-i)]+width), 
                             facecolor='orange', label='BSF, HSF > LSF > EQU', **kwgs)
            ax1.hlines(locVal['yval%d' % (7-i)], signiTs[0][-1], signiTs[1][0], linewidth=2, color='k')
            del pltT
            
            pltT = np.arange(signiTs[1][0], (signiTs[1][1]+1)) # cluster 2
            ax1.fill_between(pltT, np.ones(pltT.shape[0])*(locVal['yval%d' % (7-i)]-width), 
                             np.ones(pltT.shape[0])*(locVal['yval%d' % (7-i)]+width), 
                             facecolor='slateblue', **kwgs)
            del pltT
    del signiTs
del i, comb

ax1.vlines(TOImin, 0, pltMaxV, linewidth=1, color='k', linestyle='--')
ax1.vlines(TOImax, 0, pltMaxV, linewidth=1, color='k', linestyle='--')
ax1.set_xlim(100, 300)
ax1.set_ylim(0, pltMaxV)
ax1.spines['right'].set_color('none')
ax1.spines['top'].set_color('none')
ax1.spines['left'].set_linewidth(2)
ax1.spines['bottom'].set_linewidth(2)
ax1.set_xlabel('Time after stimulus onset (ms)',fontsize=15, labelpad=10)

yticklabels = ['SF at House', 'SF at Fearful face', 'SF at Neutral face', 'Category at EQU', 'Category at HSF', 'Category at LSF', 'Category at BSF']
yticks = [locVal['yval%d' % (i+1)] for i in np.arange(7)]

ax1.set_yticks(yticks)
ax1.set_yticklabels(yticklabels, fontsize=13)
ax1.set_xticklabels(['%d' % t for t in np.arange(100, 301, 25)], fontsize=13)

legends, labels = ax1.get_legend_handles_labels()
lg1 = ax2.legend(legends[:2], labels[:2], loc=(0.24, 0.6), prop={'size': 12}, frameon=True, framealpha=1., title='[Category]')
lg1.get_title().set_fontsize(13)
lg1.get_frame().set_linewidth(1)
lg1.get_frame().set_edgecolor('k')
ax2.set_axis_off()
lg2 = ax3.legend(legends[2:], labels[2:], loc=(0.05, 1.15), prop={'size': 12}, frameon=True, framealpha=1., title='[SF]')
lg2.get_title().set_fontsize(13)
lg2.get_frame().set_linewidth(1)
lg2.get_frame().set_edgecolor('k')
ax3.set_axis_off()

plt.gcf().subplots_adjust(top=0.915, bottom=0.13, left=0.17, right=0.94)

# save figure
os.chdir('')
plt.gcf().savefig('ResultsSummary_PosthocPairedTests.png')
plt.close(plt.gcf())





