#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
statistical test using cluster-based permutation test
(repeated measures 2-way ANOVA/source-level data)

@author: Akinori Takeda
"""

import mne
from mne.stats import f_mway_rm, spatio_temporal_cluster_test
import numpy as np
import time
import os
locVal = locals()


#--- set data path & get datafiles' name ---#
filedir = ''
ExpID = 'SF'
useFsaveModel = 'oct6'


# make subject list
SubjDir = [i for i in os.listdir(filedir) if 'Subject' in i and 'Test' not in i and '.' not in i]
rmvSubjList = []

SubjList = ['Subject%d' % (n+1) for n in np.arange(len(SubjDir)) if 'Subject%d' % (n+1) not in rmvSubjList and os.path.exists(filedir+'/Subject%d/SF' % (n+1))]
SubjN = len(SubjList)
del SubjDir


# setting some parameters
os.chdir(filedir+'/'+SubjList[0]+'/'+ExpID+'/Datafiles/EpochData')
Epochs = mne.read_epochs('ProperEpochData-epo.fif', preload=True)
epochs = Epochs.copy().pick_types(meg=True)

times = epochs.times
sfreq = epochs.info['sfreq']
del Epochs, epochs

conditions = ['NeutF_BSF', 'NeutF_LSF', 'NeutF_HSF', 'NeutF_Equ', 'FearF_BSF', 'FearF_LSF', 'FearF_HSF', 'FearF_Equ', 'House_BSF', 'House_LSF', 'House_HSF', 'House_Equ']
timemask_TOI = np.where((0.137 <= times)&(times <= 0.265))[0]


#- make directory for saving results -#
srcdir1 = 'SurfSrcEst_dSPM_forEvoked'
dirname = 'ANOVA_Fsave%s_137to265ms' % useFsaveModel.capitalize()
os.chdir(filedir+'/GrandAverage/Datafiles')
for direcname in ['StatisticalData', 'SourceData', ExpID, srcdir1, dirname]:
    if not os.path.exists('./'+direcname):
        os.mkdir('./'+direcname)
    os.chdir('./'+direcname)
savedir = os.getcwd()


#- load data & make dataset -#
MRIsubject = 'fsaverage'
subjects_dir = ''

if useFsaveModel == 'ico5':
    src = mne.read_source_spaces(subjects_dir+'/'+MRIsubject+'/bem/%s-5-src.fif' % MRIsubject)
else:
    src = mne.read_source_spaces(subjects_dir+'/'+MRIsubject+'/bem/%s-%s-src.fif' % (MRIsubject, useFsaveModel))

connectivity = mne.spatial_src_connectivity(src[-1:]) # use right hemi only
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

if useFsaveModel == 'ico5':
    excludedVerts = [i for i in np.arange(10242) if i not in CombinedLabel.get_vertices_used()]
else:
    # for making SourceEstimate instance of fsaverage
    dirname = filedir+'/Subject1/'+ExpID+'/Datafiles/SourceEstimate/'+srcdir1+'/Fsaverage_%s' % useFsaveModel
    templateSTC = mne.read_source_estimate(dirname+'/SrcEst_MeanTC_FearF_BSF_fsaverage%s' % useFsaveModel.capitalize())
    
    labelSTC = templateSTC.in_label(CombinedLabel)
    excludedVerts = [i for i, vert in enumerate(templateSTC.rh_vertno) if vert not in labelSTC.rh_vertno]
    del dirname, templateSTC, labelSTC


# load source data
print('\n< load source timecourse data >')
if useFsaveModel == 'ico5':
    os.chdir(filedir+'/GrandAverage/Datafiles/forSurfaceSrcEstAnalysis/'+srcdir1+'/%sexp_N%d' % (ExpID, SubjN))
else:
    os.chdir(filedir+'/GrandAverage/Datafiles/forSurfaceSrcEstAnalysis/'+srcdir1+'/%sexp_N%d_fsaverage%s' % (ExpID, SubjN, useFsaveModel.capitalize()))
Dataset = []
for cond in conditions:
    # load surface source data
    print(' > Loading %s data...' % cond.replace('_', '-'))
    exec('SurfData_%s = np.load(\'SurfData_%s.npy\')' % (cond, cond))
    data_post = locVal['SurfData_'+cond][:,nuseVerts:,timemask_TOI]
    X = data_post.transpose([0,2,1])
    Dataset.append(X)
    del locVal['SurfData_'+cond], data_post, X
del cond


#%%
#- Preparation for statistical test -#
# setting some parameters
Nperm = 1000
tail = 1
njobs = 6

factor_levels = [3, 4]  # Category (NeutF, FearF, House) X SF (BSF, LSF, HSF, EQU)
if factor_levels[0] <= 2 and factor_levels[1] <= 2:
    correction = False
else:
    correction = True

thresh = dict(start=0, step=0.2) # use TFCE


# define customed stat_fun
def stat_fun_Category(*args):
    return f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=factor_levels,
                     effects='A', correction=correction, return_pvals=False)[0]

def stat_fun_SF(*args):
    return f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=factor_levels,
                     effects='B', correction=correction, return_pvals=False)[0]

def stat_fun_Interaction(*args):
    return f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=factor_levels,
                     effects='A:B', correction=correction, return_pvals=False)[0]


#--- Do Cluster-based permutation test ---#
print('\n< Statistical analysis: repeated measures 2-way ANOVA >')
startT_all = time.time()
effects = ['A', 'B', 'A:B']
for ef in effects:
    if ef == 'A':
        print('[Main effect of Category]')
        stat_fun = stat_fun_Category
        savedirname = 'MainEffect_Category'
    elif ef == 'B':
        print('[Main effect of SF]')
        stat_fun = stat_fun_SF
        savedirname = 'MainEffect_SF'
    elif ef == 'A:B':
        print('[Category x SF interaction]')
        stat_fun = stat_fun_Interaction
        savedirname = 'Interaction'
    
    # ANOVA with cluster-based permutation test
    startT = time.time()
    scores, _, p_val, H0 = spatio_temporal_cluster_test(Dataset, threshold=thresh, n_permutations=Nperm, tail=tail,
                                                        connectivity=connectivity, n_jobs=njobs, buffer_size=None, 
                                                        spatial_exclude=excludedVerts, stat_fun=stat_fun)
    elapsed_time = (time.time() - startT)/60
    p_val = p_val.reshape(scores.shape)
    
    # save results
    os.chdir(savedir)
    if not os.path.exists('./'+savedirname):
        os.mkdir('./'+savedirname)
    os.chdir('./'+savedirname)
    
    np.save('TFCEscores.npy', scores)
    np.save('Pvalues.npy', p_val)
    np.save('ObservedClusterLevelStats.npy', H0)
    
    print('  -> Finished.')
    print('     Elapsed_time: {0}'.format(elapsed_time)+" [min]\n")
    del savedirname, startT, elapsed_time, scores, p_val, H0

elapsed_time_all = (time.time() - startT_all)/3600
print('\n  ==> All processes took {0}'.format(elapsed_time_all)+" hours.\n")
