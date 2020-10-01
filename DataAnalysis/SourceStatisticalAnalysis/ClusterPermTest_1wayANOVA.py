#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
statistical test using cluster-based permutation test
(1-way ANOVA for interaction/source-level data)

@author: Akinori Takeda
"""

import mne
from mne.stats import f_mway_rm, permutation_cluster_test
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

conditions = list(epochs.event_id.keys())
conditions2 = [i for i in conditions if i != 'target']
times = epochs.times
sfreq = epochs.info['sfreq']
del Epochs, epochs


#- make directory for saving results -#
TOImin = 137
TOImax = 265
srcdir1 = 'SurfSrcEst_dSPM_forEvoked'
dirname = 'PostAnalyses_Fsave%s_%dto%dms' % (useFsaveModel.capitalize(), TOImin, TOImax)
os.chdir(filedir+'/GrandAverage/Datafiles')
for direcname in ['StatisticalData', 'SourceData', ExpID, srcdir1, dirname]:
    if not os.path.exists('./'+direcname):
        os.mkdir('./'+direcname)
    os.chdir('./'+direcname)
savedir = os.getcwd()
del dirname


#- load data -#
dirname = 'ANOVA_Fsave%s_%dto%dms' % (useFsaveModel.capitalize(), TOImin, TOImax)
statdatadir = filedir+'/GrandAverage/Datafiles/StatisticalData/SourceData/'+ExpID+'/'+srcdir1+'/'+dirname+'/Interaction'
os.chdir(statdatadir)
Pval_Interaction = np.load('Pvalues.npy')
del dirname

timemask_TOI = np.where((TOImin/1000 <= times) & (times <= TOImax/1000))[0]


MRIsubject = 'fsaverage'
subjects_dir = ''

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

if useFsaveModel != 'ico5':
    # for making SourceEstimate instance of fsaverage
    dirname = filedir+'/Subject1/'+ExpID+'/Datafiles/SourceEstimate/'+srcdir1+'/Fsaverage_%s' % useFsaveModel
    templateSTC = mne.read_source_estimate(dirname+'/SrcEst_MeanTC_FearF_BSF_fsaverage%s' % useFsaveModel.capitalize())
    

#%%
#--- Do Cluster-based permutation test ---#
# setting some parameters
Nperm = 1000
tail = 1
njobs = 6

thresh = dict(start=0, step=0.2) # use TFCE


#- post analysis for interaction (repeated measures 1-way ANOVA) -#
# make combination list
Combs = [('Category', 'BSF'), ('Category', 'LSF'), ('Category', 'HSF'), ('Category', 'Equ'),
         ('SF', 'NeutF'), ('SF', 'FearF'), ('SF', 'House')]

msk = Pval_Interaction >= 0.05 # data point with p >= 0.05 will be excluded

print('\n<< Post Analyses for Interaction (repeated measures 1-way ANOVA) >>')
startT_all = time.time()
for comb in Combs:
    factor = comb[0]
    cond = comb[1]
    print('< %s at %s >' % (factor, cond))
    
    # make dataset
    print(' > Loading data...')
    if useFsaveModel == 'ico5':
        os.chdir(filedir+'/GrandAverage/Datafiles/forSurfaceSrcEstAnalysis/'+srcdir1+'/%sexp_N%d' % (ExpID, SubjN))
    else:
        os.chdir(filedir+'/GrandAverage/Datafiles/forSurfaceSrcEstAnalysis/'+srcdir1+'/%sexp_N%d_fsaverage%s' % (ExpID, SubjN, useFsaveModel.capitalize()))
    
    if factor == 'Category':
        levellist = ['NeutF','FearF','House']
        data1 = np.load('SurfData_NeutF_%s.npy' % cond)
        data2 = np.load('SurfData_FearF_%s.npy' % cond)
        data3 = np.load('SurfData_House_%s.npy' % cond)
    else:
        levellist = ['BSF','LSF','HSF','Equ']
        data1 = np.load('SurfData_%s_BSF.npy' % cond)
        data2 = np.load('SurfData_%s_LSF.npy' % cond)
        data3 = np.load('SurfData_%s_HSF.npy' % cond)
        data4 = np.load('SurfData_%s_Equ.npy' % cond)
    
    Dataset = []
    for n in np.arange(len(levellist)):
        data = locVal['data%d' % (n+1)][:,nuseVerts:,timemask_TOI]
        Dataset.append(data.transpose([0,2,1]))
        del data, locVal['data%d' % (n+1)]
    del n
    
    # define customed stat_fun
    def stat_fun_rm1wayANOVA(*args):
        return f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=[len(levellist)],
                         effects='A', correction=True, return_pvals=False)[0]
    
    # 1-way ANOVA with cluster-based permutation test procedure
    print(' > Running 1-way ANOVA...')
    startT = time.time()
    scores, _, p_val, H0 = permutation_cluster_test(Dataset, threshold=thresh, n_permutations=Nperm, tail=tail,
                                                    stat_fun=stat_fun_rm1wayANOVA, connectivity=connectivity,
                                                    n_jobs=njobs, buffer_size=None, exclude=msk.reshape(-1),
                                                    out_type='indices')
    
    # add results to data containers
    p_val = p_val.reshape(scores.shape)
    elapsed_time = (time.time() - startT)/60
    
    # save data
    os.chdir(savedir)
    if not os.path.exists('./%s_at%s' % (factor, cond)):
        os.mkdir('./%s_at%s' % (factor, cond))
    os.chdir('./%s_at%s' % (factor, cond))
    
    print(' > Saving data...')
    np.save('TFCEscores.npy', scores)
    np.save('Pvalues.npy', p_val)
    np.save('ObservedClusterLevelStats.npy', H0)
    
    print('  -> Finished.')
    print('     Elapsed_time: {0}'.format(elapsed_time)+" [min]\n")
    del factor, cond, Dataset, startT, scores, p_val, H0, elapsed_time
del comb

elapsed_time_all = (time.time() - startT_all)/3600
print('\n  ==> All processes took {0}'.format(elapsed_time_all)+" hours.\n")

