#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
statistical test using cluster-based permutation test
(within-trial comparisons/source-level data)

@author: Akinori Takeda
"""

import mne
import numpy as np
import time
import os
locVal = locals()


#--- set data path & get datafiles' name ---#
filedir = ''
ExpID = 'SF'
useFsaveModel = 'oct6'

srcdir1 = 'SurfSrcEst_dSPM_forEvoked'
RestrictedToROI = True


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


MRIsubject = 'fsaverage'
subjects_dir = ''

src = mne.read_source_spaces(subjects_dir+'/'+MRIsubject+'/bem/%s-%s-src.fif' % (MRIsubject, useFsaveModel))
connectivity = mne.spatial_src_connectivity(src[-1:]) # use right hemi only
nuseVerts = src[-1]['nuse']


#- make directory for saving results -#
os.chdir(filedir+'/GrandAverage/Datafiles')
for direcname in ['StatisticalData', 'SourceData', ExpID, srcdir1]:
    if not os.path.exists('./'+direcname):
        os.mkdir('./'+direcname)
    os.chdir('./'+direcname)
savedir = os.getcwd()

if not os.path.exists('./WithinTrialComparisons'):
    os.mkdir('./WithinTrialComparisons')
os.chdir('./WithinTrialComparisons')
savedir += '/WithinTrialComparisons'


#- parameter setting -#
Tmin = 0
Tmax = 0.3
Nperm = 1000
tail = 0
njobs = 4

thresh = dict(start=0, step=0.2)

timemask_pre = np.where(((-Tmax-1/sfreq) <= times)&(times <= (Tmin-1/sfreq)))[0]
timemask_post = np.where((Tmin <= times)&(times <= Tmax))[0]


if RestrictedToROI:
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


#--- Do Cluster-based permutation test ---#
print('\n< Statistical analysis: within-trial comparisons >')
startT_all = time.time()
for cond in conditions2:
    print('[%s data]' % cond.replace('_', '-'))
    
    os.chdir(savedir)
    if os.path.exists('./'+cond):
        print(' -> processing of the data was already finished.\n')
        continue
    
    # load surface source data
    print(' > Loading data...')
    if useFsaveModel == 'ico5':
        os.chdir(filedir+'/GrandAverage/Datafiles/forSurfaceSrcEstAnalysis/'+srcdir1+'/%sexp_N%d' % (ExpID, SubjN))
    else:
        os.chdir(filedir+'/GrandAverage/Datafiles/forSurfaceSrcEstAnalysis/'+srcdir1+'/%sexp_N%d_fsaverage%s' % (ExpID, SubjN, useFsaveModel.capitalize()))
    SurfData = np.load('SurfData_%s.npy' % cond)
    
    # make datasets
    data_pre = SurfData[:,nuseVerts:,timemask_pre]
    data_post = SurfData[:,nuseVerts:,timemask_post]
    X = data_post - data_pre
    X = X.transpose([0,2,1])
    del SurfData, data_pre, data_post
    
    # do cluster-based permutation test (using 1-sample t-test)
    print(' > Conducting cluster-based permutation test...')
    startT = time.time()
    if RestrictedToROI:
        scores, _, p_val, H0 = mne.stats.spatio_temporal_cluster_1samp_test(X, threshold=thresh, n_permutations=Nperm, tail=tail,
                                                                            connectivity=connectivity, n_jobs=njobs, buffer_size=None, 
                                                                            spatial_exclude=excludedVerts)
    else:
        scores, _, p_val, H0 = mne.stats.spatio_temporal_cluster_1samp_test(X, threshold=thresh, n_permutations=Nperm, tail=tail,
                                                                           connectivity=connectivity, n_jobs=njobs, buffer_size=None)
    
    p_val = p_val.reshape(scores.shape)
    
    # save results
    os.chdir(savedir)
    if not os.path.exists('./'+cond):
        os.mkdir('./'+cond)
    os.chdir('./'+cond)
    
    np.save('TFCEscores.npy', scores)
    np.save('Pvalues.npy', p_val)
    np.save('ObservedClusterLevelStats.npy', H0)
    
    elapsed_time = (time.time() - startT)/60
    print('  -> Finished.')
    print('     Elapsed_time: {0}'.format(elapsed_time)+" [min]\n")
    
    del X, startT, scores, p_val, H0, elapsed_time
del cond
print(' => Finished.')

elapsed_time_all = (time.time() - startT_all)/3600
print('\n  ==> All processes took {0}'.format(elapsed_time_all)+" hours.\n")

