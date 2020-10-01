#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot results of Within-trial comparisons with source waveforms

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
    del r
del roi

CombinedLabel = HCPlabellist[0] + HCPlabellist[1] + HCPlabellist[2] + HCPlabellist[3] + HCPlabellist[4]

if useFsaveModel != 'ico5':
    # for making SourceEstimate instance of fsaverage
    dirname = filedir+'/Subject1/'+ExpID+'/Datafiles/SourceEstimate/'+srcdir1+'/Fsaverage_%s' % useFsaveModel
    templateSTC = mne.read_source_estimate(dirname+'/SrcEst_MeanTC_FearF_BSF_fsaverage%s' % useFsaveModel.capitalize())
    del dirname


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
dirname = 'WithinTrialComparisons'
statdatadir = filedir+'/GrandAverage/Datafiles/StatisticalData/SourceData/'+ExpID+'/'+srcdir1+'/'+dirname
for cond in conditions2:
    os.chdir(statdatadir+'/'+cond)
    exec('Scores_'+cond+' = np.load(\'TFCEscores.npy\')')
    exec('Pval_'+cond+' = np.load(\'Pvalues.npy\')')
del cond


#%%
'''
Preparation for plotting
'''
#- parameter setting -#
pltTmin = -0.1
pltTmax = 0.3
timemask = np.where((pltTmin <= times) & (times <= pltTmax))[0]
timemask_post = np.where((0 <= times) & (times <= pltTmax))[0]

pltTimes = times[timemask]
pltTimes *= 1000

Times_post = times[timemask_post]
Times_post *= 1000

colorList = dict(BSF='black',LSF='blue',HSF='red',Equ='green')


#- value range setting -#
# get max & min values
MaxVs = []
for cond in conditions2:
    data = locVal['SurfData_'+cond][:,:,timemask]
    if useFsaveModel == 'ico5' and '_withoutSrcConnectivity' not in dirname:
        data = data[:,CombinedLabel.get_vertices_used(),:]
    elif useFsaveModel != 'ico5':
        labelSTC = templateSTC.in_label(CombinedLabel)
        if useFsaveModel == 'oct6':
            vertIdx = np.array([i for i, n in enumerate(templateSTC.rh_vertno) if n in labelSTC.rh_vertno])
            data = data[:,vertIdx,:]
        else:
            data = data[:,labelSTC.rh_vertno,:]
        del labelSTC
    
    # calculate 95% confidence interval
    cis = mne.stats.bootstrap_confidence_interval(data, ci=0.95, n_bootstraps=5000, stat_fun='mean')
    UP = cis[1,:]
    
    # add to container
    MaxVs.append(UP.max())
    
    del data, cis, UP
del cond

pltMaxV = round(np.max(MaxVs))
pltMinV = 0
del MaxVs

ylabel = 'Amplitude (${a.u.}$)'


#- directory manipulation -#
# make directory for data plot if not exist
dirlist = ['StatisticalData', 'SourceData', ExpID, srcdir1, dirname, 'VertexWaveforms']

os.chdir(filedir+'/GrandAverage/DataPlots')
for dname in dirlist:
    if not os.path.exists('./'+dname):
        os.mkdir('./'+dname)
    os.chdir('./'+dname)
currDir = os.getcwd()


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
Make individual vertex plot
'''
useMinPval = False
if not useMinPval:
    alphaP = 0.005

restrictedTW = False
if restrictedTW:
    useEachTW = True
    
    if useEachTW:
        TW_BSF = [137, 213]
        TW_LSF = [137, 208]
        TW_HSF = [177, 221]
        TW_Equ = [199, 265]
    else:
        TW_all = [137, 265]


print('\n<< Make individual vertex plots >>')
for cond in conditions2:
    os.chdir(currDir)
    if not os.path.exists('./'+cond):
        os.mkdir('./'+cond)
    os.chdir('./'+cond)
    print('> Processing %s data...' % cond.replace('_','-'))
    
    #-- preparation for plotting --#
    if useMinPval:
        Pval = locVal['Pval_'+cond] == locVal['Pval_'+cond].min()
    else:
        Pval = locVal['Pval_'+cond] < alphaP
    Pval_liberal = locVal['Pval_'+cond] < 0.05
    
    #- make data for plot -#
    data = locVal['SurfData_'+cond][:,:,timemask]
    
    #- make time mask if necessary -#
    if restrictedTW:
        if useEachTW:
            TOImin = locVal['TW_'+cond.split('_')[-1]][0]
            TOImax = locVal['TW_'+cond.split('_')[-1]][1]
        else:
            TOImin = TW_all[0]
            TOImax = TW_all[1]
        TOImask = np.where((TOImin <= pltTimes) & (pltTimes <= TOImax))[0]
        TOImask_post = np.where((TOImin <= Times_post) & (Times_post <= TOImax))[0]
    
    
    # < Plot individual data > #
    for label in HCPlabellist:
        labelname = label.name.split('_')[1]
        print('  plotting %s data...' % labelname)
        if labelname == 'PIT':
            title1 = 'IOG/OFA (\"PIT\"): '
        elif labelname == 'FFC':
            title1 = 'Lateral FG (\"FFC\"): '
        elif labelname == 'VVC':
            title1 = 'Medial FG (\"VVC\"): '
        else:
            title1 = labelname + ': '
        
        data2 = locVal['SurfData_'+cond][:,:,timemask_post]
        if useFsaveModel == 'ico5':
            data2 = data2.mean(0)[label.get_vertices_used(),:]
            vertInfo = label.get_vertices_used()
        else:
            labelSTC = templateSTC.in_label(label)
            if useFsaveModel == 'oct6':
                vertInfo = np.array([i for i, n in enumerate(templateSTC.rh_vertno) if n in labelSTC.rh_vertno])
            else:
                vertInfo = labelSTC.rh_vertno
            data2 = data2.mean(0)[vertInfo,:]
            del labelSTC
        
        if restrictedTW:
            MaxVvert = np.where(data2 == data2[:,TOImask_post].max())[0][0]
        else:
            MaxVvert = np.where(data2 == data2.max())[0][0]
        
        for idx, vert in enumerate(vertInfo):
            #- identify temporal clusters -#
            signi_tp, identified_timeclus = DetectTemporalClus(Pval[:,vert])
            signi_tp2, identified_timeclus2 = DetectTemporalClus(Pval_liberal[:,vert])
            
            #- make mean time course -#
            meanTC = data[:, vert, :].mean(0)
            
            # 95% confidence interval
            cis = mne.stats.bootstrap_confidence_interval(data[:, vert, :], ci=0.95, n_bootstraps=5000, stat_fun='mean')
            Btm = cis[0,:]
            Up = cis[1,:]
            
            # title setting 2
            if 'NeutF' in cond:
                title2 = 'Neutral face - '
            elif 'FearF' in cond:
                title2 = 'Fearful face - '
            else:
                title2 = 'House - '
            
            if '_Equ' in cond:
                title3 = 'Equiluminant'
            else:
                title3 = cond.split('_')[-1]
            title = title1 + title2 + title3
            
            margin = 0.4
            width = 0.3
            yval = pltMinV - margin - width
            pltVmin = yval - margin - width
            
            #- plot -#
            plt.figure(figsize=(8,6))
            if useFsaveModel == 'ico5':
                plt.gcf().suptitle('Vertex No.%d/%d in ' % ((idx+1), label.get_vertices_used().shape[0]) + title, fontsize=14)
            else:
                plt.gcf().suptitle('Vertex No.%d/%d in ' % ((idx+1), vertInfo.shape[0]) + title, fontsize=14)
            plt.vlines(0, pltVmin, pltMaxV, linewidth=1)
            plt.ylim(pltVmin, pltMaxV)
            
            plt.hlines(0, pltTimes[0], pltTimes[-1], linewidth=1)
            plt.plot(pltTimes, meanTC, linewidth=2, color=colorList[cond.split('_')[-1]])
            plt.fill_between(pltTimes, Btm, Up, alpha=0.3, color=colorList[cond.split('_')[-1]])
            plt.xlim(pltTimes[0], pltTimes[-1])
            plt.gcf().gca().spines['right'].set_color('none')
            plt.gcf().gca().spines['top'].set_color('none')
            plt.gcf().gca().spines['left'].set_linewidth(2)
            plt.gcf().gca().spines['bottom'].set_linewidth(2)
            plt.xlabel('Time (ms)',fontsize=22, labelpad=10)
            plt.ylabel(ylabel,fontsize=22, labelpad=10)
            plt.xticks(fontsize=19)
            plt.yticks(fontsize=19)
            del cis, Btm, title, title2, title3
            
            #- shading significant time windows -#
            # using liberal alpha (0.05)
            if signi_tp2.shape[0] >= 1:
                color = 'gray'
                alpha = 0.8
                
                for c in identified_timeclus2:
                    if len(c) == 1:
                        plt.vlines(Times_post[c[0]], yval+width, yval-width, alpha=alpha, color=color, linewidth=1)
                    elif len(c) >= 2:
                        plt.fill_between(Times_post[c], np.ones(len(c))*(yval+width), np.ones(len(c))*(yval-width), 
                                         alpha=alpha, color=color)
                del c, color, alpha
                
                # plot horizontal line
                for i, c in enumerate(identified_timeclus2):
                    if i == 0:
                        plt.hlines(yval, 0, Times_post[c[0]], linewidth=1)
                        prevEnd = c[-1]
                    else:
                        plt.hlines(yval, Times_post[prevEnd], Times_post[c[0]], linewidth=1)
                        prevEnd = c[-1]
                    
                    if i == (len(identified_timeclus2)-1) and Times_post[c[-1]] != pltTimes[-1]:
                        plt.hlines(yval, Times_post[c[-1]], pltTimes[-1], linewidth=1)
                del i, c, prevEnd
            else:
                plt.hlines(yval, 0, pltTimes[-1], linewidth=1)
            
            # using more rigolous alpha
            if signi_tp.shape[0] >= 1:
                color = 'k'
                alpha = 1
                
                for c in identified_timeclus:
                    if len(c) == 1:
                        plt.vlines(Times_post[c[0]], yval+width, yval-width, alpha=alpha, color=color, linewidth=1)
                    elif len(c) >= 2:
                        plt.fill_between(Times_post[c], np.ones(len(c))*(yval+width), np.ones(len(c))*(yval-width), 
                                         alpha=alpha, color=color)
                del c, color, alpha
            
            #- plot peak latency -#
            # detect peak latency
            if restrictedTW:
                maxAbsV = meanTC[TOImask].max()
                maskt = np.where(meanTC[TOImask] == maxAbsV)[0][0]
                t = pltTimes[TOImask][maskt]
                markY = Up[TOImask][maskt]*1.15
            else:
                maxAbsV = meanTC.max()
                maskt = np.where(meanTC == maxAbsV)[0][0]
                t = pltTimes[maskt]
                markY = Up[maskt]*1.15
            
            # check whether the peak latency exists in significant time interval
            tidx = np.where(Times_post==t)[0][0]
            tempclus = [c for c in identified_timeclus if tidx in c]
            
            notSigniLatency = False
            if tempclus == []:
                notSigniLatency = True
                peakLat = t
                useloweralpha = False
                del maxAbsV, maskt, t, tidx, tempclus
                
                ts = [i for i, k in enumerate(pltTimes) if k in Times_post[signi_tp]]
                if ts == []:
                    ts = [i for i, k in enumerate(pltTimes) if k in Times_post[signi_tp2]]
                    useloweralpha = True
                
                maxAbsV = meanTC[ts].max()
                maskt = np.where(meanTC == maxAbsV)[0][0]
                t = pltTimes[maskt]
                tidx = np.where(Times_post==t)[0][0]
                if useloweralpha:
                    tempclus = [c for c in identified_timeclus2 if tidx in c]
                else:
                    tempclus = [c for c in identified_timeclus if tidx in c]
                del ts, useloweralpha
            
            plt.scatter(t, markY, s=200, c='k', marker='v')
            plt.gcf().subplots_adjust(top=0.92, bottom=0.15, left=0.13, right=0.94)
            
            # save figure
            os.chdir(currDir+'/'+cond)
            if not os.path.exists('./'+labelname):
                os.mkdir('./'+labelname)
            os.chdir('./'+labelname)
            
            fig = plt.gcf()
            fig.savefig('Waveform_%s_VertNo%d.png' % (labelname, (idx+1)))
            plt.close(fig)
            del signi_tp, identified_timeclus, signi_tp2, identified_timeclus2, fig
        del labelname, title1, data2, MaxVvert, idx, vert
    del Pval, Pval_liberal, data
del cond
print('\n  ==> Finished.')


#%%
os.chdir(filedir+'/GrandAverage/DataPlots/StatisticalData/SourceData/'+ExpID+'/'+srcdir1+'/'+dirname)
if not os.path.exists('./StatisticalTests_ResultsSummary'):
    os.mkdir('./StatisticalTests_ResultsSummary')
os.chdir('./StatisticalTests_ResultsSummary')
currDir2 = os.getcwd()


# make datasets 
useMinPval = False
if not useMinPval:
    alphaP = 0.005

restrictedTW = True
if restrictedTW:
    useEachTW = True

# make summary output
print('\n<< Make Summary of Results from Statistical Analysis >>')
for cond in conditions2:
    os.chdir(currDir2)
    if not os.path.exists('./'+cond):
        os.mkdir('./'+cond)
    os.chdir('./'+cond)
    savedir2 = os.getcwd()
    print('> Processing %s data...' % cond.replace('_','-'))
    
    #-- preparation for plotting --#
    if useMinPval:
        Pval = locVal['Pval_'+cond] == locVal['Pval_'+cond].min()
    else:
        Pval = locVal['Pval_'+cond] < alphaP
    
    #- make data for plot -#
    data = locVal['SurfData_'+cond][:,:,timemask]
    
    #- make time mask if necessary -#
    if restrictedTW:
        if useEachTW:
            TOImin = locVal['TW_'+cond.split('_')[-1]][0]
            TOImax = locVal['TW_'+cond.split('_')[-1]][1]
        else:
            TOImin = TW_all[0]
            TOImax = TW_all[1]
        TOImask = np.where((TOImin <= pltTimes) & (pltTimes <= TOImax))[0]
        TOImask_post = np.where((TOImin <= Times_post) & (Times_post <= TOImax))[0]
    
    # title setting
    if 'NeutF' in cond:
        title2 = 'Neutral face - '
    elif 'FearF' in cond:
        title2 = 'Fearful face - '
    else:
        title2 = 'House - '
    
    if '_Equ' in cond:
        title3 = 'Equiluminant'
    else:
        title3 = cond.split('_')[-1]
    
    # < Plot individual data > #
    for label in HCPlabellist:
        labelname = label.name.split('_')[1]
        print('  plotting %s data...' % labelname)
        if labelname == 'PIT':
            title1 = 'IOG/OFA (\"PIT\")'
        elif labelname == 'FFC':
            title1 = 'Lateral FG (\"FFC\")'
        elif labelname == 'VVC':
            title1 = 'Medial FG (\"VVC\")'
        else:
            title1 = labelname
        
        data2 = locVal['SurfData_'+cond][:,:,timemask_post]
        if useFsaveModel == 'ico5':
            data2 = data2.mean(0)[label.get_vertices_used(),:]
            vertInfo = label.get_vertices_used()
        else:
            labelSTC = templateSTC.in_label(label)
            if useFsaveModel == 'oct6':
                vertInfo = np.array([i for i, n in enumerate(templateSTC.rh_vertno) if n in labelSTC.rh_vertno])
            else:
                vertInfo = labelSTC.rh_vertno
            data2 = data2.mean(0)[vertInfo,:]
            del labelSTC
        
        if restrictedTW:
            MaxVvert = np.where(data2 == data2[:,TOImask_post].max())[0][0]
        else:
            MaxVvert = np.where(data2 == data2.max())[0][0]
        
        for idx, vert in enumerate(vertInfo):
            #- identify temporal clusters -#
            signi_tp, identified_timeclus = DetectTemporalClus(Pval[:,vert])
            
            # detect peak latency
            meanTC = data[:, vert, :].mean(0)
            if restrictedTW:
                maxAbsV = meanTC[TOImask].max()
                maskt = np.where(meanTC[TOImask] == maxAbsV)[0][0]
                t = pltTimes[TOImask][maskt]
            else:
                maxAbsV = meanTC.max()
                maskt = np.where(meanTC == maxAbsV)[0][0]
                t = pltTimes[maskt]
            
            # check whether the peak latency exists in significant time interval
            tidx = np.where(Times_post==t)[0][0]
            tempclus = [c for c in identified_timeclus if tidx in c]
            
            notSigniLatency = False
            if tempclus == []:
                notSigniLatency = True
                peakLat = t
                useloweralpha = False
                del maxAbsV, maskt, t, tidx, tempclus
                
                Pval_liberal = locVal['Pval_'+cond] < 0.05
                signi_tp2, identified_timeclus2 = DetectTemporalClus(Pval_liberal[:,vert])
                
                ts = [i for i, k in enumerate(pltTimes) if k in Times_post[signi_tp]]
                if ts == []:
                    ts = [i for i, k in enumerate(pltTimes) if k in Times_post[signi_tp2]]
                    useloweralpha = True
                
                maxAbsV = meanTC[ts].max()
                maskt = np.where(meanTC == maxAbsV)[0][0]
                t = pltTimes[maskt]
                tidx = np.where(Times_post==t)[0][0]
                if useloweralpha:
                    tempclus = [c for c in identified_timeclus2 if tidx in c]
                else:
                    tempclus = [c for c in identified_timeclus if tidx in c]
                del ts, useloweralpha
            
            # make summary
            if useFsaveModel == 'ico5':
                title = '<< Summary of Results of Within-trial comparisons: %s >>\n $\\bigcirc$Vertex No.%d/%d in ' % (title2+title3, (idx+1), label.get_vertices_used().shape[0]) + title1
            else:
                title = '<< Summary of Results of Within-trial comparisons: %s >>\n $\\bigcirc$Vertex No.%d/%d in ' % (title2+title3, (idx+1), vertInfo.shape[0]) + title1
            
            if idx == MaxVvert:
                if restrictedTW:
                    title += '\n $\\bigcirc$This vertex has maximum peak amplitude within the label in %d-%d ms.' % (TOImin, TOImax)
                else:
                    title += '\n $\\bigcirc$This vertex has maximum peak amplitude in the label.'
            
            PeakSval = locVal['Scores_'+cond][tidx, vert]
            PeakPval = locVal['Pval_'+cond][tidx, vert]
            if restrictedTW:
                title += '\n $\\bigcirc$Peak latency in %d-%d ms: %d ms' % (TOImin, TOImax, t)
            else:
                title += '\n $\\bigcirc$Peak latency: %d ms' % t
            
            if notSigniLatency:
                title += '.\n   *Actual peak was %d ms, but not in the significant temporal cluster.' % peakLat
            else:
                title += ' (TFCE score = {0}, $\it{1}$ = {2})'.format(round(PeakSval, 4), 'p', PeakPval)
            
            
            # description for the results of within-condition comparison
            txt1 = '< Results >\n'
            if useMinPval:
                txt1 += '*Significance Level ($\\alpha$): the smallest $\it{0}$ value ($\it{1}$, $\it{2}$ = {3})'.format('p', 'i.e.', 'p', locVal['Pval_'+cond].min())
            else:
                txt1 += '*Significance Level: $\\alpha$ < {0}'.format(alphaP)
            
            if identified_timeclus == []:
                txt1 += '\n  ==> No significant temporal clusters were found...'
            else:
                if len(identified_timeclus) == 1:
                    txt1 += '\n  ==> 1 significant temporal cluster was found!'
                else:
                    txt1 += '\n  ==> %d significant temporal clusters were found!' % len(identified_timeclus)
                
                for i, clu in enumerate(identified_timeclus):
                    txt1 += '\n           [Cluster %d] ' % (i+1)
                    if len(clu) == 1:
                        txt1 += 'Significant Time: %d ms (only 1 t.p. was significant)' % clu[0]
                    elif len(clu) >= 2:
                        txt1 += 'Time Interval: %d-%d ms' % (clu[0], clu[-1])
                    
                    PeakSvals = locVal['Scores_'+cond][clu, vert]
                    PeakPvals = locVal['Pval_'+cond][clu, vert]
                    Svalues = np.unique(PeakSvals)
                    Pvalues = np.unique(PeakPvals)
                    
                    if useMinPval:
                        if Svalues.shape[0] > 1:
                            txt1 += '\n                            (TFCE scores = {0}-{1}, $\it{2}$ = {3})'.format(round(PeakSvals.min(), 4), round(PeakSvals.max(), 4), 'p', locVal['Pval_'+cond].min())
                        else:
                            txt1 += '\n                            (TFCE score = {0}, $\it{1}$ = {2})'.format(round(Svalues[0], 4), 'p', locVal['Pval_'+cond].min())
                    else:
                        if Svalues.shape[0] > 1:
                            if Pvalues.shape[0] > 1:
                                txt1 += '\n                            (TFCE scores = {0}-{1}, $\it{2}$ $\leqq$ {3})'.format(round(PeakSvals.min(), 4), round(PeakSvals.max(), 4), 'p', Pvalues.max())
                            else:
                                txt1 += '\n                            (TFCE scores = {0}-{1}, $\it{2}$ = {3})'.format(round(PeakSvals.min(), 4), round(PeakSvals.max(), 4), 'p', Pvalues[0])
                        else:
                            if Pvalues.shape[0] > 1:
                                txt1 += '\n                            (TFCE score = {0}, $\it{1}$ $\leqq$ {2})'.format(round(Svalues[0], 4), 'p', Pvalues.max())
                            else:
                                txt1 += '\n                            (TFCE score = {0}, $\it{1}$ = {2})'.format(round(Svalues[0], 4), 'p', Pvalues[0])
            
            
            # make output
            texts = title  + '\n\n' + txt1
            Vlen = len(identified_timeclus)/2 + 1.2
            
            plt.figure(figsize=(8.2,Vlen))
            plt.text(0, 1, texts, fontsize=11, horizontalalignment='left', verticalalignment='top')
            plt.axis('off')
            plt.subplots_adjust(left=0.04, right=0.9, top=0.98)
            
            # save output
            os.chdir(savedir2)
            if not os.path.exists('./'+labelname):
                os.mkdir('./'+labelname)
            os.chdir('./'+labelname)
            
            fig = plt.gcf()
            fig.savefig('Summary_%s_VertNo%d.png' % (labelname, (idx+1)))
            plt.close(fig)
    
    print('  ==> Finished.\n')


