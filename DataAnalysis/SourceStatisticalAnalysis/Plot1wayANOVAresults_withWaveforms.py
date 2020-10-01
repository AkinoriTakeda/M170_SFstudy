#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot Results of 1-way ANOVA with source waveforms

@author: Akinori Takeda
"""

import mne
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
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


# directory setting of statistical data 
TOImin = 137
TOImax = 265

dirname = 'PostAnalyses_Fsave%s_%dto%dms' % (useFsaveModel.capitalize(), TOImin, TOImax)
statdatadir = filedir+'/GrandAverage/Datafiles/StatisticalData/SourceData/'+ExpID+'/'+srcdir1+'/'+dirname

ANOVAdirname = 'ANOVA_Fsave%s_%dto%dms' % (useFsaveModel.capitalize(), TOImin, TOImax)
ANOVAdatadir = filedir+'/GrandAverage/Datafiles/StatisticalData/SourceData/'+ExpID+'/'+srcdir1+'/'+ANOVAdirname+'/Interaction'
os.chdir(ANOVAdatadir)
Pval_Interaction = np.load('Pvalues.npy')
del ANOVAdirname


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

TOImask = np.where((TOImin <= pltTimes) & (pltTimes <= TOImax))[0]
Tmask1 = np.where(pltTimes <= TOImin)[0]
Tmask2 = np.where(TOImax <= pltTimes)[0]


CategoryList = ['NeutF', 'FearF', 'House']
SFNameList = ['BSF', 'LSF', 'HSF', 'Equ']

Combs = [('Category', 'BSF'), ('Category', 'LSF'), ('Category', 'HSF'), ('Category', 'Equ'),
         ('SF', 'NeutF'), ('SF', 'FearF'), ('SF', 'House')]


colorList = dict(BSF='black',LSF='blue',HSF='red',Equ='green')
lineTypeList = dict(FearF='--', NeutF='-', House=':')


# get max values
MaxVs = []

for cond in conditions2:
    data = locVal['SurfData_'+cond][:,:,timemask]
    if useFsaveModel == 'ico5':
        data = data[:,CombinedLabel.get_vertices_used(),:]
    else:
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
dirlist = ['StatisticalData', 'SourceData', ExpID, srcdir1, dirname]

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
Make individual vertex plot for each effect
'''

alphaP = 0.05

# make dataset
Pv_Interaction = Pval_Interaction < alphaP
coarseXlabel = False


#- Make waveform plots -#
print('\n\n<< Make waveform plots with statistical data >>')

for comb in Combs:
    factor = comb[0]
    cond = comb[1]
    print('< %s at %s >' % (factor, cond))
    
    # load p value data
    os.chdir(statdatadir+'/%s_at%s' % (factor, cond))
    Pval_1wayANOVA = np.load('Pvalues.npy')
    
    # make dataset
    Pv_1wayANOVA = Pval_1wayANOVA < alphaP
    
    # make waveform datasets
    if factor == 'Category':
        levellist = CategoryList
        data1 = locVal['SurfData_NeutF_'+cond][:,:,timemask]
        data2 = locVal['SurfData_FearF_'+cond][:,:,timemask]
        data3 = locVal['SurfData_House_'+cond][:,:,timemask]
    else:
        levellist = SFNameList
        data1 = locVal['SurfData_%s_BSF' % cond][:,:,timemask]
        data2 = locVal['SurfData_%s_LSF' % cond][:,:,timemask]
        data3 = locVal['SurfData_%s_HSF' % cond][:,:,timemask]
        data4 = locVal['SurfData_%s_Equ' % cond][:,:,timemask]
    
    # setting for legend
    for i, condname in enumerate(levellist):
        if factor == 'Category':
            if condname == 'NeutF':
                exec('legendname%d = \'Neutral face\'' % (i+1))
            elif condname == 'FearF':
                exec('legendname%d = \'Fearful face\'' % (i+1))
            else:
                exec('legendname%d = condname' % (i+1))
        elif factor == 'SF':
            if condname == 'Equ':
                exec('legendname%d = \'EQU\'' % (i+1))
            else:
                exec('legendname%d = condname' % (i+1))
    del i, condname
    
    # make directory for saving figures
    os.chdir(currDir)
    if not os.path.exists('./%s_at%s' % (factor, cond)):
        os.mkdir('./%s_at%s' % (factor, cond))
    os.chdir('./%s_at%s' % (factor, cond))
    savedir = os.getcwd()
    
    
    # < Plot individual data > #
    for label in HCPlabellist:
        labelname = label.name.split('_')[1]
        print(' plotting %s data...' % labelname)
        if labelname == 'PIT':
            title1 = 'IOG/OFA (\"PIT\")'
        elif labelname == 'FFC':
            title1 = 'Lateral FG (\"FFC\")'
        elif labelname == 'VVC':
            title1 = 'Medial FG (\"VVC\")'
        else:
            title1 = labelname
        
        if useFsaveModel == 'ico5':
            vertInfo = label.get_vertices_used()
        else:
            labelSTC = templateSTC.in_label(label)
            if useFsaveModel == 'oct6':
                vertInfo = np.array([i for i, n in enumerate(templateSTC.rh_vertno) if n in labelSTC.rh_vertno])
            else:
                vertInfo = labelSTC.rh_vertno
            del labelSTC
        
        for idx, vert in enumerate(vertInfo):
            #- identify temporal clusters -#
            signiTP_Interaction, identified_timeclus_Interaction = DetectTemporalClus(Pv_Interaction[:,vert])
            signiTP_1wayANOVA, identified_timeclus_1wayANOVA = DetectTemporalClus(Pv_1wayANOVA[:,vert])
            
            #- make mean time course -#
            meanTC1 = data1[:,vert,:].mean(0)
            meanTC2 = data2[:,vert,:].mean(0)
            meanTC3 = data3[:,vert,:].mean(0)
            if factor == 'SF':
                meanTC4 = data4[:,vert,:].mean(0)
            
            # calculate 95% confidence interval
            for i in np.arange(len(levellist)):
                cis = mne.stats.bootstrap_confidence_interval(locVal['data%d' % (i+1)][:,vert,:], ci=0.95, n_bootstraps=5000, stat_fun='mean')
                exec('Btm%d = cis[0,:]' % (i+1))
                exec('Up%d = cis[1,:]' % (i+1))
                del cis
            del i
            
            #- setting for shading significant time windows -#
            width = 0.3
            margin = 0.4
            yval1 = -(width+margin)
            yval2 = round(yval1-width*2-margin, 1)
            
            #- plot -#
            plt.figure(figsize=(10.5,6))
            gs = gridspec.GridSpec(1,10)
            ax1 = plt.gcf().add_subplot(gs[:, :8])
            
            if useFsaveModel == 'ico5':
                plt.gcf().suptitle('Vertex No.%d/%d in ' % ((idx+1), label.get_vertices_used().shape[0]) + title1, fontsize=14)
            else:
                plt.gcf().suptitle('Vertex No.%d/%d in ' % ((idx+1), vertInfo.shape[0]) + title1, fontsize=14)
            ax1.vlines(0, (yval2-width-margin), pltMaxV, linewidth=1)
            ax1.set_ylim((yval2-width-margin), 11)
            
            ax1.hlines(0, pltTimes[0], pltTimes[-1], linewidth=1)
            for i, lv in enumerate(levellist):
                if factor == 'Category':
                    for msk in [Tmask1, TOImask, Tmask2]:
                        if (TOImin+TOImax)/2 in pltTimes[msk]:
                            ax1.plot(pltTimes[msk], locVal['meanTC%d' % (i+1)][msk], linewidth=2, color=colorList[cond], label=locVal['legendname%d' % (i+1)], linestyle=lineTypeList[lv])
                        else:
                            ax1.plot(pltTimes[msk], locVal['meanTC%d' % (i+1)][msk], linewidth=2, color=colorList[cond], alpha=0.3, linestyle=lineTypeList[lv])
                else:
                    for msk in [Tmask1, TOImask, Tmask2]:
                        if (TOImin+TOImax)/2 in pltTimes[msk]:
                            ax1.plot(pltTimes[msk], locVal['meanTC%d' % (i+1)][msk], linewidth=2, color=colorList[lv], label=locVal['legendname%d' % (i+1)], linestyle='-')
                            ax1.fill_between(pltTimes[msk], locVal['Btm%d' % (i+1)][msk], locVal['Up%d' % (i+1)][msk], alpha=0.3, color=colorList[lv])
                        else:
                            ax1.plot(pltTimes[msk], locVal['meanTC%d' % (i+1)][msk], linewidth=2, color=colorList[lv], alpha=0.3, linestyle='-')
                            ax1.fill_between(pltTimes[msk], locVal['Btm%d' % (i+1)][msk], locVal['Up%d' % (i+1)][msk], alpha=0.1, color=colorList[lv])
                del msk
            del i
            
            ax1.set_xlim(pltTimes[0], pltTimes[-1])
            ax1.set_yticks(np.arange(0, 11, 2))
            ax1.set_yticklabels([0, 2, 4, 6, 8, 10])
            ax1.spines['right'].set_color('none')
            ax1.spines['top'].set_color('none')
            ax1.spines['left'].set_linewidth(2)
            ax1.spines['bottom'].set_linewidth(2)
            
            if coarseXlabel:
                ax1.set_xticks(np.arange(-100, 301, 100))
                ax1.set_xticklabels([-100, 0, 100, 200, 300])

                ax1.set_xlabel('Time (ms)',fontsize=24, labelpad=10)
                ax1.set_ylabel(ylabel,fontsize=24, labelpad=10)
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)
            else:
                ax1.set_xlabel('Time (ms)',fontsize=22, labelpad=10)
                ax1.set_ylabel(ylabel,fontsize=22, labelpad=10)
                
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)
            
            #- shading significant time windows -#
            color = 'k'
            alpha = 1
            text = '1-way ANOVA'
            
            # 1. post analyses (1-way ANOVA) data
            if signiTP_1wayANOVA.shape[0] >= 1:
                for c in identified_timeclus_1wayANOVA:
                    if len(c) == 1:
                        ax1.vlines(c[0]+TOImin, yval1-width, yval1+width, alpha=alpha, color=color, linewidth=1)
                    elif len(c) >= 2:
                        ax1.fill_between(np.array(c)+TOImin, np.ones(len(c))*(yval1-width), np.ones(len(c))*(yval1+width), 
                                         alpha=alpha, color=color)
                del c
                
            ax1.hlines(yval1, TOImin, TOImax, linewidth=1, color='k')
            ax1.hlines(yval1, 0, TOImin, linewidth=1, color='gray')
            ax1.hlines(yval1, TOImax, pltTimes[-1], linewidth=1, color='gray')
            
            ax1.text(pltTimes[-1]+10, yval1-0.05, text, verticalalignment='center', horizontalalignment='left', fontsize=18)
            
            # 2. 2-way ANOVA interaction data
            if signiTP_Interaction.shape[0] >= 1:
                for c in identified_timeclus_Interaction:
                    if len(c) == 1:
                        ax1.vlines(c[0]+TOImin, yval2-width, yval2+width, alpha=alpha, color=color, linewidth=1)
                    elif len(c) >= 2:
                        ax1.fill_between(np.array(c)+TOImin, np.ones(len(c))*(yval2-width), np.ones(len(c))*(yval2+width), 
                                         alpha=alpha, color=color)
                del c
                
            ax1.hlines(yval2, TOImin, TOImax, linewidth=1, color='k')
            ax1.hlines(yval2, 0, TOImin, linewidth=1, color='gray')
            ax1.hlines(yval2, TOImax, pltTimes[-1], linewidth=1, color='gray')
            
            ax1.text(pltTimes[-1]+10, yval2-0.05, '2-way ANOVA interaction', verticalalignment='center', horizontalalignment='left', fontsize=18)
            
            
            legends, labels = ax1.get_legend_handles_labels()
            if factor == 'Category':
                plt.gcf().legend(legends, labels, loc=(0.69, 0.73), prop={'size': 18}, frameon=True, framealpha=1.)
            else:
                plt.gcf().legend(legends, labels, loc=(0.69, 0.68), prop={'size': 18}, frameon=True, framealpha=1.)
            plt.gcf().legends[0].get_frame().set_linewidth(1)
            plt.gcf().legends[0].get_frame().set_edgecolor('k')
            
            ax1.vlines(TOImin, (yval2-width-margin), pltMaxV, linestyle='--')
            ax1.vlines(TOImax, (yval2-width-margin), pltMaxV, linestyle='--')
            
            if coarseXlabel:
                plt.gcf().subplots_adjust(top=0.92, bottom=0.15, left=0.1, right=0.825)
            else:
                plt.gcf().subplots_adjust(top=0.92, bottom=0.14, left=0.1, right=0.825)
            
            
            # save figure
            os.chdir(savedir)
            if not os.path.exists('./'+labelname):
                os.mkdir('./'+labelname)
            os.chdir('./'+labelname)
            
            fig = plt.gcf()
            fig.savefig('Waveform_%s_VertNo%d.png' % (labelname, (idx+1)))
            plt.close(fig)
        del idx, vert
    print('  --> done.')
    del label
print('==> finished!\n')
del cond


#%%
os.chdir(filedir+'/GrandAverage/DataPlots/StatisticalData/SourceData/'+ExpID+'/'+srcdir1+'/'+dirname)
if not os.path.exists('./StatisticalTests_ResultsSummary'):
    os.mkdir('./StatisticalTests_ResultsSummary')
os.chdir('./StatisticalTests_ResultsSummary')
currDir2 = os.getcwd()

alphaP = 0.05


# make summary output
print('\n<< Make Summary of Results from Statistical Analysis >>')
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
    
    # make directory for saving figures
    os.chdir(currDir2)
    if not os.path.exists('./%s_at%s' % (factor, cond)):
        os.mkdir('./%s_at%s' % (factor, cond))
    os.chdir('./%s_at%s' % (factor, cond))
    savedir2 = os.getcwd()
    
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
        
        if useFsaveModel == 'ico5':
            vertInfo = label.get_vertices_used()
        else:
            labelSTC = templateSTC.in_label(label)
            if useFsaveModel == 'oct6':
                vertInfo = np.array([i for i, n in enumerate(templateSTC.rh_vertno) if n in labelSTC.rh_vertno])
            else:
                vertInfo = labelSTC.rh_vertno
            del labelSTC
        
        for idx, vert in enumerate(vertInfo):
            #- identify temporal clusters -#
            signiTP_1wayANOVA, identified_timeclus_1wayANOVA = DetectTemporalClus(Pv_1wayANOVA[:,vert])
            
            # make summary
            if useFsaveModel == 'ico5':
                title = '<< Summary of Results of 1-way ANOVA: %s at %s >>\n $\\bigcirc$Vertex No.%d/%d in ' % (factor, cond, (idx+1), label.get_vertices_used().shape[0]) + title1
            else:
                title = '<< Summary of Results of 1-way ANOVA: %s at %s >>\n $\\bigcirc$Vertex No.%d/%d in ' % (factor, cond, (idx+1), vertInfo.shape[0]) + title1
            
    
            # description for the results of 1-way ANOVA
            txt1 = '< Results >\n*Significance Level: $\\alpha$ < {0}'.format(alphaP)
            
            if identified_timeclus_1wayANOVA == []:
                txt1 += '\n  ==> No significant temporal clusters were found...'
            else:
                if len(identified_timeclus_1wayANOVA) == 1:
                    txt1 += '\n  ==> 1 significant temporal cluster was found!'
                else:
                    txt1 += '\n  ==> %d significant temporal clusters were found!' % len(identified_timeclus_1wayANOVA)
                
                for i, clu in enumerate(identified_timeclus_1wayANOVA):
                    txt1 += '\n           [Cluster %d] ' % (i+1)
                    if len(clu) == 1:
                        txt1 += 'Significant Time: %d ms (only 1 t.p. was significant)' % (clu[0]+TOImin)
                    elif len(clu) >= 2:
                        txt1 += 'Time Interval: %d-%d ms' % ((clu[0]+TOImin),(clu[-1]+TOImin))
                    
                    PeakScores = scores_1wayANOVA[clu, vert]
                    PeakPvals = Pval_1wayANOVA[clu, vert]
                    Scores = np.unique(PeakScores)
                    Pvalues = np.unique(PeakPvals)
                    
                    if Scores.shape[0] > 1:
                        if Pvalues.shape[0] > 1:
                            txt1 += '\n                            (TFCE scores = {0}-{1}, $\it{2}$ $\leqq$ {3})'.format(round(PeakScores.min(), 2), round(PeakScores.max(), 2), 'p', Pvalues.max())
                        else:
                            txt1 += '\n                            (TFCE scores = {0}-{1}, $\it{2}$ = {3})'.format(round(PeakScores.min(), 2), round(PeakScores.max(), 2), 'p', Pvalues[0])
                    else:
                        if Pvalues.shape[0] > 1:
                            txt1 += '\n                            (TFCE score = {0}, $\it{1}$ $\leqq$ {2})'.format(round(Scores[0], 2), 'p', Pvalues.max())
                        else:
                            txt1 += '\n                            (TFCE score = {0}, $\it{1}$ = {2})'.format(round(Scores[0], 2), 'p', Pvalues[0])
            
            
            # make output
            texts = title  + '\n\n' + txt1
            Vlen = len(identified_timeclus_1wayANOVA)/2 + 1.2
            
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

