#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot Results of 2-way ANOVA with source waveforms

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


# make subject list
SubjDir = [i for i in os.listdir(filedir) if 'Subject' in i and 'Test' not in i and '.' not in i]
rmvSubjList = []

SubjList = ['Subject%d' % (n+1) for n in np.arange(len(SubjDir)) if 'Subject%d' % (n+1) not in rmvSubjList and os.path.exists(filedir+'/Subject%d/SF' % (n+1))]
SubjN = len(SubjList)
del SubjDir


#- setting some parameters -#
srcdir1 = 'SurfSrcEst_dSPM_forEvoked'

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

dirname = 'ANOVA_Fsave%s_%dto%dms' % (useFsaveModel.capitalize(), TOImin, TOImax)


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
statdatadir = filedir+'/GrandAverage/Datafiles/StatisticalData/SourceData/'+ExpID+'/'+srcdir1+'/'+dirname
effects = ['MainEffect_Category','MainEffect_SF','Interaction']
for cond in effects:
    os.chdir(statdatadir+'/'+cond)
    exec('Scores_'+cond+' = np.load(\'TFCEscores.npy\')')
    exec('Pval_'+cond+' = np.load(\'Pvalues.npy\')')
del cond



#%%
'''
Preparation for plotting
'''
#- setting Exp & condition which will be plotted -#
#pltType = 'acrossSF'
pltType = 'acrossCategory'
 
'''
[memo] pltType
 'acrossSF' ---> plot all SF conditions data within selected category
 'acrossCategory' ---> plot selected SF condition data within all category
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


CategoryList = ['FearF', 'NeutF', 'House']
SFNameList = ['BSF', 'LSF', 'HSF', 'Equ']

colorList = dict(BSF='black',LSF='blue',HSF='red',Equ='green')
lineTypeList = dict(FearF='-', NeutF='--', House=':')

# dataset setting
if pltType == 'acrossSF':
    condType1 = CategoryList
    condType2 = SFNameList
elif pltType == 'acrossCategory':
    condType1 = SFNameList
    condType2 = CategoryList


#- value range setting -#
# get max value
MaxVs = []

for cond in conditions2:
    data = locVal['SurfData_'+cond][:,:,timemask]
    
    if useFsaveModel == 'ico5':
        data = data[:,CombinedLabel.get_vertices_used(),:]
    elif useFsaveModel != 'ico5':
        labelSTC = templateSTC.in_label(CombinedLabel)
        if useFsaveModel == 'oct6':
            vertIdx = np.array([i for i, n in enumerate(templateSTC.rh_vertno) if n in labelSTC.rh_vertno])
            data = data[:,vertIdx,:]
        else:
            data = data[:,labelSTC.rh_vertno,:]
        del labelSTC
    meanTC = data.mean(0)
    
    # calculate 95% confidence interval
    cis = mne.stats.bootstrap_confidence_interval(data, ci=0.95, n_bootstraps=5000, stat_fun='mean')
    UP = cis[1,:]
    
    # add to container
    MaxVs.append(UP.max())
    
    del data, meanTC, cis, UP
del cond

pltMaxV = round(np.max(MaxVs))
pltMinV = 0
ylabel = 'Amplitude (${a.u.}$)'
del MaxVs


#- directory manipulation -#
# make directory for data plot if not exist
dirlist = ['StatisticalData', 'SourceData', ExpID, srcdir1, dirname, 'VertexWaveforms_'+pltType]

os.chdir(filedir+'/GrandAverage/DataPlots')
for dname in dirlist:
    if not os.path.exists('./'+dname):
        os.mkdir('./'+dname)
    os.chdir('./'+dname)
currDir = os.getcwd()


#%%
'''
Make individual vertex plot
'''
useMinPval = False
if not useMinPval:
    alphaP = 0.05
    includeItself = False


# make datasets
for cond in effects:
    if useMinPval:
        exec('Pv_'+cond+' = locVal[\'Pval_\'+\''+cond+'\'] == locVal[\'Pval_\'+\''+cond+'\'].min()')
    else:
        if includeItself:
            exec('Pv_'+cond+' = locVal[\'Pval_\'+\''+cond+'\'] <= alphaP')
        else:
            exec('Pv_'+cond+' = locVal[\'Pval_\'+\''+cond+'\'] < alphaP')
    exec('Pv_'+cond+'_liberal = locVal[\'Pval_\'+\''+cond+'\'] < 0.05')
    
    #- detect time points & sensors with siginificant statistical values -#
    exec('timeIdx_'+cond+' = np.unique(np.where(Pv_'+cond+')[0])')
    exec('spaceIdx_'+cond+' = [np.where(Pv_'+cond+'[t,:])[0] for t in timeIdx_'+cond+']')
    
    exec('timeIdx_'+cond+'_lib = np.unique(np.where(Pv_'+cond+'_liberal)[0])')
    exec('spaceIdx_'+cond+'_lib = [np.where(Pv_'+cond+'_liberal[t,:])[0] for t in timeIdx_'+cond+'_lib]')
del cond


print('\n<< Make individual vertex plots >>')
for I in condType1:
    os.chdir(currDir)
    if not os.path.exists('./'+I):
        os.mkdir('./'+I)
    os.chdir('./'+I)
    print('> plotting %s data...' % I)
    
    if pltType == 'acrossSF':
        if I == 'FearF':
            condname = 'Fearful face'
        elif I == 'NeutF':
            condname = 'Neutral face'
        else:
            condname = I
    else:
        if I == 'Equ':
            condname = 'Equiluminant'
        else:
            condname = I
    
    #- make waveform data for individual vertex plots -#
    for n, ii in enumerate(condType2):
        if pltType == 'acrossSF':
            cond = I+'_'+ii
        else:
            cond = ii+'_'+I
        
        exec('data_'+ii+' = locVal[\'SurfData_\'+\''+cond+'\'][:,:,timemask]')
        del cond
    del n, ii

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
        
        if useFsaveModel == 'ico5':
            if '_withoutSrcConnectivity' in dirname:
                vertInfo = np.array([i for i, n in enumerate(CombinedLabel.get_vertices_used()) if n in label.get_vertices_used()])
            else:
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
            for cond in effects:
                exec('signiTP_'+cond+' = np.where(locVal[\'Pv_\'+\''+cond+'\'][:,vert])[0]')
                exec('identified_timeclus_'+cond+' = []')
                if locVal['signiTP_'+cond].shape[0] == 1:
                    locVal['identified_timeclus_'+cond].append(locVal['signiTP_'+cond].tolist())
                elif locVal['signiTP_'+cond].shape[0] >= 2:
                    cluster_tp = []
                    for i, tp in enumerate(locVal['signiTP_'+cond][:-1]):
                        cluster_tp.append(tp)
                        
                        if (locVal['signiTP_'+cond][(i+1)] - tp) > 1:
                            locVal['identified_timeclus_'+cond].append(cluster_tp)
                            del cluster_tp
                            cluster_tp = []
                        
                        if i == (len(locVal['signiTP_'+cond][:-1])-1):
                            locVal['identified_timeclus_'+cond].append(cluster_tp)
                    del i, tp, cluster_tp
                    
                    if locVal['identified_timeclus_'+cond][-1]==[] or locVal['signiTP_'+cond][-1] == (locVal['identified_timeclus_'+cond][-1][-1]+1):
                        locVal['identified_timeclus_'+cond][-1].append(locVal['signiTP_'+cond][-1])
                
                # also identify temporal clusters in more liberal data
                exec('signiTP2_'+cond+' = np.where(locVal[\'Pv_\'+\''+cond+'\'+\'_liberal\'][:,vert])[0]')
                exec('identified_timeclus2_'+cond+' = []')
                if locVal['signiTP2_'+cond].shape[0] == 1:
                    locVal['identified_timeclus2_'+cond].append(locVal['signiTP2_'+cond].tolist())
                elif locVal['signiTP2_'+cond].shape[0] >= 2:
                    cluster_tp2 = []
                    for i, tp in enumerate(locVal['signiTP2_'+cond][:-1]):
                        cluster_tp2.append(tp)
                        
                        if (locVal['signiTP2_'+cond][(i+1)] - tp) > 1:
                            locVal['identified_timeclus2_'+cond].append(cluster_tp2)
                            del cluster_tp2
                            cluster_tp2 = []
                        
                        if i == (len(locVal['signiTP2_'+cond][:-1])-1):
                            locVal['identified_timeclus2_'+cond].append(cluster_tp2)
                    del i, tp, cluster_tp2
                    
                    if locVal['identified_timeclus2_'+cond][-1]==[] or locVal['signiTP2_'+cond][-1] == (locVal['identified_timeclus2_'+cond][-1][-1]+1):
                        locVal['identified_timeclus2_'+cond][-1].append(locVal['signiTP2_'+cond][-1])
            del cond
            
            #- make mean time course -#
            for ii in condType2:
                exec('meanTC_'+ii+' = locVal[\'data_\'+\''+ii+'\'][:,vert,:].mean(0)')
                
                # calculate 95% confidence interval
                cis = mne.stats.bootstrap_confidence_interval(locVal['data_'+ii][:,vert,:], ci=0.95, n_bootstraps=5000, stat_fun='mean')
                exec('Btm_'+ii+' = cis[0,:]')
                exec('Up_'+ii+' = cis[1,:]')
                del cis
            del ii
            
            
            #- setting for shading significant time windows -#
            width = 0.3
            margin = 0.4
            
            yval1 = -(width+margin)
            yval2 = round(yval1-width*2-margin, 1)
            yval3 = round(yval2-width*2-margin, 1)
            
            #- plot -#
            plt.figure(figsize=(8.5,5.5))
            gs = gridspec.GridSpec(1,10)
            ax1 = plt.gcf().add_subplot(gs[:, :8])
            
            if useFsaveModel == 'ico5':
                plt.gcf().suptitle('Vertex No.%d/%d in ' % ((idx+1), label.get_vertices_used().shape[0]) + title1 + condname, fontsize=14)
            else:
                plt.gcf().suptitle('Vertex No.%d/%d in ' % ((idx+1), vertInfo.shape[0]) + title1 + condname, fontsize=14)
            ax1.vlines(0, (yval3-width-margin), pltMaxV, linewidth=1)
            ax1.set_ylim((yval3-width-margin), pltMaxV)
            
            ax1.hlines(0, pltTimes[0], pltTimes[-1], linewidth=1)
            for ii in condType2:
                if pltType == 'acrossSF':
                    if ii == 'Equ':
                        sf = 'Equiluminant'
                    else:
                        sf = ii
                    
                    for msk in [Tmask1, TOImask, Tmask2]:
                        if (TOImin+TOImax)/2 in pltTimes[msk]:
                            ax1.plot(pltTimes[msk], locVal['meanTC_'+ii][msk], linewidth=2, color=colorList[ii], label=sf)
                            ax1.fill_between(pltTimes[msk], locVal['Btm_'+ii][msk], locVal['Up_'+ii][msk], alpha=0.3, color=colorList[ii])
                        else:
                            ax1.plot(pltTimes[msk], locVal['meanTC_'+ii][msk], linewidth=2, color=colorList[ii], alpha=0.3)
                            ax1.fill_between(pltTimes[msk], locVal['Btm_'+ii][msk], locVal['Up_'+ii][msk], alpha=0.1, color=colorList[ii])
                else:
                    if ii == 'FearF':
                        legendname = 'Fearful face'
                    elif ii == 'NeutF':
                        legendname = 'Neutral face'
                    else:
                        legendname = ii
                    
                    for msk in [Tmask1, TOImask, Tmask2]:
                        if (TOImin+TOImax)/2 in pltTimes[msk]:
                            ax1.plot(pltTimes[msk], locVal['meanTC_'+ii][msk], linewidth=2, color=colorList[I], label=legendname, linestyle=lineTypeList[ii])
                        else:
                            ax1.plot(pltTimes[msk], locVal['meanTC_'+ii][msk], linewidth=2, color=colorList[I], alpha=0.3, linestyle=lineTypeList[ii])
            ax1.set_xlim(pltTimes[0], pltTimes[-1])
            ax1.set_yticks(np.arange(0, pltMaxV+1, 2.5))
            ax1.set_yticklabels([0, 2.5, 5, 7.5, 10, 12.5, 15])
            ax1.spines['right'].set_color('none')
            ax1.spines['top'].set_color('none')
            ax1.set_xlabel('Time (ms)',fontsize=12)
            ax1.set_ylabel(ylabel,fontsize=12)
            
            #- shading significant time windows -#
            # using liberal alpha (0.05)
            for cond in effects:
                if cond == 'MainEffect_Category':
                    yval = yval1
                    text = '$\it{Category}$'
                elif cond == 'MainEffect_SF':
                    yval = yval2
                    text = '$\it{SF}$'
                else:
                    yval = yval3
                    text = 'Interaction'
                color = 'gray'
                alpha = 0.8
                    
                if locVal['signiTP2_'+cond].shape[0] >= 1:
                    for c in locVal['identified_timeclus2_'+cond]:
                        if len(c) == 1:
                            ax1.vlines(c[0]+TOImin, yval-width, yval+width, alpha=alpha, color=color, linewidth=1)
                        elif len(c) >= 2:
                            ax1.fill_between(np.array(c)+TOImin, np.ones(len(c))*(yval-width), np.ones(len(c))*(yval+width), 
                                             alpha=alpha, color=color)
                    del c
                    
                    # plot horizontal line
                    for i, c in enumerate(locVal['identified_timeclus2_'+cond]):
                        if i == 0:
                            if c[0] != 0:
                                ax1.hlines(yval, TOImin, c[0]+TOImin, linewidth=1, color='k')
                            prevEnd = c[-1]+TOImin
                        else:
                            ax1.hlines(yval, prevEnd, c[0]+TOImin, linewidth=1)
                            prevEnd = c[-1]+TOImin
                        
                        if i == (len(locVal['identified_timeclus2_'+cond])-1) and (c[-1]+TOImax) != pltTimes[-1]:
                            ax1.hlines(yval, c[-1]+TOImin, TOImax, linewidth=1)
                    del i, c, prevEnd
                else:
                    ax1.hlines(yval, TOImin, TOImax, linewidth=1, color='k')
                ax1.hlines(yval, 0, TOImin, linewidth=1, color='gray')
                ax1.hlines(yval, TOImax, pltTimes[-1], linewidth=1, color='gray')
                
                ax1.text(pltTimes[-1]+10, yval-0.1, text, verticalalignment='center', horizontalalignment='left', fontsize=11)
                
                del yval, color, alpha, text
            del cond
            
            # using more rigolous alpha
            for cond in effects:
                if cond == 'MainEffect_Category':
                    yval = yval1
                elif cond == 'MainEffect_SF':
                    yval = yval2
                else:
                    yval = yval3
                color = 'k'
                alpha = 1
                
                if locVal['signiTP_'+cond].shape[0] >= 1:
                    for c in locVal['identified_timeclus_'+cond]:
                        if len(c) == 1:
                            ax1.vlines(c[0]+TOImin, yval-width, yval+width, alpha=alpha, color=color, linewidth=1)
                        elif len(c) >= 2:
                            ax1.fill_between(np.array(c)+TOImin, np.ones(len(c))*(yval-width), np.ones(len(c))*(yval+width), 
                                             alpha=alpha, color=color)
                            
                    del c
                del yval, color, alpha
            del cond
            
            legends, labels = ax1.get_legend_handles_labels()
            if pltType == 'acrossSF':
                plt.gcf().legend(legends, labels, loc=(0.775, 0.75), prop={'size': 12}, frameon=True, framealpha=1.)
            else:
                plt.gcf().legend(legends, labels, loc=(0.775, 0.75), prop={'size': 12}, frameon=True, framealpha=1.)
            
            ax1.vlines(TOImin, (yval3-width-margin), pltMaxV, linestyle='--')
            ax1.vlines(TOImax, (yval3-width-margin), pltMaxV, linestyle='--')
            
            plt.gcf().subplots_adjust(top=0.92, bottom=0.12)
            
            # save figure
            os.chdir(currDir+'/'+I)
            if not os.path.exists('./'+labelname):
                os.mkdir('./'+labelname)
            os.chdir('./'+labelname)
            
            fig = plt.gcf()
            fig.savefig('Waveform_%s_VertNo%d.png' % (labelname, (idx+1)))
            plt.close(fig)


#%%
os.chdir(filedir+'/GrandAverage/DataPlots/StatisticalData/SourceData/'+ExpID+'/'+srcdir1+'/'+dirname)
if not os.path.exists('./StatisticalTests_ResultsSummary'):
    os.mkdir('./StatisticalTests_ResultsSummary')
os.chdir('./StatisticalTests_ResultsSummary')
currDir2 = os.getcwd()


# make datasets 
useMinPval = False
if not useMinPval:
    alphaP = 0.05
    includeItself = False


for cond in effects:
    if useMinPval:
        exec('Pv_'+cond+' = locVal[\'Pval_\'+\''+cond+'\'] == locVal[\'Pval_\'+\''+cond+'\'].min()')
    else:
        if includeItself:
            exec('Pv_'+cond+' = locVal[\'Pval_\'+\''+cond+'\'] <= alphaP')
        else:
            exec('Pv_'+cond+' = locVal[\'Pval_\'+\''+cond+'\'] < alphaP')
del cond


# make summary output
print('\n<< Make Summary of Results from Statistical Analysis >>')
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
        for cond in effects:
            exec('signiTP_'+cond+' = np.where(locVal[\'Pv_\'+\''+cond+'\'][:,vert])[0]')
            exec('identified_timeclus_'+cond+' = []')
            if locVal['signiTP_'+cond].shape[0] == 1:
                locVal['identified_timeclus_'+cond].append(locVal['signiTP_'+cond].tolist())
            elif locVal['signiTP_'+cond].shape[0] >= 2:
                cluster_tp = []
                for i, tp in enumerate(locVal['signiTP_'+cond][:-1]):
                    cluster_tp.append(tp)
                    
                    if (locVal['signiTP_'+cond][(i+1)] - tp) > 1:
                        locVal['identified_timeclus_'+cond].append(cluster_tp)
                        del cluster_tp
                        cluster_tp = []
                    
                    if i == (len(locVal['signiTP_'+cond][:-1])-1):
                        locVal['identified_timeclus_'+cond].append(cluster_tp)
                del i, tp, cluster_tp
                
                if locVal['identified_timeclus_'+cond][-1]==[] or locVal['signiTP_'+cond][-1] == (locVal['identified_timeclus_'+cond][-1][-1]+1):
                    locVal['identified_timeclus_'+cond][-1].append(locVal['signiTP_'+cond][-1])
        del cond
        
        
        # make summary
        if useFsaveModel == 'ico5':
            title = '<< Summary of Results from Repeated Measures 2-way ANOVA >>\n $\\bigcirc$Vertex No.%d/%d in ' % ((idx+1), label.get_vertices_used().shape[0]) + title1
        else:
            title = '<< Summary of Results from Repeated Measures 2-way ANOVA >>\n $\\bigcirc$Vertex No.%d/%d in ' % ((idx+1), vertInfo.shape[0]) + title1
        
        # [1] Main effect of Category
        txt1 = '\n\n< Main effects of $\it{Category}$ >\n'
        if useMinPval:
            txt1 += '*Significance Level ($\\alpha$): the smallest $\it{0}$ value ($\it{1}$, $\it{2}$ = {3})'.format('p', 'i.e.', 'p', locVal['Pval_MainEffect_Category'].min())
        else:
            if includeItself:
                txt1 += '*Significance Level: $\\alpha$ $\leqq$ {0}'.format(alphaP)
            else:
                txt1 += '*Significance Level: $\\alpha$ < {0}'.format(alphaP)
        
        if locVal['identified_timeclus_MainEffect_Category'] == []:
            txt1 += '\n  ==> No significant temporal clusters were found...'
        else:
            if len(locVal['identified_timeclus_MainEffect_Category']) == 1:
                txt1 += '\n  ==> 1 significant temporal cluster was found!'
            else:
                txt1 += '\n  ==> %d significant temporal clusters were found!' % len(locVal['identified_timeclus_MainEffect_Category'])
            
            for i, clu in enumerate(locVal['identified_timeclus_MainEffect_Category']):
                txt1 += '\n           [Cluster %d] ' % (i+1)
                if len(clu) == 1:
                    txt1 += 'Significant Time: %d ms (only 1 t.p. was significant)' % (clu[0]+TOImin)
                elif len(clu) >= 2:
                    txt1 += 'Time Interval: %d-%d ms' % ((clu[0]+TOImin),(clu[-1]+TOImin))
                
                PeakSvals = locVal['Scores_MainEffect_Category'][clu, vert]
                PeakPvals = locVal['Pval_MainEffect_Category'][clu, vert]
                Svalues = np.unique(PeakSvals)
                Pvalues = np.unique(PeakPvals)
                
                if useMinPval:
                    if Svalues.shape[0] > 1:
                        txt1 += '\n                            (TFCE scores = {0}-{1}, $\it{2}$ = {3})'.format(round(PeakSvals.min(), 2), round(PeakSvals.max(), 2), 'p', locVal['Pval_MainEffect_Category'].min())
                    else:
                        txt1 += '\n                            (TFCE score = {0}, $\it{1}$ = {2})'.format(round(Svalues[0], 2), 'p', locVal['Pval_MainEffect_Category'].min())
                else:
                    if Svalues.shape[0] > 1:
                        if Pvalues.shape[0] > 1:
                            txt1 += '\n                            (TFCE scores = {0}-{1}, $\it{2}$ $\leqq$ {3})'.format(round(PeakSvals.min(), 2), round(PeakSvals.max(), 2), 'p', Pvalues.max())
                        else:
                            txt1 += '\n                            (TFCE scores = {0}-{1}, $\it{2}$ = {3})'.format(round(PeakSvals.min(), 2), round(PeakSvals.max(), 2), 'p', Pvalues[0])
                    else:
                        if Pvalues.shape[0] > 1:
                            txt1 += '\n                            (TFCE score = {0}, $\it{1}$ $\leqq$ {2})'.format(round(Svalues[0], 2), 'p', Pvalues.max())
                        else:
                            txt1 += '\n                            (TFCE score = {0}, $\it{1}$ = {2})'.format(round(Svalues[0], 2), 'p', Pvalues[0])
        
        # [2] Main effect of Category
        txt2 = '< Main effects of $\it{SF}$ >\n'
        if useMinPval:
            txt2 += '*Significance Level ($\\alpha$): the smallest $\it{0}$ value ($\it{1}$, $\it{2}$ = {3})'.format('p', 'i.e.', 'p', locVal['Pval_MainEffect_SF'].min())
        else:
            if includeItself:
                txt2 += '*Significance Level: $\\alpha$ $\leqq$ {0}'.format(alphaP)
            else:
                txt2 += '*Significance Level: $\\alpha$ < {0}'.format(alphaP)
        
        if locVal['identified_timeclus_MainEffect_SF'] == []:
            txt2 += '\n  ==> No significant temporal clusters were found...'
        else:
            if len(locVal['identified_timeclus_MainEffect_SF']) == 1:
                txt2 += '\n  ==> 1 significant temporal cluster was found!'
            else:
                txt2 += '\n  ==> %d significant temporal clusters were found!' % len(locVal['identified_timeclus_MainEffect_SF'])
            
            for i, clu in enumerate(locVal['identified_timeclus_MainEffect_SF']):
                txt2 += '\n           [Cluster %d] ' % (i+1)
                if len(clu) == 1:
                    txt2 += 'Significant Time: %d ms (only 1 t.p. was significant)' % (clu[0]+TOImin)
                elif len(clu) >= 2:
                    txt2 += 'Time Interval: %d-%d ms' % ((clu[0]+TOImin),(clu[-1]+TOImin))
                
                PeakSvals = locVal['Scores_MainEffect_SF'][clu, vert]
                PeakPvals = locVal['Pval_MainEffect_SF'][clu, vert]
                Svalues = np.unique(PeakSvals)
                Pvalues = np.unique(PeakPvals)
                
                if useMinPval:
                    if Svalues.shape[0] > 1:
                        txt2 += '\n                            (TFCE scores = {0}-{1}, $\it{2}$ = {3})'.format(round(PeakSvals.min(), 2), round(PeakSvals.max(), 2), 'p', locVal['Pval_MainEffect_SF'].min())
                    else:
                        txt2 += '\n                            (TFCE score = {0}, $\it{1}$ = {2})'.format(round(Svalues[0], 2), 'p', locVal['Pval_MainEffect_SF'].min())
                else:
                    if Svalues.shape[0] > 1:
                        if Pvalues.shape[0] > 1:
                            txt2 += '\n                            (TFCE scores = {0}-{1}, $\it{2}$ $\leqq$ {3})'.format(round(PeakSvals.min(), 2), round(PeakSvals.max(), 2), 'p', Pvalues.max())
                        else:
                            txt2 += '\n                            (TFCE scores = {0}-{1}, $\it{2}$ = {3})'.format(round(PeakSvals.min(), 2), round(PeakSvals.max(), 2), 'p', Pvalues[0])
                    else:
                        if Pvalues.shape[0] > 1:
                            txt2 += '\n                            (TFCE score = {0}, $\it{1}$ $\leqq$ {2})'.format(round(Svalues[0], 2), 'p', Pvalues.max())
                        else:
                            txt2 += '\n                            (TFCE score = {0}, $\it{1}$ = {2})'.format(round(Svalues[0], 2), 'p', Pvalues[0])
            
        # [3] Category x SF interaction
        txt3 = '< $\it{Category}$ $\\times$ $\it{SF}$ interaction >\n'
        if useMinPval:
            txt3 += '*Significance Level ($\\alpha$): the smallest $\it{0}$ value ($\it{1}$, $\it{2}$ = {3})'.format('p', 'i.e.', 'p', locVal['Pval_Interaction'].min())
        else:
            if includeItself:
                txt3 += '*Significance Level: $\\alpha$ $\leqq$ {0}'.format(alphaP)
            else:
                txt3 += '*Significance Level: $\\alpha$ < {0}'.format(alphaP)
        
        if locVal['identified_timeclus_Interaction'] == []:
            txt3 += '\n  ==> No significant temporal clusters were found...'
        else:
            if len(locVal['identified_timeclus_Interaction']) == 1:
                txt3 += '\n  ==> 1 significant temporal cluster was found!'
            else:
                txt3 += '\n  ==> %d significant temporal clusters were found!' % len(locVal['identified_timeclus_Interaction'])
            
            for i, clu in enumerate(locVal['identified_timeclus_Interaction']):
                txt3 += '\n           [Cluster %d] ' % (i+1)
                if len(clu) == 1:
                    txt3 += 'Significant Time: %d ms (only 1 t.p. was significant)' % (clu[0]+TOImin)
                elif len(clu) >= 2:
                    txt3 += 'Time Interval: %d-%d ms' % ((clu[0]+TOImin),(clu[-1]+TOImin))
                
                PeakSvals = locVal['Scores_Interaction'][clu, vert]
                PeakPvals = locVal['Pval_Interaction'][clu, vert]
                Svalues = np.unique(PeakSvals)
                Pvalues = np.unique(PeakPvals)
                
                if useMinPval:
                    if Svalues.shape[0] > 1:
                        txt3 += '\n                            (TFCE scores = {0}-{1}, $\it{2}$ = {3})'.format(round(PeakSvals.min(), 2), round(PeakSvals.max(), 2), 'p', locVal['Pval_Interaction'].min())
                    else:
                        txt3 += '\n                            (TFCE score = {0}, $\it{1}$ = {2})'.format(round(Svalues[0], 2), 'p', locVal['Pval_Interaction'].min())
                else:
                    if Svalues.shape[0] > 1:
                        if Pvalues.shape[0] > 1:
                            txt3 += '\n                            (TFCE scores = {0}-{1}, $\it{2}$ $\leqq$ {3})'.format(round(PeakSvals.min(), 2), round(PeakSvals.max(), 2), 'p', Pvalues.max())
                        else:
                            txt3 += '\n                            (TFCE scores = {0}-{1}, $\it{2}$ = {3})'.format(round(PeakSvals.min(), 2), round(PeakSvals.max(), 2), 'p', Pvalues[0])
                    else:
                        if Pvalues.shape[0] > 1:
                            txt3 += '\n                            (TFCE score = {0}, $\it{1}$ $\leqq$ {2})'.format(round(Svalues[0], 2), 'p', Pvalues.max())
                        else:
                            txt3 += '\n                            (TFCE score = {0}, $\it{1}$ = {2})'.format(round(Svalues[0], 2), 'p', Pvalues[0])
        
        
        # make output
        texts = title + txt1 + '\n\n' + txt2 + '\n\n' + txt3
        vlen = len(locVal['identified_timeclus_MainEffect_Category']) + len(locVal['identified_timeclus_MainEffect_SF']) + len(locVal['identified_timeclus_Interaction'])
        Vlen = vlen/2 + 3
        
        plt.figure(figsize=(6,Vlen))
        plt.text(0, 1, texts, fontsize=11, horizontalalignment='left', verticalalignment='top')
        plt.axis('off')
        plt.subplots_adjust(left=0.08, right=0.9, top=0.98)
        
        # save output
        os.chdir(currDir2)
        if not os.path.exists('./'+labelname):
            os.mkdir('./'+labelname)
        os.chdir('./'+labelname)
        
        fig = plt.gcf()
        fig.savefig('Summary_%s_VertNo%d.png' % (labelname, (idx+1)))
        plt.close(fig)

print('\n  ==> Finished.')

