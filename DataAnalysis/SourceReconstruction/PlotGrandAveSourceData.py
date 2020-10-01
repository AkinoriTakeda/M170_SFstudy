#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot grand-average source-level data with labels

@author: Akinori Takeda
"""

import mne
import numpy as np
import os
locVal = locals()


#--- set data path ---#
filedir = ''
ExpID = 'SF'
useFsaveModel = 'oct6'
usedMethod = 'dSPM'


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
nuseVerts = src[-1]['nuse']


Tmin = -0.1
Tmax = 0.3
timemask = np.where((Tmin <= times) & (times <= Tmax))[0]


#--- load grand-average data ---#
print('\n<< Load Grand-average Datasets (%s Exp, N=%d, method:%s) >>' % (ExpID, SubjN, usedMethod))
srcdir1 = 'SurfSrcEst_dSPM_forEvoked'
if useFsaveModel == 'ico5':
    os.chdir(filedir+'/GrandAverage/Datafiles/forSurfaceSrcEstAnalysis/'+srcdir1+'/%sexp_N%d' % (ExpID, SubjN))
else:
    os.chdir(filedir+'/GrandAverage/Datafiles/forSurfaceSrcEstAnalysis/'+srcdir1+'/%sexp_N%d_fsaverage%s' % (ExpID, SubjN, useFsaveModel.capitalize()))
for cond in conditions2:
    print(' > Loading %s data...' % cond)
    
    # load surface source data
    exec('SurfData_%s = np.load(\'SurfData_%s.npy\')' % (cond, cond))
    locVal['SurfData_'+cond] = locVal['SurfData_'+cond][:,:,timemask]
del cond
print(' => Finished.')


# for making SourceEstimate instance of fsaverage
dirname = filedir+'/Subject1/'+ExpID+'/Datafiles/SourceEstimate/'+srcdir1
if useFsaveModel == 'ico5':
    templateSTC = mne.read_source_estimate(dirname+'/Fsaverage/SrcEst_MeanTC_FearF_BSF_fsaverage')
else:
    templateSTC = mne.read_source_estimate(dirname+'/Fsaverage_%s/SrcEst_MeanTC_FearF_BSF_fsaverage%s' % (useFsaveModel, useFsaveModel.capitalize()))


#- prepare some labels -#
HCPlabels = mne.read_labels_from_annot(MRIsubject, parc='HCPMMP1', hemi='both',
                                   surf_name='inflated', subjects_dir=subjects_dir)

HCPlabellist = []
ROIname = ['VVC','PIT','FFC','V8','V4_']
for roi in ROIname:
    for r in [i for i in HCPlabels if roi in i.name and i.hemi == 'rh']:
        HCPlabellist.append(r)

CombinedLabel = HCPlabellist[0] + HCPlabellist[1] + HCPlabellist[2] + HCPlabellist[3] + HCPlabellist[4]


#- directory setting for saving figures -#
os.chdir(filedir+'/GrandAverage/DataPlots')
if not os.path.exists('./SurfaceSrcEstDataPlots_withLabels'):
    os.mkdir('./SurfaceSrcEstDataPlots_withLabels')
os.chdir('./SurfaceSrcEstDataPlots_withLabels')

if not os.path.exists('./'+srcdir1):
    os.mkdir('./'+srcdir1)
os.chdir('./'+srcdir1)

if not os.path.exists('./%sexp_N%d' % (ExpID, SubjN)):
    os.mkdir('./%sexp_N%d' % (ExpID, SubjN))
os.chdir('./%sexp_N%d' % (ExpID, SubjN))
savedir = os.getcwd()


#%%
TW = [80, 300]
pltTimes = np.arange(TW[0], TW[1]+1, 1)
plotLabel = True


cond = 'House_Equ'

os.chdir(savedir)
if not os.path.exists('./'+cond):
    os.mkdir('./'+cond)
os.chdir('./'+cond)

if plotLabel:
    savedirname = './SrcPlots_%dto%dms_withLabels' % (TW[0], TW[1])
else:
    savedirname = './SrcPlots_%dto%dms' % (TW[0], TW[1])

if not os.path.exists('./'+savedirname):
    os.mkdir('./'+savedirname)
os.chdir('./'+savedirname)


meanTC = locVal['SurfData_%s' % cond].mean(0)
SrcEst = mne.SourceEstimate(meanTC, vertices=templateSTC.vertices, tmin=Tmin, 
                            tstep=templateSTC.tstep, subject=MRIsubject)

if '_Equ' in cond:
    kwargs = dict(colormap='hot', clim=dict(kind='value', lims=[2.4, 2.7, 3]))
else:
    kwargs = dict(colormap='hot', clim=dict(kind='value', lims=[3.8, 4.2, 4.6]))


print('\n<< Plot Source Activity >>')
print('< %s data >' % cond.replace('_','-'))
for t in pltTimes:
    SrcPlot = SrcEst.plot(subject=MRIsubject, surface='inflated', hemi='rh',
                          subjects_dir=subjects_dir, time_unit='ms', views='ven', 
                          initial_time=round(t), time_label=None, colorbar=True, size=(600,800), 
                          **kwargs)
    SrcPlot.show_view(view=dict(azimuth=-270, elevation=-150, roll=0), distance=460)
    
    if plotLabel:
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
    
    # save figure
    SrcPlot.save_single_image('SrcActivity_%dms.png' % t)
    SrcPlot.close()
    del SrcPlot

print('   ==> Done.')

