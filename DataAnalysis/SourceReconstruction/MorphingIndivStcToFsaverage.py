#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Morph individual stc data to the decimated fsaverage stc data

*Prior to running this program, it is necessary to create SrcMorph instance of 
 decimated fsaverage model, which can be obtained by 'MakeSrcMorphInstance.py'

@author: Akinori Takeda
"""

import mne
import os
locVal = locals()


#--- set data path & load MEG data ---#
filedir = ''
SubjID = 'Subject18'
ExpID = 'SF'
useFsaveModel = 'oct6'


# load epoch data to get information
os.chdir(filedir+'/'+SubjID+'/'+ExpID+'/Datafiles/EpochData')
Epochs = mne.read_epochs('ProperEpochData-epo.fif', preload=True)
epochs = Epochs.copy().pick_types(meg=True)
del Epochs

conditions = list(epochs.event_id.keys())
conditions2 = [i for i in conditions if i != 'target']
del epochs

# load SourceMorph instance
morph = mne.read_source_morph(filedir+'/'+SubjID+'/SrcMorph_%stoFsaverage%s-morph.h5' % (SubjID, useFsaveModel.capitalize()))


#- directory manipulation -#
savedir = filedir+'/'+SubjID+'/'+ExpID+'/Datafiles/SourceEstimate/SurfSrcEst_dSPM_forEvoked'
os.chdir(savedir)
if not os.path.exists('./Fsaverage_%s' % useFsaveModel):
    os.mkdir('./Fsaverage_%s' % useFsaveModel)


# do morphing
print('\n<< Morph individual stc data to decimated fsaverage >>')
nRep = 0
for cond in conditions2:
    print('< Processing %s condition data >' % (cond.replace('_','-')))
    
    # load individual stc 
    print('  > loading data...')
    os.chdir(savedir+'/IndivBrain')
    stc = mne.read_source_estimate('SrcEst_MeanTC_%s' % cond, subject=morph.subject_from)
    
    # morph data to fsaverage
    print('  > morphing data...')
    stc_morphed = morph.apply(stc)
    
    # save SourceEstimate instance
    print('  > saving data...')
    os.chdir(savedir+'/Fsaverage_%s' % useFsaveModel)
    stc_morphed.save('SrcEst_MeanTC_%s_fsaverage%s' % (cond, useFsaveModel.capitalize()))
    
    nRep += 1
    print('  => finished (%d/%d done).\n' % (nRep, len(conditions2)))
    del stc, stc_morphed
print('\n   ==> All processings were done!')
del cond, nRep

    
