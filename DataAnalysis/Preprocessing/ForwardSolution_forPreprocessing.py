#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Computing forward solution for ICA processing in preprocessing
using mixed source space (surface + subcortical)

@author: Akinori Takeda
"""

import mne
import os


#--- set data path & get datafiles' name ---#
#- 1. MEG data -#
filedir = ''
SubjID = 'Subject18'

os.chdir(filedir+'/'+SubjID)
TargetExpID = 'SF'


#- 2. MRI data -#
# get file name of -trans.fif 
trans = [ii for ii in os.listdir(filedir+'/'+SubjID) if '-trans.fif' in ii][0]

# get MRI subject name
MRIsubject = ''

# set directory name of MRI data
subjects_dir = ''

# for forward solution
src = subjects_dir + '/' + MRIsubject + '/bem/%s-oct-6-src.fif' % MRIsubject
bem_sol = subjects_dir + '/' + MRIsubject + '/bem/%s-5120-5120-5120-bem-sol.fif' % MRIsubject  # 3-shell model
bem_model = subjects_dir + '/' + MRIsubject + '/bem/%s-5120-5120-5120-bem.fif' % MRIsubject  # 3-shell model
vol_src = subjects_dir + '/' + MRIsubject + '/mri/aseg.mgz'


#--- setting up source space ---#
src_space = mne.read_source_spaces(src)

VolLabel = ['Left-Amygdala', 'Left-Hippocampus', 'Right-Amygdala', 'Right-Hippocampus']
volsrc_space = mne.setup_volume_source_space(MRIsubject, mri=vol_src, volume_label=VolLabel,
                                             bem=bem_model, subjects_dir=subjects_dir, verbose=True)
src_space += volsrc_space


#--- load data files ---#
datafile = filedir+'/'+SubjID+'/'+TargetExpID+'/Datafiles/OTPprocessed/ParticipantNoise_OTPprocessed_raw.fif'
rawdata = mne.io.read_raw_fif(datafile, preload=True)


#--- forward solution ---#
# calculate free orientation forward solution
fwd = mne.make_forward_solution(rawdata.info, trans=(filedir+'/'+SubjID+'/'+trans),
                                src=src_space, bem=bem_sol, meg=True, eeg=False)


#--- save data file ---#
newfile = 'MixedSourceSpace_%s_3shell_forICA-fwd.fif' % SubjID
newsrc = 'MixedSourceSpace_%s_forICA%s' % (SubjID, src[-14:])

mne.write_forward_solution(newfile, fwd, overwrite=True)
mne.write_source_spaces(newsrc, src_space, overwrite=True)
