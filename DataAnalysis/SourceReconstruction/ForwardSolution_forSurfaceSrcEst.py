#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Computing forward solution for source-level analysis (source estimation)
using surface source space

@author: Akinori Takeda
"""

import mne
import os


#--- set data path & get datafiles' name ---#
filedir = ''
SubjID = 'Subject18'
TargetExpID = 'SF'

#- MRI data -#
# get file name of -trans.fif 
trans = [ii for ii in os.listdir(filedir+'/'+SubjID) if '-trans.fif' in ii][0]
TransFile = filedir+'/'+SubjID+'/'+trans

# get MRI subject name
MRIsubject = ''

# set directory name of MRI data
subjects_dir = ''


#--- setting up source space ---#
src = subjects_dir + '/' + MRIsubject + '/bem/%s-oct-6-src.fif' % MRIsubject
src_space = mne.read_source_spaces(src)


#--- load MEG data information ---#
datafile = filedir+'/'+SubjID+'/'+TargetExpID+'/Datafiles/OTPprocessed/SubjectNoise_OTPprocessed_raw.fif'
rawinfo = mne.io.read_info(datafile)


#--- calculate forward solution ---#
bemsolfile = subjects_dir + '/' + MRIsubject + '/bem/%s-5120-5120-5120-bem-sol.fif' % MRIsubject

# calculate free orientation forward solution
fwd = mne.make_forward_solution(rawinfo, trans=TransFile, src=src_space, 
                                bem=bemsolfile, meg=True, eeg=False)


#--- save data file ---#
os.chdir(filedir+'/'+SubjID)
newfile = 'SurfaceSourceSpace_%s_3shell_forSrcEst-fwd.fif' % SubjID
newsrc = 'SurfaceSourceSpace_%s_forSrcEst-oct-6-src.fif' % SubjID

mne.write_forward_solution(newfile, fwd, overwrite=True)
mne.write_source_spaces(newsrc, src_space, overwrite=True)
