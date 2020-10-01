#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make SourceMorph instance to morph individiual brain to fsaverage

@author: Akinori Takeda
"""

import mne
import os


#--- set data path & load MEG data ---#
filedir = ''
SubjID = ''
useFsaveModel = 'oct6' # 'ico5 (standard fsaverage model)', 'oct6', 'ico4'

#--- load data files ---#
# get MRI subject name
MRIsubject = ''

# set directory name of MRI data
subjects_dir = ''

# load Surface Source Space data
src = subjects_dir + '/' + MRIsubject + '/bem/%s-oct-6-src.fif' % MRIsubject
SurfSrcSpace = mne.read_source_spaces(src)


#--- Make SourceMorph instance ---#
if 'ico' in useFsaveModel:
    if useFsaveModel == 'ico5':
        morph = mne.compute_source_morph(SurfSrcSpace, subject_from=MRIsubject,
                                         subject_to='fsaverage', subjects_dir=subjects_dir)
    elif useFsaveModel == 'ico4':
        morph = mne.compute_source_morph(SurfSrcSpace, subject_from=MRIsubject,
                                         subject_to='fsaverage', spacing=4, subjects_dir=subjects_dir)
else:
    fsave_file = subjects_dir + '/fsaverage/bem/fsaverage-%s-src.fif' % useFsaveModel
    fsave_src = mne.read_source_spaces(fsave_file)
    fsave_vertices = [s['vertno'] for s in fsave_src]
    morph = mne.compute_source_morph(SurfSrcSpace, subject_from=MRIsubject,
                                     subject_to='fsaverage', spacing=fsave_vertices,
                                     subjects_dir=subjects_dir)

# save SourceMorph instance
os.chdir(filedir+'/'+SubjID)
if useFsaveModel == 'ico5':
    morph.save('SrcMorph_%stoFsaverage' % SubjID, overwrite=True)
else:
    morph.save('SrcMorph_%stoFsaverage%s' % (SubjID, useFsaveModel.capitalize()), overwrite=True)

