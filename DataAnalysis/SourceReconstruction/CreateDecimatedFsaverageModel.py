#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create decimated fsaverage model

@author: Akinori Takeda
"""

import mne
import os


# set MRI subject name
MRIsubject = 'fsaverage'

# set directory name of MRI data
subjects_dir = ''


# create decimated fsaverage data
spacing = 'oct6'
src_space = mne.setup_source_space(MRIsubject, spacing=spacing, add_dist=True,
                                   subjects_dir=subjects_dir, n_jobs=2)


# save created source space data
os.chdir(subjects_dir+'/'+MRIsubject+'/bem')
newsrc = 'fsaverage-%s-src.fif' % spacing
mne.write_source_spaces(newsrc, src_space, overwrite=True)


#%%
#- check data -#
import numpy as np
from mayavi import mlab
from surfer import Brain

# load data
src = subjects_dir + '/' + MRIsubject + '/bem/fsaverage-%s-src.fif'  % spacing
src_space = mne.read_source_spaces(src)

src2 = subjects_dir + '/' + MRIsubject + '/bem/fsaverage-5-src.fif'
src_space2 = mne.read_source_spaces(src2)


# plot vertices
os.chdir('')

brain = Brain(MRIsubject, 'rh', 'inflated', subjects_dir=subjects_dir, size=(600,800))
surf = brain.geo['rh']
vertidx = np.where(src_space[1]['inuse'])[0]
mlab.points3d(surf.x[vertidx], surf.y[vertidx], surf.z[vertidx], color=(1,1,0), scale_factor=1.5)
brain.show_view(view=dict(azimuth=-270, elevation=-125, roll=0), distance=350)

brain.save_single_image('Fsaverage_%s.png' % spacing)
brain.close()
del brain, surf, vertidx



brain2 = Brain(MRIsubject, 'rh', 'inflated', subjects_dir=subjects_dir, size=(600,800))
surf2 = brain2.geo['rh']
vertidx2 = np.where(src_space2[1]['inuse'])[0]
mlab.points3d(surf2.x[vertidx2], surf2.y[vertidx2], surf2.z[vertidx2], color=(1,1,0), scale_factor=1.5)
brain2.show_view(view=dict(azimuth=-270, elevation=-125, roll=0), distance=350)

brain2.save_single_image('Fsaverage_ico5.png')
brain2.close()
del brain2, surf2, vertidx2

