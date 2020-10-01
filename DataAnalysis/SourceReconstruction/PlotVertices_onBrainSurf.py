#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot vertices on brain surface

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

srcdir1 = 'SurfSrcEst_dSPM_forEvoked'

MRIsubject = 'fsaverage'
subjects_dir = ''


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
    del r
del roi, ROIname, HCPlabels

CombinedLabel = HCPlabellist[0] + HCPlabellist[1] + HCPlabellist[2] + HCPlabellist[3] + HCPlabellist[4]


#- directory setting for saving figures -#
os.chdir(filedir+'/GrandAverage/DataPlots')
if not os.path.exists('./SurfaceSrcEstDataPlots'):
    os.mkdir('./SurfaceSrcEstDataPlots')
os.chdir('./SurfaceSrcEstDataPlots')

if not os.path.exists('./VerticesLocations'):
    os.mkdir('./VerticesLocations')
os.chdir('./VerticesLocations')
savedir = os.getcwd()


#%%
'''
plot vertices on surface source space model
'''

meanTC = np.ones(templateSTC.shape)
SrcEst = mne.SourceEstimate(meanTC, vertices=templateSTC.vertices, tmin=templateSTC.tmin, 
                            tstep=templateSTC.tstep, subject=templateSTC.subject)
kwargs = dict(colormap='mne', clim=dict(kind='value', pos_lims=[4.5, 5.5, 6.5]))

print('\n< Make source space plot indicating the locations of vertices included in the ROI labels >')
for label in HCPlabellist:
    labelname = label.name.split('_')[1]
    print(' > Making plots of vertices in %s...' % labelname)
    
    if useFsaveModel == 'ico5':
        VertsInLabel = label.get_vertices_used()
    else:
        labelSTC = templateSTC.in_label(label)
        if useFsaveModel == 'oct6':
            VertsInLabel = np.array([i for i, n in enumerate(templateSTC.rh_vertno) if n in labelSTC.rh_vertno])
        else:
            VertsInLabel = labelSTC.rh_vertno
        del labelSTC
    
    for idx, vert in enumerate(VertsInLabel):
        # plot
        SrcPlot = SrcEst.plot(subject=MRIsubject, surface='inflated', hemi='rh',
                              subjects_dir=subjects_dir, time_unit='ms', views='ven', 
                              initial_time=0, time_label=None, colorbar=False, size=(600,800), 
                              background='w', **kwargs)
#        SrcPlot.show_view(view=dict(azimuth=-270, elevation=-125, roll=0), distance=350)
#        SrcPlot.show_view(view=dict(azimuth=-270, elevation=-125, roll=0), distance=250)
        SrcPlot.show_view(view=dict(azimuth=-270, elevation=-150, roll=0), distance=460)
        
        # add labels
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
                color = 'magenta'
            
            SrcPlot.add_label(l, borders=True, color=color)
            del color
        del l
        
        # show location
        if useFsaveModel == 'oct6':
            SrcPlot.add_foci(templateSTC.rh_vertno[vert], coords_as_verts=True, scale_factor=0.22, color='k', hemi='rh')
        else:
            SrcPlot.add_foci([vert], coords_as_verts=True, scale_factor=0.22, color='k', hemi='rh')
        
        # save figure
        os.chdir(savedir)
        if not os.path.exists('./'+labelname):
            os.mkdir('./'+labelname)
        os.chdir('./'+labelname)
        
        SrcPlot.save_single_image('VertLoc_in%s_No%d.png' % (labelname, (idx+1)))
        SrcPlot.close()
        del SrcPlot
    del labelname, idx, vert
del label

print('   ==> Done.')


#%%
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('ticks')

fig = plt.figure(figsize=(10,10))

# [1] plot labels
SrcPlot = SrcEst.plot(subject=MRIsubject, surface='inflated', hemi='rh',
                      subjects_dir=subjects_dir, time_unit='ms', views='ven', 
                      initial_time=0, time_label=None, colorbar=False, background='w', size=(900,1200), 
                      **kwargs)
#SrcPlot.show_view(view=dict(azimuth=-270, elevation=-125, roll=0), distance=400)
SrcPlot.show_view(view=dict(azimuth=-270, elevation=-150, roll=0), distance=460)

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
    
    SrcPlot.add_label(l, borders=False, color=color)
    del color
del l

# screenshot source plot
screenshot = SrcPlot.screenshot()
SrcPlot.close()

# preparation for plotting screenshot
nonwhitePix = (screenshot != 255).any(-1)
nonwhite_row = nonwhitePix.any(1)
nonwhite_col = nonwhitePix.any(0)
cropped_screenshot = screenshot[nonwhite_row][:, nonwhite_col]

plt.subplot(1, 2, 1)
plt.imshow(cropped_screenshot)
plt.axis('off')


# [2] plot vertices
if useFsaveModel == 'ico5':
    VertsInLabel = CombinedLabel.get_vertices_used()
else:
    labelSTC = templateSTC.in_label(CombinedLabel)
    if useFsaveModel == 'oct6':
        VertsInLabel = np.array([i for i, n in enumerate(templateSTC.rh_vertno) if n in labelSTC.rh_vertno])
    else:
        VertsInLabel = labelSTC.rh_vertno
    del labelSTC

SrcPlot = SrcEst.plot(subject=MRIsubject, surface='inflated', hemi='rh',
                      subjects_dir=subjects_dir, time_unit='ms', views='ven', 
                      initial_time=0, time_label=None, colorbar=False, background='w', size=(900,1200), 
                      **kwargs)
#SrcPlot.show_view(view=dict(azimuth=-270, elevation=-125, roll=0), distance=400)
SrcPlot.show_view(view=dict(azimuth=-270, elevation=-150, roll=0), distance=460)

SrcPlot.add_label(CombinedLabel, borders=True, color=CombinedLabel.color)

if useFsaveModel == 'oct6':
    SrcPlot.add_foci(templateSTC.rh_vertno[VertsInLabel], coords_as_verts=True, scale_factor=0.22, color='gold', hemi='rh')
else:
    SrcPlot.add_foci(VertsInLabel, coords_as_verts=True, scale_factor=0.22, color='gold', hemi='rh')

# screenshot source plot
screenshot = SrcPlot.screenshot()
SrcPlot.close()

# preparation for plotting screenshot
nonwhitePix = (screenshot != 255).any(-1)
nonwhite_row = nonwhitePix.any(1)
nonwhite_col = nonwhitePix.any(0)
cropped_screenshot = screenshot[nonwhite_row][:, nonwhite_col]

plt.subplot(1, 2, 2)
plt.imshow(cropped_screenshot)
plt.axis('off')

fig.tight_layout()

# save figure
os.chdir(filedir+'/GrandAverage/DataPlots/SurfaceSrcEstDataPlots')
fig.savefig('SrcSpaceModel.png')
plt.close(fig)

