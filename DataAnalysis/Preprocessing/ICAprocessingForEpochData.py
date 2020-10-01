#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data preprocessing: ICA processing of epoch data

@author: Akinori Takeda
"""

import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse_raw
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import seaborn as sns
import os
sns.set(style="whitegrid")


#--- set data path & get datafiles' name ---#
#- 1. MEG data -#
filedir = ''
SubjID = 'Subject18'
ExpID = 'SF'

# get MEG data
os.chdir(filedir+'/'+SubjID+'/'+ExpID+'/Datafiles')
rootdir = os.getcwd()

os.chdir(rootdir+'/ICAforConcatenatedRaws')
EpochData = mne.read_epochs('ICAprocessed_EpochData-epo.fif', preload=True)
print('\n')
print(EpochData)


#- 2. MRI data -#
# get MRI subject name
MRIsubject = ''

# set directory name of MRI data
subjects_dir = ''


# directory handling
os.chdir(rootdir)
if not os.path.exists('./EpochData'):
    os.mkdir('./EpochData')
    

#%%
'''
preparation for epoch ICA
'''

picks_meg = mne.pick_types(EpochData.info, meg=True, eeg=False, stim=False, eog=False, exclude='bads')

#- parameters for ICA -#
ICAmethod = 'fastica'
n_components = 0.99


#- parameters for ICA projection to surface source space -#
# load Covariance data
os.chdir(filedir+'/'+SubjID+'/'+ExpID)
covfile = 'NoiseCov_fromEmptyRoom-cov.fif'
NoiseCov = mne.read_cov(covfile)


#- forward solution setting -#
os.chdir(filedir+'/'+SubjID)
fwdfile = 'MixedSourceSpace_%s_3shell_forICA-fwd.fif' % SubjID
fwd = mne.read_forward_solution(fwdfile)

# convert to surface-based source orientation
fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=False)


srcfile = [i for i in os.listdir(os.getcwd()) if '_forICA-oct-6-src.fif' in i][0]
srcspace = mne.read_source_spaces(srcfile)
nvert_insurf = srcspace[0]['nuse'] + srcspace[1]['nuse']
vertices = [srcspace[0]['vertno'], srcspace[1]['vertno']]

import matplotlib.gridspec as gridspec
locVal = locals()
vollabels = mne.get_volume_labels_from_src(srcspace, MRIsubject, subjects_dir)
nVert = [len(i.vertices) for i in vollabels]
vertNo = np.cumsum(nVert)


# make inverse operators:
InvOperator = make_inverse_operator(EpochData.info, forward=fwd, noise_cov=NoiseCov, 
                                    loose='auto', depth=0.8, fixed=False, limit_depth_chs=False, 
                                    verbose=None) 

snr = 3.0
lambda2 = 1.0 / snr ** 2
method = "MNE" 


#%%
'''
run ICA
'''
#- setting -#
ICAoperator = mne.preprocessing.ICA(n_components=n_components, method=ICAmethod, max_iter=300)

#- run ICA processing -#
ICAoperator.fit(EpochData, picks=picks_meg, reject=None)
#      :need a few seconds...


#- save ICA solution -#
os.chdir(rootdir+'/EpochData')

savefilename = 'ICAsolution_EpochData-ica.fif'
ICAoperator.save(savefilename)


#%%
#- check components -#
# < 1. plot topomap of each component >
if n_components < 1:
    Ncomps = ICAoperator.n_components_
else:
    Ncomps = n_components
ICAoperator.plot_components(picks=np.arange(0, Ncomps), cmap='jet')


# < 2. plot waveforms of each component >
ICAoperator.plot_sources(EpochData.copy().pick_types(meg=True))


# < 3. calculate evoked responses of each IC >
ICAepochs = ICAoperator.get_sources(EpochData.copy().pick_types(meg=True))
picks_icaComp = mne.pick_types(ICAepochs.info, include=ICAepochs.ch_names)
times = ICAepochs.times*1000

ICA_evoked = ICAepochs.average(picks=picks_icaComp)
sd = ICAepochs.get_data().std(axis=0)
vmin = round(np.min(ICA_evoked.data-sd),1)
vmax = round(np.max(ICA_evoked.data+sd),1)

plt.figure(figsize=(25,13))
plt.suptitle('Evoked response waves of ICs', fontsize=15)
for i in np.arange(0,ICA_evoked.data.shape[0]):
    if Ncomps%10 != 0:
        plt.subplot(int((Ncomps-(Ncomps%10))/10+1),10,(i+1))
    else:
        plt.subplot(int(Ncomps/10),10,(i+1))
    plt.plot(times, ICA_evoked.data[i])
    plt.fill_between(times, ICA_evoked.data[i]-sd[i], ICA_evoked.data[i]+sd[i], alpha=0.3)
    plt.title("IC %d" % i)
    plt.xlim([times[0], times[-1]])
    plt.ylim([vmin*1.1, vmax*1.1])
    plt.xlabel('time (ms)')
    plt.ylabel('AU')
plt.tight_layout()
plt.subplots_adjust(top=0.93)


# < 4. calculate FFT of each IC epoch >
sfreq = EpochData.info['sfreq']
CompsEpochs = ICAepochs.get_data()

# set parameters
N = CompsEpochs.shape[-1]
dt = 1 / sfreq
freqList = sp.fftpack.fftfreq(N, dt)
freqList = freqList[np.arange(0,(N/2), dtype='int')] # use only positive frequency
CompsEpochFFTs = np.zeros([CompsEpochs.shape[0], CompsEpochs.shape[1], CompsEpochs.shape[2]])

# run FFT
for trial in np.arange(0,CompsEpochs.shape[0]):
    Comps = CompsEpochs[trial]
    Compsfft = np.zeros([CompsEpochs.shape[1], CompsEpochs.shape[2]])
    for ic in np.arange(0,CompsEpochs.shape[1]):
        CompsFFT = sp.fftpack.fft(Comps[ic]) / (N / 2)
        CompsFFT[0] = CompsFFT[0] / 2
        CompsFFT = np.abs(CompsFFT)
        Compsfft[ic] = CompsFFT
    CompsEpochFFTs[trial] = Compsfft
    del CompsFFT, Compsfft, Comps

CompsEpochFFTs_ave = CompsEpochFFTs.mean(axis=0)
CompsEpochFFTs_sd = CompsEpochFFTs.std(axis=0)

# leave components in the frequency band of interest
fmax = 100
#fmax = 60
freqList = freqList[np.where(freqList<=fmax)]
CompsEpochFFTs_ave = CompsEpochFFTs_ave[:, 0:freqList.shape[0]]
CompsEpochFFTs_sd = CompsEpochFFTs_sd[:, :freqList.shape[0]]

# plot the spectra
plt.figure(figsize=(23,13))
plt.suptitle('FFT spectra of ICs\' epochs (1~%d Hz)' % fmax, fontsize=15)
for i in np.arange(0,CompsEpochFFTs_ave.shape[0]):
    if Ncomps%10 != 0:
        plt.subplot(int((Ncomps-(Ncomps%10))/10+1),10,(i+1))
    else:
        plt.subplot(int(Ncomps/10),10,(i+1))
    plt.plot(freqList, CompsEpochFFTs_ave[i])
    plt.fill_between(freqList, CompsEpochFFTs_ave[i]-CompsEpochFFTs_sd[i], CompsEpochFFTs_ave[i]+CompsEpochFFTs_sd[i], alpha=0.3)
    plt.title("IC %d" % i)
    plt.xlim([0,fmax])
    plt.ylim([0, round(CompsEpochFFTs_ave.max(),2)])
    plt.xlabel('frequency (Hz)')
    plt.ylabel('Amplitude (a.u.)')
plt.tight_layout()
plt.subplots_adjust(top=0.93)


# < 5. ICA components' projection to surface source space >
CompsSpatDist = np.dot(ICAoperator.mixing_matrix_.T, ICAoperator.pca_components_[:ICAoperator.n_components_])
CompsSpatDist = CompsSpatDist.T
PseudoRaw = mne.io.RawArray(CompsSpatDist, EpochData.copy().pick_types(meg=True).info)

CompsSrc = apply_inverse_raw(PseudoRaw, InvOperator, lambda2, method=method)

surfdata = CompsSrc.data[:nvert_insurf, :]
stcdata = mne.SourceEstimate(surfdata, vertices=vertices, tmin=CompsSrc.tmin, 
                             tstep=1/sfreq, subject=MRIsubject)
brainView = stcdata.plot(subject=MRIsubject, hemi='both', time_label='IC #%03d',
                         time_viewer=True, subjects_dir=subjects_dir, time_unit='ms')
brainView.toggle_toolbars(show=True)

# also plot subcortical information
Subcort=CompsSrc.data[nvert_insurf:,:]
LAmyg = Subcort[:vertNo[0], :]
LHipp = Subcort[vertNo[0]:vertNo[1], :]
RAmyg = Subcort[vertNo[1]:vertNo[2], :]
RHipp = Subcort[vertNo[2]:, :]
MaxV = np.max([LAmyg.mean(0).max(), LHipp.mean(0).max(), RAmyg.mean(0).max(), RHipp.mean(0).max()])+20
MinV = 0

sns.set(style="whitegrid")
fig = plt.figure(figsize=(14,10))
fig.suptitle('ICA components\' projection to subcortical sources', fontsize=25)

gs = gridspec.GridSpec(19,22)
ax1 = fig.add_subplot(gs[:8, :10])
ax2 = fig.add_subplot(gs[11:, :10])
ax3 = fig.add_subplot(gs[:8, 12:])
ax4 = fig.add_subplot(gs[11:, 12:])

datanames = ['LAmyg', 'LHipp', 'RAmyg', 'RHipp']
for i, n in enumerate(datanames):
    pltdata = locVal[n]
    exec('ax%d.plot(np.arange(Ncomps), pltdata.mean(0), \'o-\')' % (i+1))
    exec('ax%d.fill_between(np.arange(Ncomps), pltdata.mean(0)-pltdata.std(0), pltdata.mean(0)+pltdata.std(0), alpha=0.3)' % (i+1))
    exec('ax%d.set_xlim(0, Ncomps-1)' % (i+1))
    exec('ax%d.set_ylim(MinV, MaxV)' % (i+1))
    exec('ax%d.set_title(vollabels[i].name+\' (nVert: %d)\')' % ((i+1), nVert[i]))


#%%
# remove bad ICs #
badIC = []
ICAoperator.apply(EpochData, exclude=badIC)


'''
Save preprocessed data
'''
newfile = 'ICAprocessed_EpochData2-epo.fif'
print('Name of new data file: %s' % newfile)

os.chdir(rootdir+'/EpochData')
if newfile in os.listdir(os.getcwd()):
    print('The data file already exists. ---> File will be overwritten...')
else:
    print('Saving data...')
EpochData.save(newfile)

# also save indices of bad ICs
np.save('badICsIdx.npy', np.array(badIC))


