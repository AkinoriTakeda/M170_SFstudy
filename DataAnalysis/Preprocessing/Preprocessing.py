#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data preprocessing: Filtering & ICA for concatenated raw data

@author: Akinori Takeda
"""

import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse_raw
import numpy as np
import scipy as sp
import pickle
import shutil
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
sns.set(style="whitegrid")


#--- set data path & get datafiles' name ---#
filedir = ''
SubjID = 'Subject18'
ExpID = 'SF'


#- event info -#
os.chdir(filedir+'/'+SubjID+'/'+ExpID)
with open('ProperStimIdcs.txt', 'rb') as fp:
    PropStimIdcs = pickle.load(fp)

with open('ProperTargetIdcs.txt', 'rb') as fp:
    PropTargetIdcs = pickle.load(fp)


#- MEG data -#
# directory management (if necessary)
os.chdir(filedir+'/'+SubjID+'/'+ExpID+'/Datafiles')
rootdir = os.getcwd()

if not os.path.exists('./MaxfilterWithMaxmoved'):
    os.mkdir('./MaxfilterWithMaxmoved')
    
    os.chdir(rootdir+'/OTPprocessed')
    filelist = [i for i in os.listdir(os.getcwd()) if '_tsss.fif' in i]
    for fname in filelist:
        shutil.move(fname, rootdir+'/MaxfilterWithMaxmoved')


#- MRI data -#
# get file name of -trans.fif 
trans = [ii for ii in os.listdir(filedir+'/'+SubjID) if '-trans.fif' in ii][0]

# get MRI subject name
MRIsubject = ''

# set directory name of MRI data
subjects_dir = ''


#%%
#--- make concatenated raw data ---#
print('\n<< Making concatenated raw data >>')
rawdatalist = []

#- 1. Participant noise data -#
print('< 1. Participant Noise data >')

# load data
datafile = [i for i in os.listdir(rootdir+'/MaxfilterWithMaxmoved') if 'ParticipantNoise' in i][0]
raw = mne.io.read_raw_fif(rootdir+'/MaxfilterWithMaxmoved/'+datafile, preload=True)
sfreq = raw.info['sfreq']
picks = mne.pick_types(raw.info, meg=True, eog=True)

# preprocessing: filtering
print('\n> Filtering data...')
raw.filter(1.0, 200.0, picks=picks)
raw.notch_filter(np.arange(60.0, 301.0, 60), picks=picks)
del picks, datafile

# make raw data of 3 s
picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=True, stim=True, exclude='bads')
events = mne.find_events(raw, stim_channel='STI101', min_duration=(2 / sfreq))

tmin = 0
tmax = 180

epoch = mne.Epochs(raw, events, tmin=tmin, tmax=tmax, proj=True,
                   picks=picks, baseline=None, preload=True)
data = epoch.get_data()[0]

pick_stim = mne.pick_types(raw.info, meg=False, eeg=False, eog=False, stim=True, exclude='bads')
data[pick_stim[0], :] = 0

rawdata = mne.io.RawArray(data, epoch.info)
del epoch, data, pick_stim

print('> Making data for 3 seconds (%d ~ %d sec)' % (tmin, tmax))
del picks, events, tmin, tmax, raw

# append to data container
rawdatalist.append(rawdata)
del rawdata
print(' --> done.')


#- 2. Main experiment data -#
# get names of raw datafiles
datafiles = [i for i in os.listdir(rootdir+'/MaxfilterWithMaxmoved') if 'run' in i]
datafiles.sort()

# manipulate raw data
marginTime = 1 # unit: sec
AccumEveN = 0
print('\n< 2. Main Experiment data (margin: %d sec) >' % marginTime)
for runNo, datafile in enumerate(datafiles):
    # load raw data
    print('[Run %d]' % (runNo+1))
    rawdata = mne.io.read_raw_fif(rootdir+'/MaxfilterWithMaxmoved/'+datafile, preload=True)
    
    # preprocessing: filtering
    print('\n> Filtering data...')
    picks = mne.pick_types(rawdata.info, meg=True, eog=True)
    rawdata.filter(1.0, 200.0, picks=picks)
    rawdata.notch_filter(np.arange(60.0, 301.0, 60), picks=picks)
    del picks
    
    # find the onset times of last trigger in the former part of run
    print('> Cropping data & add to data container...')
    events = mne.find_events(rawdata, stim_channel='STI101', min_duration=(2 / sfreq))
    lasteveIdx = [n for n in np.arange(events.shape[0]-1) if (events[(n+1), 0] - events[n, 0]) >= 4000][0]
    
    firstTrig_onset1 = round(((events[0, 0]-rawdata.first_samp) / sfreq), 4)
    lastTrig_onset1 = round(((events[lasteveIdx, 0]-rawdata.first_samp) / sfreq), 4)
    firstTrig_onset2 = round(((events[(lasteveIdx+1), 0]-rawdata.first_samp) / sfreq), 4)
    lastTrig_onset2 = round(((events[-1, 0]-rawdata.first_samp) / sfreq), 4)
    
    print('\nThe former part: %.3f ~ %.3f sec' % (firstTrig_onset1, lastTrig_onset1))
    print('The latter part: %.3f ~ %.3f sec' % (firstTrig_onset2, lastTrig_onset2))
    
    # add to data container
    data = rawdata.copy().crop(tmin=(firstTrig_onset1-marginTime), tmax=(lastTrig_onset1+marginTime))
    raw = mne.io.RawArray(data.get_data(), data.info)
    rawdatalist.append(raw)
    del data, raw
    
    data = rawdata.copy().crop(tmin=(firstTrig_onset2-marginTime), tmax=(lastTrig_onset2+marginTime))
    raw = mne.io.RawArray(data.get_data(), data.info)
    rawdatalist.append(raw)
    del data, raw
    
    # event data manipulation
    propStimIdx = PropStimIdcs[runNo]
    propTargetIdx = PropTargetIdcs[runNo]
    propImgIdx = np.concatenate((propStimIdx, propTargetIdx))
    propImgIdx.sort()
    
    propImgIdx += AccumEveN
    if runNo == 0:
        PropImgIdx = propImgIdx
    else:
        PropImgIdx = np.concatenate((PropImgIdx, propImgIdx))
    
    AccumEveN += events.shape[0]
    
    del propStimIdx, propTargetIdx, propImgIdx
    del events, lasteveIdx, firstTrig_onset1, lastTrig_onset1, firstTrig_onset2, lastTrig_onset2, rawdata
    print(' --> done.\n')
del runNo, datafile
print(' ==> All processing were done.\n')


# concatenate raw data
rawdata = mne.concatenate_raws(rawdatalist)
rawdata.set_annotations(None)
del rawdatalist


#- visualize and check -trans.fif file -#
mne.viz.plot_alignment(rawdata.info, trans=(filedir+'/'+SubjID+'/'+trans), subject=MRIsubject,
                       subjects_dir=subjects_dir, surfaces=['head','brain'], 
                       meg='helmet', dig=True, eeg=False, ecog=False, coord_frame='mri')


#%%
'''
Preprocessing 2: ICA processing for raw data
'''
#--- Parameter setting for preprocessing ---#
reject = dict(grad=4000e-13, mag=4e-12)

picks = mne.pick_types(rawdata.info, meg=True, eeg=False, stim=True, exclude='bads')
picks_meg = mne.pick_types(rawdata.info, meg=True, eeg=False, stim=False, eog=False, exclude='bads')


#- event data -#
Allevents = mne.find_events(rawdata, stim_channel='STI101', min_duration=(2 / sfreq))
events = Allevents[Allevents[:,-1]!=20]
tmin=-0.75
tmax=0.75
baseline = (-0.1, 0) # tuple


#- parameters for ICA -#
ICAmethod = 'fastica'
n_components = 0.99


#- parameters for ICA projection to surface source space -#
# load Covariance data:
os.chdir(filedir+'/'+SubjID+'/'+ExpID)
covfile = 'NoiseCov_fromEmptyRoom-cov.fif'
NoiseCov = mne.read_cov(covfile)


#- forward solution setting -#
os.chdir(filedir+'/'+SubjID)
fwdfile = 'MixedSourceSpace_%s_3shell_forICA-fwd.fif' % SubjID
fwd = mne.read_forward_solution(fwdfile)

# convert to surface-based source orientation
fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=False)

# preparation for subcortical sources
srcfile = [i for i in os.listdir(os.getcwd()) if '_forICA-oct-6-src.fif' in i][0]
srcspace = mne.read_source_spaces(srcfile)
nvert_insurf = srcspace[0]['nuse'] + srcspace[1]['nuse']
vertices = [srcspace[0]['vertno'], srcspace[1]['vertno']]

locVal = locals()
vollabels = mne.get_volume_labels_from_src(srcspace, MRIsubject, subjects_dir)
nVert = [len(i.vertices) for i in vollabels]
vertNo = np.cumsum(nVert)


# make inverse operators:
InvOperator = make_inverse_operator(rawdata.info, forward=fwd, noise_cov=NoiseCov, 
                                    loose='auto', depth=0.8, fixed=False, limit_depth_chs=False, 
                                    verbose=None) 

snr = 3.0
lambda2 = 1.0 / snr ** 2
method = "MNE"


#%%
#- setting -#
ICA_raw = mne.preprocessing.ICA(n_components=n_components, method=ICAmethod, max_iter=300)

#- run ICA processing -#
ICA_raw.fit(rawdata, picks=picks_meg, reject=reject)
#      :need a few time...


#- save ICA solution -#
os.chdir(rootdir)
if not os.path.exists('./ICAforConcatenatedRaws'):
    os.mkdir('./ICAforConcatenatedRaws')
os.chdir('./ICAforConcatenatedRaws')

savefilename = 'ICAsolution_ConcatenatedRaws-ica.fif'
ICA_raw.save(savefilename)


#%%
#- check components -#
# < 1. plot topomap of each component >
if n_components < 1:
    Ncomps = ICA_raw.n_components_
else:
    Ncomps = n_components
ICA_raw.plot_components(picks=np.arange(0, Ncomps), cmap='jet')

# < 2. plot waveforms of each component >
ICA_raw.plot_sources(rawdata.copy().pick_types(meg=True))


# < 3. calculate evoked responses of each IC >
ICAraw = ICA_raw.get_sources(rawdata.copy().pick_types(meg=True))
picks_icaComp = mne.pick_types(ICAraw.info, include=ICAraw.ch_names)
ICAepochs = mne.Epochs(ICAraw, events, tmin=tmin, tmax=tmax, proj=True,
                       picks=picks_icaComp,baseline=baseline, preload=True)
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
CompsSpatDist = np.dot(ICA_raw.mixing_matrix_.T, ICA_raw.pca_components_[:ICA_raw.n_components_])
CompsSpatDist = CompsSpatDist.T
PseudoRaw = mne.io.RawArray(CompsSpatDist, rawdata.copy().pick_types(meg=True).info)
CompsSrc = apply_inverse_raw(PseudoRaw, InvOperator, lambda2, method=method)

# plot surface source space data
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
#-- remove bad ICs --#
print('\n< Remove bad ICs >')
badIC = []

ICA_raw.apply(rawdata, exclude=badIC)


#-- make Epoch data --#
print('\n< Make Epoch data >')
eveID = dict(FearF_BSF=1, FearF_LSF=2, FearF_HSF=4, FearF_Equ=8,
             NeutF_BSF=3, NeutF_LSF=5, NeutF_HSF=6, NeutF_Equ=9,
             House_BSF=10, House_LSF=12, House_HSF=16, House_Equ=17, target=18)

picks = mne.pick_types(rawdata.info, meg=True, eeg=False, stim=True, eog=True, exclude='bads')

# epoching
EpochData = mne.Epochs(rawdata, Allevents[PropImgIdx,:], event_id=eveID, tmin=tmin, tmax=tmax,
                       baseline=None, picks=picks, preload=True, reject=None)


#-- save Epoch data --#
print('\n< Save Epoch data >')
newfile = 'ICAprocessed_EpochData-epo.fif'
print('Name of new data file: %s' % newfile)

os.chdir(rootdir+'/ICAforConcatenatedRaws')
if newfile in os.listdir(os.getcwd()):
    print('The data file already exists. ---> File will be overwritten...')
else:
    print('Saving data...')
EpochData.save(newfile)

# also save indices of bad ICs
np.save('badICsIdx.npy', np.array(badIC))


#-- also make ICA-processed Participant Noise data --#
print('\n< Make & save ICA-processed Participant Noise data >')
ParticipantNoise = rawdata.copy().crop(tmin=0, tmax=180)

newfile = 'ParticipantNoise_Preprocessed_trans_raw_tsss.fif'
print('Name of new data file: %s' % newfile)

if newfile in os.listdir(os.getcwd()):
    print('The data file already exists. ---> File will be overwritten...')
    ParticipantNoise.save(newfile, overwrite=True)
else:
    print('Saving data...')
    ParticipantNoise.save(newfile)




