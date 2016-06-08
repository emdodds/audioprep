# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 13:17:43 2015

@author: Eric Dodds
"""

import os
import scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import specgram
from librosa.core import load as wavload
from librosa.core import resample
#import pca
import sys
#sys.modules['pca.pca'] = pca
from pca.pca import PCA # Jesse Livezey's PCA class
import pickle

import pca.pca
sys.modules['pca'] = pca.pca

#### Constants
# desired sample rate
srfinal = 16000

# number of time points in each spectrogram
ntimepoints = 25

# frequencies to sample
nfreqs = 256
fmin = 100
fmax = 4000
#bins_per_octave = (-1.+nfreqs)/np.log2(fmax/fmin)
freqs = np.logspace(np.log10(fmin), np.log10(fmax), num=nfreqs)
# interpolation method to go from linearly spaced frequencies to those specified above
interpolation = 'linear'

# FFT window parameters, in seconds
window = .016
overlap = .008

# save spectrogram chunks convolutionally with this stride length
stride = 12

# cutoff for eliminating silent portions
cutoff = 10**(-6) # Carlson used 10**(-6), I've found 10^(-12) gives more reasonable-looking spectrograms

########
    
def interp_logpsd(data, rate, NFFT, noverlap, freqs, window=None, interpolation='linear'):
    """Computes linear-frequency power spectral density, then uses interpolation
    (linear by default) to estimate the psd at the desired frequencies."""
    if window is None:
        window = np.hamming(NFFT)
    stft, linfreqs, times = specgram(data, NFFT, Fs=rate, window=window, noverlap=noverlap)
    ntimes = len(times)
    logpsd = np.log10(np.abs(stft.T)**2)
    interps = [scipy.interpolate.interp1d(linfreqs, logpsd[t,:], kind=interpolation) for t in range(ntimes)]
    interped_logpsd = np.array([interps[t](freqs) for t in range(ntimes)])
    return interped_logpsd, freqs, times


def wav_to_logPSD(infile):
    """Read in .wav file, return the log power spectral density with the frequency axis
determined by the constant freqs."""           
    signal, sr = wavload(infile, sr = None)
    signal = resample(signal, sr, srfinal)
    
    signal = signal/(10*np.var(signal))
    signal = signal - np.mean(signal)        
    
    logflogpsd, logffreqs, times = interp_logpsd(signal, sr, int(window*sr), int(overlap*sr), freqs)

    # remove segments of the PSD with total power below cutoff
    #abovethebar = (np.sum(10**logflogpsd,axis=1) > cutoff)
    return logflogpsd#[abovethebar,:]
           
 

# This is currently the main thing   
# Note to self: the reason I gave up on all this and went back to matlab is that matlab seems to be doing something smoother and more sophisticated
# than linear interpolation from the linear-frequency spectrogram...I don't know what it is and I don't want to waste more time trying to figure it out
def wav_to_PCA(infolder='../speech_corpora/', outfile='../Data/processedspeech12.npy', 
               pcafilename = '../Data/spectropca12.pickle', testfile = 'test12.npy', ncomponents = 200, whiten = True, maxspectros=100000):
    """Do the whole preprocessing scheme at once, saving a pickled PCA object and a .npy array with the data in the reduced
representation. Unreduced spectrograms are not saved. Since these are all stored at once and the covariance matrix for all of them
is computed, this method requires a substantial amount of RAM (something like 8GB for the TIMIT data set)."""
    infilelist = []
    for pth, subd, files in os.walk(infolder):
        for fname in files:
            fstring = os.path.join(pth,fname)
            if fstring.lower().endswith('.wav'):
                infilelist.append(fstring)
   # infilelist = listdir(infolder)
    
    allspectros = [] # don't know length in advance, use list for flexible append. there's probably a faster way
    for infilename in infilelist:
        logflogpsd = wav_to_logPSD(infilename)
        
        nchunks = int((logflogpsd.shape[0] - ntimepoints)*(stride/logflogpsd.shape[0]))
        for chunk in range(nchunks):
            # convert each chunk to a vector and store. throw out any chunk with average power below cutoff
            start = chunk*stride #ntimepoints*chunk
            finish = chunk*stride + ntimepoints#ntimepoints*(chunk+1)
            temp = logflogpsd[start:finish,:]
            if np.mean(10**temp) > cutoff/nfreqs:
                allspectros.append(temp.flatten())
        if len(allspectros) > maxspectros:
            break
    allspectros = np.array(allspectros)
    
    # regularize, normalize spectrograms
    allspectros = np.nan_to_num(allspectros)
    allspectros = np.clip(allspectros,-1000,1000)
#    datamean = np.mean(allspectros, axis=0)
#    allspectros = allspectros - datamean
#    datastd = np.std(allspectros, axis=0)
#    allspectros = allspectros/datastd
    allspectros = allspectros - allspectros.mean(axis=1)[:,np.newaxis]
    #this is just for compatibility with other code
    datamean = 0
    datastd = 1

    # do PCA
    pca = PCA(dim=ncomponents, whiten=whiten)
    print ("Fitting the PCA...")
    pca.fit(allspectros)
    print ("Done. Transforming and saving vectors...")
    reduced = pca.transform(allspectros)
    
    np.save(outfile, reduced)    
    with open(pcafilename, 'wb') as f:
        pickle.dump([pca, (ntimepoints, nfreqs), datamean, datastd], f)    
    print ("Done.")

    # save a file with 9 example spectrograms and their reconstructions
    comparison = allspectros[:9,:]
    recons = pca.inverse_transform(reduced[:9,:])
    np.save(testfile, np.concatenate((comparison, recons),axis=0))
    
    return reduced, pca, (ntimepoints, nfreqs), datamean, datastd

# Functions for testing how the preprocessing worked
    
def view_PCs(pcafile = '../Data/speechpca.pickle', first=0):
    with open(pcafile,'rb') as f:
        pca, origshape, datamean, datastd = pickle.load(f)
    PCs = pca.eVectors
    plt.figure()
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(PCs[i+first,:].reshape(origshape).T, interpolation= 'nearest', cmap='jet', aspect='auto')
        plt.gca().invert_yaxis()
    plt.show()
 
def sample_recons(infile='../Data/processedspeech.npy', pcafile = '../Data/speechpca.pickle'):
    vectors = np.load(infile)
    with open(pcafile,'rb') as f:
        pca, origshape, datamean, datastd = pickle.load(f)
    recons = []
    plt.figure()
    for i in range(9):
        plt.subplot(3,3,i+1)
        recon = pca.inverse_transform(vectors[i,:]).reshape(origshape)
        recons.append(recon)
        plt.imshow(recon.T, interpolation= 'nearest', cmap='jet', aspect='auto')
        plt.gca().invert_yaxis()
    plt.show()
    return np.array(recons)
