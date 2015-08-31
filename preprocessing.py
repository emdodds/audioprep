# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 13:17:43 2015

@author: Eric Dodds
"""

from os import listdir
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import specgram
from librosa.core import constantq
from librosa.core import load as wavload

def logfpsd(data, rate, window, noverlap, fmin, bins_per_octave):
    """Computes ordinary linear-frequency power spectral density, then multiplies by a matrix
    that converts to log-frequency space.
    Returns the log-frequency PSD, the centers of the frequency bins,
    and the time points.
    Adapted from Matlab code by Dan Ellis (Columbia):
    http://www.ee.columbia.edu/ln/rosa/matlab/sgram/logfsgram.m"""
    stft, linfreqs, times = specgram(data, window, Fs=rate, noverlap=noverlap)
    
    # construct matrix for mapping to log-frequency space
    fratio = 2**(1/bins_per_octave) # ratio between adjacent frequencies
    nbins = np.floor(np.log((rate/2)/fmin)/np.log(fratio))
    fftfreqs = (rate/window)*np.arange(window/2+1)
    nfftbins = window/2+1
    logffreqs = fmin*np.exp(np.log(2)*np.arange(nbins)/bins_per_octave)
    logfbws = logffreqs*(fratio-1)
    logfbws = np.maximum(logfbws, rate/window)
    bandoverlapconstant = 0.5475 # controls adjacent band overlap. set by hand by Dan Ellis
    freqdiff = (np.repeat(logffreqs[:,np.newaxis],nfftbins,axis=1) - np.repeat(fftfreqs[np.newaxis,:],nbins,axis=0))
    freqdiff = freqdiff / np.repeat(bandoverlapconstant*logfbws[:,np.newaxis],nfftbins,axis=1)
    mapping = np.exp(-0.5*freqdiff**2)
    rowEs = np.sqrt(2*np.sum(mapping**2,axis=1))
    mapping = mapping/np.repeat(rowEs[:,np.newaxis],nfftbins,axis=1)
    
    # perform mapping
    logfpsd = np.sqrt(np.dot(mapping,(np.abs(stft)**2)))
    
    return logfpsd, logffreqs, times

def CQTPSD(signal, sr, fmin, fmax, bins_per_octave, res=.1):
    """Plots and returns the analog of a power spectal density from a constant Q transform."""
    nbins = int(np.log2(fmax/fmin))*bins_per_octave
    lhop = (2**0)*2**(nbins/bins_per_octave)
    # pseudo constant Q transform. shape is (nbins, t)
    CQT = constantq.pseudo_cqt(signal, sr=sr, hop_length=lhop, fmin=fmin, n_bins=nbins,
                               bins_per_octave=bins_per_octave, resolution=res)
    logpsdish = np.log10(np.abs(CQT)**2)
    plt.figure(3)
    plt.imshow(logpsdish, interpolation='nearest', aspect='auto', cmap='jet',origin='lower')
    plt.title("logPSD-like thing from CQT")
    return logpsdish

def wav_to_spectro(infolder='../speech_corpora/TIMIT/', outfolder='../Spectrograms/'):
    """Takes all the .wav files in the given folder and makes files containing
    spectrograms according to the parameters in Carlson et al."""
    # number of time points in each spectrogram
    ntimepoints = 25
    
    # frequencies to sample
    nfreqs = 256
    fmin = 100
    fmax = 4000
    bins_per_octave = (-1.+nfreqs)/np.log2(fmax/fmin)
    freqs = np.logspace(np.log10(fmin), np.log10(fmax), num=nfreqs)
    
    # FFT window parameters, in seconds
    window = .016
    overlap = .008
    
    infilelist = listdir(infolder)
    for infilename in infilelist:
    #filename = "../speech_corpora/TIMIT/SA1_FADG0_DR4_TEST.WAV"
        infile = infolder+infilename
        if infile[-4:].lower() != '.wav':
            continue
        signal, sr = wavload(infile, sr = None)
        
        logfpsd_, logffreqs, times = logfpsd(signal,sr,int(window*sr*2),int(overlap*sr*2),fmin,bins_per_octave)
        try:
            assert np.allclose(freqs, logffreqs[:nfreqs])
        except AssertionError:
            print (logffreqs[:nfreqs])
            print (freqs)
            raise ValueError
        logflogpsd = np.log(logfpsd_[:,:nfreqs])
        
        nchunks = int(len(times)/ntimepoints)
        for chunk in range(nchunks):
            start = ntimepoints*chunk
            finish = ntimepoints*(chunk+1)
            outfile = outfolder + infilename[:-4] + str(chunk)
            np.save(outfile, logflogpsd[start:finish,:])



#print ("Sampling rate: " + str(sr))

#spect, freqs, times = specgram(signal, NFFT = int(window*sr*2), Fs = sr, 
#                                     #window = mlab.window_hanning(lwindow*rate), 
#                                     noverlap = int(overlap*sr*2))
#logpsd = np.log(np.abs(spect)**2)                                     
#                                     
#plt.imshow(logpsd,interpolation='nearest', aspect='auto', cmap='jet')
#plt.title("Actually the logPSD")
#plt.gca().invert_yaxis()


#plt.imshow(logflogpsd, interpolation='nearest', aspect='auto', cmap='jet')
#plt.gca().invert_yaxis()
#plt.title("Log-frequency logPSD")
#plt.show()

