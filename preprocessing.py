# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 13:17:43 2015

@author: Eric Dodds
"""

from os import listdir
import scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import specgram
from librosa.core import constantq
from librosa.core import load as wavload
from pca.pca import PCA # Jesse Livezey's PCA class
import pickle

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
    #fftfreqs = (rate/window)*np.arange(window/2+1)
    nfftbins = window/2+1
    logffreqs = fmin*np.exp(np.log(2)*np.arange(nbins)/bins_per_octave)
    logfbws = logffreqs*(fratio-1)
    logfbws = np.maximum(logfbws, rate/window)
    bandoverlapconstant = 0.5475 # controls adjacent band overlap. set by hand by Dan Ellis
    freqdiff = (np.repeat(logffreqs[:,np.newaxis],nfftbins,axis=1) - np.repeat(linfreqs[np.newaxis,:],nbins,axis=0))
    freqdiff = freqdiff / np.repeat(bandoverlapconstant*logfbws[:,np.newaxis],nfftbins,axis=1)
    mapping = np.exp(-0.5*freqdiff**2)
    rowEs = np.sqrt(2*np.sum(mapping**2,axis=1))
    mapping = mapping/np.repeat(rowEs[:,np.newaxis],nfftbins,axis=1)
    
    # perform mapping
    logfpsd = np.sqrt(np.dot(mapping,(np.abs(stft)**2)))
    
    return logfpsd.T, logffreqs, times
    
def interp_logpsd(data, rate, window, noverlap, freqs, interpolation='linear'):
    """Computes linear-frequency power spectral density, then uses interpolation
    (linear by default) to estimate the psd at the desired frequencies."""
    stft, linfreqs, times = specgram(data, window, Fs=rate, noverlap=noverlap)
    ntimes = len(times)
    logpsd = np.log10(np.abs(stft.T)**2)
    interps = [scipy.interpolate.interp1d(linfreqs, logpsd[t,:], kind=interpolation) for t in range(ntimes)]
    interped_logpsd = np.array([interps[t](freqs) for t in range(ntimes)])
    return interped_logpsd, freqs, times

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

def wav_to_spectro(infolder='../speech_corpora/TIMIT/', outfolder='../Spectrograms/', interpolation = 'linear', max_number=None):
    """Takes all the .wav files in the given folder and makes files containing
    spectrograms according to the parameters in Carlson et al.
    Stops after max_number spectrograms if specified."""
    # number of time points in each spectrogram
    ntimepoints = 25
    
    # frequencies to sample
    nfreqs = 256
    fmin = 100
    fmax = 4000
    #bins_per_octave = (-1.+nfreqs)/np.log2(fmax/fmin)
    freqs = np.logspace(np.log10(fmin), np.log10(fmax), num=nfreqs)
    
    # FFT window parameters, in seconds
    window = .016
    overlap = .008
    
    infilelist = listdir(infolder)
    count = 0
    for infilename in infilelist:
    #filename = "../speech_corpora/TIMIT/SA1_FADG0_DR4_TEST.WAV"
        infile = infolder+infilename
        if infile[-4:].lower() != '.wav':
            # ignore anything that isn't a .wav file
            continue
        signal, sr = wavload(infile, sr = None)
        
        #logfpsd_, logffreqs, times = logfpsd(signal,sr,int(window*sr*2),int(overlap*sr*2),fmin,bins_per_octave)
        logflogpsd, logffreqs, times = interp_logpsd(signal, sr, int(window*sr*2), int(overlap*sr*2), freqs, interpolation)
        
        try:
            assert np.allclose(freqs, logffreqs[:nfreqs])
        except AssertionError:
            print (logffreqs[:nfreqs])
            print (freqs)
            raise AssertionError
        #logflogpsd = np.log10(logfpsd_[:,:nfreqs])
        
        nchunks = int(logflogpsd.shape[0]/ntimepoints)
        for chunk in range(nchunks):
            start = ntimepoints*chunk
            finish = ntimepoints*(chunk+1)
            outfile = outfolder + infilename[:-4] + str(chunk)
            np.save(outfile, logflogpsd[start:finish,:])
            count = count + 1
        if max_number is not None and count > max_number:
            return
            
def view_sample_spectros(infolder='../Spectrograms/'):
    infilelist = listdir(infolder)
    infilelist = [f for f in infilelist if f.lower().endswith('.npy')]
    plt.figure()
    plt.clf()
    array = None
    for i in range(9):
        infile = infolder + infilelist.pop()
        array = np.load(infile).T
        plt.subplot(3,3,i+1)
        plt.imshow(array, interpolation= 'nearest', cmap='jet', aspect='auto')
        plt.gca().invert_yaxis()
    plt.show()
    return array
    
def view_old_spectros():
    spectros = scipy.io.loadmat("../speechdata.mat")["speechdata0"][:,:,:9]
    plt.figure()
    plt.clf()
    for i in range(9):
        array = spectros[:,:,i]
        plt.subplot(3,3,i+1)
        plt.imshow(array, interpolation= 'nearest', cmap='jet', aspect='auto')
        plt.gca().invert_yaxis()
    plt.show()

def pca_reduce(infolder='../Spectrograms/', num_to_fit=30000, outfile='../Data/processedspeech',
               ncomponents=200, pcafilename = '../Data/speechpca.pickle', whiten = True):
    """Read in (at most) num_to_fit of the arrays in infolder, fit a PCA object to them,
    save the pca object. Save the transformed, dimensionality-reduced representations
    of all arrays in infolder to outfile.
    Set num_to_fit to None to fit to all arrays. The only reason not to use all
    of them is insufficient RAM."""
    
    # build the array for fitting the PCA object
    infilelist = listdir(infolder)
    infilelist = [f for f in infilelist if f.lower().endswith('.npy')]
    nfiles = len(infilelist)
    origshape = np.load(infolder+infilelist[0]).shape
    veclength = np.prod(origshape)
    np.random.shuffle(infilelist)
    if num_to_fit is None:
        num_to_fit = nfiles
    trainingdata = np.zeros((num_to_fit, veclength))

    for count in range(num_to_fit):
        infile = infolder + infilelist.pop()
        array = np.load(infile)
        trainingdata[count] = array.flatten()
    
    trainingdata = trainingdata[:count+1,:]
    trainingdata = np.nan_to_num(trainingdata)
    trainingdata = np.clip(trainingdata,-1000,1000) # just in case something funny happened with logs
    
    # center and normalize the training data
    trainingdatamean = np.mean(trainingdata, axis=0)
    trainingdata = trainingdata - trainingdatamean
    trdata_std = np.std(trainingdata, axis=0)
    trainingdata = trainingdata/trdata_std
    
    # create and fit PCA object
    pca = PCA(dim=ncomponents, whiten=whiten)
    print ("Fitting the PCA...")
    pca.fit(trainingdata)
    print ("Done.")
    # transform all the input arrays
    print ("Transforming...")
    allvectors = np.zeros((nfiles,ncomponents))
    allvectors[:count+1,:] = pca.transform(trainingdata)
    del trainingdata # clear up memory
    for infilename in infilelist:
        count = count+1
        infile = infolder + infilename
        vector = np.load(infile).flatten()
        allvectors[count,:] = pca.transform(vector)
    
    np.save(outfile, allvectors)    
    
    with open(pcafilename, 'wb') as f:
        pickle.dump([pca, origshape, trainingdatamean, trdata_std], f)
    
    print ("Done")

    # plot the first 9 principal components
    PCs = pca.eVectors
    plt.figure()
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(PCs[i,:].reshape(origshape), interpolation= 'nearest', cmap='jet', aspect='auto')
        plt.gca().invert_yaxis()
    plt.show()
    
    return allvectors, pca, origshape, trainingdatamean, trdata_std
    
    
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
    plt.figure()
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(pca.inverse_transform(vectors[i,:]).reshape(origshape).T, interpolation= 'nearest', cmap='jet', aspect='auto')
        plt.gca().invert_yaxis()
    plt.show()
    
# TODO: the above function doesn't work, don't know why. Also need to run pca_reduce again since it had
    # a really dumb error in it before
    