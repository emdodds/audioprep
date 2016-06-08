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
from librosa.core import resample
from pca.pca import PCA # Jesse Livezey's PCA class
import pickle

#### Constants ####
# number of time points in each spectrogram
ntimepoints = 25

# sample rate; all waveforms resampled to this rate
samplerate = 16000

# frequencies to sample
nfreqs = 256
fmin = 100
fmax = 4000
bins_per_octave = (-1.+nfreqs)/np.log2(fmax/fmin) #used for logfpsd
freqs = np.logspace(np.log10(fmin), np.log10(fmax), num=nfreqs)
# interpolation method to go from linearly spaced frequencies to those specified above
interpolation = 'linear'

# FFT window parameters, in seconds
window = .016
overlap = .008

# cutoff for eliminating silent portions
cutoff = 10**(-12) # Carlson used 10**(-6)

########

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
    stft, linfreqs, times = specgram(data, window, Fs=rate, noverlap=noverlap, window = np.hamming(window))
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

def wav_to_logPSD(infile):
    """Read in .wav file, return the log power spectral density with the frequency axis
determined by the constant freqs."""           
    signal, nativerate = wavload(infile, sr = None)
    signal = resample(signal, nativerate, samplerate)
    # Carlson's code seems to do this. I don't know why and I think it muddies the spectrograms but maybe I misunderstand something
    #signal = signal/(10*np.var(signal))
    #signal = signal - np.mean(signal)        
    
    sr = samplerate
    #logfpsd_, logffreqs, times = logfpsd(signal,sr,int(window*sr*2),int(overlap*sr*2),fmin,bins_per_octave)
    logflogpsd, logffreqs, times = interp_logpsd(signal, sr, int(window*sr), int(overlap*sr), freqs, interpolation)
    #9/25/2015: removed factors of 2 in int() things, to match Carlson's code

    # remove segments of the PSD with total power below cutoff
    abovethebar = (np.sum(10**logflogpsd,axis=1) > cutoff)
    return logflogpsd[abovethebar,:]

def wav_to_spectro(infolder='../speech_corpora/TIMIT/', outfolder='../Spectrograms/', max_number=None):
    """Takes all the .wav files in the given folder and makes files containing
    spectrograms according to the parameters in Carlson et al.
    Stops after max_number spectrograms if specified."""
            
    infilelist = listdir(infolder)
    count = 0
    for infilename in infilelist:
    #filename = "../speech_corpora/TIMIT/SA1_FADG0_DR4_TEST.WAV"
        if not infilename.lower().endswith('.wav'):
            continue
            # ignore anything that isn't a .wav file
        logflogpsd = wav_to_logPSD(infolder+infilename)
              
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
        infile = infolder + infilelist[i]
        array = np.load(infile).T
        plt.subplot(3,3,i+1)
        plt.imshow(array, interpolation= 'nearest', cmap='jet', aspect='auto')
        plt.gca().invert_yaxis()
    plt.show()
    return array
    
def view_old_spectros(start=0):
    spectros = scipy.io.loadmat("../speechdata.mat")["speechdata0"][:,:,start:start+9]
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
        # center and normalize
        vector = vector - trainingdatamean
        vector = vector/trdata_std
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
    
def wav_to_PCA(infolder='../speech_corpora/TIMIT/', outfile='../Data/processedspeech2.npy', 
               pcafilename = '../Data/spectropca2.pickle', testfile = '../Data/test2.npy', ncomponents = 200, whiten = True):
    """Do the whole preprocessing scheme at once, saving a pickled PCA object and a .npy array with the data in the reduced
representation. Unreduced spectrograms are not saved. Since these are all stored at once and the covariance matrix for all of them
is computed, this method requires a substantial amount of RAM (something like 8GB for the TIMIT data set)."""
    infilelist = listdir(infolder)
    
    allspectros = [] # don't know length in advance, use list for flexible append
    for infilename in infilelist:
        if not infilename.lower().endswith('.wav'):
            continue # ignore anything that isn't a .wav file
        logflogpsd = wav_to_logPSD(infolder+infilename)
        
        nchunks = int(logflogpsd.shape[0]/ntimepoints)
        somespectros = np.zeros((nchunks,ntimepoints*nfreqs))
        for chunk in range(nchunks):
            # convert each chunk to a vector and store. the last chunk is ignored if it's incomplete
            start = ntimepoints*chunk
            finish = ntimepoints*(chunk+1)
            somespectros[chunk] = logflogpsd[start:finish,:].flatten()
        allspectros.append(somespectros)
    allspectros = np.array(allspectros)
    
    # center and normalize spectrograms
    allspectros = np.nan_to_num(allspectros)
    allspectros = np.clip(allspectros,-1000,1000)
#    datamean = np.mean(allspectros, axis=0)
#    allspectros = allspectros - datamean
#    datastd = np.std(allspectros, axis=0)
#    allspectros = allspectros/datastd
    # Nicole's code seems to do this instead, centering each spectrogram
    allspectros = allspectros - allspectros.mean(axis=1)[:,np.newaxis]
    # for backwards compatibility, at least for now...
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
 
def sample_recons(infile='../Data/processedspeech.npy', pcafile = '../Data/speechpca.pickle', which=None):
    vectors = np.load(infile)
    if which is not None:
        vectors = vectors[which]
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