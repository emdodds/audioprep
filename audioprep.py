# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 14:35:05 2016

@author: Eric
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from librosa.core import load as wavload
from librosa.core import resample
from pca import PCA # Jesse Livezey's PCA class
import pickle

try:
    from matlab import double as mdouble
    import matlab.engine    
except OSError as e:
    print(e)
    print("Can't make spectrograms because Matlab engine not compatible. Try using Python 3.4.")
    
class PrepParams:
    def __init__(self, transform='fourier', max_data = 80000, 
                 max_at_once=80000, sample_rate=16000,
                 silence_cutoff = 1e-6, cutoff_type = 'pointwise', 
                 segment_length = 25, segment_overlap = 0.5,
                 nPCs=200, whiten=True, **kwargs):
        self.max_data = max_data
        self.max_at_once = max_at_once
        self.silence_cutoff = silence_cutoff
        self.sample_rate = sample_rate
        self.transform = transform
        self.segment_overlap = segment_overlap
        self.segment_length = segment_length
        self.specific = kwargs
        self.cutoff_type = cutoff_type
        self.nPCs = nPCs
        self.whiten = whiten
        
class Transformer:
    def transform(self, signal):
        raise NotImplementedError
        
class Spectrogram(Transformer):
    def __init__(self, sample_rate=16000, window_length=256, 
                 window_overlap= None, nfreqs=256, pointwise_cutoff = 0):
        self.eng = matlab.engine.start_matlab()
        self.window_length = window_length
        if window_overlap is None:
            self.window_overlap = int(np.floor(window_length/2))
        else:
            self.window_overlap = window_overlap
        self.sample_rate = sample_rate
        self.nfreqs = nfreqs
        self.freqs = np.logspace(2,np.log10(sample_rate/4),nfreqs)
        self.mfreqs = mdouble(self.freqs.tolist())
        self.pointwise_cutoff = pointwise_cutoff
        
    def get_size(self):
        """Size per timepoint."""
        return self.nfreqs
               
    def transform(self, signal):
        msignal = mdouble(signal.tolist())
        _,_,timebins,psd = self.eng.spectrogram(msignal, self.window_length, 
                                                self.window_overlap, self.mfreqs, 
                                                self.sample_rate, nargout=4)
        result = np.flipud(np.array(psd._data).reshape(psd.size[::-1]))
        if self.pointwise_cutoff>0:
            mask = result.sum(1) > self.pointwise_cutoff
            result = result[mask,:]
        return np.log10(result)
        
def get_file_list(directory):
    """Get a list of the audio files in the given directory and its subdirectories."""
    file_list = []
    for pth, subd, files in os.walk(directory):
        for fname in files:
            if fname.lower().endswith('.wav'):
                file_list.append(os.path.join(pth,fname))
    return file_list            
    
def file_to_waveform(file, desired_rate):
    signal, sr = wavload(file, sr = None)
    signal = resample(signal, sr, desired_rate)
    
    signal = signal/(10*np.var(signal))
    signal = signal - np.mean(signal)
    
    return signal
    
def get_data(file_list, params, transformer, stride):
    ndata = 0
    data = np.zeros((params.max_at_once, params.segment_length*transformer.get_size()))
    while len(file_list)>0:
        file = file_list.pop()
        #print()
        #print('Processing ' + file + '. Segments so far: ' + str(ndata))
        signal = file_to_waveform(file, params.sample_rate)
        processed = transformer.transform(signal)        
        
        for t in range(int(processed.shape[0]/stride)-1):
            #print("Of this file, trying segment " + str(t)+'\r',end='')
            start = t*stride
            end = start + params.segment_length
            segment = processed[start:end,:]
            segment = segment - segment.mean()
            
            if params.cutoff_type != 'segmentwise' or np.power(10,segment).mean() > params.silence_cutoff:
                data[ndata,:] = segment.flatten()
                
                ndata += 1
                if ndata >= params.max_at_once:
                    break
                
        if ndata >= params.max_at_once:
            break
        
    #print('Produced ' + str(ndata) + ' data.')        
    if ndata<params.max_data:
        data = data[:ndata]
    return data

    
def files_to_data(directory, outfile, pcafile, params, prefit=False):
    """If prefit is True, attempt to load the prefit pca object. 
    Otherwise a new one will be fit to all the data, which requires a lot of RAM."""
    print ('Looking for files...')
    file_list = get_file_list(directory)
    print ('Found ' + str(len(file_list)) + ' wav files. Processing...')
    
    if params.transform == 'fourier':
        if params.cutoff_type == 'pointwise':
            pointwise_cutoff = params.silence_cutoff
        else:
            pointwise_cutoff = 0
        transformer = Spectrogram(params.sample_rate, 
                                  pointwise_cutoff = pointwise_cutoff,
                                  **params.specific)
    else:
        raise ValueError('Transform type not supported.')
        return
        
    
    stride = params.segment_length*params.segment_overlap
    ndata = 0
    analyzer=None
    reduced = np.zeros((0,params.nPCs))
    while ndata < params.max_data:
        data = get_data(file_list, params, transformer, stride)
        if data.shape[0] == 0:
            break
        ndata += data.shape[0]
        print('Total segments: ', str(ndata), ', ', str(len(file_list)),' files to go...')
    
        if analyzer is None:
            if prefit:
                try:
                    with open(pcafile, 'rb') as f:  
                        analyzer, origshape = pickle.load(f)
                except:
                    print("Failed to load PCA. Returning unreduced data.")
                    return data
                print("Loaded prefit PCA.")
                savepca=False
            else:
                analyzer = PCA(dim=params.nPCs, whiten=params.whiten)
                print ("Fitting the PCA...")
                analyzer.fit(data)
                savepca=True
        print ("Transforming vectors...")
        reduced = np.concatenate([reduced,analyzer.transform(data)],axis=0)
    
    np.save(outfile, reduced)    
    origshape = (params.segment_length, transformer.get_size())
    if savepca:        
        with open(pcafile, 'wb') as f:
            pickle.dump([analyzer, origshape], f)   
        
    nsamples=10
    samplerecons = analyzer.inverse_transform(reduced[-nsamples:])
    plt.figure()
    for ii in range(nsamples):
        plt.subplot(2,10,ii+1)
        plt.imshow(samplerecons[ii].reshape(origshape).T,interpolation='nearest', cmap='jet', aspect='auto', origin='lower')
        plt.subplot(2,10,nsamples+ii+1)        
        plt.imshow(data[-nsamples+ii].reshape(origshape).T,interpolation='nearest', cmap='jet', aspect='auto', origin='lower')
    plt.savefig('comparison.png', bbox_inches='tight')
        
    print ("Done.")
                
    return reduced, analyzer, data
        