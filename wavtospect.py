# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 17:41:38 2015

@author: Eric Dodds

Convert .wav files to spectrograms.
"""

import scipy.io.wavfile as wf
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np

def wavtospect(filename = 'speech_corpora/TIMIT/SA1_FAEM0_DR2_TRAIN.wav',
               window = .016, overlap = .008):   
    """
    Read in .wav file and create spectrogram.
    filename = string containing filename with .wav extension
    window = length of each FFT window in seconds
    overlap = overlap of each pair of FFT windows in seconds
    """
    rate, data = wf.read(filename)
     
    spect, freq, time = mlab.specgram(data, NFFT = int(window*rate*2), Fs = rate, 
                                     #window = mlab.window_hanning(lwindow*rate), 
                                     noverlap = int(overlap*rate*2))
    #spect, freq, time = mlab.specgram(data, Fs = rate)
    plt.figure()
    plt.imshow(np.log10(spect))
    plt.gca().invert_yaxis()
    plt.show()
    print(freq)
    
wavtospect()    