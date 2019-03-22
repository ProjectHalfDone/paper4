# Code for computing, data, models and Graphs for Paper 3.
#
#
# @ Alexander Tureczek, Technical University of Denmark, 2017
#
# Description: functions for wavelets

import pandas as pd
import numpy as np
import pywt

from statsmodels.robust import mad


#------------------------------------------------------
#                 wavelet data prep
#------------------------------------------------------
#sigma on the lowest level
def wave(data, wavelet='haar', mode = 'soft'):#, pyr=0, wav = 0):
    """Wavelet coefficients of the input data, and subsequent pyramide plotting
    build using pywt.

    threshold is using universal thresholding. Refer to jseabold wavelet regression
    http://jseabold.net/blog/2012/02/23/wavelet-regression-in-python/
    

    SYNTAX:
        [true_coef, signal, denoised] = wavelets.wave(ava['101'], mode='hard')
        
    INPUT:
        data: 1D-array or list with values for wavelets.

        wavelet: the mother wavelet. default = 'Haar', refer to pywt.wavelist() for wavelets.

        mode: 'hard', 'soft', 'less', refer to pywt.threshold for details.

        
    Output:
    
        true_coef: the coefficients for the original signal.     
    
        signal: wavelet transformed data.

        denoised: the denoised coefficients.

    Ex.
        [tr, signal, dn] = wavelets.wave(ava['101'], wavelet='coif16', mode='hard')
        
    """
    true_coefs = pywt.wavedec(data, wavelet, mode='per')

    #Evaluate data
        #Pyramid plot
    '''
    if pyr ==1:
        
        fig = cpp.coef_pyramid_plot(true_coefs[1:])
        fig.show()
    '''

    #Calculating Mean-absolute-deviation
    
    sigma = mad(true_coefs[-1])
    #Calculating the universal thresholding.
    uthresh = sigma*np.sqrt(2*np.log(len(data)))
    #uthresh = sigma*np.sqrt(2*np.log(len(data))/len(data))

    #denoising data using universal thresholding, resulting in denoised signal. 
    denoised = true_coefs[:]
    denoised[1:] = (pywt.threshold(i, value=uthresh, mode=mode, substitute=0) for i in denoised[1:])
    signal = pywt.waverec(denoised, wavelet, mode='per')

    return true_coefs, signal, denoised


    '''
    #Number of coefficients
    comp = cmpt(denoised)

    #Evaluate Chosen Wavelet
    #let = pywt.Wavelet(wavelet)
    #sca, wave, x = let.wavefun()
    '''


    '''
    #Plotting
    if wav ==1:
        fig = plt.figure()
        gs = gridspec.GridSpec(1,2, width_ratios=[3,1])
        t_str = ('Wavelet Family: '+wavelet+', Compression ratio:'+str(comp/len(true_coefs))+', final coef #: '+str(comp))
        fig.suptitle(t_str, fontsize=30)
        ax0 = plt.subplot(gs[0])
        plt.title('Wavelet fit to original data')
        ax1 = plt.subplot(gs[1])
        plt.title('Wavelet')

        ax0.plot(data, 'red', signal, 'black')
        ax1.plot(x, wave)
            
        plt.show(block=False)

    return true_coefs, signal, denoised
    '''
