import logging
logger = logging.getLogger(f'fr.{__name__}')
import numpy as np
import jax.numpy as jnp

from scipy.signal import butter, filtfilt

from typing import Tuple, Sequence
from ._typing import *


def movemean(signal: Array, window_size:int, start_idx:int) -> Array:
    '''Compute the running average of the signal.
    
    Arguments:\n
        signal: a 1D array of signal.\n
        window_size: the window size for computing average. If window is even, then window/2 number of points on each side of the centre point will be used. If window is odd, then (window-1)/2 number of points on each side will be used.\n
        start_idx: from where to start computing the running average. \n
    Return:\n
        the smoothed signal with the same dimension as the original.
    '''

    if len(signal.shape) > 1:
        raise ValueError('Running average only works on 1D array.')

    l = len(signal)
    w2 = int(window_size/2)
    smooth_signal = np.zeros_like(signal)
    
    if start_idx > 0:
        smooth_signal[:start_idx] = signal[:start_idx]
    
    if w2 > start_idx: # if there are not enough points in the window
        logger.debug('Not enough points to compute the running average for the user defined window size. Using increasing window size.')        
        start_from = w2
        for j in range(start_idx,w2):
            smooth_signal[j] = np.mean(signal[j-start_idx:j+1])
    else:
        start_from = start_idx
    
    # Now using the user defined window size
    for j in range(start_from,l):
        smooth_signal[j] = np.mean(signal[j-w2:j+w2])

    return smooth_signal



def butter_lowpass_filter(
        signal:Array, 
        cutoff:Scalar, 
        fs:Scalar, 
        order:int = 2
    ) -> Array:
    '''Butterworth low pass filter.
    
    Arguments:\n
        signal: a 1D array representing a time series.\n
        cutoff: cutoff frequency.\n
        fs: sampling frequency of the signal.\n
        order: order of the filter, see scipy.signal.butter. A higher order will make the filter attenuate signal quicker.\n
    Returns:\n
        Filtered signal.
    '''
    
    normal_cutoff = cutoff / fs / 2
    b, a = butter(order, normal_cutoff, btype='low', analog=False)# Get the filter coefficients 
    y = filtfilt(b, a, signal)

    return y


# From the paper by Oxlade et al. 2012. De-noising of time-resolved PIV.
def estimate_noise_floor(
        signal, 
        fs, 
        f1=None, 
        window_fn='hanning',
        convergence='movemean',
        return_full=False,
        **kwargs
    ) -> Tuple[Scalar, Array]: 
    '''Estimate the power of the white noise.
    
    Arguments:\n
        signal: a 1D array of time series.\n
        fs: the sampling frequency.\n
        f1: lowest expected noise frequency.\n
        window_fn: window function for use in fft.\n
        convergence: 'standard'-using fft power without futher processing. 'movemean'-using running average to improve convergence.\n
        return_full: return the frequency range over which the noise floor is calculated in addition to other return values.
        **kwargs: kwargs for use in convergence functions.\n
    Returns:\n
        estimated_noise_floor, power spectrum of the signal (if return_full is True, then also returns [frequency lower bound, frequency upper bound])
    '''

    dt = 1/fs
    ls = len(signal)

    if ls%2 != 0:
        nfft = ls+1
    else:
        nfft = ls

    if window_fn == 'hanning':
        window = jnp.hanning(ls)
    else:
        raise NotImplementedError

    fftfreq = jnp.fft.fftfreq(nfft,dt)
    df = fs/nfft
    power = jnp.abs(
        jnp.fft.fft(signal*window, nfft)
    )**2 

    if convergence == 'movemean':
        power = movemean(power,**kwargs)
    elif convergence == 'standard':
        pass
    else:
        raise ValueError

    f2 = 0.5*fs - df
    if not f1:
        f1 = 0.46*(fs/2) # this is the value given in the paper
    idx_lower = (jnp.abs(fftfreq - f1)).argmin()
    idx_upper = (jnp.abs(fftfreq - f2)).argmin()

    if return_full:
        return jnp.mean(power[idx_lower:idx_upper]), power, [f1, f2]
    else:    
        return jnp.mean(power[idx_lower:idx_upper]), power


def estimate_noise_cutoff_frequency(
        power:Array, 
        noise_estimate:Scalar, 
        df:Scalar, 
        slack:Scalar
    ) -> Scalar:

    '''Find the cut-off frequency for a low-pass filter based on the estimated noise floor. 
    
    Arguments:\n
        power: the power spectrum of the signal.\n
        noise_estimate: the estimated noise floor from the power spectrum.\n
        df: frequency resolution.\n
        slack: how much to raise the cut-off frequency by. For example, 0.1*Nyquist frequency.\n
    Returns:\n
        the cut-off frequency to use with a low-pass filter.
    '''

    return np.argmax(power < noise_estimate) * df + slack 


def rfftn_filter(a:Array, axes:Sequence, function:str, *args, value_k0:float = 0.):
    _shape = np.array(a.shape)
    nx = _shape[axes]
    fftfreq = []
    for i in range(len(nx)-1):
        fftfreq.append(
            np.fft.fftfreq(nx[i], d=1/nx[i])
        )
    fftfreq.append(
        np.fft.rfftfreq(nx[-1], d=1/nx[-1])
    )
    kgrid = np.meshgrid(*fftfreq, indexing='ij')
    kgrid = np.array(kgrid)
    kgrid_magnitude = np.sqrt(np.einsum('n... -> ...', kgrid**2))
    
    filter_functions = {
        'exponential_decay': lambda x: 1 / (np.exp(x*args[0])+1),
        'no_change': lambda x: np.ones_like(x),
        'polynomial_decay': lambda x: 1/(x**args[0])
    }
    mask = np.where(kgrid_magnitude==0, value_k0, filter_functions[function](kgrid_magnitude))

    expanded_shape = np.ones(len(a.shape), dtype=int)
    expanded_shape[axes] = mask.shape
    mask = np.reshape(mask, expanded_shape)
    
    return mask