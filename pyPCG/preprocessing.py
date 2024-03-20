from pyPCG import pcg_signal
import scipy.signal as signal
import numpy as np
import pywt
import copy
import emd
from typing import Callable, TypedDict

def envelope(sig: pcg_signal) -> pcg_signal:
    """Calculates the envelope of the signal based on Hilbert transformation

    Args:
        sig (pcg_signal): input signal

    Returns:
        pcg_signal: envelope
    """
    ret_sig = copy.deepcopy(sig)
    ret_sig.processing_log.append("Envelope")
    ret_sig.data = np.abs(signal.hilbert(ret_sig.data)) # type: ignore
    return ret_sig

def homomorphic(sig: pcg_signal, filt_ord: int = 6, filt_cutfreq: float = 8) -> pcg_signal:
    """Calculate the homomoprphic envelope of the signal

    Args:
        sig (pcg_signal): input signal
        filt_ord (int, optional): lowpass filter order. Defaults to 6.
        filt_cutfreq (float, optional): lowpass filter cutoff frequency. Defaults to 8.

    Raises:
        ValueError: The cutoff frequency exceeds the Nyquist limit

    Returns:
        pcg_signal: homomoprhic envelope
    """
    if filt_cutfreq>sig.fs/2:
        raise ValueError("Filter cut frequency exceeds Nyquist limit")
    
    ret_sig = copy.deepcopy(sig)
    env = envelope(sig).data
    ret_sig.processing_log.append(f"Homomorphic envelope (order-{filt_ord},cut-{filt_cutfreq})")
    lp = signal.butter(filt_ord,filt_cutfreq,output='sos',fs=sig.fs,btype='lowpass')
    env[env<=0] = np.finfo(float).eps
    filt = signal.sosfiltfilt(lp,np.log(env))
    ret_sig.data = np.exp(filt)
    return ret_sig

def filter(sig: pcg_signal, filt_ord: int, filt_cutfreq: float, filt_type: str = "LP") -> pcg_signal:
    """Filters the signal based on the input parameters with a Butterworth filter design

    Args:
        sig (pcg_signal): input signal
        filt_ord (int): filter order
        filt_cutfreq (float): filter cutoff frequency
        filt_type (str, optional): filter type: `"LP"` (low-pass) or `"HP"` (high-pass). Defaults to `"LP"`.

    Raises:
        NotImplementedError: Other filter type
        ValueError: Filter cutoff exceeds Nyquist limit

    Returns:
        pcg_signal: filtered signal
    """
    longname = ""
    if filt_type == "LP":
        longname = "lowpass"
    elif filt_type == "HP":
        longname = "highpass"
    else:
        raise NotImplementedError("Only HP and LP filters are supported right now")
    if filt_cutfreq>sig.fs/2:
        raise ValueError("Filter cut frequency exceeds Nyquist limit")
    filt = signal.butter(filt_ord,filt_cutfreq,output='sos',fs=sig.fs,btype=longname)
    ret_sig = copy.deepcopy(sig)
    ret_sig.data = signal.sosfiltfilt(filt,ret_sig.data)
    ret_sig.processing_log.append(f"{filt_type} Filter (order-{filt_ord}, cut-{filt_cutfreq})")
    return ret_sig

def wt_denoise(sig: pcg_signal, th: float=0.2, wt_family: str = "coif4", wt_level: int = 5) -> pcg_signal:
    """Denoise the signal with a wavelet thresholding method

    Args:
        sig (pcg_signal): input noisy signal
        th (float, optional): threshold value given as a percentage of maximum. Defaults to 0.2.
        wt_family (str, optional): wavelet family. Defaults to "coif4".
        wt_level (int, optional): wavelet decomposition level. Defaults to 5.

    Returns:
        pcg_signal: denoised signal
    """
    ret_sig = copy.deepcopy(sig)
    th_coeffs = []
    coeffs = pywt.wavedec(ret_sig.data,wt_family,level=wt_level)
    for coeff in coeffs:
        th_coeffs.append(pywt.threshold(coeff,th*max(coeff)))
    ret_sig.data = pywt.waverec(th_coeffs,wt_family)
    ret_sig.processing_log.append(f"Wavelet denoise (family-{wt_family}, level-{wt_level}, th-{th})")
    return ret_sig

def slice_signal(sig: pcg_signal, time_len: float=60, overlap: float=0) -> list[pcg_signal]:
    """Slice long signal into a list of shorter signals

    Args:
        sig (pcg_signal): input long signal
        time_len (float, optional): desired short timelength [seconds]. Defaults to 60.
        overlap (float, optional): overlap percentage. Defaults to 0.

    Returns:
        list[pcg_signal]: list of shorter signals
    """
    time_len_s = round(time_len*sig.fs)
    step = time_len_s-round(time_len_s*overlap)
    start = 0
    acc = []
    at_end = False
    while(not at_end):
        end = start+time_len_s
        if end >= len(sig.data):
            end = len(sig.data)
            at_end = True
        sliced = pcg_signal(sig.data[start:end],sig.fs)
        acc.append(sliced)
        start += step
    return acc

def emd_denoise_sth(sig: pcg_signal) -> pcg_signal:
    """EMD based denoising with soft threshold method.
    
    Based on: Boudraa, Abdel-O & Cexus, Jean-Christophe & Saidi, Zazia. (2005). EMD-Based Signal Noise Reduction. Signal Processing. 1.
    
    Note: Tau parameter calculation modified

    Args:
        sig (pcg_signal): input signal

    Returns:
        pcg_signal: denoised signal
    """
    imf = emd.sift.sift(sig.data)
    mad = np.median(np.abs(imf - np.median(imf,axis=0)),axis=0) #type: ignore
    sigma = mad/0.6745
    # MODIFIED TAU
    tau = sigma*np.sqrt(2)
    th_imf = pywt.threshold(imf,tau) #type: ignore
    ret_sig = copy.deepcopy(sig)
    ret_sig.data = np.sum(th_imf,axis=1)
    ret_sig.processing_log.append("EMD denoising (soft th)")
    return ret_sig

def emd_denoise_savgol(sig: pcg_signal, window: int=10, poly: int=3) -> pcg_signal:
    """EMD based denoising method with Savoy-Golatzky filter method.
    
    Based on: Boudraa, Abdel-O & Cexus, Jean-Christophe & Saidi, Zazia. (2005). EMD-Based Signal Noise Reduction. Signal Processing. 1.

    Args:
        sig (pcg_signal): input signal
        window (int, optional): savgol filter window size [samples]. Defaults to 10.
        poly (int, optional): savgol polynomial degree to fit. Defaults to 3.

    Returns:
        pcg_signal: denoised signal
    """
    imf = emd.sift.sift(sig.data)
    th_imf = signal.savgol_filter(imf,window,poly,mode="nearest")
    ret_sig = copy.deepcopy(sig)
    ret_sig.data = np.sum(th_imf,axis=1)
    ret_sig.processing_log.append("EMD denoising (savgol)")
    return ret_sig

def wt_denoise_sth(sig: pcg_signal, wt_family: str = "coif4", wt_level: int = 5) -> pcg_signal:
    """Denoise the signal with automatic wavelet thresholding method. Threshold is calculated automatically based on...

    Args:
        sig (pcg_signal): input noisy signal
        wt_family (str, optional): wavelet family. Defaults to "coif4".
        wt_level (int, optional): wavelet decomposition level. Defaults to 5.

    Returns:
        pcg_signal: denoised signal
    """
    ret_sig = copy.deepcopy(sig)
    th_coeffs = []
    coeffs = pywt.wavedec(ret_sig.data,wt_family,level=wt_level)
    for coeff in coeffs:
        mad = np.median(np.abs(coeff - np.median(coeff))) #type: ignore
        sigma = mad/0.6745
        # MODIFIED TAU
        tau = sigma*np.sqrt(2)
        th_coeffs.append(pywt.threshold(coeff,tau))
    ret_sig.data = pywt.waverec(th_coeffs,wt_family)
    ret_sig.processing_log.append(f"Wavelet denoise (family-{wt_family}, level-{wt_level})")
    return ret_sig

def resample(sig: pcg_signal, target_fs:int) -> pcg_signal:
    """Resample signal to target samplerate

    Args:
        sig (pcg_signal): input signal
        target_fs (int): target samplerate

    Returns:
        pcg_signal: resampled signal
    """
    ret_sig = copy.deepcopy(sig)
    ret_sig.data = signal.resample_poly(ret_sig.data,target_fs,ret_sig.fs)
    ret_sig.fs = target_fs
    ret_sig.processing_log.append(f"Resample to {target_fs} Hz")
    return ret_sig

class process_config(TypedDict):
    """Type to hold processing calculation configs"""
    step: Callable
    """function for the calculation"""
    params: dict[str,int|float|str]
    """parameters to pass to the function as keyword arguments"""

class process_pipeline:
    """Processing pipeline. One step's input is the previous step's output
    
    Attributes:
        steps (list[Callable | process_config]): List of steps as functions or function and parameters as keyword dictionary
    
    Example:
        Creating a simple pipeline
        
        >>> import pyPCG
        >>> my_pipeline = pyPCG.process_pipeline(pyPCG.zero_center, pyPCG.unit_scale)
        >>> print(my_pipeline)
        PCG processing pipeline [2 steps]
        
        Creating a pipeline with parameters:
        
        Option 1: Create a dictionary with the appropriate function with the parameters passed as keyword arguments
        
        For an easier experience, use the `process_config` type
        
        >>> import pyPCG
        >>> import pyPCG.preprocessing as preproc
        >>> step_1 = {"step":preproc.filter,"params":{"filt_ord":6,"filt_cutfreq":100,"filt_type":"LP"}}
        >>> step_2 = {"step":preproc.filter,"params":{"filt_ord":6,"filt_cutfreq":20,"filt_type":"HP"}}
        >>> my_pipeline = pyPCG.process_pipeline(step_1,step_2)
        
        Option 2:
        Using ``functools.partial``
        
        >>> import pyPCG
        >>> import pyPCG.preprocessing as preproc
        >>> from functools import partial
        >>> step_1 = partial(preproc.filter, filt_ord=6, filt_cutfreq=100, filt_type="LP")
        >>> step_2 = partial(preproc.filter, filt_ord=6, filt_cutfreq=20, filt_type="HP")
        >>> my_pipeline = pyPCG.process_pipeline(step_1,step_2)
    """
    def __init__(self, *configs: Callable|process_config) -> None:
        """Create processing pipeline object"""
        self.steps = []
        for k in configs:
            self.steps.append(k)
            
    def __repr__(self) -> str:
        return f"PCG processing pipeline [{len(self.steps)} steps]"
    
    def run(self, input: pcg_signal) -> pcg_signal:
        """Run the processing pipeline

        Args:
            input (pcg_signal): input signal

        Returns:
            pcg_signal: processed signal
        """
        out = input
        for step in self.steps:
            if type(step) is process_config or type(step) is dict:
                out = step["step"](out,**step["params"])
            else:
                out = step(out)
        return out

if __name__ == '__main__':
    print("Preprocessing functions")