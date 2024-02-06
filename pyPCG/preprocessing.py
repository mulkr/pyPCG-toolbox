import pyPCG as pcg
import scipy.signal as signal
import numpy as np
import pywt
import copy
import emd

def envelope(sig: pcg.pcg_signal) -> pcg.pcg_signal:
    """Calculates the envelope of the signal based on Hilbert transformation

    Args:
        sig (pcg.pcg_signal): input signal

    Returns:
        pcg.pcg_signal: envelope
    """
    ret_sig = copy.deepcopy(sig)
    ret_sig.processing_log.append("Envelope")
    ret_sig.data = np.abs(signal.hilbert(ret_sig.data)) # type: ignore
    return ret_sig

def homomorphic(sig: pcg.pcg_signal, filt_ord: int = 6, filt_cutfreq: float = 8) -> pcg.pcg_signal:
    """Calculate the homomoprphic envelope of the signal

    Args:
        sig (pcg.pcg_signal): input signal
        filt_ord (int, optional): lowpass filter order. Defaults to 6.
        filt_cutfreq (float, optional): lowpass filter cutoff frequency. Defaults to 8.

    Raises:
        ValueError: The cutoff frequency exceeds the Nyquist limit

    Returns:
        pcg.pcg_signal: homomoprhic envelope
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

def filter(sig: pcg.pcg_signal, filt_ord: int, filt_cutfreq: float, filt_type: str = "LP") -> pcg.pcg_signal:
    """Filters the signal based on the input parameters

    Args:
        sig (pcg.pcg_signal): input signal
        filt_ord (int): filter order
        filt_cutfreq (float): filter cutoff frequency
        filt_type (str, optional): filter type: "LP" or "HP". Defaults to "LP".

    Raises:
        NotImplementedError: Other filter type
        ValueError: Filter cutoff exceeds Nyquist limit

    Returns:
        pcg.pcg_signal: filtered signal
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

def wt_denoise(sig: pcg.pcg_signal, th: float=0.2, wt_family: str = "coif4", wt_level: int = 5) -> pcg.pcg_signal:
    """Denoise the signal with a wavelet thresholding method

    Args:
        sig (pcg.pcg_signal): input noisy signal
        th (float, optional): threshold value given as a percentage of maximum. Defaults to 0.2.
        wt_family (str, optional): wavelet family. Defaults to "coif4".
        wt_level (int, optional): wavelet decomposition level. Defaults to 5.

    Returns:
        pcg.pcg_signal: denoised signal
    """
    ret_sig = copy.deepcopy(sig)
    th_coeffs = []
    coeffs = pywt.wavedec(ret_sig.data,wt_family,level=wt_level)
    for coeff in coeffs:
        th_coeffs.append(pywt.threshold(coeff,th*max(coeff)))
    ret_sig.data = pywt.waverec(th_coeffs,wt_family)
    ret_sig.processing_log.append(f"Wavelet denoise (family-{wt_family}, level-{wt_level}, th-{th})")
    return ret_sig

def slice_signal(sig: pcg.pcg_signal, time_len: float=60, overlap: float=0):
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
        sliced = pcg.pcg_signal(sig.data[start:end],sig.fs)
        acc.append(sliced)
        start += step
    return acc

def emd_denoise_sth(sig: pcg.pcg_signal) -> pcg.pcg_signal:
    """Boudraa, Abdel-O & Cexus, Jean-Christophe & Saidi, Zazia. (2005). EMD-Based Signal Noise Reduction. Signal Processing. 1."""
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

def emd_denoise_savgol(sig: pcg.pcg_signal, window: int=10, poly: int=3) -> pcg.pcg_signal:
    """Boudraa, Abdel-O & Cexus, Jean-Christophe & Saidi, Zazia. (2005). EMD-Based Signal Noise Reduction. Signal Processing. 1."""
    imf = emd.sift.sift(sig.data)
    th_imf = signal.savgol_filter(imf,window,poly,mode="nearest")
    ret_sig = copy.deepcopy(sig)
    ret_sig.data = np.sum(th_imf,axis=1)
    ret_sig.processing_log.append("EMD denoising (savgol)")
    return ret_sig

def wt_denoise_sth(sig: pcg.pcg_signal, wt_family: str = "coif4", wt_level: int = 5) -> pcg.pcg_signal:
    """Denoise the signal with automatic wavelet thresholding method

    Args:
        sig (pcg.pcg_signal): input noisy signal
        wt_family (str, optional): wavelet family. Defaults to "coif4".
        wt_level (int, optional): wavelet decomposition level. Defaults to 5.

    Returns:
        pcg.pcg_signal: denoised signal
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

if __name__ == '__main__':
    print("Preprocessing functions")