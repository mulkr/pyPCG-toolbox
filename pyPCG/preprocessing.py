import pyPCG as pcg
import scipy.signal as signal
import numpy as np
import pywt
import copy

def envelope(sig: pcg.pcg_signal) -> np.ndarray:
    """Calculates the envelope of the signal based on Hilbert transformation

    Args:
        sig (pcg.pcg_signal): input signal

    Returns:
        np.ndarray: envelope
    """
    if "env" in sig.features.keys():
        return sig.features["env"]
    env = np.abs(signal.hilbert(sig.data)) # type: ignore
    sig.features["env"] = env
    return env

def homomorphic(sig: pcg.pcg_signal, filt_ord: int = 6, filt_cutfreq: float = 8) -> np.ndarray:
    """Calculate the homomoprphic envelope of the signal

    Args:
        sig (pcg.pcg_signal): input signal
        filt_ord (int, optional): lowpass filter order. Defaults to 6.
        filt_cutfreq (float, optional): lowpass filter cutoff frequency. Defaults to 8.

    Raises:
        ValueError: The cutoff frequency exceeds the Nyquist limit

    Returns:
        np.ndarray: homomoprhic envelope
    """
    if "h_env" in sig.features.keys():
        return sig.features["h_env"]
    env = envelope(sig)
    if filt_cutfreq>sig.fs/2:
        raise ValueError("Filter cut frequency exceeds Nyquist limit")
    lp = signal.butter(filt_ord,filt_cutfreq,output='sos',fs=sig.fs,btype='lowpass')
    env[env<=0] = np.finfo(float).eps
    filt = signal.sosfiltfilt(lp,np.log(env))
    h_env = np.exp(filt)
    sig.features["h_env"] = h_env
    return h_env

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
    ret_sig.features = {}
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
    ret_sig.features = {}
    return ret_sig