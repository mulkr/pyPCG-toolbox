import warnings
import numpy as np
import pyPCG as pcg
import numpy.typing as npt
import scipy.signal as signal
import scipy.fft as fft
import pywt

def _check_start_end(start,end):
    if len(start) != len(end):
        warnings.warn("Start and end arrays not the same size. Converting to same size...")
        if len(start) > len(end):
            start = start[:len(end)]
        else:
            end = end[:len(start)]
    return start, end

def time_delta(start: npt.NDArray[np.int_],end: npt.NDArray[np.int_], sig: pcg.pcg_signal) -> npt.NDArray[np.float_]:
    """Calculate time differences between pairs of points

    Args:
        start (np.ndarray): start points in samples
        end (np.ndarray): end points in samples
        sig (pcg.pcg_signal): input signal

    Returns:
        np.ndarray: time differences in seconds
    """
    start, end = _check_start_end(start,end)
    return (end-start)/sig.fs

def ramp_time(start: npt.NDArray[np.int_],end: npt.NDArray[np.int_],envelope: pcg.pcg_signal,type: str="onset") -> npt.NDArray[np.float_]:
    """Calculate ramp time (onset or exit), the time difference between the boundary and peak

    Args:
        start (np.ndarray): start points in samples
        end (np.ndarray): end points in samples
        envelope (pcg.pcg_signal): precalculated envelope signal
        type (str, optional): ramp type "onset" or "exit". Defaults to "onset".

    Raises:
        ValueError: ramp type not "onset" or "exit"

    Returns:
        np.ndarray: ramp times in seconds
    """
    start, end = _check_start_end(start,end)
    ret = []
    for s,e in zip(start,end):
        peak = np.argmax(envelope.data[s:e])
        l = e-s
        if type=="onset":
            ret.append(peak)
        elif type=="exit":
            ret.append(l-peak)
        else:
            raise ValueError("Unrecognized ramp type")
    return np.array(ret)/envelope.fs

def zero_cross_rate(start: npt.NDArray[np.int_],end: npt.NDArray[np.int_],sig: pcg.pcg_signal) -> npt.NDArray[np.float_]:
    """Calculate zero cross rate

    Args:
        start (np.ndarray): start times in samples
        end (np.ndarray): end times in samples
        sig (pcg.pcg_signal): input signal

    Returns:
        np.ndarray: zero cross rates
    """
    start, end = _check_start_end(start,end)
    ret = []
    for s, e in zip(start,end):
        crosses = len(np.nonzero(np.diff(sig.data[s:e] > 0))[0])+0.5
        ret.append(crosses/(e-s))
    return np.array(ret)

def peak_spread(start: npt.NDArray[np.int_],end: npt.NDArray[np.int_],envelope: pcg.pcg_signal,factor: float=0.7) -> npt.NDArray[np.int_]:
    """Calculate peak width, the amount of area under the peak with a given percentage of the total and time differences between the beginning and end

    Args:
        start (np.ndarray): start times in samples
        end (np.ndarray): end times in samples
        envelope (pcg.pcg_signal): input envelope signal
        factor (float, optional): percentage of total area. Defaults to 0.7.

    Returns:
        np.ndarray: time difference between the beginning and end of the percentage area
    """
    start, end = _check_start_end(start,end)
    ret = []
    for s, e in zip(start, end):
        win = envelope.data[s:e]
        th = np.sum(win)*factor
        vals = np.sort(win*-1)*-1
        for val in vals:
            filt = win>val
            if np.sum(win[filt])>=th:
                idx = np.nonzero(filt)[0]
                ret.append(idx[-1]-idx[0])
                break
    return np.array(ret)

def peak_centroid(start: npt.NDArray[np.int_],end: npt.NDArray[np.int_],envelope: pcg.pcg_signal) -> tuple[npt.NDArray[np.int_],npt.NDArray[np.float_]]:
    """Calculate centroid (center of mass) of the envelope

    Args:
        start (np.ndarray): start times in samples
        end (np.ndarray): end times in samples
        envelope (pcg.pcg_signal): input envelope signal

    Returns:
        np.ndarray: time delays from start to centroid
        np.ndarray: envelope values at centroid
    """
    start, end = _check_start_end(start,end)
    power = envelope.data**2
    loc, val = [], []
    for s, e in zip(start,end):
        win = power[s:e]
        th = np.sum(win)*0.5 #type: ignore
        centr = np.nonzero(np.cumsum(win)>th)[0][0]
        loc.append(centr)
        val.append(win[centr])
    return np.array(loc), np.array(val)

def peak_width(start: npt.NDArray[np.int_],end: npt.NDArray[np.int_],envelope: pcg.pcg_signal,factor: float=0.7) -> npt.NDArray[np.int_]:
    start, end = _check_start_end(start,end)
    ret = []
    for s,e in zip(start,end):
        loc = np.argmax(envelope.data[s:e])+s
        val = envelope.data[loc]
        th = val*factor
        w_s = np.nonzero(envelope.data[:loc]<th)[0][-1]
        w_e = np.nonzero(envelope.data[loc:]<th)[0][0]+loc
        ret.append(w_e-w_s)
    return np.array(ret)

def max_freq(start: npt.NDArray[np.int_],end: npt.NDArray[np.int_],sig: pcg.pcg_signal,nfft: int=512) -> tuple[npt.NDArray[np.float_],npt.NDArray[np.float_]]:
    """Calculate frequency with maximum amplitude of the segment

    Args:
        start (np.ndarray): start times in samples
        end (np.ndarray): end times in samples
        sig (pcg.pcg_signal): input signal
        nfft (int, optional): fft width parameter. Defaults to 512.

    Returns:
        np.ndarray: frequencies of the maximum amplitude
        np.ndarray: values of the maximum amplitude frequency
    """
    start, end = _check_start_end(start,end)
    freqs = np.linspace(0,sig.fs//2,nfft//2)
    loc, val = [],[]
    for s,e in zip(start,end):
        spect = abs(fft.fft(sig.data[s:e],n=nfft)) #type: ignore
        spect = spect[:nfft//2]
        loc.append(freqs[np.argmax(spect)])
        val.append(np.max(spect))
    return np.array(loc), np.array(val)

def spectral_spread(start: npt.NDArray[np.int_],end: npt.NDArray[np.int_],sig: pcg.pcg_signal, factor: float=0.7, nfft: int=512) -> npt.NDArray[np.int_] -> npt.NDArray[np.int_]:
    """Calculate spectral width of the segments, percentage of the total power of the segment and the frequency difference between the beginning and end of the calculated area

    Args:
        start (np.ndarray): start times in samples
        end (np.ndarray): end times in samples
        sig (pcg.pcg_signal): input signal
        factor (float, optional): percentage of total power. Defaults to 0.7.
        nfft (int, optional): fft width parameter. Defaults to 512.

    Returns:
        np.ndarray: difference of the beginning and end of the given area
    """
    start, end = _check_start_end(start,end)
    ret = []
    for s, e in zip(start, end):
        spect = abs(fft.fft(sig.data[s:e],n=nfft)) #type: ignore
        spect = spect[:nfft//2]
        power = spect**2
        th = np.sum(power)*factor
        vals = np.sort(power*-1)*-1
        for val in vals:
            filt = power>val
            if np.sum(power[filt])>=th:
                idx = np.nonzero(filt)[0]
                ret.append(idx[-1]-idx[0])
                break
    return np.array(ret)

def spectral_centroid(start: npt.NDArray[np.int_],end: npt.NDArray[np.int_],sig: pcg.pcg_signal,nfft: int=512) -> tuple[npt.NDArray[np.float_],npt.NDArray[np.float_]]:
    """Calculate spectral centroid (center of mass)

    Args:
        start (np.ndarray): start times in samples
        end (np.ndarray): end times in samples
        sig (pcg.pcg_signal): input signal
        nfft (int, optional): fft width parameter. Defaults to 512.

    Returns:
        np.ndarray: spectral centroid locations in Hz
        np.ndarray: spectral centroid values
    """
    start, end = _check_start_end(start,end)
    freqs = np.linspace(0,sig.fs//2,nfft//2)
    loc, val = [],[]
    for s,e in zip(start,end):
        spect = abs(fft.fft(sig.data[s:e],n=nfft)) #type: ignore
        spect = spect[:nfft//2]
        th = np.sum(spect)*0.5
        idx = np.nonzero(np.cumsum(spect)>th)[0][0]
        loc.append(freqs[idx])
        val.append(spect[idx])
    return np.array(loc), np.array(val)

def spectral_width(start: npt.NDArray[np.int_],end: npt.NDArray[np.int_],sig: pcg.pcg_signal, factor: float=0.7, nfft: int=512) -> npt.NDArray[np.int_]:
    start, end = _check_start_end(start,end)
    ret = []
    for s, e in zip(start, end):
        spect = abs(fft.fft(sig.data[s:e],n=nfft)) #type: ignore
        spect = spect[:nfft//2]
        power = spect**2
        loc = np.argmax(power)
        val = power[loc]
        th = val*factor
        w_s = np.nonzero(power[:loc]<th)[0][-1]
        w_e = np.nonzero(power[loc:]<th)[0][0]+loc
        ret.append(w_e-w_s)
    return np.array(ret)

def spectrum_raw(start: npt.NDArray[np.int_],end: npt.NDArray[np.int_],sig: pcg.pcg_signal,nfft:int=512) -> npt.NDArray[np.float_]:
    start, end = _check_start_end(start,end)
    ret = []
    for s,e in zip(start,end):
        spect = abs(fft.fft(sig.data[s:e],n=nfft)) #type: ignore
        spect = spect[:nfft//2]
        ret.append(spect)
    return np.array(ret)

def max_cwt(start: npt.NDArray[np.int_],end: npt.NDArray[np.int_],sig: pcg.pcg_signal) -> tuple[npt.NDArray[np.float_],npt.NDArray[np.float_]]:
    warnings.warn("CWT calculation done with PyWT which has parity problems with Matlab")
    start, end = _check_start_end(start,end)
    time,freq = [],[]
    for s,e in zip(start,end):
        coef, fr = pywt.cwt(sig.data[s:e],np.arange(1,100),"cmor1.0-1.5",sampling_period=1/sig.fs)
        coef = np.abs(coef)
        loc = np.unravel_index(np.argmax(coef),coef.shape)
        time.append(loc[0])
        freq.append(fr[loc[1]])
    return np.array(time),np.array(freq)

def dwt(start,end,sig):
    raise NotImplementedError("This feature calculation not implemented yet")

def katz_fd(start,end,sig):
    raise NotImplementedError("This feature calculation not implemented yet")

def lyapunov(start,end,sig):
    raise NotImplementedError("This feature calculation not implemented yet")

if __name__ == '__main__':
    print("Feature calculation")