import warnings
import numpy as np
import pyPCG as pcg
import numpy.typing as npt
import scipy.signal as signal
import scipy.fft as fft

def _check_start_end(start,end):
    if len(start) != len(end):
        warnings.warn("Start and end arrays not the same size. Converting to same size...")
        if len(start) > len(end):
            start = start[:len(end)]
        else:
            end = end[:len(start)]
    return start, end

def time_delta(start: npt.NDArray[np.int_],end: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
    start, end = _check_start_end(start,end)
    return end-start

def ramp_time(start: npt.NDArray[np.int_],end: npt.NDArray[np.int_],envelope: pcg.pcg_signal,type: str="onset") -> npt.NDArray[np.int_]:
    start, end = _check_start_end(start,end)
    peak = np.argmax(envelope.data[start:end])
    ret = np.array([])
    if type=="onset":
        ret = peak-start
    elif type=="exit":
        ret = end-peak
    else:
        raise ValueError("Unrecognized ramp type")
    return ret

def zero_cross_rate(start: npt.NDArray[np.int_],end: npt.NDArray[np.int_],sig: pcg.pcg_signal) -> npt.NDArray[np.float_]:
    start, end = _check_start_end(start,end)
    ret = []
    for s, e in zip(start,end):
        crosses = len(np.nonzero(np.diff(sig.data[s:e] > 0))[0])
        ret.append(crosses/e-s)
    return np.array(ret)
    

def peak_width(start: npt.NDArray[np.int_],end: npt.NDArray[np.int_],envelope: pcg.pcg_signal,factor: float=0.7) -> npt.NDArray[np.int_]:
    start, end = _check_start_end(start,end)
    sums = np.sum(envelope.data[start:end],axis=1)
    ret = []
    for summ, s, e in zip(sums, start, end):
        th = summ*factor
        win = envelope.data[s:e]
        vals = np.sort(win*-1)*-1
        for val in vals:
            filt = win>val
            if np.sum(win[filt])>=th:
                idx = np.nonzero(filt)[0]
                ret.append(idx[-1]-idx[0])
                break
    return np.array(ret)
    
    
def peak_centroid(start: npt.NDArray[np.int_],end: npt.NDArray[np.int_],envelope: pcg.pcg_signal) -> tuple[npt.NDArray[np.int_],npt.NDArray[np.float_]]:
    start, end = _check_start_end(start,end)
    power = envelope.data**2
    loc, val = [], []
    for s, e in zip(start,end):
        win = power[s,e]
        th = np.sum(win)*0.5 #type: ignore
        centr = np.nonzero(np.cumsum(win)>th)[0][0]
        loc.append(centr)
        val.append(win[centr])
    return np.array(loc), np.array(val)

def max_freq(start: npt.NDArray[np.int_],end: npt.NDArray[np.int_],sig: pcg.pcg_signal,nfft: int=512) -> tuple[npt.NDArray[np.float_],npt.NDArray[np.float_]]:
    start, end = _check_start_end(start,end)
    spect = abs(fft.fft(sig.data[start:end],n=nfft)) #type: ignore
    spect = spect[:,:nfft//2]
    freqs = np.linspace(0,sig.fs//2,nfft//2)
    return freqs[np.argmax(spect,axis=1)], np.max(spect,axis=1)
    

def spectral_width(start: npt.NDArray[np.int_],end: npt.NDArray[np.int_],sig: pcg.pcg_signal, factor: float=0.7, nfft: int=512):
    start, end = _check_start_end(start,end)
    spect = abs(fft.fft(sig.data[start:end],n=nfft)) #type: ignore
    spect = spect[:,:nfft//2]
    power = spect**2
    sums = np.sum(power[start:end],axis=1)
    ret = []
    for summ, s, e in zip(sums, start, end):
        th = summ*factor
        win = power[s:e]
        vals = np.sort(win*-1)*-1
        for val in vals:
            filt = win>val
            if np.sum(win[filt])>=th:
                idx = np.nonzero(filt)[0]
                ret.append(idx[-1]-idx[0])
                break
    return np.array(ret)

def spectral_centroid(start: npt.NDArray[np.int_],end: npt.NDArray[np.int_],sig: pcg.pcg_signal,nfft: int=512) -> tuple[npt.NDArray[np.float_],npt.NDArray[np.float_]]:
    start, end = _check_start_end(start,end)
    spect = abs(fft.fft(sig.data[start:end],n=nfft)) #type: ignore
    spect = spect[:,:nfft//2]
    freqs = np.linspace(0,sig.fs//2,nfft//2)
    th = np.sum(spect,axis=1)*0.5
    locs = np.nonzero(np.cumsum(spect,axis=1)>th)[0][:,0] #? Nem tudom hogy jó-e lol
    return freqs[locs], spect[locs]
    

def cwt(start,end,sig):
    pass

def dwt(start,end,sig):
    pass

def katz_fd(start,end,sig):
    pass

def lyapunov(start,end,sig):
    pass

if __name__ == '__main__':
    print("Feature calculation")