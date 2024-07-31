import warnings
import nolds
import numpy as np
import numpy.typing as npt
import scipy.ndimage as ndimage
import scipy.fft as fft
import pywt
from typing import Callable, Literal, TypedDict
from pyPCG import pcg_signal

def _check_start_end(start,end):
    if len(start) != len(end):
        warnings.warn("Start and end arrays not the same size. Converting to same size...")
        if len(start) > len(end):
            start = start[:len(end)]
        else:
            end = end[:len(start)]
    return start, end

def time_delta(start: npt.NDArray[np.int_],end: npt.NDArray[np.int_], sig: pcg_signal) -> npt.NDArray[np.float64]:
    """Calculate time differences between pairs of points

    Args:
        start (np.ndarray): start points in samples
        end (np.ndarray): end points in samples
        sig (pcg_signal): input signal

    Returns:
        np.ndarray: time differences in seconds
    """
    start, end = _check_start_end(start,end)
    return (end-start)/sig.fs

def ramp_time(start: npt.NDArray[np.int_],end: npt.NDArray[np.int_],envelope: pcg_signal,type: str="onset") -> npt.NDArray[np.float64]:
    """Calculate ramp time (onset or exit), the time difference between the boundary and peak

    Args:
        start (np.ndarray): start points in samples
        end (np.ndarray): end points in samples
        envelope (pcg_signal): precalculated envelope signal
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

def zero_cross_rate(start: npt.NDArray[np.int_],end: npt.NDArray[np.int_],sig: pcg_signal) -> npt.NDArray[np.float64]:
    """Calculate zero cross rate

    Args:
        start (np.ndarray): start times in samples
        end (np.ndarray): end times in samples
        sig (pcg_signal): input signal

    Returns:
        np.ndarray: zero cross rates
    """
    start, end = _check_start_end(start,end)
    ret = []
    for s, e in zip(start,end):
        cross_log = np.abs(np.diff(np.sign(sig.data[s:e])))>1
        cross_ind = np.nonzero(cross_log)[0]
        crosses = len(cross_ind)+0.5
        ret.append(crosses/(e-s))
    return np.array(ret)

def peak_spread(start: npt.NDArray[np.int_],end: npt.NDArray[np.int_],envelope: pcg_signal,factor: float=0.7) -> npt.NDArray[np.int_]:
    """Calculate peak spread, the amount of area under the peak with a given percentage of the total and time differences between the beginning and end

    Args:
        start (np.ndarray): start times in samples
        end (np.ndarray): end times in samples
        envelope (pcg_signal): input envelope signal
        factor (float, optional): percentage of total area. Defaults to 0.7.

    Returns:
        np.ndarray: time difference between the beginning and end of the percentage area in ms
    """
    start, end = _check_start_end(start,end)
    ret = []
    for s, e in zip(start, end):
        win = envelope.data[s:e]
        th = np.sum(win)*factor
        grow_left = True
        offset_left, offset_right = 0, 0
        peak = np.argmax(win)
        for _ in win:
            if np.sum(win[peak-offset_left:peak+offset_right])>=th:
                break
            if grow_left and peak-offset_left>0:
                offset_left += 1
                grow_left = not grow_left
                continue
            elif not grow_left and peak+offset_right<len(win)-2:
                offset_right += 1
                grow_left = not grow_left
                continue
        spread = offset_right-offset_left
        ret.append(spread/envelope.fs * 1000)
    return np.array(ret)

def peak_centroid(start: npt.NDArray[np.int_],end: npt.NDArray[np.int_],envelope: pcg_signal) -> tuple[npt.NDArray[np.int_],npt.NDArray[np.float64]]:
    """Calculate centroid (center of mass) of the envelope

    Args:
        start (np.ndarray): start times in samples
        end (np.ndarray): end times in samples
        envelope (pcg_signal): input envelope signal

    Returns:
        np.ndarray: time delays from start to centroid
        np.ndarray: envelope values at centroid
    """
    start, end = _check_start_end(start,end)
    loc, val = [], []
    for s, e in zip(start,end):
        win = envelope.data[s:e]
        th = np.sum(win)*0.5 #type: ignore
        centr = np.nonzero(np.cumsum(win)>th)[0][0]
        loc.append(centr)
        val.append(win[centr])
    return np.array(loc), np.array(val)

def peak_width(start: npt.NDArray[np.int_],end: npt.NDArray[np.int_],envelope: pcg_signal,factor: float=0.7) -> npt.NDArray[np.int_]:
    """Calculate conventional width of the given peak, the difference between the preceding and succeeding time locations where the value is at a given proportion of the peak value

    Args:
        start (np.ndarray): start times in samples
        end (np.ndarray): end times in samples
        envelope (pcg_signal): input envelope signal
        factor (float, optional): proportionality factor. Defaults to 0.7.

    Returns:
        np.ndarray: peak width in ms
    """
    start, end = _check_start_end(start,end)
    ret = []
    for s,e in zip(start,end):
        win = envelope.data[s:e]
        loc = np.argmax(win)
        th = win[loc]*factor
        w_prev = np.nonzero(win[:loc]<th)[0]
        w_next = np.nonzero(win[loc:]<th)[0]
        w_s = w_prev[-1] if len(w_prev)!=0 else 0
        w_e = w_next[0]+loc if len(w_next)!=0 else e-s
        width = w_e-w_s
        ret.append(width/envelope.fs * 1000)
    return np.array(ret)

def max_freq(start: npt.NDArray[np.int_],end: npt.NDArray[np.int_],sig: pcg_signal,nfft: int=512) -> tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]]:
    """Calculate frequency with maximum amplitude of the segment

    Args:
        start (np.ndarray): start times in samples
        end (np.ndarray): end times in samples
        sig (pcg_signal): input signal
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

def spectral_spread(start: npt.NDArray[np.int_],end: npt.NDArray[np.int_],sig: pcg_signal, factor: float=0.7, nfft: int=512) -> npt.NDArray[np.int_]:
    """Calculate spectral spread of the segments, percentage of the total power of the segment and the frequency difference between the beginning and end of the calculated area
    
    For a more detailed definition, see `peak_spread`.

    Args:
        start (np.ndarray): start times in samples
        end (np.ndarray): end times in samples
        sig (pcg_signal): input signal
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
        grow_left = True
        offset_left, offset_right = 0, 0
        peak = np.argmax(power)
        for _ in power:
            if np.sum(power[peak-offset_left:peak+offset_right])>=th:
                break
            if grow_left and peak-offset_left>0:
                offset_left += 1
                grow_left = not grow_left
                continue
            elif not grow_left and peak+offset_right<len(power)-2:
                offset_right += 1
                grow_left = not grow_left
                continue
        spread = offset_right-offset_left
        ret.append(spread)
    return np.array(ret)

def spectral_centroid(start: npt.NDArray[np.int_],end: npt.NDArray[np.int_],sig: pcg_signal,nfft: int=512) -> tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]]:
    """Calculate spectral centroid (center of mass)
    
    For a more detailed definition, see `peak_centroid`.

    Args:
        start (np.ndarray): start times in samples
        end (np.ndarray): end times in samples
        sig (pcg_signal): input signal
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

def spectral_width(start: npt.NDArray[np.int_],end: npt.NDArray[np.int_],sig: pcg_signal, factor: float=0.7, nfft: int=512) -> npt.NDArray[np.int_]:
    """Calculate conventional width of the spectrum, the difference between the preceding and succeeding frequency locations where the value is at a given proportion of the maximum value
    
    For a more detailed definition, see `peak_width`.

    Args:
        start (np.ndarray): start times in samples
        end (np.ndarray): end times in samples
        sig (pcg_signal): input signal
        factor (float, optional): proportionality factor. Defaults to 0.7.
        nfft (int, optional): fft width parameter. Defaults to 512

    Returns:
        np.ndarray: spectral width
    """
    start, end = _check_start_end(start,end)
    ret = []
    for s, e in zip(start, end):
        spect = abs(fft.fft(sig.data[s:e],n=nfft)) #type: ignore
        spect = spect[:nfft//2]
        power = spect**2
        loc = np.argmax(power)
        val = power[loc]
        th = val*factor
        w_prev = np.nonzero(power[:loc]<th)[0]
        w_next = np.nonzero(power[loc:]<th)[0]
        w_s = w_prev[-1] if len(w_prev)!=0 else 0
        w_e = w_next[0]+loc if len(w_next)!=0 else len(power)
        ret.append(w_e-w_s)
    return np.array(ret)

def spectrum_raw(start: npt.NDArray[np.int_],end: npt.NDArray[np.int_],sig: pcg_signal,nfft:int=512) -> npt.NDArray[np.float64]:
    """Calculate spectra of all input segments

    Args:
        start (np.ndarray): start times in samples
        end (np.ndarray): end times in samples
        sig (pcg_signal): input signal
        nfft (int, optional): fft width parameter. Defaults to 512.

    Returns:
        np.ndarray: spectrum of each segment (2D)
    """
    start, end = _check_start_end(start,end)
    ret = []
    for s,e in zip(start,end):
        spect = abs(fft.fft(sig.data[s:e],n=nfft)) #type: ignore
        spect = spect[:nfft//2]
        ret.append(spect)
    return np.array(ret)

def max_cwt(start: npt.NDArray[np.int_],end: npt.NDArray[np.int_],sig: pcg_signal) -> tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]]:
    """Calculate the maximum cwt coeffitient location in both directions

    Args:
        start (np.ndarray): start times in samples
        end (np.ndarray): end times in samples
        sig (pcg_signal): input signal

    Returns:
        np.ndarray: maximum locations in time
        np.ndarray: maximum locations in frequency
    """
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

def cwt_peakdist(start: npt.NDArray[np.int_],end: npt.NDArray[np.int_],sig: pcg_signal, size:int=8) -> npt.NDArray[np.float64]:
    """Calculate the distance of the largest two peaks in the continuous wavelet transform of the segment.
    
    Note:
        This currently has parity problems with Matlab's CWT function

    Args:
        start (np.ndarray): start times in samples
        end (np.ndarray): end times in samples
        sig (pcg_signal): input signal
        size (int, optional): size of maximum filter, i.e. peak neighborhood to check local maximum. Defaults to 8.

    Returns:
        np.ndarray: distance of two largest peaks in CWT, if only one peak is detected the value is 0
    """
    warnings.warn("CWT calculation done with PyWT, which has parity problems with Matlab")
    start, end = _check_start_end(start,end)
    ret = []
    for s,e in zip(start,end):
        coef, fr = pywt.cwt(sig.data[s:e],np.arange(1,40),"cmor1.0-1.5",sampling_period=1/sig.fs)
        coef = np.abs(coef)
        ind = ndimage.maximum_filter(coef,size)==coef
        val = coef[ind]
        asc = np.argsort(val)[::-1]
        x,y = np.nonzero(ind)
        max_x = x[asc[:2]]
        max_y = y[asc[:2]]
        if len(max_x)<2 or len(max_y)<2:
            ret.append(0)
        else:
            ret.append(np.sqrt((max_x[0]-max_x[1])**2+(max_y[0]-max_y[1])**2))
    return np.array(ret)

def dwt_intensity(start: npt.NDArray[np.int_],end: npt.NDArray[np.int_],sig: pcg_signal,wt_family:str="db6",decomp_level:int=4,select_level:int=2) -> npt.NDArray[np.float64]:
    """Calculate the 'intensity' of discrete wavelet transform

    Args:
        start (np.ndarray): start times in samples
        end (np.ndarray): end times in samples
        sig (pcg_signal): input signal
        wt_family (str, optional): wavelet family to use for decomposition. Defaults to "db6".
        decomp_level (int, optional): wavelet decomposition level. Defaults to 4.
        select_level (int, optional): selected decomposition level. Defaults to 2.

    Returns:
        np.ndarray: intensities of selected wavelet detail
    """
    start, end = _check_start_end(start,end)
    ret = []
    decomp = pywt.wavedec(sig.data,wt_family,level=decomp_level)
    select = decomp[-select_level]
    scale = 2**select_level
    for s,e in zip(start,end):
        win = select[s//scale:e//scale]
        intens = np.sqrt(np.sum(win**2))
        ret.append(intens)
    return np.array(ret)

def dwt_entropy(start: npt.NDArray[np.int_],end: npt.NDArray[np.int_],sig: pcg_signal,wt_family:str="db6",decomp_level:int=4,select_level:int=2)  -> npt.NDArray[np.float64]:
    """Calculate Shannon entropy of selected discrete wavelet decomposition level

    Args:
        start (np.ndarray): start times in samples
        end (np.ndarray): end times in samples
        sig (pcg_signal): input signal
        wt_family (str, optional): wavelet family to use for decomposition. Defaults to "db6".
        decomp_level (int, optional): wavelet decomposition level. Defaults to 4.
        select_level (int, optional): selected decomposition level. Defaults to 2.

    Returns:
        np.ndarray: entropies of selected wavelet detail
    """
    start, end = _check_start_end(start,end)
    ret = []
    decomp = pywt.wavedec(sig.data,wt_family,level=decomp_level) #wavedec first, then window?
    select = decomp[-select_level]
    scale = 2**select_level
    for s,e in zip(start,end):
        win = select[s//scale:e//scale]
        ent = np.sqrt(-np.sum(win**2*np.log2(win**2)))
        ret.append(ent)
    return np.array(ret)

def katz_fd(start: npt.NDArray[np.int_],end: npt.NDArray[np.int_],sig: pcg_signal) -> npt.NDArray[np.float64]:
    """Calculate Katz fractal dimension

    Args:
        start (np.ndarray): start times in samples
        end (np.ndarray): end times in samples
        sig (pcg_signal): input signal

    Returns:
        np.ndarray: Katz fractal dimension estimates
    """
    start, end = _check_start_end(start,end)
    ret = []
    for s,e in zip(start,end):
        win = sig.data[s:e]
        ind = np.arange(len(win)-1)
        A = np.stack((ind,win[:-1]))
        B = np.stack((ind+1,win[1:]))
        dists = np.linalg.norm(B-A,axis=0)
        ind = np.arange(len(win))
        A = np.stack((ind,win))
        first = np.reshape([0,win[0]],(2,1))
        aux_d = np.linalg.norm(A-first,axis=0)
        L = np.sum(dists)
        a = np.mean(dists)
        d = np.max(aux_d)
        D = np.log10(L/a)/np.log10(d/a)
        ret.append(D)
    return np.array(ret)

def lyapunov(start: npt.NDArray[np.int_],end: npt.NDArray[np.int_],sig: pcg_signal,dim:int=4,lag:int=3) -> npt.NDArray[np.float64]:
    """Estimate Lyapunov exponent with nolds
    
    Note:
        This currently has parity problems with Matlab's lyapunovExponent function

    Args:
        start (np.ndarray): start times in samples
        end (np.ndarray): end times in samples
        sig (pcg_signal): input signal
        dim (int, optional): Lyapunov embedding dimension. Defaults to 4.
        lag (int, optional): time-lag (tau) [samples]. Defaults to 3.

    Returns:
        np.ndarray: estimated Lyapunov exponents
    """
    warnings.warn("Lyapunov exponent calculation done with nolds, which has parity problems with Matlab")
    start, end = _check_start_end(start,end)
    ret = []
    for s,e in zip(start,end):
        win = sig.data[s:e]
        ly = nolds.lyap_r(win,emb_dim=dim,lag=lag,tau=1/sig.fs) #type: ignore
        ret.append(ly)
    return np.array(ret)

class _config(TypedDict):
    calc_fun: Callable
    """funtcion for calculation"""
    name: str
    """name of feature"""
    input: Literal["env"]|Literal["raw"]
    """input signal type, `env` for envelope, `raw` for non-envelope signal"""

class feature_config(_config,total=False):
    """Type to hold feature calculation configs"""
    params: dict[str,int|float|str]
    """parameters to pass to feature calculation function"""

class feature_group:
    """Group feature calculations together for reuse
    
    Attributes:
        feature_configs (list[feature_config]): List of configurations for feature calculation.
        
    Example:
        Create a feature group:
        
        For an easier experience, use the `feature_config` type
        
        >>> import pyPCG.features as ftr
        >>> ftr_1 = {"calc_fun":ftr.time_delta,"name":"length","input":"raw"}
        >>> ftr_2 = {"calc_fun":ftr.ramp_time,"name":"onset","input":"env"}
        >>> ftr_3 = {"calc_fun":ftr.ramp_time,"name":"exit","input":"env","params":{"type":"exit"}}
        >>> my_group = ftr.feature_group(ftr_1,ftr_2,ftr_3)
        
        Run the created group with a regular signal `raw_sig` and its envelope `env_sig` for all detected S1 events (boundaries given as `s1_start`, `s1_end`):
        
        >>> my_features = my_group.run(raw_sig,env_sig,s1_start,s1_end)
    """
    def __init__(self,*configs: feature_config) -> None:
        self.feature_configs = []
        for config in configs:
            self.feature_configs.append(config)

    def run(self, raw_sig: pcg_signal, env_sig: pcg_signal, starts: npt.NDArray[np.int_], ends: npt.NDArray[np.int_]) -> dict[str,npt.NDArray[np.float64]]:
        """Run feature calculations on the input signal based on the elements of feature_configs

        Args:
            raw_sig (pcg_signal): input signal
            env_sig (pcg_signal): envelope of input signal
            starts (np.ndarray): start times in samples
            ends (np.ndarray): end times in samples

        Returns:
            dict[str,np.ndarray]: calculated features, with the names given in `feature_configs` field
        """
        ret_dict = {}
        for ftr in self.feature_configs:
            in_sig = raw_sig if ftr["input"] == "raw" else env_sig
            calc = ftr["calc_fun"](starts,ends,in_sig) if len(ftr) == 3 else ftr["calc_fun"](starts,ends,in_sig,**ftr["params"])
            ret_dict[ftr["name"]] = calc[0] if type(calc) is tuple else calc
        return ret_dict

if __name__ == '__main__':
    print("Feature calculation")