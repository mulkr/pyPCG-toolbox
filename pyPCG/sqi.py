import numpy as np
import pyPCG as pcg
import scipy.stats as scistat
import pyPCG.preprocessing as preproc
from itertools import combinations
from math import log

# SQI definitions from: https://www.hindawi.com/journals/bmri/2021/7565398/


# https://en.wikipedia.org/wiki/Sample_entropy#Implementation
def _construct_templates(timeseries_data:list, m:int=2):
    num_windows = len(timeseries_data) - m + 1
    return [timeseries_data[x:x+m] for x in range(0, num_windows)]

def _get_matches(templates:list, r:float):
    return len(list(filter(lambda x: _is_match(x[0], x[1], r), combinations(templates, 2))))

def _is_match(template_1:list, template_2:list, r:float):
    return all([abs(x - y) < r for (x, y) in zip(template_1, template_2)])

def _sample_entropy(timeseries_data, window_size:int, r:float):
    B = _get_matches(_construct_templates(timeseries_data, window_size), r)
    A = _get_matches(_construct_templates(timeseries_data, window_size+1), r)
    return -log(A/B)

def _autocorr(x):
    result = np.correlate(x, x, mode='full')
    result = result/np.max(result)
    return result[result.size//2+1:]

def _time_autocorr(x,period):
    X = np.array(x)
    X = np.append(X,X[::-1])
    R_tt = []
    for i in range(len(x)):
        ofs = np.arange(0,len(x),period)
        win = X[i+ofs]
        R_tt.append(_autocorr(win))
    return np.array(R_tt)

def _cfsd(x,period,nfft,fs):
    R_tt = _time_autocorr(x,period)
    R_tt = R_tt-np.mean(R_tt)
    r = np.fft.rfft(R_tt,nfft,axis=0)
    alfa = np.fft.rfftfreq(nfft,1/fs)
    end = np.fft.fft(r,nfft,axis=1)
    return np.sum(np.abs(end),axis=1), alfa

def periodicity_score(sig: pcg.pcg_signal, period:int, nfft:int) -> float:
    gamma,alfa = _cfsd(sig.data,period,nfft,sig.fs)
    d = np.max(gamma[1:])/np.median(gamma[1:])
    return d

def sentropy(sig: pcg.pcg_signal, win:int=2, r:float|None=None) -> float:
    env = preproc.resample(preproc.envelope(sig),30)
    if r is None:
        r = np.std(env.data)*0.2 #type: ignore
    s = _sample_entropy(env.data,win,r) #type: ignore
    return s

def autocorr_max(sig: pcg.pcg_signal, bpm_min: float=100, bpm_max: float=200) -> float:
    ar = _autocorr(sig.data)
    expect_min = round((1/bpm_max)*sig.fs)
    expect_max = round((1/bpm_min)*sig.fs)
    m = np.max(ar[expect_min:expect_max])
    return m

def env_std(sig: pcg.pcg_signal) -> float:
    env = preproc.envelope(sig)
    s = np.std(env.data).astype(float)
    return s

def raw_kurt(sig: pcg.pcg_signal) -> float:
    return scistat.kurtosis(sig.data) #type: ignore