""" SQI definitions and some internal calculations based on:

H. Tang, M. Wang, Y. Hu, B. Guo, and T. Li, “Automated Signal Quality Assessment for Heart Sound Signal by Novel Features and Evaluation in Open Public Datasets,” BioMed Research International, vol. 2021. Hindawi Limited, pp. 1–15, Feb. 24, 2021.

https://www.hindawi.com/journals/bmri/2021/7565398/
"""

import numpy as np
import math
import scipy.stats as scistat
import scipy.fft as fft
import pyPCG.preprocessing as preproc
from scipy.linalg import hankel
from scipy.spatial.distance import pdist
from pyPCG import pcg_signal

def _sample_entropy_fast(x, m, r):
    # from Tang et al.
    x = x-np.mean(x)
    x = x/np.std(x)
    N = len(x)
    indm = hankel(np.arange(N-m), np.arange(N-m,N-1))
    inda = hankel(np.arange(N-m), np.arange(N-m,N))
    ym = x[indm]
    ya = x[inda]
    cheb = pdist(ym, "chebychev")
    cm = np.sum(cheb<=r)*2/ (ym.shape[0]*(ym.shape[0]-1))
    cheb = pdist(ya, "chebychev")
    ca = np.sum(cheb<=r)*2/ (ya.shape[0]*(ya.shape[0]-1))
    return -math.log(ca/cm)

def _autocorr(x):
    result = np.correlate(x, x, mode='full')
    result = result/np.max(result)
    return result[result.size//2+1:]

def _nextpow2(x):
    return math.ceil(math.log(x, 2))

def _fast_cfsd(sig,f1,f2,k):
    # from Tang et al.
    w = np.exp(-1j*2*np.pi*(f2-f1)/(k*sig.fs))
    a = np.exp(1j*2*np.pi*f1/sig.fs)
    x = preproc.envelope(sig).data
    x = x-np.mean(x)
    m = len(x)
    nfft = 2**_nextpow2(m+k-1)
    kk = np.arange(-m,max(k,m))
    kk2 = (kk**2)/2
    ww = w**kk2
    nn = np.arange(m)
    aa = a**(-nn)
    aa = aa*ww[m+nn]
    y = x * aa
    fy = fft.fft(y,nfft)
    fv = fft.fft(1/ww[:k-1+m],nfft)
    fy = fy * fv #type: ignore
    g = fft.ifft(fy)
    g = g[m:m+k-1] * ww[m:m+k-1]
    return np.abs(g)

def periodicity_score(sig: pcg_signal, f1:float=0.3, f2:float=2.5, k:int=200) -> float:
    """Calculate SQI based on cylcic periodicity.
    
    Larger -> Better

    Args:
        sig (pcg_signal): input signal (envelope)
        f1 (float, optional): min Hz to search. Defaults to 0.3.
        f2 (float, optional): max Hz to search. Defaults to 2.5.
        k (int, optional): weight parameter. Defaults to 200.

    Returns:
        float: Periodicity score
    """
    gamma = _fast_cfsd(sig,f1,f2,k)
    return np.max(gamma)/np.median(gamma)

def sentropy(sig: pcg_signal, win:int=2, r:float=0.2, precalc:bool=False) -> float:
    """Calculate sample entropy SQI
    
    Smaller -> Better

    Args:
        sig (pcg_signal): input signal
        win (int, optional): window size. Defaults to 2.
        r (float, optional): weight parameter. Defaults to 0.2.
        precalc (bool, optional): is the input a precalculated envelope. Defaults to False.

    Returns:
        float: sample entropy
    """
    temp = preproc.envelope(sig) if not precalc else sig
    env = preproc.resample(temp,30)
    return _sample_entropy_fast(env.data,win,r)

def autocorr_max(sig: pcg_signal, bpm_min: float=100, bpm_max: float=200) -> float:
    """Calculate SQI based on autocorrelation.
    
    Larger -> Better

    Args:
        sig (pcg_signal): input signal (envelope)
        bpm_min (float, optional): minimum expected BPM. Defaults to 100.
        bpm_max (float, optional): maximum expected BPM. Defaults to 200.

    Returns:
        float: autocorrelation maximum
    """
    ar = _autocorr(sig.data)
    expect_min = round((1/bpm_max)*sig.fs)
    expect_max = round((1/bpm_min)*sig.fs)
    m = np.max(ar[expect_min:expect_max])
    return m

def env_std(sig: pcg_signal, precalc: bool=False) -> float:
    """Calculate standard deviation of the envelope values
    
    Smaller -> Better

    Args:
        sig (pcg_signal): input signal
        precalc (bool, optional): is the input a precalculated envelope. Defaults to False.

    Returns:
        float: standard deviation of envelope
    """
    env = preproc.envelope(sig) if not precalc else sig
    s = np.std(env.data).astype(float)
    return s

def raw_kurt(sig: pcg_signal) -> float:
    """Calculate kurtosis of raw signal values
    
    Larger -> Better

    Args:
        sig (pcg_signal): input signal

    Returns:
        float: kurtosis of signal
    """
    return scistat.kurtosis(sig.data) #type: ignore