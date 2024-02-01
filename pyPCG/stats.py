import numpy as np
import numpy.typing as npt
import scipy.stats as sts
from scipy.special import erfcinv
from typing import Callable

def trim_transform(data: npt.NDArray[np.float_], trim_precent: float) -> npt.NDArray[np.float_]:
    """Trim the upper and lower percentage of values

    Args:
        data (np.ndarray): input data to trim
        trim_precent (float): percentage to trim away

    Returns:
        np.ndarray: trimmed values
    """
    return sts.trimboth(data,trim_precent/100)

def outlier_remove_transform(data: npt.NDArray[np.float_], dist: float=3.0) -> npt.NDArray[np.float_]:
    """Remove outliers based on the MAD (median of absolute differences)

    Args:
        data (np.ndarray): input data
        dist (float, optional): MAD score threshold. Defaults to 3.0.

    Returns:
        np.ndarray: data without outliers
    """
    d = np.abs(data - np.median(data))
    c = -1/(np.sqrt(2)*erfcinv(3/2))
    mdev = c*np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<dist]

def mean(data: npt.NDArray[np.float_]) -> npt.NDArray[np.float_] | np.float_:
    """Calculate mean of inputs

    Args:
        data (np.ndarray): input data (can be 2 dimensional array)

    Returns:
        np.ndarray | float: mean value of data, if input is 2D then return the value along of 1st axis
    """
    if len(data.shape) == 1:
        return np.mean(data)
    else: return np.mean(data,axis=1)

def std(data: npt.NDArray[np.float_]) -> npt.NDArray[np.float_] | np.float_:
    """Calculate standard deviation of inputs

    Args:
        data (np.ndarray): input data (can be 2 dimensional array)

    Returns:
        np.ndarray | float: standard deviation of data, if input is 2D then return the value along of 1st axis
    """
    if len(data.shape) == 1:
        return np.std(data)
    else: return np.std(data,axis=1)

def rms(data: npt.NDArray[np.float_]) -> npt.NDArray[np.float_] | np.float_:
    """Calculate root mean square of inputs

    Args:
        data (np.ndarray): input data (can be 2 dimensional array)

    Returns:
        np.ndarray | float: root mean square of data, if input is 2D then return the value along of 1st axis
    """
    if len(data.shape) == 1:
        return np.sqrt(np.mean(data**2))
    else: return np.sqrt(np.mean(data**2,axis=1))

def med(data: npt.NDArray[np.float_]) -> npt.NDArray[np.float_] | np.float_:
    """Calculate median of inputs

    Args:
        data (np.ndarray): input data (can be 2 dimensional array)

    Returns:
        np.ndarray | float: median of data, if input is 2D then return the value along of 1st axis
    """
    if len(data.shape) == 1:
        return np.median(data)
    else: return np.median(data,axis=1)

def skew(data: npt.NDArray[np.float_]) -> npt.NDArray[np.float_] | np.float_:
    """Calculate skewness of inputs

    Args:
        data (np.ndarray): input data (can be 2 dimensional array)

    Returns:
        np.ndarray | float: skewness of data, if input is 2D then return the value along of 1st axis
    """
    if len(data.shape) == 1:
        return sts.skew(data)
    else: return sts.skew(data,axis=1)

def kurt(data: npt.NDArray[np.float_]) -> npt.NDArray[np.float_] | np.float_:
    """Calculate kurtosis of inputs

    Args:
        data (np.ndarray): input data (can be 2 dimensional array)

    Returns:
        np.ndarray | float: kurtosis of data, if input is 2D then return the value along of 1st axis
    """
    if len(data.shape) == 1:
        return sts.kurtosis(data).astype(np.float_)
    else: return sts.kurtosis(data,axis=1) #type: ignore

def max(data: npt.NDArray[np.float_],k: int=1) -> npt.NDArray[np.float_] | np.float_:
    """Get maximum values from input

    Args:
        data (np.ndarray): input data
        k (int, optional): number of largest values to return. Defaults to 1.

    Returns:
        np.ndarray | float: maximum value(s)
    """
    s_data = np.sort(data)
    select = s_data[-k-1:-1]
    return select[::-1]

def min(data: npt.NDArray[np.float_],k: int=1) -> npt.NDArray[np.float_] | np.float_:
    """Get minimum values from input

    Args:
        data (np.ndarray): input data
        k (int, optional): number of smallest values to return. Defaults to 1.

    Returns:
        np.ndarray | float: minimum value(s)
    """
    s_data = np.sort(data)
    return s_data[0:k]

def percentile(data: npt.NDArray[np.float_], perc: float=25) -> npt.NDArray[np.float_] | np.float_:
    """Calculate given percentile of inputs

    Args:
        data (np.ndarray): input data (can be 2 dimensional array)
        perc (float): selected percentile to calculate. Defaults to 25.

    Returns:
        np.ndarray | float: given percentile of data, if input is 2D then return the value along of 1st axis
    """
    if len(data.shape) == 1:
        return np.percentile(data,perc)
    else: return np.percentile(data,perc,axis=1)

def window_operator(data: npt.NDArray[np.float_],win_size: int,fun: Callable,overlap_percent: float=0.5) -> tuple[npt.NDArray[np.int_],npt.NDArray[np.float_]]:
    """Apply given statistical function over a sliding window on the input

    Args:
        data (np.ndarray): input data
        win_size (int): window size
        fun (Callable): statistical function to apply
        overlap_percent (float, optional): window overlap as a ratio to the window size. Defaults to 0.5.

    Returns:
        np.ndarray: window sample locations (usually used as time dimension)
        np.ndarray: calculated values in the windows
    """
    step = win_size-round(win_size*overlap_percent)
    val,loc = [], []
    for i in range(0,len(data) - win_size,step):
        val.append(fun(data[i:i+win_size]))
        loc.append(i)
    return np.array(loc), np.array(val)

if __name__ == '__main__':
    print("Statistics")