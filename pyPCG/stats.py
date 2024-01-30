import numpy as np
import scipy.stats as sts

def trim_transform(data,trim_precent):
    k = round(len(data)*(trim_precent/100)/2)
    return data[k:-k]

def outlier_remove_transform(data,dist=3.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<dist]

def mean(data):
    return np.mean(data)

def std(data):
    return np.std(data)

def rms(data):
    return np.sqrt(np.mean(data**2))

def med(data):
    return np.median(data)

def skew(data):
    return sts.skew(data)

def kurt(data):
    return sts.kurtosis(data)

def max(data,k=1):
    s_data = np.sort(data)
    return s_data[-k-1:-1]

def min(data,k=1):
    s_data = np.sort(data)
    return s_data[k-1:k]

def percentile(data,perc=25):
    return np.percentile(data,perc)

def window_operator(data,win_size,fun,overlap_percent=0.5):
    step = win_size-round(win_size*overlap_percent)
    val,loc = [], []
    for i in range(0,len(data) - win_size,step):
        val.append(fun(data[i:i+win_size]))
        loc.append(i)
    return np.array(loc), np.array(val)

if __name__ == '__main__':
    print("Statistics")