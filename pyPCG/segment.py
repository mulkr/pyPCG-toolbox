import numpy as np
import scipy.signal as signal

def adv_peak(sig,percent_th=0.5):
    """Peak detection by local maxima and following drop in value
    """
    peaks = []
    loc_max,_ = signal.find_peaks(sig)
    for loc_ind in range(len(loc_max)):
        search_start = loc_max[loc_ind]
        search_end = loc_max[loc_ind+1] if loc_ind+1 < len(loc_max) else len(sig)
        th_val = sig[search_start]*percent_th
        if np.any(sig[search_start:search_end]<th_val):
            peaks.append(search_start)
    return np.array(sig[peaks]), np.array(peaks)

def peak_sort_diff(peak_locs):
    if len(peak_locs)<2:
        raise ValueError("Too few peak locations (<2)")
    
    interval_diff = np.append(np.diff(peak_locs,2),[0,0]) # type: ignore
    s1_loc = peak_locs[interval_diff>0]
    s2_loc = peak_locs[interval_diff<0]
    return s1_loc, s2_loc

def segment_peaks(peak_locs,envelope,start_drop=0.6,end_drop=0.6):
    starts, ends = [],[]
    for peak_ind,peak_loc in enumerate(peak_locs):
        prev_peak = peak_locs[peak_ind-1] if peak_ind>0 else 0
        next_peak = peak_locs[peak_ind+1] if peak_ind+1<len(peak_locs) else len(envelope)
        start_th = envelope[peak_loc]*start_drop
        end_th = envelope[peak_loc]*end_drop
        starts.append(np.nonzero(envelope[prev_peak:peak_loc]<start_th)[0][-1]+prev_peak)
        ends.append(np.nonzero(envelope[peak_loc:next_peak]<end_th)[0][0]+peak_loc)
    return np.array(starts), np.array(ends)