import numpy as np
import scipy.signal as sgnl
import pyPCG as pcg
import pyPCG.lr_hsmm as hsmm

def adv_peak(signal: pcg.pcg_signal, percent_th:float=0.5) -> tuple[np.ndarray,np.ndarray]:
    """Adaptive peak detection, based on local maxima and following value drop

    Args:
        signal (pcg.pcg_signal): input signal to detect peaks (usually this is the envelope)
        percent_th (float, optional): percent drop in value to be considered a real peak. Defaults to 0.5.

    Returns:
        np.ndarray: detected peak values
        np.ndarray: detected peak locations
    """
    peaks = []
    sig = signal.data
    loc_max,_ = sgnl.find_peaks(sig)
    for loc_ind in range(len(loc_max)):
        search_start = loc_max[loc_ind]
        search_end = loc_max[loc_ind+1] if loc_ind+1 < len(loc_max) else len(sig)
        th_val = sig[search_start]*percent_th
        if np.any(sig[search_start:search_end]<th_val):
            peaks.append(search_start)
    return np.array(sig[peaks]), np.array(peaks)

def peak_sort_diff(peak_locs: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    """Sort detected peaks based on time differences.
    A short time difference corresponds with systole -> S1-S2, a long time difference corresponds with diastole -> S2-S1

    Args:
        peak_locs (np.ndarray): Detected peak locations in samples

    Raises:
        ValueError: Less than two peaks detected

    Returns:
        np.ndarray: S1 locations
        np.ndarray: S2 locations
    """
    if len(peak_locs)<2:
        raise ValueError("Too few peak locations (<2)")
    
    interval_diff = np.append(np.diff(peak_locs,2),[0,0]) # type: ignore
    s1_loc = peak_locs[interval_diff>0]
    s2_loc = peak_locs[interval_diff<0]
    return s1_loc, s2_loc

def segment_peaks(peak_locs: np.ndarray, envelope_signal: pcg.pcg_signal ,start_drop:float=0.6, end_drop:float=0.6) -> tuple[np.ndarray,np.ndarray]:
    """Create start and end locations from the detected peaks based on the provided envelope.
    The relative drop in envelope value is marked as the start and end positions of the given heartsound

    Args:
        peak_locs (np.ndarray): detected peak locations in samples
        envelope_signal (pcg.pcg_signal): precalculated envelope of the signal (homomorphic recommended)
        start_drop (float, optional): precent drop in value for start location. Defaults to 0.6.
        end_drop (float, optional): percent drop in value for end location. Defaults to 0.6.

    Returns:
        np.ndarray: heartsound start locations
        np.ndarray: heartsound end locations
    """
    envelope = envelope_signal.data
    starts, ends = [],[]
    for peak_ind,peak_loc in enumerate(peak_locs):
        prev_peak = peak_locs[peak_ind-1] if peak_ind>0 else 0
        next_peak = peak_locs[peak_ind+1] if peak_ind+1<len(peak_locs) else len(envelope)
        start_th = envelope[peak_loc]*start_drop
        end_th = envelope[peak_loc]*end_drop
        starts.append(np.nonzero(envelope[prev_peak:peak_loc]<start_th)[0][-1]+prev_peak)
        ends.append(np.nonzero(envelope[peak_loc:next_peak]<end_th)[0][0]+peak_loc)
    return np.array(starts), np.array(ends)

def load_hsmm(path:str) -> hsmm.LR_HSMM:
    """Load pretrained LR-HSMM model. (Training is done internally, it is not recommended to use it right now)

    Args:
        path (str): path to pretrained model json file

    Returns:
        hsmm.LR_HSMM: pretrained model loaded in
    """
    model = hsmm.LR_HSMM()
    model.load_model(path)
    return model

def segment_hsmm(model:hsmm.LR_HSMM,signal:pcg.pcg_signal) -> np.ndarray:
    """Use a trained LR-HSMM model to segment a pcg signal

    Args:
        model (hsmm.LR_HSMM): trained LR-HSMM model
        signal (pcg.pcg_signal): input signal to be segmented

    Raises:
        ValueError: Samplerate discrepancy

    Returns:
        np.ndarray: heartcycle states [1-S1, 2-sys, 3-S2, 4-dia]
    """
    if(model.signal_fs!=signal.fs):
        raise ValueError(f"Unexpected signal samplerate {signal.fs}, LR-HSMM expects {model.signal_fs}")
    states, _ = model.segment_single(signal.data)
    return states