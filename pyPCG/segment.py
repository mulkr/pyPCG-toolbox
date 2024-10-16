from enum import Enum
import numpy as np
import numpy.typing as npt
import scipy.signal as sgnl
from pyPCG import pcg_signal
import pyPCG.lr_hsmm as lr_hsmm

def adv_peak(signal: pcg_signal, percent_th:float=0.5) -> tuple[npt.NDArray[np.float64],npt.NDArray[np.int_]]:
    """Adaptive peak detection, based on local maxima and following value drop

    Args:
        signal (pcg_signal): input signal to detect peaks (usually this is the envelope)
        percent_th (float, optional): percent drop in value to be considered a real peak. Defaults to 0.5.

    Returns:
        tuple[np.ndarray,np.ndarray]: detected peak values and their locations
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

def peak_sort_diff(peak_locs: npt.NDArray[np.int_]) -> tuple[npt.NDArray[np.int_],npt.NDArray[np.int_]]:
    """Sort detected peaks based on time differences.
    
    A short time difference corresponds with systole -> S1-S2, a long time difference corresponds with diastole -> S2-S1

    Args:
        peak_locs (np.ndarray): Detected peak locations in samples

    Raises:
        ValueError: Less than two peaks detected

    Returns:
        tuple[np.ndarray,np.ndarray]: S1 and S2 locations
    """
    if len(peak_locs)<2:
        raise ValueError("Too few peak locations (<2)")

    interval_diff = np.append(np.diff(peak_locs,2),[0,0]) # type: ignore
    s1_loc = peak_locs[interval_diff>0]
    s2_loc = peak_locs[interval_diff<0]
    return s1_loc, s2_loc

def peak_sort_centroid(peak_locs: npt.NDArray[np.int_],raw: pcg_signal,envelope: pcg_signal) -> tuple[npt.NDArray[np.int_],npt.NDArray[np.int_]]:
    from pyPCG.features import spectral_centroid
    starts, ends = segment_peaks(peak_locs,envelope,0.8,0.8)
    freqs, _= spectral_centroid(starts,ends,raw)
    mfreq = np.mean(freqs)
    
    s1_loc = (starts[freqs<mfreq]+ends[freqs<mfreq])//2
    s2_loc = (starts[freqs>mfreq]+ends[freqs>mfreq])//2
    return s1_loc, s2_loc

def segment_peaks(peak_locs: npt.NDArray[np.int_], envelope_signal: pcg_signal ,start_drop:float=0.6, end_drop:float=0.6) -> tuple[npt.NDArray[np.int_],npt.NDArray[np.int_]]:
    """Create start and end locations from the detected peaks based on the provided envelope.
    
    The relative drop in envelope value is marked as the start and end positions of the given heartsound

    Args:
        peak_locs (np.ndarray): detected peak locations in samples
        envelope_signal (pcg_signal): precalculated envelope of the signal (homomorphic recommended)
        start_drop (float, optional): precent drop in value for start location. Defaults to 0.6.
        end_drop (float, optional): percent drop in value for end location. Defaults to 0.6.

    Returns:
        tuple[np.ndarray,np.ndarray]: heartsound boundary locations
    """
    envelope = envelope_signal.data
    starts, ends = [],[]
    for peak_ind,peak_loc in enumerate(peak_locs):
        prev_peak = peak_locs[peak_ind-1] if peak_ind>0 else 0
        next_peak = peak_locs[peak_ind+1] if peak_ind+1<len(peak_locs) else len(envelope)
        start_th = envelope[peak_loc]*start_drop
        end_th = envelope[peak_loc]*end_drop
        prev_drop = np.nonzero(envelope[prev_peak:peak_loc]<start_th)[0]
        next_drop = np.nonzero(envelope[peak_loc:next_peak]<end_th)[0]
        if len(prev_drop)!=0 and len(next_drop)!=0:
            starts.append(prev_drop[-1]+prev_peak)
            ends.append(next_drop[0]+peak_loc)
    return np.array(starts), np.array(ends)

def load_hsmm(path:str) -> lr_hsmm.LR_HSMM:
    """Load pretrained LR-HSMM model.
    
    Note:
        Training is done internally, it is not recommended to use it right now

    Args:
        path (str): path to pretrained model json file

    Returns:
        lr_hsmm.LR_HSMM: pretrained model loaded in
    """
    model = lr_hsmm.LR_HSMM()
    model.load_model(path)
    return model

def segment_hsmm(model:lr_hsmm.LR_HSMM,signal:pcg_signal) -> npt.NDArray[np.float64]:
    """Use a trained LR-HSMM model to segment a pcg signal

    Args:
        model (lr_hsmm.LR_HSMM): trained LR-HSMM model
        signal (pcg_signal): input signal to be segmented

    Raises:
        ValueError: Samplerate discrepancy

    Returns:
        np.ndarray: heartcycle states
    """
    if(model.signal_fs!=signal.fs):
        raise ValueError(f"Unexpected signal samplerate {signal.fs}, LR-HSMM expects {model.signal_fs}")
    states, _ = model.segment_single(signal.data)
    return states

class heart_state(Enum):
    """Heart states enum"""
    S1 = 1
    SYS = 2
    S2 = 3
    DIA = 4
    unknown = 0
heart_state.S1.__doc__ = "First heartsound"
heart_state.S2.__doc__ = "Second heartsound"
heart_state.SYS.__doc__ = "Systole section"
heart_state.DIA.__doc__ = "Diastole section"
heart_state.unknown.__doc__ = "Default value, unknown state"

def convert_hsmm_states(states: npt.NDArray[np.float64], state_id: int|heart_state) -> tuple[npt.NDArray[np.int_],npt.NDArray[np.int_]]:
    """Convert selected LR-HSMM state to start and end times

    Args:
        states (np.ndarray): output states of LR-HSMM
        state_id (int, heart_state): selected state to convert

    Raises:
        ValueError: Unrecognized heart cycle state

    Returns:
        tuple[np.ndarray,np.ndarray]: state boundaries in samples
    """
    if (type(state_id) is int) and (state_id not in [0,1,2,3,4]):
        raise ValueError(f"Unrecognized heart cycle state: {state_id}")
    if type(state_id) is heart_state:
        state_id = int(state_id.value)
    select = np.zeros_like(states)
    select[states==state_id] = 1
    states_diff = np.diff(select)
    state_start = np.nonzero(states_diff>0)[0]
    state_end = np.nonzero(states_diff<0)[0]
    return state_start, state_end

if __name__ == '__main__':
    print("Segmenting and peak detection")