import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import copy
from typing import Callable

class pcg_signal:
    """PCG signal object
    
    Attributes:
    ----------
        data (np.ndarray): signal data
        fs (int): sampling rate
        processing_log (list[str]): processing steps and parameters on the signal
    
    Methods:
    -------
        get_timelength: get the length of the signal in seconds
    """
    def __init__(self, data:npt.NDArray=np.array([]), fs:int=1, log:list[str]=["File read in"]) -> None:
        self.data = data.astype(float)
        self.fs = fs
        self.processing_log = log
        
    def __repr__(self) -> str:
        return f"PCG signal [{self.fs}] {self.processing_log}"

    def get_timelength(self) -> float:
        """Get legth of signal in seconds

        Returns:
            float: length of signal in seconds
        """
        return len(self.data)/self.fs
    
class process_pipeline:
    """Processing pipeline. One step's input is the prevous step's output
    
    Attributes:
    ----------
        steps (list[Callable | tuple[Callable, dict]]): List of steps as functions or function and parameters as keyword dictionary
    
    Methods:
    -------
        run(input): Run the process pipeline as described in steps
    """
    def __init__(self, *configs: Callable|tuple[Callable, dict[str,int|float|str]]) -> None:
        self.steps = []
        for k in configs:
            self.steps.append(k)
    
    def run(self, input: pcg_signal) -> pcg_signal:
        """Run the processing pipeline

        Args:
            input (pcg_signal): input signal

        Returns:
            pcg_signal: processed signal
        """
        out = input
        for step in self.steps:
            if type(step) is tuple:
                out = step[0](out,**step[1])
            else:
                out = step(out)
        return out

def zero_center(sig: pcg_signal) -> pcg_signal:
    """Center signal to zero

    Args:
        sig (pcg_signal): Input signal

    Returns:
        pcg_signal: Centered signal
    """
    ret_sig = copy.deepcopy(sig)
    ret_sig.data -= np.mean(ret_sig.data)
    ret_sig.processing_log.append("Zero center")
    return ret_sig

def unit_scale(sig: pcg_signal) -> pcg_signal:
    """Scale signal to [-1,1] interval

    Args:
        sig (pcg_signal): Input signal

    Returns:
        pcg_signal: Scaled signal
    """
    ret_sig = copy.deepcopy(sig)
    ret_sig.data /=np.max(np.abs(ret_sig.data))
    ret_sig.processing_log.append("Unit scale")
    return ret_sig

def std_scale(sig: pcg_signal) -> pcg_signal:
    """Scale signal to 1 std

    Args:
        sig (pcg_signal): Input signal

    Returns:
        pcg_signal: Scaled signal
    """
    ret_sig = copy.deepcopy(sig)
    ret_sig.data /=np.std(ret_sig.data)
    ret_sig.processing_log.append("Std scale")
    return ret_sig

def normalize(sig: pcg_signal) -> pcg_signal:
    """Center to zero and scale signal to [-1,1] interval

    Args:
        sig (pcg_signal): Input signal

    Returns:
        pcg_signal: Normalized signal
    """
    return zero_center(unit_scale(sig))

def plot(sig: pcg_signal) -> None:
    t = sig.get_timelength()
    plt.plot(np.linspace(0,t,len(sig.data)),sig.data)
    plt.title(sig.processing_log[-1])

def multiplot(*args):
    for sig in args:
        time = np.linspace(0,sig.get_timelength,len(sig.data))
        plt.plot(time,sig.data)

if __name__ == '__main__':
    print("Signal container and process pipeline builder")