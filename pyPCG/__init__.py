import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import copy
from typing import Callable

__version__ = "0.1-a"

class pcg_signal:
    """PCG signal object. This is used to store the signal data, as well as its sampling rate. The object also includes a processing log to track the processing steps and settings done to the signal.
    
    When using the plot function the last step is used as a default figure title.
    
    All processing steps require an object like this as input, and their output is also a signal object.
    
    Attributes:
        data (np.ndarray): signal data
        fs (int): sampling rate in Hz
        processing_log (list[str]): processing steps and parameters on the signal
    
    Example:
        Creating a signal from numpy array:
        
        >>> import pyPCG
        >>> import numpy as np 
        >>> sig = np.arange(10)
        >>> signal = pyPCG.pcg_signal(sig,100,["Example signal"])
        >>> print(signal)
        PCG signal [0.1s 100Hz] ['Example signal']
        
        Using `pyPCG.io.read_signal_file()`:
        
        >>> import pyPCG
        >>> import pyPCG.io
        >>> # reading in a 60 second long wav file with 333 Hz sampling rate
        >>> sig, fs = pyPCG.io.read_signal_file("example.wav","wav")
        >>> signal = pyPCG.pcg_signal(sig,fs)
        >>> # or optionally using the splat operator
        >>> signal = pyPCG.pcg_signal(*pyPCG.io.read_signal_file("example.wav","wav"))
        >>> print(signal)
        PCG signal [60s 333Hz] ['File read in']
    """
    def __init__(self, data:npt.NDArray[np.int_|np.float_], fs:int=1, log:list[str]|None=None) -> None:
        """Create PCG signal object

        Args:
            data (np.ndarray): raw signal data
            fs (int, optional): sampling rate. Defaults to 1.
            log (list[str] | None, optional): processing log. Defaults to None.

        Raises:
            ValueError: Tried to create PCG signal with no data
        """
        if len(data) == 0:
            raise ValueError("Tried to create PCG signal with no data")
        self.data = data.astype(float)
        self.fs = fs
        self.processing_log = ["File read in"] if log is None else log
        
    def __repr__(self) -> str:
        return f"PCG signal [{self.get_timelength()}s {self.fs}Hz] {self.processing_log}"

    def get_timelength(self) -> float:
        """Get length of signal in seconds

        Returns:
            float: length of signal in seconds
        """
        return len(self.data)/self.fs
    
class process_pipeline:
    """Processing pipeline. One step's input is the previous step's output
    
    Attributes:
        steps (list[Callable | tuple[Callable, dict]]): List of steps as functions or function and parameters as keyword dictionary
    
    Example:
        Creating a simple pipeline
        
        >>> import pyPCG
        >>> my_pipeline = pyPCG.process_pipeline(pyPCG.zero_center, pyPCG.unit_scale)
        >>> print(my_pipeline)
        PCG processing pipeline [2 steps]
        
        Creating a pipeline with parameters:
        
        >>> import pyPCG
        >>> import pyPCG.preprocessing as preproc
        >>> # Option 1:
        >>> # Create a tuple with the appropriate function a dictionary with the parameters passed as keyword arguments
        >>> step_1 = (preproc.filter,{"filt_ord":6,"filt_cutfreq":100,"filt_type":"LP"})
        >>> step_2 = (preproc.filter,{"filt_ord":6,"filt_cutfreq":20,"filt_type":"HP"})
        >>> my_pipeline = pyPCG.process_pipeline(step_1,step_2)
        >>> # Option 2:
        >>> # Using `functools.partial`
        >>> from functools import partial
        >>> step_a = partial(preproc.filter, filt_ord=6, filt_cutfreq=100, filt_type="LP")
        >>> step_b = partial(preproc.filter, filt_ord=6, filt_cutfreq=20, filt_type="HP")
        >>> my_pipeline = pyPCG.process_pipeline(step_a,step_b)
    """
    def __init__(self, *configs: Callable|tuple[Callable, dict[str,int|float|str]]) -> None:
        """Create processing pipeline object"""
        self.steps = []
        for k in configs:
            self.steps.append(k)
            
    def __repr__(self) -> str:
        return f"PCG processing pipeline [{len(self.steps)} steps]"
    
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
    return unit_scale(zero_center(sig))

def plot(sig: pcg_signal, zeroline: bool=False) -> None:
    """Plot pcg signal with appropriate dimensions

    Args:
        sig (pcg_signal): signal to plot
        zeroline (bool, optional): plot a dashed line at zero. Defaults to False.
    """
    t = sig.get_timelength()
    plt.plot(np.linspace(0,t,len(sig.data)),sig.data)
    if zeroline:
        z_line = np.zeros_like(sig.data)
        plt.plot(np.linspace(0,t,len(sig.data)),z_line,"k:")
    plt.title(sig.processing_log[-1])
    plt.xlabel("Time (s)")
    plt.ylabel("PCG (a.u.)")

def multiplot(*args):
    for sig in args:
        time = np.linspace(0,sig.get_timelength,len(sig.data))
        plt.plot(time,sig.data)

if __name__ == '__main__':
    print("Signal container and process pipeline builder")