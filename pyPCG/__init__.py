import numpy as np
import matplotlib.pyplot as plt
import copy

class pcg_signal:
    def __init__(self, data=np.array([]), fs=1, log=[]) -> None:
        self.data = data.astype(float)
        self.fs = fs
        self.processing_log = log

    def get_timelength(self) -> float:
        return len(self.data)/self.fs
    
class process_pipeline:
    def __init__(self, *args) -> None:
        self.steps = []
        for k in args:
            # TODO: import inspect -> check for valid signature
            self.steps.append(k)
    
    def run(self, input: pcg_signal) -> pcg_signal:
        out = input
        for step in self.steps:
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