import numpy as np
import matplotlib.pyplot as plt

class pcg_signal:
    def __init__(self, data=np.array([]), fs=1) -> None:
        self.data = data.astype(float)
        self.fs = fs
        self.features = {}

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
    sig.data -= np.mean(sig.data)
    return sig

def unit_scale(sig: pcg_signal) -> pcg_signal:
    """Scale signal to [-1,1] interval

    Args:
        sig (pcg_signal): Input signal

    Returns:
        pcg_signal: Scaled signal
    """
    sig.data /=np.max(np.abs(sig.data))
    return sig

def std_scale(sig: pcg_signal) -> pcg_signal:
    """Scale signal to 1 std

    Args:
        sig (pcg_signal): Input signal

    Returns:
        pcg_signal: Scaled signal
    """
    sig.data /=np.std(sig.data)
    return sig

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
    if "env" in sig.features.keys():
        plt.plot(np.linspace(0,t,len(sig.data)),sig.features["env"])
    if "h_env" in sig.features.keys():
        plt.plot(np.linspace(0,t,len(sig.data)),sig.features["h_env"])

if __name__ == '__main__':
    print("Signal container and process pipeline builder")