import numpy as np

class pcg_signal:
    def __init__(self, data=np.array([]), fs=1) -> None:
        self.data = data
        self.fs = fs
        self.features = {}
        pass
    def get_timelength(self) -> float:
        return len(self.data)/self.fs
    
class process_pipeline:
    def __init__(self, input, *args) -> None:
        self.steps = []

def zero_center(sig: pcg_signal) -> pcg_signal:
    """Center to zero

    Args:
        sig (pcg_signal): Input signal

    Returns:
        pcg_signal: Centered signal
    """
    temp = sig
    temp.data -= np.mean(temp.data)
    return temp


def unit_scale(sig: pcg_signal) -> pcg_signal:
    """Scale signal to [-1,1] interval

    Args:
        sig (pcg_signal): Input signal

    Returns:
        pcg_signal: Scaled signal
    """
    temp = sig
    temp.data /=np.max(np.abs(temp.data))
    return temp

def normalize(sig: pcg_signal) -> pcg_signal:
    """Center to zero and scale signal to [-1,1] interval

    Args:
        sig (pcg_signal): Input signal

    Returns:
        pcg_signal: Normalized signal
    """
    return zero_center(unit_scale(sig))

if __name__ == '__main__':
    print("Signal container and process pipeline builder")