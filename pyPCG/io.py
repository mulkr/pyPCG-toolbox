import csv
import warnings
import numpy as np
import numpy.typing as npt
import scipy.io as sio
from math import floor

def read_signal_file(path: str, format: str) -> tuple[npt.NDArray[np.int_],int]:
    """Read in fetal heartsound containing file
    
    Supported file formats:
    - `wav`: wav file
    - `raw`: raw binary (headerless)
    - `mat`: MATLAB file (with two variables: `fs`-samplerate, `sig`-signal data)
    - `FETA`: every second byte is PCG data (headerless)
    - `1k`: 1 kB chunks (!untested!)

    Args:
        path (str): Path to input file
        format (str): File format identification

    Returns:
        signal (numpy.ndarray[int]): Unprocessed heartsound signal read in from file
    """
    
    signal = np.array([])
    fs = 0

    with open(path, 'rb') as dat:
        data = np.array(list(dat.read()))
        if format == 'raw':
            signal = data
        elif format == '1k':
            warnings.warn("1k format is not yet tested")
            BLOCK_SIZE = 1024
            block_count = floor(len(data)/BLOCK_SIZE)
            for i in range(block_count):
                signal = np.append(signal,data[25+i*BLOCK_SIZE:1024+i*BLOCK_SIZE])
            if block_count*BLOCK_SIZE+24 < len(data):
                signal = np.append(signal,data[25+block_count*BLOCK_SIZE:])
        elif format == '2nd':
            
            def _get_byte_offset_start(b_sig):
                test = b_sig[:16]
                mark_00 = 0
                for i in range(8):
                    if sum(test[i::8][:2])==0:
                        mark_00 = i
                        break
                test_ofs = mark_00
                read_ofs = mark_00+1%2
                return test_ofs, read_ofs

            def _find_byte_shift(b_sig,offset=0):
                test = b_sig[offset::8]
                mask = test!=0
                loc = np.where(mask)[0]
                if len(loc)>0:
                    res = ((loc[0])*8+offset)
                    # print(f"Byte shift detected at:{res}")
                    return res
                return -1

            def _correct_byte_shift(b_sig,t_offset=0):
                has_shift = True
                while has_shift:
                    shift = _find_byte_shift(b_sig,t_offset)
                    if shift<0:
                        has_shift = False
                    else:
                        b_sig = np.append(b_sig[:shift-1],b_sig[shift:])
                return b_sig
            
            t_offset,r_offset = _get_byte_offset_start(data)
            corr = _correct_byte_shift(data,t_offset)
            signal = corr[r_offset::2]
        elif format == 'wav':
            fs, signal = sio.wavfile.read(path)
        elif format == 'mat':
            mat = sio.loadmat(path)
            fs = mat["fs"][0,0]
            signal = mat["sig"][0]
        else:
            raise ValueError('Format not recognized')
    return np.array(signal).astype(int), fs

def read_hsannot_file(fpath: str) -> tuple[list,list]:
    """Reads manually labeled heartsounds from annotation csv file
    
    The csv format has to be the following:
    - `Location`: heartsound location [seconds]
    - `Value`: heartsound type [`"S1"` or `"S2"`]

    Args:
        fpath (str): path to annotation file

    Returns:
        s1_loc (list[float]): S1 annotation locations
        s2_loc (list[float]): S2 annotation locations
    """
    
    s1_loc, s2_loc = [],[]
    with open(fpath,'r') as annot:
        reader = csv.DictReader(annot,delimiter=';')
        for line in reader:
            if line['Value']=='S1':
                s1_loc.append(float(line['Location']))
            elif line['Value']=='S2': 
                s2_loc.append(float(line['Location']))
            else:
                print("Unknown label. Skipping...")
    return s1_loc,s2_loc

if __name__ == '__main__':
    print("Data Loader")