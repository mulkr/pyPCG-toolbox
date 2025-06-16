import numpy as np
import pandas as pd
import numpy.typing as npt
import scipy.stats as sts
from scipy.special import erfcinv
from typing import Callable, TypedDict

def trim_transform(data: npt.NDArray[np.float64], trim_precent: float) -> npt.NDArray[np.float64]:
    """Trim the upper and lower percentage of values

    Args:
        data (np.ndarray): input data to trim
        trim_precent (float): percentage to trim away

    Returns:
        np.ndarray: trimmed values
    """
    return sts.trimboth(data,trim_precent/100) #type: ignore

def outlier_remove_transform(data: npt.NDArray[np.float64], dist: float=3.0) -> npt.NDArray[np.float64]:
    """Remove outliers based on the MAD (median of absolute differences)

    Args:
        data (np.ndarray): input data
        dist (float, optional): MAD score threshold. Defaults to 3.0.

    Returns:
        np.ndarray: data without outliers
    """
    d = np.abs(data - np.median(data))
    c = -1/(np.sqrt(2)*erfcinv(3/2))
    mdev = c*np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<dist]

def mean(data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64] | np.float64:
    """Calculate mean of inputs

    Args:
        data (np.ndarray): input data (can be 2 dimensional array)

    Returns:
        np.ndarray | float: mean value of data, if input is 2D then return the value along of 1st axis
    """
    if len(data.shape) == 1:
        return np.mean(data)
    else: return np.mean(data,axis=1)

def std(data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64] | np.float64:
    """Calculate standard deviation of inputs

    Args:
        data (np.ndarray): input data (can be 2 dimensional array)

    Returns:
        np.ndarray | float: standard deviation of data, if input is 2D then return the value along of 1st axis
    """
    if len(data.shape) == 1:
        return np.std(data)
    else: return np.std(data,axis=1)

def rms(data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64] | np.float64:
    """Calculate root mean square of inputs

    Args:
        data (np.ndarray): input data (can be 2 dimensional array)

    Returns:
        np.ndarray | float: root mean square of data, if input is 2D then return the value along of 1st axis
    """
    if len(data.shape) == 1:
        return np.sqrt(np.mean(data**2))
    else: return np.sqrt(np.mean(data**2,axis=1))

def med(data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64] | np.float64:
    """Calculate median of inputs

    Args:
        data (np.ndarray): input data (can be 2 dimensional array)

    Returns:
        np.ndarray | float: median of data, if input is 2D then return the value along of 1st axis
    """
    if len(data.shape) == 1:
        return np.median(data)
    else: return np.median(data,axis=1)

def skew(data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64] | np.float64:
    """Calculate skewness of inputs

    Args:
        data (np.ndarray): input data (can be 2 dimensional array)

    Returns:
        np.ndarray | float: skewness of data, if input is 2D then return the value along of 1st axis
    """
    if len(data.shape) == 1:
        return sts.skew(data)
    else: return sts.skew(data,axis=1)

def kurt(data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64] | np.float64:
    """Calculate kurtosis of inputs

    Args:
        data (np.ndarray): input data (can be 2 dimensional array)

    Returns:
        np.ndarray | float: kurtosis of data, if input is 2D then return the value along of 1st axis
    """
    if len(data.shape) == 1:
        return sts.kurtosis(data) #type: ignore
    else: return sts.kurtosis(data,axis=1) #type: ignore

def max(data: npt.NDArray[np.float64],k: int=1) -> npt.NDArray[np.float64] | np.float64:
    """Get maximum values from input

    Args:
        data (np.ndarray): input data
        k (int, optional): number of largest values to return. Defaults to 1.

    Returns:
        np.ndarray | float: maximum value(s)
    """
    s_data = np.sort(data)
    select = s_data[-k-1:-1]
    ret = select[::-1]
    if len(ret)==1:
        ret=ret[0]
    return ret

def min(data: npt.NDArray[np.float64],k: int=1) -> npt.NDArray[np.float64] | np.float64:
    """Get minimum values from input

    Args:
        data (np.ndarray): input data
        k (int, optional): number of smallest values to return. Defaults to 1.

    Returns:
        np.ndarray | float: minimum value(s)
    """
    s_data = np.sort(data)
    ret = s_data[0:k]
    if len(ret)==1:
        ret=ret[0]
    return ret

def percentile(data: npt.NDArray[np.float64], perc: float=25) -> npt.NDArray[np.float64] | np.float64:
    """Calculate given percentile of inputs

    Args:
        data (np.ndarray): input data (can be 2 dimensional array)
        perc (float): selected percentile to calculate. Defaults to 25.

    Returns:
        np.ndarray | float: given percentile of data, if input is 2D then return the value along of 1st axis
    """
    if len(data.shape) == 1:
        return np.percentile(data,perc)
    else: return np.percentile(data,perc,axis=1)

def window_operator(data: npt.NDArray[np.float64],win_size: int,fun: Callable,overlap_percent: float=0.5) -> tuple[npt.NDArray[np.int_],npt.NDArray[np.float64]]:
    """Apply given statistical function over a sliding window on the input

    Args:
        data (np.ndarray): input data
        win_size (int): window size
        fun (Callable): statistical function to apply
        overlap_percent (float, optional): window overlap as a ratio to the window size. Defaults to 0.5.

    Returns:
        tuple[np.ndarray,np.ndarray]: window sample locations (usually used as time dimension), and calculated values in the windows
    """
    step = win_size-round(win_size*overlap_percent)
    val,loc = [], []
    for i in range(0,len(data) - win_size,step):
        val.append(fun(data[i:i+win_size]))
        loc.append(i)
    return np.array(loc), np.array(val)

def iqr(data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64] | np.float64:
    """Calculate interquartile range

    Args:
        data (np.ndarray): input data

    Returns:
        np.ndarray | float: interquartile range of data
    """
    return percentile(data,75)-percentile(data,25)

class stats_config(TypedDict):
    """Type to hold statistic calculation configs"""
    calc_fun: Callable
    """function for calculation"""
    name: str
    """name of the calculated statistic"""

class stats_group:
    """Group statistic calculations together for reuse
    
    Attributes:
        configs (list[stats_config]): List of statistic calculations with names
        signal_stats (dict[str, dict[str, list[float]]]): Signal statistics by segment, #TODO: this will become its own type in the future
        dataframe (pd.DataFrame): Pandas dataframe container of statistics for utility
        
    Example:
        Create a statistic group calculation:
        
        For an easier experience use the `stats_config` type
        
        >>> import pyPCG.stats as sts
        >>> stat_1 = {"calc_fun":sts.mean,"name":"Mean"}
        >>> stat_2 = {"calc_fun":sts.std,"name":"Std"}
        >>> mean_std = sts.stats_group(stat_1,stat_2)
        
        Run the created group with some dummy features:
        
        >>> import numpy as np
        >>> dummy = {"length":np.arange(10),"max freq":np.arange(10)}
        >>> basic_stats = mean_std.run(dummy)
        >>> print(basic_stats)
        {'Feature': ['length', 'max freq'], 'Mean': [4.5, 4.5], 'Std': [2.8722813232690143, 2.8722813232690143]}
        
        Add statistics for a given segment, and export it as xlsx:
        
        >>> mean_std.add_stat("Test",basic_stats)
        >>> mean_std.export("test.xlsx")
        
    """
    def __init__(self,*stats: stats_config) -> None:
        self.configs = []
        self.signal_stats = {}
        self.dataframe = pd.DataFrame()
        for stat in stats:
            self.configs.append(stat)

    def _update_df(self):
        first_key = next(iter(self.signal_stats.keys()))
        ftr_and_stats = list(self.signal_stats[first_key].keys())
        temp = {"Segment":[]}
        for key in ftr_and_stats:
            temp[key] = []
        for segment,dat in self.signal_stats.items():
            for key,vals in dat.items():
                for val in vals:
                    temp[key].append(val)
            for _ in dat["Feature"]:
                temp["Segment"].append(segment)
        self.dataframe = pd.DataFrame(temp)

    def run(self,ftr_dict: dict[str,npt.NDArray[np.float64]]) -> dict[str,list[float]]:
        """Run the statistic calculation based on configuration

        Args:
            ftr_dict (dict[str,np.ndarray]): feature dictionary, same format as the output of feature_group.run

        Returns:
            dict[str,list[float]]: Calculated statistics, named
        """
        ret = {"Feature":[]}
        for stat in self.configs:
            ret[stat["name"]] = []
        for name,value in ftr_dict.items():
            ret["Feature"].append(name)
            for stat in self.configs:
                ret[stat["name"]].append(stat["calc_fun"](value))
        return ret
    
    def add_stat(self,segment:str, stats:dict[str,list[float]]):
        """Add calculated statistics to signal_stats with the given segment name

        Args:
            segment (str): segment name to save to
            stats (dict[str,list[float]]): calculated statistics
        """
        self.signal_stats[segment] = stats
        self._update_df()
    
    def calc_group_stats(self,total_ftr_dict: dict[str,dict[str,npt.NDArray[np.float64]]]):
        """Calculate all stats on all given features and segments

        Args:
            total_ftr_dict (dict[str,dict[str,np.ndarray]]): Feature dictionaries named by segment
        """
        for segment,ftr_dict in total_ftr_dict.items():
            self.add_stat(segment,self.run(ftr_dict))

    def export(self,filename: str):
        """Export statistics to excel file

        Args:
            filename (str): filename to save to
        """
        with pd.ExcelWriter(filename) as writer:
            self.dataframe.to_excel(excel_writer=writer,sheet_name="Summary",index=False)
            for segment in self.dataframe["Segment"].unique():
                sub = self.dataframe[self.dataframe["Segment"]==segment][self.dataframe.columns.difference(["Segment"],sort=False)]
                sub.to_excel(excel_writer=writer,sheet_name=segment,index=False)

def calc_group_stats(ftr_dict: dict[str,dict[str,npt.NDArray[np.float64]]], *configs: tuple[Callable,str]) -> dict[str,list[float]]:
    """Calculate the same statistics for different segments and their features

    Args:
        ftr_dict (dict[str,dict[str,np.ndarray]]): input segment features, <segment name>:<feature dict from featuregroup.run>

    Returns:
        dict[str,list[float]]: statistics for each segment and its features
    """
    cols = {"Segment":[],"Feature":[]}
    for config in configs:
        cols[config[1]] = []
    for seg, ftrs in ftr_dict.items():
        for ftr,val in ftrs.items():
            cols["Segment"].append(seg)
            cols["Feature"].append(ftr)
            for config in configs:
                cols[config[1]].append(config[0](val))
    return cols

def export_stats(filename: str, group_stats: dict[str,list[float]]):
    """Export statistics calculated with calc_group_stats to excel

    Args:
        filename (str): name of excel file
        group_stats (dict[str,list[float]]): input statistics
    """
    df = pd.DataFrame(group_stats)
    with pd.ExcelWriter(filename) as writer:
        df.to_excel(excel_writer=writer,sheet_name="Summary",index=False)
        for segment in df["Segment"].unique():
            sub = df[df["Segment"]==segment][df.columns.difference(["Segment"],sort=False)]
            sub.to_excel(excel_writer=writer,sheet_name=segment,index=False)

if __name__ == '__main__':
    print("Statistics")