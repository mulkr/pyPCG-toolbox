"""
# INTERNAL CODE
## LR-HSMM segmenter
### Implementation: K. Müller, PPCU-ITK (PhD student) 2023.
Correspondance: muller.kristof@itk.ppke.hu

Based on:
D. B. Springer, L. Tarassenko and G. D. Clifford, "Logistic Regression-HSMM-Based Heart Sound Segmentation," in IEEE Transactions on Biomedical Engineering, vol. 63, no. 4, pp. 822-832, April 2016, doi: 10.1109/TBME.2015.2475278.
"""

import pywt
import json
import warnings
import numpy as np
import scipy.signal as sgn
from math import ceil
from sklearn.linear_model import LogisticRegression
from hsmmlearn.emissions import AbstractEmissions
from hsmmlearn.hsmm import HSMMModel
from scipy.stats import norm

INF = 999999999

class LR_HSMM():
    """
    Main segmenter object

    Attributes:
    ----------
        signal_fs (int): Sampling frequency of the input signal. Default: 1000 [Hz]
        feature_fs (int): Sampling frequency for feature calculations. Default: 50 [Hz]
        mean_s1_len (float): Average S1 duration. Default: 122 [ms]
        mean_s2_len (float): Average S2 duration. Default: 99 [ms]
        std_s1_len (float): Standard deviation of S1 durations. Default: 22 [ms]
        std_s2_len (float): Standard deviation of S2 durations. Default: 22 [ms]
        bandpass_frq (tuple[float,float]): Cutoff frequencies for pre-processing band-pass filtering (Butterworth, 4th order). Default: (25,400) [Hz]
        expected_hr_range (tuple[float,float]): Minimum and maximum expected heartrates. Default: (30,120) [bpm]
        
        hsmm_model (hsmmlearn.hsmm.HSMMModel): State predictor model
        lr_model (hsmmlear.emissions.AbstractEmissions): Probability emissions for the HSMM states. Includes a LogisticRegression model for each state
            
    Methods:
    -------
        train_model(train_data, train_s1_annot, train_s2_annot)
            Trains the model on the specified data with S1 and S2 location annotations
        
        segment_single(sig)
            Predicts the states for the given PCG signal
            
        save_model(filename)
            Saves the model parameters to a json file
            
        load_model(filename)
            Loads the model parameters from a json file
    """
    
    def __init__(self) -> None:        
        self.signal_fs = 1000
        self.feature_fs = 50
        self.mean_s1_len = 122
        self.mean_s2_len = 99
        self.std_s1_len = 22
        self.std_s2_len = 22
        self.bandpass_frq = (25,400)
        self.expected_hr_range = (30,120)
        
        self.hsmm_model = None
        self.lr_model = _LREmission()
    
    def train_model(self,train_data,train_s1_annot,train_s2_annot):
        """Trains the model on the specified data with S1 and S2 location annotations 

        Args:
            train_data (numpy.ndarray): Array of input signals
            train_s1_annot (numpy.ndarray): Array of S1 annotations for each signal
            train_s2_annot (numpy.ndarray): Array of S2 annotations for each signal
        """
        
        tmat = np.array([[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.],[1.,0.,0.,0.]])
        states = np.array([])
        f_henv, f_env, f_psd, f_wt = np.array([]),np.array([]),np.array([]),np.array([])
        d_hr, d_sys = np.array([]),np.array([])
        print("Generating features...")
        for i,(data,s1_annot,s2_annot) in enumerate(zip(train_data,train_s1_annot,train_s2_annot)):
            hr, sys = _get_hr_sys(data,self.signal_fs,self.bandpass_frq,self.expected_hr_range[0],self.expected_hr_range[1])
            sts = _generate_states(data,s1_annot,s2_annot,self.signal_fs,self.feature_fs,self.mean_s1_len,self.mean_s2_len,self.std_s1_len,self.std_s2_len)
            henv, env, psd, wt = _generate_features(data,self.signal_fs,self.feature_fs,self.bandpass_frq)
            states = np.append(states,sts)
            f_henv = np.append(f_henv,henv)
            f_env = np.append(f_env,env)
            f_psd = np.append(f_psd,psd)
            f_wt = np.append(f_wt,wt)
            d_hr = np.append(d_hr,hr)
            d_sys = np.append(d_sys,sys)
        features = np.array([f_henv,f_env,f_psd,f_wt])
        durs = _get_duration_distributions(np.mean(d_hr),np.mean(d_sys),self.feature_fs,self.mean_s1_len,self.mean_s2_len,self.std_s1_len,self.std_s2_len)
        print("Training model...")
        self.lr_model = _LREmission(features.T, states)
        self.hsmm_model = HSMMModel(self.lr_model,durs,tmat)
        
    def segment_single(self,sig):
        """Predicts the states for the given PCG signal

        Args:
            sig (numpy.ndarray): Input signal to be segmented

        Returns:
            e_states (numpy.ndarray): Predicted state for each sample
            seg_features (numpy.ndarray): Calculated features for each sample
        """
        
        henv, env, psd, wt = _generate_features(sig,self.signal_fs,self.feature_fs,self.bandpass_frq)
        seg_features = np.array([henv, env, psd, wt])
        if self.hsmm_model is None:
            warnings.warn("Attempting to segment with untrained model. Returning empty states...",RuntimeWarning)
            return [] , []
        d_states = self.hsmm_model.decode(seg_features.T)
        e_states = _expand_states(d_states+1,self.feature_fs,self.signal_fs,len(sig))
        return e_states, seg_features
        
    def save_model(self,filename):
        """Saves the model parameters to a json file

        Args:
            filename (str): Name of file to be saved
        """
        if self.hsmm_model is None:
            warnings.warn("Attempting to save untrained model. No file will be written. Returning...",RuntimeWarning)
            return
        durs = self.hsmm_model.durations.tolist() #type: ignore
        tmat = self.hsmm_model.tmat.tolist() #type: ignore
        emissions = self.hsmm_model.emissions.serialize() #type: ignore
        config = {
            "sig_fs":self.signal_fs,
            "f_fs": self.feature_fs,
            "preproc_bp": self.bandpass_frq,
            "mean_s1": self.mean_s1_len,
            "mean_s2" : self.mean_s2_len,
            "std_s1" : self.std_s1_len,
            "std_s2" : self.std_s2_len,
            "min_hr": self.expected_hr_range[0],
            "max_hr": self.expected_hr_range[1]
        }
        serialized_hsmm = {"config":config,"durations":durs,"transition":tmat,"emissions":emissions}
        with open(filename,"w") as save:
            save.write(json.dumps(serialized_hsmm))
            
    def load_model(self,filename):
        """Loads the model parameters from a json file

        Args:
            filename (str): Name of file containing model parameters
        """
        
        with open(filename,"r") as load:
            data = json.loads(load.read())
            durs = np.array(data["durations"])
            tmat = np.array(data["transition"])
            config = data["config"]
            
            self.signal_fs = config["sig_fs"]
            self.feature_fs = config["f_fs"]
            self.bandpass_frq = config["preproc_bp"]
            self.mean_s1_len = config["mean_s1"]
            self.mean_s2_len = config["mean_s2"]
            self.std_s1_len = config["std_s1"]
            self.std_s2_len = config["std_s2"]
            self.expected_hr_range = (config["min_hr"],config["max_hr"])
            
            emission = _LREmission()
            emission.unserialize(data["emissions"])
            self.lr_model = emission
            self.hsmm_model = HSMMModel(self.lr_model,durs,tmat)
            
class _LREmission(AbstractEmissions):
    
    def __init__(self, features=None, states=None) -> None:
        self.LRmodel_s1 = LogisticRegression()
        self.LRmodel_s2 = LogisticRegression()
        self.LRmodel_sys = LogisticRegression()
        self.LRmodel_dia = LogisticRegression()
        self.predictors = [self.LRmodel_s1,self.LRmodel_sys,self.LRmodel_s2,self.LRmodel_dia]
        if(features is None or states is None):
            return
        
        s1_train, s2_train, sys_train, dia_train = np.zeros_like(states), np.zeros_like(states), np.zeros_like(states), np.zeros_like(states)
        s1_train[states != 1] = 1
        s2_train[states != 3] = 1
        sys_train[states != 2] = 1
        dia_train[states != 4] = 1
        
        self.LRmodel_s1 = LogisticRegression(random_state=0,max_iter=1000,class_weight="balanced").fit(features,s1_train)
        self.LRmodel_s2 = LogisticRegression(random_state=0,max_iter=1000,class_weight="balanced").fit(features,s2_train)
        self.LRmodel_sys = LogisticRegression(random_state=0,max_iter=1000,class_weight="balanced").fit(features,sys_train)
        self.LRmodel_dia = LogisticRegression(random_state=0,max_iter=1000,class_weight="balanced").fit(features,dia_train)
        self.predictors = [self.LRmodel_s1,self.LRmodel_sys,self.LRmodel_s2,self.LRmodel_dia]
        
    def likelihood(self, obs):
        l_s1 = self.LRmodel_s1.predict_proba(obs)[:,0]
        l_s2 = self.LRmodel_s2.predict_proba(obs)[:,0]
        l_sys = self.LRmodel_sys.predict_proba(obs)[:,0]
        l_dia = self.LRmodel_dia.predict_proba(obs)[:,0]
        return np.array([l_s1,l_sys,l_s2,l_dia])
    
    def serialize(self):
        serialized_models = {"lr_s1":None, "lr_sys":None, "lr_s2":None, "lr_dia":None}
        for lr_type,model in zip(serialized_models.keys(),self.predictors):
            params = model.get_params()
            attrs = [i for i in dir(model) if i.endswith('_') and not i.endswith('__') and not i.startswith('_')]
            attr_dict = {i: getattr(model, i) for i in attrs}
            for k in attr_dict:
                if isinstance(attr_dict[k], np.ndarray):
                    attr_dict[k] = attr_dict[k].tolist()
            serialized_lr = {"params":params,"attrs":attr_dict}
            serialized_models[lr_type] = serialized_lr #type: ignore
        return serialized_models
    
    def unserialize(self,serial):
        for loaded,lr in zip(serial.keys(),self.predictors):
            params = serial[loaded]["params"]
            attrs = serial[loaded]["attrs"]
            lr.set_params(**params)
            for k in attrs:
                if isinstance(attrs[k],list):
                    setattr(lr,k,np.array(attrs[k]))
                else:
                    setattr(lr,k,attrs[k])

def _envelope_feature(sig):
    env = abs(sgn.hilbert(sig)) #type: ignore
    return env

def _h_envelope_feature(sig,sig_fs):
    env = _envelope_feature(sig)
    lp = sgn.butter(6,8,output='sos',fs=sig_fs,btype='lowpass')
    filt = np.exp(sgn.sosfiltfilt(lp,np.log(env)))
    filt[0] = filt[1] #?
    return filt

def _psd_feature(sig,sig_fs):
    f_lo = 40
    f_hi = 60
    [f,_,Zxx] = sgn.stft(sig,fs=sig_fs,window="hamming",nperseg=sig_fs//40,scaling="psd",nfft=1024)
    lo_pos = np.argmin(np.abs(f-f_lo))
    hi_pos = np.argmin(np.abs(f-f_hi))
    psd = np.mean(np.abs(Zxx[lo_pos:hi_pos,:]),axis=0)
    psd_re = sgn.resample_poly(psd,len(sig),len(psd))
    return psd_re

def _wt_feature(sig):
    coefs = pywt.wavedec(sig,"rbio3.9",level=3)
    cD = coefs[1:]
    cD.reverse()
    detail = np.zeros((3,len(sig)))
    for i in range(3):
        d = np.tile(cD[i],(2**(i+1),1)).T.ravel()
        start = len(d)-len(sig)-1//2
        end = start + len(sig)
        detail[i,:] = d[start:end]
        
    d3 = np.abs(detail[2,:])
    return d3
    
def _normalize(sig):
    m = np.mean(sig)
    s = np.std(sig)
    n_sig = (sig-m)/s
    return n_sig    

def _spike_removal(sig,sig_fs):
    window_s = round(sig_fs/2)
    trailing = len(sig) % window_s
    frames = np.reshape(sig[:-trailing],(window_s,-1))
    MAAs = np.max(np.abs(frames),axis=0)
    while(np.count_nonzero(MAAs>np.median(MAAs)*3)>0):
        # val = np.max(MAAs,axis=1)
        framenum = np.argmax(MAAs,axis=0)
        if type(framenum) != np.int64:
            if len(framenum)>1:
                framenum = framenum[1]
        pos = np.argmax(np.abs(frames[:,framenum]))
        zerocross = np.where(np.abs(np.diff(np.sign(frames[:,framenum])))>0)[0]
        last = np.nonzero(zerocross[0:pos])[0][-1]
        start = max(1,last)
        zerocross[0:pos] = 0
        end = min(np.nonzero(zerocross)[0][0],window_s)
        frames[start:end,framenum] = 0.0001
        MAAs = np.max(np.abs(frames),axis=0)
    removed = np.reshape(frames,(-1,1))
    removed = np.append(removed,sig[len(removed):])
    return removed

def _generate_features(sig,sig_fs,f_fs,preproc=(25,400)):
    bpf = sgn.butter(4,preproc,"bandpass",output="sos",fs=sig_fs)
    f_sig = sgn.sosfiltfilt(bpf,sig)
    # rem_sig = _spike_removal(f_sig,sig_fs)
    rem_sig = f_sig
    
    h_env = _normalize(_h_envelope_feature(rem_sig,sig_fs))
    env = _normalize(_envelope_feature(rem_sig))
    psd = _normalize(_psd_feature(rem_sig,sig_fs))
    wt = _normalize(_wt_feature(rem_sig))
    
    d_h_env = sgn.resample_poly(h_env,sig_fs,f_fs)
    d_env = sgn.resample_poly(env,sig_fs,f_fs)
    d_psd = sgn.resample_poly(psd,sig_fs,f_fs)
    d_wt = sgn.resample_poly(wt,sig_fs,f_fs)

    return d_h_env,d_env,d_psd,d_wt

def _generate_states(sig,annot_s1,annot_s2,sig_fs,f_fs,mean_s1=122,mean_s2=99,std_s1=22,std_s2=22):
    henv = _h_envelope_feature(sig,sig_fs)
    env = sgn.resample_poly(henv,sig_fs,f_fs)
    states = np.zeros_like(env)
    ms_scale = sig_fs/1000
    fs_scale = sig_fs/f_fs
    scale = f_fs/1000
    as1 = np.round(np.array(annot_s1)*scale).astype(int) # *ms_scale/fs_scale
    as2 = np.round(np.array(annot_s2)*scale).astype(int)
    ms1 = round(mean_s1*scale)
    ms2 = round(mean_s2*scale)
    ss1 = round(std_s1*scale)
    ss2 = round(std_s2*scale)
    for s1 in as1:
        upper_s1 = min(len(states)-1,s1+ms1+ss1) # +std_s1
        lower_s1 = max(1,s1-ms1-ss1)
        search = env[lower_s1:upper_s1]
        s1_ind = np.argmax(search)
        s1_ind = min(len(states)-1,lower_s1+s1_ind) #type: ignore
        upper_s1 = min(len(states)-1,ceil(s1_ind+(ms1/2)))
        lower_s1 = max(0,ceil(s1_ind-(ms1/2)))
        
        states[lower_s1:upper_s1] = 1
    for s2 in as2:
        upper_s2 = min(len(states)-1,s2+ms2+ss2)
        lower_s2 = max(0,s2-ms2-ss2)
        search = env[lower_s2:upper_s2]*(1-states[lower_s2:upper_s2])
        s2_ind = np.argmax(search)
        s2_ind = min(len(states)-1,lower_s2+s2_ind) #type: ignore
        
        upper_s2 = min(len(states)-1,ceil(s2_ind+(ms2/2)))
        lower_s2 = max(0,ceil(s2_ind-(ms2/2)))
        states[lower_s2:upper_s2] = 3
        
        s1_labels = np.nonzero(states == 1)[0]
        diffs = s1_labels - s2
        diffs[diffs<0] = INF
        end_pos = 0
        if len(diffs<INF)==0:
            end_pos = len(states)-1
        else:
            end_pos = s1_labels[np.argmin(diffs)]
        states[ceil(s2_ind+(ms2/2)):end_pos] = states[ceil(s2_ind+(ms2/2)):end_pos]+4
        
    first_definite = np.nonzero(states)[0][0]
    if first_definite > 0:
        if states[first_definite+1] == 1:
            states[0:first_definite] = 4
        if states[first_definite+1] == 3:
            states[0:first_definite] = 2
    last_definite = np.nonzero(np.array(states))[0][-1]
    if last_definite > 0:
        if states[last_definite] == 1:
            states[last_definite:] = 2
        if states[last_definite] == 3:
            states[last_definite:] = 4
            
    states[states==0] = 2
    return states

def _get_hr_sys(sig,sig_fs,preproc=(25,400),min_hr=30,max_hr=120):
    bpf = sgn.butter(4,preproc,"bandpass",output="sos",fs=sig_fs)
    f_sig = sgn.sosfiltfilt(bpf,sig)
    # rem_sig = spike_removal(f_sig,sig_fs)
    h_env = _h_envelope_feature(f_sig,sig_fs)
    y = h_env - np.mean(h_env)
    coef = sgn.correlate(y,y)
    coef = coef[len(h_env):]/np.max(coef)
    min_ind = round((60/max_hr)*sig_fs)
    max_ind = round((60/min_hr)*sig_fs)
    ind = np.argmax(coef[min_ind:max_ind]) + min_ind
    heartrate = 60/(ind/sig_fs)
    
    max_sys = round(((60/heartrate)*sig_fs)/2)
    min_sys = round(0.2*sig_fs)
    ind = np.argmax(coef[min_sys:max_sys]) + min_sys
    systole = ind/sig_fs
    # print(systole)
    # systole = 0.1 # temporary hack
    
    return heartrate, systole

def _get_duration_params(hr,sys,mean_s1=122,mean_s2=99,std_s1=22,std_s2=22):
    mean_sys = round(sys*1000)
    std_sys = 25
    # mean_dia = round((60/hr)-sys)*1000-94
    mean_dia = round(((60/hr)-sys)*1000)-130
    # std_dia = round(0.07*mean_dia) + 6
    std_dia = round(0.15*mean_dia)
    
    min_sys = mean_sys - 3*(std_sys+std_s1)
    max_sys = mean_sys + 3*(std_sys+std_s1)
    min_dia = mean_dia - 3*std_dia
    max_dia = mean_dia + 3*std_dia
    min_s1 = mean_s1 - 3*std_s1
    max_s1 = mean_s1 + 3*std_s1
    min_s2 = mean_s2 - 3*std_s2
    max_s2 = mean_s2 + 3*std_s2
    
    # max_duration = (60/hr)*1000
    max_duration = max([max_s1+2*std_s1,max_s2+2*std_s2,max_sys+2*(std_sys+std_s1),max_dia+2*std_dia])
    # min_duration = min([min_s1-2*std_s1,min_s2-2*std_s2,min_sys-2*(std_sys-std_s1),min_dia-2*std_dia])
    return max_duration, mean_sys, std_sys, mean_dia, std_dia
    
def _get_duration_distributions(hr,sys,f_fs,mean_s1=122,mean_s2=99,std_s1=22,std_s2=22):
    max_duration, mean_sys, std_sys, mean_dia, std_dia = _get_duration_params(hr,sys,mean_s1,mean_s2,std_s1,std_s2)
    # fs_scale = round(sig_fs/f_fs)
    time_scale = f_fs/1000
    max_duration = round(max_duration*time_scale)
    s1_dur, s2_dur, sys_dur, dia_dur = np.zeros((max_duration)),np.zeros((max_duration)),np.zeros((max_duration)),np.zeros((max_duration))
    for i in range(max_duration):
        s1_dur[i] = norm.pdf(i,loc=mean_s1*time_scale,scale=std_s1*time_scale)
        s2_dur[i] = norm.pdf(i,loc=mean_s2*time_scale,scale=std_s2*time_scale)
        sys_dur[i] = norm.pdf(i,loc=mean_sys*time_scale,scale=std_sys*time_scale)
        dia_dur[i] = norm.pdf(i,loc=mean_dia*time_scale,scale=std_dia*time_scale)
    return np.array([s1_dur,sys_dur,s2_dur,dia_dur])

def _diff(sig):
    sub = np.insert(sig,0,0)
    sig = sig-sub[:-1]
    return sig[1:]

def _expand_states(states,orig_fs,new_fs,new_len):
    expanded = np.zeros((new_len,1))
    changes = np.nonzero(_diff(states))[0]
    changes = np.append(changes,len(states)-1)
    start = 0
    for end in changes:
        mid = round((end-start)/2) + start
        mid_val = states[mid]
        exp_start = round((start/orig_fs)*new_fs)
        exp_end = round((end/orig_fs)*new_fs) if round((end/orig_fs)*new_fs) < new_len else new_len
        expanded[exp_start:exp_end] = mid_val
        start = end
    return expanded

if __name__ == '__main__':
    print("LR-HSMM model based on Springer et al.")