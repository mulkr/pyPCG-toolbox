Validation
==========

This shows the validation process for the segmentation methods. An interactive version can be found in a `Colab Notebook <https://colab.research.google.com/drive/1TQrgCDys47YJ_bu-iYo_HcB3LF03ixVv>`_.

Here, we assume that all data is available and pyPCG is installed with its dependencies, as well as NeuroKit2 and detection results from the original Springer HSMM. For more information about the data please refer to the Colab Notebook or contact us via email.

Import modules
--------------

.. code-block:: python

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.signal as sgn
    import pyPCG, pyPCG.segment, pyPCG.io, pyPCG.preprocessing
    import neurokit2 as nk

Detection
---------

Define detection methods
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def detect_naive(sig):
        bp_signal = pyPCG.preprocessing.filter(pyPCG.preprocessing.filter(sig,6,100,"LP"),6,20,"HP")
        env = pyPCG.preprocessing.homomorphic(bp_signal)
        d = round(0.27*sig.fs)
        locs,_ = sgn.find_peaks(env.data,distance=d)
        s1_detect = locs/sig.fs
        s2_detect = np.array([])
        return s1_detect, s2_detect

    def detect_adaptive(sig):
        bp_signal = pyPCG.preprocessing.filter(pyPCG.preprocessing.filter(sig,6,100,"LP"),6,20,"HP")
        denoise_signal = pyPCG.preprocessing.wt_denoise(bp_signal)
        env = pyPCG.preprocessing.homomorphic(denoise_signal)

        _, locs = pyPCG.segment.adv_peak(env)
        s1_detect, s2_detect = pyPCG.segment.peak_sort_diff(locs)
        s1_detect = s1_detect/sig.fs
        s2_detect = s2_detect/sig.fs
        return s1_detect, s2_detect

    def detect_neurokit(sig):
        env = pyPCG.preprocessing.homomorphic(sig)
        info = nk.signal_findpeaks(env.data,height_min=np.median(env.data))
        s1_detect = info["Peaks"]/sig.fs
        s2_detect = info["Peaks"]/sig.fs
        return s1_detect, s2_detect

    def detect_hsmm(sig, model):
        states = pyPCG.segment.segment_hsmm(model,sig)
        s1_s, s1_e = pyPCG.segment.convert_hsmm_states(states,pyPCG.segment.heart_state.S1)
        s2_s, s2_e = pyPCG.segment.convert_hsmm_states(states,pyPCG.segment.heart_state.S2)
        if len(s1_s)<len(s1_e):
            s1_e = s1_e[1:]
        if len(s2_s)<len(s2_e):
            s2_e = s2_e[1:]
        if len(s2_s)<len(s2_e):
            s2_e = s2_e[1:]
        s1_detect = (s1_s+s1_e)/2
        s2_detect = (s2_s+s2_e)/2
        s1_detect = s1_detect/sig.fs
        s2_detect = s2_detect/sig.fs
        return s1_detect, s2_detect

Read in pretrained HSMM model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    hsmm = pyPCG.segment.load_hsmm("pyPCG/data/pre_trained_fpcg.json")

Set up directories for detections, and paths for reading in data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    data_path = "data"
    label_path = "label"

    methods = ["naive","adaptive","neurokit","pyhsmm"]
    os.mkdir("detections")
    for method in methods:
        os.mkdir(f"detections/{method}")

Run each detection method
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    for filename in sorted(os.listdir(data_path)):
        if not filename.endswith("wav"):
            continue
        data, fs = pyPCG.io.read_signal_file(os.path.join(data_path,filename),"wav")
        print(filename)

        raw = pyPCG.pcg_signal(data,fs)
        signal = pyPCG.normalize(raw)

        for method in methods:
            s1_detect, s2_detect = [], []
            if method=="naive":
                s1_detect, s2_detect = detect_naive(signal)
            elif method=="adaptive":
                s1_detect, s2_detect = detect_adaptive(signal)
            elif method=="pyhsmm":
                s1_detect, s2_detect = detect_hsmm(raw,hsmm)
            elif method=="neurokit":
                s1_detect, s2_detect = detect_neurokit(signal)
            else:
                print(f"Unrecognized detection method type: {method}. No detection will be generated!")
                continue

            with open(f"detections/{method}/{filename[:-4]}.csv","w") as detectfile:
                detectfile.write("Location;Value\n")
                for s1 in s1_detect:
                    detectfile.write(f"{s1};S1\n")
                for s2 in s2_detect:
                    detectfile.write(f"{s2};S2\n")

Read in ground truth from manual labels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    GT_S1, GT_S2 = [], []
    for filename in os.listdir(label_path):
        if not filename.endswith("csv"):
            continue
        S1, S2 = pyPCG.io.read_hsannot_file(os.path.join(label_path,filename))
        GT_S1.append(S1)
        GT_S2.append(S2)

Calculation
-----------

Define helper functions
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def get_detections(detection):
        DL_S1, DL_S2 = [], []
        for filename in os.listdir(detection):
            if not filename.endswith("csv"):
                continue
            S1, S2 = pyPCG.io.read_hsannot_file(os.path.join(detection,filename))
            DL_S1.append(S1)
            DL_S2.append(S2)
        return DL_S1,DL_S2
    
    def tolerance_detect(detect, label, tolerance=0.06):
        tp = np.sum(np.min(np.abs(np.subtract.outer(label,detect)),axis=0)<tolerance)
        fp = np.sum(np.min(np.abs(np.subtract.outer(label,detect)),axis=0)>=tolerance)
        fn = len(label)-tp
        tn = 0
        return tp,fp,tn,fn

    def get_tolerance_scores(detect,gt,tols):
        sens,spec,ppv,f1,b_acc = [],[],[],[],[]
        for tol in tols:
            tp,fp,tn,fn = 0,0,0,0
            for GT,DL in zip(gt,detect):
                A = tolerance_detect(np.array(DL),np.array(GT),tolerance=tol)
                tp+=A[0]
                fp+=A[1]
                tn+=A[2]
                fn+=A[3]
            B = acc_measure(tp,fp,tn,fn)
            sens.append(B[0])
            spec.append(B[1])
            ppv.append(B[2])
            f1.append(B[3])
            b_acc.append(B[4])
        return sens,spec,ppv,f1,b_acc

    def abs_error(detect, label):
        mae = np.mean(np.min(np.abs(np.subtract.outer(label,detect)),axis=0))
        rmse = np.sqrt(np.mean(np.min(np.square(np.subtract.outer(label,detect)),axis=0)))
        return mae,rmse

    def mean_std(dat):
        m,s = np.mean(dat), np.std(dat)
        fstr = f"{m:.3f}±{s:.3f}"
        return m,s,fstr

    def mae(gt,detect):
        error=[]
        for GT,DL in zip(gt,detect):
            e, _ = abs_error(np.array(GT)*1000,np.array(DL)*1000)
            error.append(e)
        _,_,ret=mean_std(np.array(error))
        return ret

    def acc_measure(tp=0,fp=0,tn=0,fn=0):
        sens = tp/(tp+fn)
        spec = tn/(tn+fp)
        ppv = tp/(tp+fp)
        f1 = (2*sens*ppv)/(sens+ppv)
        b_acc = (sens+spec)/2
        return sens,spec,ppv,f1,b_acc

    def tol(gt,detect):
        tp,fp,tn,fn = 0,0,0,0
        for GT,DL in zip(gt,detect):
            A = tolerance_detect(np.array(DL),np.array(GT),0.03)
            tp+=A[0]
            fp+=A[1]
            tn+=A[2]
            fn+=A[3]
        _,_,ppv,f1,_ = acc_measure(tp,fp,tn,fn)
        return ppv,f1

Read generated detections
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    naive_s1, naive_s2 = utils.get_detections("detections/naive")
    adv_s1, adv_s2 = utils.get_detections("detections/adaptive")
    neurokit_s1, neurokit_s2 = utils.get_detections("detections/neurokit")
    springer_s1, springer_s2 = utils.get_detections("detections/springer")
    pyhsmm_s1, pyhsmm_s2 = utils.get_detections("detections/pyhsmm")

Set up tolerance scale
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    tolerances = np.arange(0.003,0.09,0.003)

Calculate F1 score-vs-tolerance for S1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # sens,spec,ppv,f1,b_acc
    _,_,_,naive_f1,_ = utils.get_tolerance_scores(naive_s1,GT_S1,tolerances)
    _,_,_,adv_f1,_ = utils.get_tolerance_scores(adv_s1,GT_S1,tolerances)
    _,_,_,neurokit_f1,_ = utils.get_tolerance_scores(neurokit_s1,GT_S1,tolerances)
    _,_,_,springer_f1,_ = utils.get_tolerance_scores(springer_s1,GT_S1,tolerances)
    _,_,_,pyhsmm_f1,_ = utils.get_tolerance_scores(pyhsmm_s1,GT_S1,tolerances)

Calculate F1 score-vs-tolerance for S2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # sens,spec,ppv,f1,b_acc
    _,_,_,adv_f1_2,_ = utils.get_tolerance_scores(adv_s2,GT_S2,tolerances)
    _,_,_,neurokit_f1_2,_ = utils.get_tolerance_scores(neurokit_s2,GT_S2,tolerances)
    _,_,_,springer_f1_2,_ = utils.get_tolerance_scores(springer_s2,GT_S2,tolerances)
    _,_,_,pyhsmm_f1_2,_ = utils.get_tolerance_scores(pyhsmm_s2,GT_S2,tolerances)

Calculate MAE and accuracy scores for S1 (constant tolerance: 30 ms)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    naive_mae = utils.mae(naive_s1,GT_S1)
    adv_mae = utils.mae(adv_s1,GT_S1)
    neurokit_mae = utils.mae(neurokit_s1,GT_S1)
    springer_mae = utils.mae(springer_s1,GT_S1)
    pyhsmm_mae = utils.mae(pyhsmm_s1,GT_S1)

    naive_tol = utils.tol(naive_s1,GT_S1)
    adv_tol = utils.tol(adv_s1,GT_S1)
    neurokit_tol = utils.tol(neurokit_s1,GT_S1)
    springer_tol = utils.tol(springer_s1,GT_S1)
    pyhsmm_tol = utils.tol(pyhsmm_s1,GT_S1)

Calculate MAE and accuracy scores for S2 (constant tolerance: 30 ms)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    adv_mae_2 = utils.mae(adv_s2,GT_S2)
    neurokit_mae_2 = utils.mae(neurokit_s2,GT_S2)
    springer_mae_2 = utils.mae(springer_s2,GT_S2)
    pyhsmm_mae_2 = utils.mae(pyhsmm_s2,GT_S2)

    adv_tol_2 = utils.tol(adv_s2,GT_S2)
    neurokit_tol_2 = utils.tol(neurokit_s2,GT_S2)
    springer_tol_2 = utils.tol(springer_s2,GT_S2)
    pyhsmm_tol_2 = utils.tol(pyhsmm_s2,GT_S2)

Results
-------

Print results for S1
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    print("S1 MAE")
    print(f"naive {naive_mae} ms")
    print(f"adaptive {adv_mae} ms")
    print(f"neurokit {neurokit_mae} ms")
    print(f"hsmm {springer_mae} ms")
    print(f"pyhsmm {pyhsmm_mae} ms")

    print()

    print("S1 Acc")
    print(f"naive ppv={naive_tol[0]:.3f} F1={naive_tol[1]:.3f}")
    print(f"adaptive ppv={adv_tol[0]:.3f} F1={adv_tol[1]:.3f}")
    print(f"neurokit ppv={neurokit_tol[0]:.3f} F1={neurokit_tol[1]:.3f}")
    print(f"hsmm ppv={springer_tol[0]:.3f} F1={springer_tol[1]:.3f}")
    print(f"pyhsmm ppv={pyhsmm_tol[0]:.3f} F1={pyhsmm_tol[1]:.3f}")

Print results for S2
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    print("S2 MAE")
    print(f"adaptive {adv_mae_2} ms")
    print(f"neurokit {neurokit_mae_2} ms")
    print(f"hsmm {springer_mae_2} ms")
    print(f"pyhsmm {pyhsmm_mae_2} ms")

    print()

    print("S2 Acc")
    print(f"adaptive ppv={adv_tol_2[0]:.3f} F1={adv_tol_2[1]:.3f}")
    print(f"neurokit ppv={neurokit_tol_2[0]:.3f} F1={neurokit_tol_2[1]:.3f}")
    print(f"hsmm ppv={springer_tol_2[0]:.3f} F1={springer_tol_2[1]:.3f}")
    print(f"pyhsmm ppv={pyhsmm_tol_2[0]:.3f} F1={pyhsmm_tol_2[1]:.3f}")

Define plotting functions
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def plot_score_v_tolerance(score,name,score_name,th=0.8,tols=tolerances):
        temp = np.nonzero(np.array(score)>th)[0]
        plt.figure(figsize=(5,3))
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.plot(tols*1000,score,linewidth=3)
        if len(temp)!=0:
            th_point = temp[0]
            th_tol = tols[th_point]*1000
            plt.axvline(th_tol,color="k",linestyle="--") #type: ignore
            plt.text(th_tol+1,0.5,f">{th} at {th_tol:.2f} ms") #type: ignore
            plt.plot(th_tol,score[th_point],'o')
        plt.ylim((np.min(score)-0.02,1.02))
        plt.xlim((tols[0]*1000-2,tols[-1]*1000+2))
        plt.xlabel("Tolerance [ms]")
        plt.ylabel(score_name)
        plt.title(name)
        plt.show()

    def bland_altman(detect,gt,name):
        def diff_mean(detect,gt):
            def flatten(matrix):
                return [item for row in matrix for item in row]
            l1,l2=[],[]
            for t in gt:
                l1.append(np.diff(t))
            for t in detect:
                l2.append(np.diff(t))

            difs,ms = [],[]
            for g,d in zip(l1,l2):
                if len(g)>len(d):
                    difs.append(g[:len(d)]-d)
                    ms.append((g[:len(d)]+d)/2)
                elif len(g)<len(d):
                    difs.append(g-d[:len(g)])
                    ms.append((g+d[:len(g)])/2)
                else:
                    difs.append(g-d)
                    ms.append((g+d)/2)
            return flatten(difs),flatten(ms)

        plt.figure()
        dif, mean = diff_mean(detect,gt)
        md = np.mean(dif)
        sd = np.std(dif)
        mm = np.mean(mean)
        sm = np.std(mean)
        plt.scatter(mean,dif,facecolors='none', edgecolors='r')
        plt.axhline(md,color="b")
        plt.axhline(md+sd*1.96,linestyle="--",color="b")
        plt.axhline(md-sd*1.96,linestyle="--",color="b")
        plt.xlabel("Mean of values")
        plt.ylabel("Difference of values")
        plt.title(name)
        plt.legend(["Differences vs Means","Mean difference","Mean difference Â± 1.96*std"],loc="upper right")
        plt.show()

F1 Score-vs-Tolerance plots for S1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    utils.plot_score_v_tolerance(naive_f1,"Naive S1","F1",tols=tolerances)
    utils.plot_score_v_tolerance(adv_f1,"Adaptive S1","F1",tols=tolerances)
    utils.plot_score_v_tolerance(neurokit_f1,"Neurokit S1","F1",tols=tolerances)
    utils.plot_score_v_tolerance(springer_f1,"HSMM S1","F1",tols=tolerances)
    utils.plot_score_v_tolerance(pyhsmm_f1,"pyHSMM S1","F1",tols=tolerances)

F1 Score-vs-Tolerance plots for S2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    utils.plot_score_v_tolerance(adv_f1_2,"Adaptive S2","F1",th=0.75,tols=tolerances)
    utils.plot_score_v_tolerance(neurokit_f1_2,"Neurokit S2","F1",th=0.75,tols=tolerances)
    utils.plot_score_v_tolerance(springer_f1_2,"HSMM S2","F1",th=0.75,tols=tolerances)
    utils.plot_score_v_tolerance(pyhsmm_f1_2,"pyHSMM S2","F1",th=0.75,tols=tolerances)

Bland-Altman plots for S1
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    utils.bland_altman(naive_s1,GT_S1,"Naive S1")
    utils.bland_altman(adv_s1,GT_S1,"Adaptive S1")
    utils.bland_altman(neurokit_s1,GT_S1,"Neurokit S1")
    utils.bland_altman(springer_s1,GT_S1,"HSMM S1")
    utils.bland_altman(pyhsmm_s1,GT_S1,"pyHSMM S1")

Bland-Altman plots for S2
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    utils.bland_altman(adv_s2,GT_S2,"Adaptive S2")
    utils.bland_altman(neurokit_s2,GT_S2,"Neurokit S2")
    utils.bland_altman(springer_s2,GT_S2,"HSMM S2")
    utils.bland_altman(pyhsmm_s2,GT_S2,"pyHSMM S2")