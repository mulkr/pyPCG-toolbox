Feature group tutorial
======================

Setup steps
-----------

.. code:: ipython3

    import pyPCG as pcg
    import pyPCG.io as pcg_io
    import pyPCG.preprocessing as preproc
    import pyPCG.segment as sgm
    import pyPCG.features as ftr

Read in signal and calculate its envelope
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    from importlib.resources import files
    data, fs = pcg_io.read_signal_file(str(files('pyPCG').joinpath("data").joinpath("example.wav")),"wav")
    signal = pcg.pcg_signal(data,fs)
    
    signal = pcg.normalize(signal)
    bp_signal = preproc.filter(preproc.filter(signal,6,100,"LP"),6,20,"HP")
    denoise_signal = preproc.wt_denoise(bp_signal)
    env_signal = preproc.homomorphic(denoise_signal)

Segment S1 sounds from the signal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    hsmm = sgm.load_hsmm("pre_trained.json")
    states = sgm.segment_hsmm(hsmm,signal)
    s1_start, s1_end = sgm.convert_hsmm_states(states,1)

Extracting features from S1 segments
------------------------------------

All feature calculations require two numpy arrays with the boundary
timings of the desired segments. These timings are in samples. The
outputs of the feature calculations are numpy arrays containing the
calculated feature for every segment. Sometimes the output is two
arrays, usually corresponding to a location and the value at that
location.

Note: some features expect the input signal to be the envelope of the
PCG for proper functionality.

.. code:: ipython3

    s1_len = ftr.time_delta(s1_start,s1_end,env_signal)
    s1_maxfreq, s1_maxfreq_val = ftr.max_freq(s1_start,s1_end,signal,nfft=1024)
    
    print(len(s1_len),f"{s1_len[0]:.3f}")
    print(len(s1_maxfreq),f"{s1_maxfreq[0]:.3f}")


.. parsed-literal::

    134 0.087
    134 28.912
    

However, if the same feature calculations need to be run on multiple
types of segments (e.g.: timelength and frequency for S1, S2, systole,
diastole each), calling each function one-by-one can get quite tedious,
more so when changes need to be made to the calculated features. This is
prone to errors due to human error.

To circumvent the previously mentioned problems, we can create a feature
group object

Feature group object
--------------------

The feature group object takes an arbitrary number of so-called *feature
configs*. Each feature config must contain a feature calculation
function, the name of the calculated feature, the expected input (raw
signal or envelope). Optionally additional parameters can be provided as
key-value pairs in a dictionary.

.. code:: ipython3

    timing_group = ftr.feature_group({"calc_fun":ftr.time_delta,"name":"length","input":"raw"},
                                     {"calc_fun":ftr.ramp_time,"name":"onset","input":"env"},
                                     {"calc_fun":ftr.ramp_time,"name":"exit","input":"env","params":{"type":"exit"}})

This feature group will calculate the time length of the segments, the
onset times (time from start of segment to the maximum location), and
exit times (time from maximum location to end of segment).

To run a feature group, we use its ``run`` method. Which takes both
types of expected input, and the segment boundaries.

The output will be a dictionary containing the calculated feature arrays
for each segment with the names of the features provided in the feature
configs.

.. code:: ipython3

    timings = timing_group.run(signal,env_signal,s1_start,s1_end)
    
    for key,vals in timings.items():
        print(key,len(vals),f"{vals[0]:.3f}")

.. parsed-literal::

    length 134 0.087
    onset 134 0.042
    exit 134 0.045
    

If a feature returns multiple values (similar to ``max_freq``), only the
first output is considered in the output of the feature group.

Note: this is likely to change in a future version

.. code:: ipython3

    freq_group = ftr.feature_group({"calc_fun":ftr.max_freq, "name":"max frequency","input":"raw","params":{"nfft":1024}},
                                   {"calc_fun":ftr.spectral_centroid, "name":"center frequency", "input":"raw"})
    
    frequencies = freq_group.run(signal,env_signal,s1_start,s1_end)
    
    for key,vals in frequencies.items():
        print(key,len(vals),f"{vals[0]:.3f}")

.. parsed-literal::

    max frequency 134 28.912
    center frequency 134 41.663
    

If we want to combine the results, we can do so with the Python
dictionary union operation

.. code:: ipython3

    total_features = timings | frequencies
    
    print(total_features.keys())


.. parsed-literal::

    dict_keys(['length', 'onset', 'exit', 'max frequency', 'center frequency'])
    

Additional notes
----------------

The previous ``total_features`` could also be calculated with a unified
feature group. However, it may be advantageous to separate certain
features to different groups to reduce unnecessary calculations. For
example, calculating the onset time of the systole does not make much
sense, since there is no expected peak in the segment.

Using feature groups may be not necessary if only one type of segment is
considered, or if the difference between the sets of desired features
for segment types is large.
