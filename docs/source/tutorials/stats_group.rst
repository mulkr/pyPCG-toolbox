Stats group tutorial
====================

Setup steps
-----------

.. code:: ipython3

    import pyPCG as pcg
    import pyPCG.io as pcg_io
    import pyPCG.preprocessing as preproc
    import pyPCG.segment as sgm
    import pyPCG.features as ftr
    import pyPCG.stats as sts

.. code:: ipython3

    from importlib.resources import files
    data, fs = pcg_io.read_signal_file(str(files('pyPCG').joinpath("data").joinpath("example.wav")),"wav")
    signal = pcg.pcg_signal(data,fs)
    
    signal = pcg.normalize(signal)
    bp_signal = preproc.filter(preproc.filter(signal,6,100,"LP"),6,20,"HP")
    denoise_signal = preproc.wt_denoise(bp_signal)
    env_signal = preproc.homomorphic(denoise_signal)

.. code:: ipython3

    hsmm = sgm.load_hsmm(str(files('pyPCG').joinpath("data").joinpath("pre_trained_fpcg.json")))
    states = sgm.segment_hsmm(hsmm,signal)
    s1_start, s1_end = sgm.convert_hsmm_states(states,sgm.heart_state.S1)

Calculate some example features
-------------------------------

.. code:: ipython3

    s1_len = ftr.time_delta(s1_start,s1_end,env_signal)
    s1_maxfreq, s1_maxfreq_val = ftr.max_freq(s1_start,s1_end,signal,nfft=1024)

Basic statistics
----------------

To create statistics from the calculated features, call the appropriate
statistic function.

Let’s calculate the mean and standard deviation of the example features
above

.. code:: ipython3

    mean_len = sts.mean(s1_len)
    std_len = sts.std(s1_len)
    print(f"{mean_len=:.3f} {std_len=:.3f}")
    
    mean_maxfreq = sts.mean(s1_maxfreq)
    std_maxfreq = sts.std(s1_maxfreq)
    print(f"{mean_maxfreq=:.3f} {std_maxfreq=:.3f}")


.. parsed-literal::

    mean_len=0.095 std_len=0.009
    mean_maxfreq=29.278 std_maxfreq=3.078

Statistics group object
-----------------------

If a large amount of different statistics is required for multiple
features and multiple segment types, then a statistics group can be
created to reduce the repeated code and lessen the possibility human
error.

The ``stats_group`` object takes an arbitrary amount of *stats configs*,
which is a dictionary of the statistic measure and its name.

For example, let’s create a common statistic measure group with the mean
and standard deviation, as seen above.

.. code:: ipython3

    mean_std = sts.stats_group({"calc_fun":sts.mean,"name":"Mean"},
                               {"calc_fun":sts.std,"name":"Std"})

To run the calculations call the ``run`` method on the statistics group.

The input is a dictionary containing the features with their names.

The output will be a dictionary with a ``Feature`` field containing a
list of the names of the features, and the calculated statistics with
the names described in the configs. The values are in the same order as
in the ``Feature`` list.

.. code:: ipython3

    basic_stats = mean_std.run({"length":s1_len,"max freq":s1_maxfreq})
    print(basic_stats)


.. parsed-literal::

    {'Feature': ['length', 'max freq'], 'Mean': [0.09477387835596791, 29.278003329730993], 'Std': [0.009471572524317224, 3.0778448376432928]}
    

The required input format for running a statistics group is the same as
the output of a feature group object.

Let’s create a feature group for demonstration. (For additional details,
see the feature group tutorial)

.. code:: ipython3

    example_group = ftr.feature_group({"calc_fun":ftr.time_delta, "name":"length", "input":"raw"},
                                      {"calc_fun":ftr.ramp_time, "name":"onset", "input":"env"},
                                      {"calc_fun":ftr.max_freq, "name":"max frequency", "input":"raw","params":{"nfft":1024}})
    
    example_features = example_group.run(signal,env_signal,s1_start,s1_end)

Now the statistic calculation will look like the following

.. code:: ipython3

    example_stats = mean_std.run(example_features)
    print(example_stats)

.. parsed-literal::

    {'Feature': ['length', 'onset', 'max frequency'], 'Mean': [0.09477387835596791, 0.061852897673793185, 29.278003329730993], 'Std': [0.009471572524317224, 0.011814973307823076, 3.0778448376432928]}
    

Exporting statistics
--------------------

Each statistics group can store statistics from different segments. To
do this, call the ``add_stat`` method with the name of the segment and
the calculated statistics.

As an example, let’s store the previous statistics as *S1*

.. code:: ipython3

    mean_std.add_stat("S1",example_stats)

For further analysis, the statistics group contains a pandas dataframe,
which contains the added statistics

.. code:: ipython3

    mean_std.dataframe

.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Segment</th>
          <th>Feature</th>
          <th>Mean</th>
          <th>Std</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>S1</td>
          <td>length</td>
          <td>0.094774</td>
          <td>0.009472</td>
        </tr>
        <tr>
          <th>1</th>
          <td>S1</td>
          <td>onset</td>
          <td>0.061853</td>
          <td>0.011815</td>
        </tr>
        <tr>
          <th>2</th>
          <td>S1</td>
          <td>max frequency</td>
          <td>29.278003</td>
          <td>3.077845</td>
        </tr>
      </tbody>
    </table>
    </div>

The stored statistics can also be exported to an Excel spreadsheet

.. code:: ipython3

    mean_std.export("example.xlsx")
