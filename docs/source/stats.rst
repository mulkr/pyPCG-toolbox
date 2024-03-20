Statistics
==========
Basic statistic calculations
----------------------------
.. autofunction:: pyPCG.stats.max
.. autofunction:: pyPCG.stats.min
.. autofunction:: pyPCG.stats.mean
.. autofunction:: pyPCG.stats.std
.. autofunction:: pyPCG.stats.med
.. autofunction:: pyPCG.stats.percentile
.. autofunction:: pyPCG.stats.rms
.. autofunction:: pyPCG.stats.skew
.. autofunction:: pyPCG.stats.kurt
.. autofunction:: pyPCG.stats.iqr

Operators
---------
These functions take a given statistic calculation function and extend it

.. autofunction:: pyPCG.stats.window_operator

Transformations
---------------
These functions take the input data and give a new dataset

.. autofunction:: pyPCG.stats.trim_transform
.. autofunction:: pyPCG.stats.outlier_remove_transform

Statistic calculation grouping
------------------------------
.. autotypeddict:: pyPCG.stats.stats_config
.. autoclass:: pyPCG.stats.stats_group
    :members: