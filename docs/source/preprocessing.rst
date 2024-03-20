Preprocessing
=============
Transform the signal
--------------------
.. autofunction:: pyPCG.preprocessing.slice_signal
.. autofunction:: pyPCG.preprocessing.resample

Envelope calculation
--------------------
.. autofunction:: pyPCG.preprocessing.envelope
.. autofunction:: pyPCG.preprocessing.homomorphic

Denoising functions
-------------------
.. autofunction:: pyPCG.preprocessing.wt_denoise
.. autofunction:: pyPCG.preprocessing.wt_denoise_sth
.. autofunction:: pyPCG.preprocessing.emd_denoise_sth
.. autofunction:: pyPCG.preprocessing.emd_denoise_savgol

Filtering
---------
.. autofunction:: pyPCG.preprocessing.filter

Processing pipeline
-------------------
.. autotypeddict:: pyPCG.preprocessing.process_config 
.. autoclass:: pyPCG.preprocessing.process_pipeline
    :members: