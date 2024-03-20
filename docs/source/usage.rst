Usage
=====
Installation
------------
Clone the git repository, or download the code

Navigate to the directory and install with pip

.. code-block::

    pip install .

For validating the installation, check the version of the toolbox:

.. code-block:: python3

    >>> import pyPCG
    >>> print(pyPCG.__version__)
    '0.1-a'

Example
-------
The following code will read in the specified signal file, and visualize the first four seconds.

.. code-block:: python3

    >>> import pyPCG
    >>> data, fs = pyPCG.read_signal_file("example.wav","wav")
    >>> example = pyPCG.pcg_signal(data,fs)
    >>> pyPCG.plot(example,xlim=(0,4))