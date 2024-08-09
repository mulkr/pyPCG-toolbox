# pyPCG toolbox
Common processing algorithms and segmemtation for PCG signals

## Installation
**IMPORTANT!**

pyPCG is dependent on a package `hsmmlearn` which is currently not available on PyPi.

To solve this issue, install `hsmmlearn` separately from https://github.com/jvkersch/hsmmlearn
```
pip install git+https://github.com/jvkersch/hsmmlearn@master
```
*Note:* Python 3.10 is supported but 3.11 is recommended

### From PyPi
Install `pyPCG` using pip

```
pip install pyPCG_toolbox
```

### From source
Clone the git repository
```
git clone https://github.com/mulkr/pyPCG-toolbox.git
```
Change directory to the cloned repository
```
cd pyPCG-tooblox
```
Install with pip
```
pip install .
```

## Building the documentation
Install required packages
```
pip install -r docs/requirements.txt
```
Call Sphinx build command
```
sphinx-build -M html docs/source docs/build
```
On Windows you can also run the `make.bat` file
```
.\docs\make.bat html
```

The documentation should be available in the `docs/build` directory as html files<br>
This includes the example codes as tutorials

## Correspondence
Kristóf Müller (muller.kristof@itk.ppke.hu)
Márton Áron Goda (goda.marton.aron@itk.ppke.hu)