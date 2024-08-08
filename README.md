# pyPCG toolbox
Common processing algorithms and segmemtation for PCG signals

## Installation
**Important:** Python 3.10 is supported but 3.11 is recommended

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