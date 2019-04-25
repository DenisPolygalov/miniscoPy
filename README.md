__This fork is archived. All contributions to this fork are now available in the CaFFlow project:__

https://github.com/DenisPolygalov/CaFFlow


A package to analyse calcium imaging data recorded with the __[Miniscope](http://miniscope.org)__. 

* miniscoPy is a Python 3 implementation of CNMF-E, an efficient algorithm to deconvolve calcium transients ([Zhou et al, 2016](https://pdfs.semanticscholar.org/0c68/5753c9379f71b5120fe3b51589458c4de695.pdf)).
* It is based on the [CaImAn package](https://github.com/flatironinstitute/CaImAn). The CaImAn package is a versatile tool that can do several things (two photons, one photons, online analysis, etc). Here, the main idea is to have a simple package for straightforward analysis of Miniscope data with CNMF-E.
* __What does it do?__
	* Loads Miniscope AVI files
	* Creates one HDF5 file that includes everything
	* Runs the motion correction algorithm
	* Runs CNMF-E. That's it.
* __What it does not do?__
	* Loads all the available movie format on the Planet.
	* Creates memory map.
	* Analyses two-photon imaging 
	* Other methods such as PCA/ICA, etc.

Yet, the architecture of this package is almost the same as the CaImAn package. If needed, it is thus very easy to insert CaImAn functions. Besides, the visualization tools of miniscoPy have been kept at its bare minimum purposefully.

### Overview

To take a glimpse at the package, you can look at two Jupyter notebooks :
* __[main_cnmfe_e.ipynb](main_cnmfe_e.ipynb)__ for a minimal working example with actual data.
* __[main_test_hd_generated.ipynb](main_test_hd_generated.ipynb)__ for an example with simulated data of the head-direction system (requires [BRIAN2](https://brian2.readthedocs.io/en/stable/) and [XGBoost](http://xgboost.readthedocs.io/en/latest/python/python_intro.html))

## Getting Started

### Prerequisites

This installation protocol has been tested with a freshly installed Ubuntu 18.04 LTS. It is not guaranted if you have conflicting packages already installed. In this case, it may work better within a Miniconda environment (indications coming soon).
First start with cloning and installing packages:

```
git clone https://github.com/peyrachelab/miniscoPy
cd miniscoPy
sudo apt-get install python3-pip
```

Most packages can be installed in one single command line.

```
pip3 install --user numpy scipy matplotlib jupyter pandas cython scikit-learn scikit-image sklearn h5py opencv-python tqdm future
```

The most troublesome to install is __[pyav](https://mikeboers.github.io/PyAV/)__. This package allows for the fast loading of the avi files generated by the Miniscope recording software. 
According to the [PyAV installation protocol](https://mikeboers.github.io/PyAV/installation.html) and for Ubuntu >= 14.04 LTS, dependencies can be installed with:
```
sudo apt-get install -y python-dev pkg-config

sudo apt-get install -y libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libavresample-dev libavfilter-dev

pip3 install av
```
If it does not work, the best strategy is to start a conda environment and do:
```
conda install av -c conda-forge
```
If it still doens't work (and you cannot stand Conda environment and have already recompiled a linux kernel), the ultimate strategy is to compile the lib in a Conda environment and then to link it by hand. 

Brian2 and xgboost are additional packages required to run the Jupyter notebook using simulated data, they are easy to install:
```
pip3 install --user brian2 xgboost
```

### Installing

To install miniscoPy:

```
python3 setup.py build_ext -i
python3 setup.py install --user
```

### Building and testing on Windows (experimental)

In order to build miniscoPy on Windows you will need Microsoft Visual Studio 2015 or 2017 installed (Community Edition is OK) and Miniconda(preferably) or Anaconda.

First make sure you have conda-forge channel added to your conda installation:

```
conda config --show channels
```
If not - add the conda-forge channel by executing:

```
conda config --add channels conda-forge
```
Then create new conda environment and activate it:
```
conda create -n miniscoPy
activate miniscoPy
```
Install minimal set of packages required for data processing (without IPython and Jupyter):
```
conda install av scipy pandas cython scikit-learn scikit-image h5py opencv tqdm future pyyaml
```
CD to a directory where files of this repository located and build C extention:
```
python setup.py build_ext -i
```
Check your environment by executing all test scripts located in the "testbench" directory:
```
python -m unittest discover testbench
```
Finally, process sample data:
```
python main_cnmf_e.py
```

## Deployment

CNMF-E has a lot of free parameters and two recording sessions from the same animal may be best analyzed with slightly different parameters. Thus, to keep track of the parameter set for each session, miniscoPy looks for a parameter file in the recording folder in __[yaml text format](http://yaml.org/start.html)__. An example of such file can be found in _[miniscoPy/example_movies/parameters.yaml](parameters.yaml)_.

The yaml file is loaded as a dictionary at the beginning of an analysis in a Python session:
```python
import yaml
parameters = yaml.load(open('/example_movies/parameters.yaml', 'r'))
```
The [Miniscope recording software](https://github.com/daharoni/Miniscope_DAQ_Software) saves several avi files in the recording folder. Before starting the analysis pipeline, it is required to list all the avi files in a recording folder:
```python
import glob
files = glob.glob('/example_movies/*.avi')
```
Then, the first step is the motion correction algorithm :
```python
from miniscopy.base.motion_correction import *
data, video_info = normcorre(files, None, parameters['motion_correction'])
```
The second step is CNMF-E :
```python
from miniscopy import CNMFE
cnm = CNMFE(data, parameters['cnmfe'])
cnm.fit()
```

A minimal working example can be found in [main_cnmf_e.py](main_cnmf_e.py). Another example with generated data can be found in [main_test_hd_generated.py](main_test_hd_generated.py).

## Built With

* [numpy](https://numpy.org) and [scipy](https://scipy.org)
* [pandas](https://pandas.pydata.org/) - Python Data Analysis Library
* [OpenCv](https://opencv.org/) - Open Source Computer Vision Library
* [h5py](https://www.h5py.org/) - Pythonic interface to the HDF5 binary data format
* [av](https://github.com/mikeboers/PyAV) - Pythonic binding for FFmpeg or Libav
* [scikit-learn](http://scikit-learn.org/stable/) - Machine learning in Python
* [tqdm](https://github.com/noamraph/tqdm)

Besides, codes from several toolboxes are used : 
* [CaImAn](https://github.com/flatironinstitute/CaImAn)
* [SIMA](http://www.losonczylab.org/sima/1.3.2/)
* [pyfluo](https://github.com/bensondaled/pyfluo)

## Version

* Future development
  * Integration of the great [neuroseries](https://github.com/MemDynLab/neuroseries)
  * Frame-by-frame dynamic visualization
  * Conforming to [Neurodata Without Borders](https://www.nwb.org/)
* 0.0.1 (Current)
  * Work in progress

## Troubleshooting

Send us an email or post an Issue.

## Author

* Guillaume Viejo - *Initial work*
* Elena Kerjean (Sciences Sorbonne Université)

## License

This project is licensed under the GNU License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

* Hat tip to all the contributors of the different packages used here. Would not have been possible from scratch!
* Guillaume Etter from Sylvain Williams's lab for the miniscope calcium imaging movie example.
* Thanks Denis Polygalov for troubleshooting the Windows installation.




