# pycox

[![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://github.com/havakv/pycox/blob/master/LICENSE)

Time-to-event prediction (survival analysis) with [PyTorch](https://pytorch.org).

This python package contains implementations of various survival models, and some useful evaluation metrics. 

## Content

The package contains implementations for 

Methods:
- [Cox-Time](https://github.com/havakv/pycox/blob/master/examples/cox_models_1_introduction.ipynb):  a non-proportional relative risk model \[[paper](https://arxiv.org/abs/1907.00825)\]
- [Cox-CC](https://github.com/havakv/pycox/blob/master/examples/cox_models_1_introduction.ipynb): a Cox-PH model \[[paper](https://arxiv.org/abs/1907.00825)\]
- [DeepSurv](https://github.com/havakv/pycox/blob/master/examples/cox_models_1_introduction.ipynb): a Cox-PH model \[[paper](https://doi.org/10.1186/s12874-018-0482-1)\]
- [DeepHit](https://github.com/havakv/pycox/blob/master/examples/deephit.ipynb) (single event): a discrete time model \[[paper](http://medianetlab.ee.ucla.edu/papers/AAAI_2018_DeepHit)\]

Evaluation metrics:
- Time-dependent concordance index \[[paper](https://doi.org/10.1002/sim.2427)\]
- Brier score (IPCW) \[[paper1](https://onlinelibrary.wiley.com/doi/abs/10.1002/%28SICI%291097-0258%2819990915/30%2918%3A17/18%3C2529%3A%3AAID-SIM274%3E3.0.CO%3B2-5)\] \[[paper2](https://onlinelibrary.wiley.com/doi/abs/10.1002/bimj.200610301?sid=nlm%3Apubmed)\]
- Binomial log-likelihood (IPCW)



## Installation

The package only works for python 3.6+.

Before installing pycox, please install [PyTorch](https://pytorch.org/get-started/locally/) (version >= 1.1).
You can then run the following command to install the package, but we recommend to instead install from source (see below)
```sh
pip install -e git+git://github.com/havakv/pycox.git#egg=pycox git+git://github.com/havakv/torchtuples.git
```

### Install from source

Installation from source depends on [PyTorch](https://pytorch.org/get-started/locally/), in addition to [torchtuples](https://github.com/havakv/torchtuples) which can be installed with
```sh
pip install git+git://github.com/havakv/torchtuples.git
```
Next, clone and install with
```sh
git clone https://github.com/havakv/pycox.git
cd pycox
python setup.py install
```


