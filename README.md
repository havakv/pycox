# pycox

[![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://github.com/havakv/torchtuples/blob/master/LICENSE)

Time-to-event prediction (survival analysis) with with [PyTorch](https://pytorch.org).

## Content

The package contains implementations for 

Methods:
- [Cox-Time](https://github.com/havakv/pycox/blob/master/examples/cox_models_1_introduction.ipynb)
- [Cox-CC](https://github.com/havakv/pycox/blob/master/examples/cox_models_1_introduction.ipynb)
- [DeepSurv](https://github.com/havakv/pycox/blob/master/examples/cox_models_1_introduction.ipynb) \[[paper](https://doi.org/10.1186/s12874-018-0482-1)\]
- [DeepHit](https://github.com/havakv/pycox/blob/master/examples/deephit.ipynb) for single event cause \[[paper](http://medianetlab.ee.ucla.edu/papers/AAAI_2018_DeepHit)\]

Evaluation metrics:
- Time-dependent concordance index
- Brier score (IPCW)
- Binomial log-likelihood (IPCW)



## Installation

The package only works for python 3.6+.

Before installing pycox, please install [PyTorch](https://pytorch.org/). We now only support pytorch 1.1.

```sh
pip install git+git://github.com/havakv/pycox.git git+git://github.com/havakv/torchtuples.git
```

