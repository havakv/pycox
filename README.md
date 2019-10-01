# pycox

[![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://github.com/havakv/pycox/blob/master/LICENSE)

Time-to-event prediction (survival analysis) with [PyTorch](https://pytorch.org).

The python package contains implementations of various survival models, some useful evaluation metrics, and a collection of [event-time datasets](https://github.com/havakv/pycox/tree/master/datasets).

## Content

The package contains implementations for 

**Methods:**
- **Cox-Time**:  a non-proportional relative risk model. \[1\] \[[example](https://github.com/havakv/pycox/blob/master/examples/cox_models_1_introduction.ipynb)\]

- **Cox-CC**: a Cox-PH model. \[1\] \[[example](https://github.com/havakv/pycox/blob/master/examples/cox_models_1_introduction.ipynb)\]

- **DeepSurv**: a Cox-PH model. \[2\] \[[example](https://github.com/havakv/pycox/blob/master/examples/cox_models_1_introduction.ipynb)\]

- **DeepHit** (single event or competing risks): a discrete time model for improved ranking. \[3\] \[[example](https://github.com/havakv/pycox/blob/master/examples/deephit.ipynb)\] 

**Evaluation metrics:**
- Time-dependent concordance index. \[4\]

- Brier score IPCW (inverse probability of censoring weighting). \[5\] \[6\]

- Binomial log-likelihood IPCW.

**Datasets:**
- For available data sets see [datasets](https://github.com/havakv/pycox/tree/master/datasets) or the module `pycox.datasets`.


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

## References

  \[1\] Håvard Kvamme, Ørnulf Borgan, and Ida Scheel. Time-to-event prediction with neural networks and Cox regression. *Journal of Machine Learning Research*, 20(129):1–30, 2019. \[[paper](http://jmlr.org/papers/v20/18-424.html)\]

  \[2\] Jared L. Katzman, Uri Shaham, Alexander Cloninger, Jonathan Bates, Tingting Jiang, and Yuval Kluger. Deepsurv: personalized treatment recommender system using a Cox proportional hazards deep neural network. *BMC Medical Research Methodology*, 18(1), 2018. \[[paper](https://doi.org/10.1186/s12874-018-0482-1)\]

  \[3\] Changhee Lee, William R Zame, Jinsung Yoon, and Mihaela van der Schaar. Deephit: A deep learning approach to survival analysis with competing risks. *In Thirty-Second AAAI Conference on Artificial Intelligence*, 2018. \[[paper](http://medianetlab.ee.ucla.edu/papers/AAAI_2018_DeepHit)\]
  
  \[4\] Laura Antolini, Patrizia Boracchi, and Elia Biganzoli. A time-dependent discrimination index for survival data. *Statistics in Medicine*, 24(24):3927–3944, 2005. \[[paper](https://doi.org/10.1002/sim.2427)\]

  \[5\] Erika Graf, Claudia Schmoor, Willi Sauerbrei, and Martin Schumacher. Assessment and comparison of prognostic classification schemes for survival data. *Statistics in Medicine*, 18(17-18):2529–2545, 1999. \[[paper](https://onlinelibrary.wiley.com/doi/abs/10.1002/%28SICI%291097-0258%2819990915/30%2918%3A17/18%3C2529%3A%3AAID-SIM274%3E3.0.CO%3B2-5)\]

  \[6\] Thomas A. Gerds and Martin Schumacher. Consistent estimation of the expected brier score in general survival models with right-censored event times. *Biometrical Journal*, 48 (6):1029–1040, 2006. \[[paper](https://onlinelibrary.wiley.com/doi/abs/10.1002/bimj.200610301?sid=nlm%3Apubmed)\]
