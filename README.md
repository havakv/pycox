
<h1 align="center">
    <font size="7">pycox</font>
</h1>

<p align="center">
    <strong>Time-to-event prediction with PyTorch</strong>
</p>

<p align="center">
     <a href="https://github.com/havakv/pycox/actions"><img alt="GitHub Actions status" src="https://github.com/havakv/pycox/workflows/Python%20package/badge.svg"></a>
    <a href="https://github.com/havakv/pycox/blob/master/LICENSE" title="License"><img src="https://img.shields.io/badge/License-BSD%202--Clause-orange.svg"></a>
</p>


<p align="center">
    <a href="#get-started">Get Started</a> •
    <a href="#methods">Methods</a> •
    <a href="#evaluation-criteria">Evaluation Criteria</a> •
    <a href="#datasets">Datasets</a> •
    <a href="#installation">Installation</a> •
    <a href="#references">References</a>
</p>


**pycox** is a python package for survival analysis and time-to-event prediction with [PyTorch](https://pytorch.org), built on the [torchtuples](https://github.com/havakv/torchtuples) package for training PyTorch models.

The package contains implementations of various [survival models](#methods), some useful [evaluation metrics](#evaluation-criteria), and a collection of [event-time datasets](#datasets).
In addition, some useful preprocessing tools are available in the `pycox.preprocessing` module.

## Get Started

To get started you first need to install [PyTorch](https://pytorch.org/get-started/locally/).
You can then install **pycox** with 
```sh
pip install pycox
```

We then recommend to start with [THIS INTRODUCTION NOTEBOOK](https://nbviewer.jupyter.org/github/havakv/pycox/blob/master/examples/01_introduction.ipynb), which explains the general usage of the package in terms of preprocessing, creation of neural networks, model training, and evaluation procedure.
The notebook use the `LogisticHazard` method for illustration, but most of the principles generalize to the other methods.

Alternatively, there are many examples listed in the [examples folder](https://nbviewer.jupyter.org/github/havakv/pycox/tree/master/examples).


## Methods

The following methods are available in the `pycox.methods` module.

### Continuous-Time Models:
<table>
    <tr>
        <th>Method</th>
        <th>Description</th>
        <th>Example</th>
    </tr>
    <tr>
        <td>CoxTime</td>
        <td>
        Cox-Time is a relative risk model that extends Cox regression beyond the proportional hazards <a href="#references">[1]</a>.
        </td>
        <td><a href="https://nbviewer.jupyter.org/github/havakv/pycox/blob/master/examples/cox-time.ipynb">notebook</a></td>
    </tr>
    <tr>
        <td>CoxCC</td>
        <td>
        Cox-CC is a proportional version of the Cox-Time model <a href="#references">[1]</a>.
        </td>
        <td><a href="https://nbviewer.jupyter.org/github/havakv/pycox/blob/master/examples/cox-cc.ipynb">notebook</a></td>
    </tr>
    <tr>
        <td>CoxPH (DeepSurv)</td>
        <td>
        CoxPH is a Cox proportional hazards model also referred to as DeepSurv <a href="#references">[2]</a>.
        </td>
        <td><a href="https://nbviewer.jupyter.org/github/havakv/pycox/blob/master/examples/cox-ph.ipynb">notebook</a></td>
    </tr>
    <tr>
        <td>PCHazard</td>
        <td>
        The Piecewise Constant Hazard (PC-Hazard) model <a href="#references">[12]</a> assumes that the continuous-time hazard function is constant in predefined intervals.
        It is similar to the Piecewise Exponential Models <a href="#references">[11]</a> and PEANN <a href="#references">[14]</a>, but with a softplus activation instead of the exponential function.
        </td>
        <td><a href="https://nbviewer.jupyter.org/github/havakv/pycox/blob/master/examples/pc-hazard.ipynb">notebook</a>
        </td>
    </tr>
</table>

### Discrete-Time Models:
<table>
    <tr>
        <th>Method</th>
        <th>Description</th>
        <th>Example</th>
    </tr>
    <tr>
        <td>LogisticHazard (Nnet-survival)</td>
        <td>
        The Logistic-Hazard method parametrize the discrete hazards and optimize the survival likelihood <a href="#references">[12]</a> <a href="#references">[7]</a>.
        It is also called Partial Logistic Regression <a href="#references">[13]</a> and Nnet-survival <a href="#references">[8]</a>.
        </td>
        <td><a href="https://nbviewer.jupyter.org/github/havakv/pycox/blob/master/examples/01_introduction.ipynb">notebook</a>
        </td>
    </tr>
    <tr>
        <td>PMF</td>
        <td>
        The PMF method parametrize the probability mass function (PMF) and optimize the survival likelihood <a href="#references">[12]</a>. It is the foundation of methods such as DeepHit and MTLR.
        </td>
        <td><a href="https://nbviewer.jupyter.org/github/havakv/pycox/blob/master/examples/pmf.ipynb">notebook</a>
        </td>
    </tr>
    <tr>
        <td>DeepHit, DeepHitSingle</td>
        <td>
        DeepHit is a PMF method with a loss for improved ranking that 
        can handle competing risks <a href="#references">[3]</a>.
        </td>
        <td><a href="https://nbviewer.jupyter.org/github/havakv/pycox/blob/master/examples/deephit.ipynb">single</a>
        <a href="https://nbviewer.jupyter.org/github/havakv/pycox/blob/master/examples/deephit_competing_risks.ipynb">competing</a></td>
    </tr>
    <tr>
        <td>MTLR (N-MTLR)</td>
        <td>
        The (Neural) Multi-Task Logistic Regression is a PMF methods proposed by 
        <a href="#references">[9]</a> and <a href="#references">[10]</a>.
        </td>
        <td><a href="https://nbviewer.jupyter.org/github/havakv/pycox/blob/master/examples/mtlr.ipynb">notebook</a>
        </td>
    </tr>
</table>

## Evaluation Criteria

The following evaluation metrics are available with `pycox.evalutation.EvalSurv`.

<table>
    <tr>
        <th>Metric</th>
        <th>Description</th>
    </tr>
    <tr>
        <td>concordance_td</td>
        <td>
        The time-dependent concordance index evaluated at the event times <a href="#references">[4]</a>.
        </td>
    </tr>
    <tr>
        <td>brier_score</td>
        <td>
        The IPCW Brier score (inverse probability of censoring weighted Brier score) <a href="#references">[5]</a><a href="#references">[6]</a>.
        </td>
    </tr>
    <tr>
        <td>nbll</td>
        <td>
        The IPCW (negative) binomial log-likelihood <a href="#references">[5]</a><a href="#references">[1]</a>. I.e., this is minus the binomial log-likelihood and should not be confused with the negative binomial distribution.
        </td>
    </tr>
    <tr>
        <td>integrated_brier_score</td>
        <td>
        The integrated IPCW Brier score. Numerical integration of the `brier_score` <a href="#references">[5]</a><a href="#references">[6]</a>.
        </td>
    </tr>
    <tr>
        <td>integrated_nbll</td>
        <td>
        The integrated IPCW (negative) binomial log-likelihood. Numerical integration of the `nbll` <a href="#references">[5]</a><a href="#references">[1]</a>.
        </td>
    </tr>
</table>

## Datasets

A collection of datasets are available through the `pycox.datasets` module.
For example, the following code will download the `metabric` dataset and load it in the form of a pandas dataframe
```python
from pycox import datasets
df = datasets.metabric.read_df()
```

### Real Datasets:
<table>
    <tr>
        <th>Dataset</th>
        <th>Size</th>
        <th>Dataset</th>
        <th>Data source</th>
    </tr>
    <tr>
        <td>flchain</td>
        <td>6,524</td>
        <td>
        The Assay of Serum Free Light Chain (FLCHAIN) dataset. See 
        <a href="#references">[1]</a> for preprocessing.
        </td>
        <td><a href="https://github.com/vincentarelbundock/Rdatasets">source</a>
    </tr>
    <tr>
        <td>gbsg</td>
        <td>2,232</td>
        <td>
        The Rotterdam & German Breast Cancer Study Group.
        See <a href="#references">[2]</a> for details.
        </td>
        <td><a href="https://github.com/jaredleekatzman/DeepSurv/tree/master/experiments/data">source</a>
    </tr>
    <tr>
        <td>kkbox_v1</td>
        <td>2,646,746</td>
        <td>
        A survival dataset created from the WSDM - KKBox's Churn Prediction Challenge 2017. 
        See <a href="#references">[1]</a> for details.
        Note: You need 
        <a href="https://github.com/Kaggle/kaggle-api#api-credentials">Kaggle credentials</a> to access the dataset.
        </td>
        <td><a href="https://www.kaggle.com/c/kkbox-churn-prediction-challenge/data">source</a>
    </tr>
    <tr>
        <td>metabric</td>
        <td>1,904</td>
        <td>
        The Molecular Taxonomy of Breast Cancer International Consortium (METABRIC).
        See <a href="#references">[2]</a> for details.
        </td>
        <td><a href="https://github.com/jaredleekatzman/DeepSurv/tree/master/experiments/data">source</a>
    </tr>
    <tr>
        <td>nwtco</td>
        <td>4,028</td>
        <td>
        Data from the National Wilm's Tumor (NWTCO).
        </td>
        <td><a href="https://github.com/vincentarelbundock/Rdatasets">source</a>
    </tr>
    <tr>
        <td>support</td>
        <td>8,873</td>
        <td>
        Study to Understand Prognoses Preferences Outcomes and Risks of Treatment (SUPPORT).
        See <a href="#references">[2]</a> for details.
        </td>
        <td><a href="https://github.com/jaredleekatzman/DeepSurv/tree/master/experiments/data">source</a>
    </tr>
</table>

### Simulated Datasets:

<table>
    <tr>
        <th>Dataset</th>
        <th>Size</th>
        <th>Dataset</th>
        <th>Data source</th>
    </tr>
    <tr>
        <td>rr_nl_nph</td>
        <td>25,000</td>
        <td>
        Dataset from simulation study in <a href="#references">[1]</a>.
        This is a continuous-time simulation study with event times drawn from a
        relative risk non-linear non-proportional hazards model (RRNLNPH).
        </td>
        <td><a href="https://github.com/havakv/pycox/blob/master/pycox/simulations/relative_risk.py">SimStudyNonLinearNonPH</a>
    </tr>
    <tr>
        <td>sac3</td>
        <td>100,000</td>
        <td>
        Dataset from simulation study in <a href="#references">[12]</a>.
        This is a discrete time dataset with 1000 possible event-times.
        </td>
        <td><a href="https://github.com/havakv/pycox/tree/master/pycox/simulations/discrete_logit_hazard.py">SimStudySACCensorConst</a>
    </tr>
</table>


## Installation

**Note:** *This package is still in its early stages of development, so please don't hesitate to report any problems you may experience.* 

The package only works for python 3.6+.

Before installing **pycox**, please install [PyTorch](https://pytorch.org/get-started/locally/) (version >= 1.1).
You can then run the following command to install the package (consider adding `--force-reinstall`):
```sh
pip install git+git://github.com/havakv/pycox.git
```

### Install from Source

Installation from source depends on [PyTorch](https://pytorch.org/get-started/locally/), so make sure a it is installed.
Next, clone and install with
```sh
git clone https://github.com/havakv/pycox.git
cd pycox
pip install .
```

## References

  \[1\] Håvard Kvamme, Ørnulf Borgan, and Ida Scheel. Time-to-event prediction with neural networks and Cox regression. *Journal of Machine Learning Research*, 20(129):1–30, 2019. \[[paper](http://jmlr.org/papers/v20/18-424.html)\]

  \[2\] Jared L. Katzman, Uri Shaham, Alexander Cloninger, Jonathan Bates, Tingting Jiang, and Yuval Kluger. Deepsurv: personalized treatment recommender system using a Cox proportional hazards deep neural network. *BMC Medical Research Methodology*, 18(1), 2018. \[[paper](https://doi.org/10.1186/s12874-018-0482-1)\]

  \[3\] Changhee Lee, William R Zame, Jinsung Yoon, and Mihaela van der Schaar. Deephit: A deep learning approach to survival analysis with competing risks. *In Thirty-Second AAAI Conference on Artificial Intelligence*, 2018. \[[paper](http://medianetlab.ee.ucla.edu/papers/AAAI_2018_DeepHit)\]
  
  \[4\] Laura Antolini, Patrizia Boracchi, and Elia Biganzoli. A time-dependent discrimination index for survival data. *Statistics in Medicine*, 24(24):3927–3944, 2005. \[[paper](https://doi.org/10.1002/sim.2427)\]

  \[5\] Erika Graf, Claudia Schmoor, Willi Sauerbrei, and Martin Schumacher. Assessment and comparison of prognostic classification schemes for survival data. *Statistics in Medicine*, 18(17-18):2529–2545, 1999. \[[paper](https://onlinelibrary.wiley.com/doi/abs/10.1002/%28SICI%291097-0258%2819990915/30%2918%3A17/18%3C2529%3A%3AAID-SIM274%3E3.0.CO%3B2-5)\]

  \[6\] Thomas A. Gerds and Martin Schumacher. Consistent estimation of the expected brier score in general survival models with right-censored event times. *Biometrical Journal*, 48 (6):1029–1040, 2006. \[[paper](https://onlinelibrary.wiley.com/doi/abs/10.1002/bimj.200610301?sid=nlm%3Apubmed)\]

\[7\] Charles C. Brown. On the use of indicator variables for studying the time-dependence of parameters in a response-time model. *Biometrics*, 31(4):863–872, 1975.
\[[paper](https://www.jstor.org/stable/2529811?seq=1#metadata_info_tab_contents)\]

\[8\] Michael F. Gensheimer and Balasubramanian Narasimhan. A scalable discrete-time survival model for neural networks. *PeerJ*, 7:e6257, 2019.
\[[paper](https://peerj.com/articles/6257/)\]

\[9\] Chun-Nam Yu, Russell Greiner, Hsiu-Chin Lin, and Vickie Baracos. Learning patient- specific cancer survival distributions as a sequence of dependent regressors. *In Advances in Neural Information Processing Systems 24*, pages 1845–1853. Curran Associates, Inc., 2011.
\[[paper](https://papers.nips.cc/paper/4210-learning-patient-specific-cancer-survival-distributions-as-a-sequence-of-dependent-regressors)\]

\[10\] Stephane Fotso. Deep neural networks for survival analysis based on a multi-task framework. *arXiv preprint arXiv:1801.05512*, 2018.
\[[paper](https://arxiv.org/pdf/1801.05512.pdf)\]

\[11\] Michael Friedman. Piecewise exponential models for survival data with covariates. *The Annals of Statistics*, 10(1):101–113, 1982.
\[[paper](https://projecteuclid.org/euclid.aos/1176345693)\]

\[12\] Håvard Kvamme and Ørnulf Borgan. Continuous and discrete-time survival prediction with neural networks. *arXiv preprint arXiv:1910.06724*, 2019.
\[[paper](https://arxiv.org/pdf/1910.06724.pdf)\]

\[13\] Elia Biganzoli, Patrizia Boracchi, Luigi Mariani, and Ettore Marubini. Feed forward neural networks for the analysis of censored survival data: a partial logistic regression approach. *Statistics in Medicine*, 17(10):1169–1186, 1998.
\[[paper](https://onlinelibrary.wiley.com/doi/abs/10.1002/(SICI)1097-0258(19980530)17:10%3C1169::AID-SIM796%3E3.0.CO;2-D)\]

\[14\] Marco Fornili, Federico Ambrogi, Patrizia Boracchi, and Elia Biganzoli. Piecewise exponential artificial neural networks (PEANN) for modeling hazard function with right censored data. *Computational Intelligence Methods for Bioinformatics and Biostatistics*, pages 125–136, 2014.
\[[paper](https://link.springer.com/chapter/10.1007%2F978-3-319-09042-9_9)\]
