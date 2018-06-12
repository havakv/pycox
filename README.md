# pycox


[![Build Status](https://img.shields.io/travis/havakv/pycox.svg?branch=master)](https://travis-ci.org/havakv/pycox)
[![PyPI - License](https://img.shields.io/pypi/l/Django.svg)](./LICENSE)
[![Read the Docs](https://img.shields.io/readthedocs/pip.svg)](https://pycox.readthedocs.io/en/latest/?badge=latest)



Time-to-event prediction (survival analysis) with Cox regression in pytorch. 

This is an implementation of \<link to paper\> in pytorch.
In short, we train relative risk models with neural networks to model the event times of future events.


<img src="./Time-to-event.svg" width="40%">


------------------
## Quick-start

A quick example of Cox prortional hazards model parameterized by a 2 layer MLP:
```python
from pycox.models.high_level import CoxPHReluNet

# df_train, df_val, and df_test are pandas dataframes
# with covariates, event times, and event indicators.

cox_mlp = CoxPHReluNet(input_size, n_layers=2, n_nodes=32, dropout=False, batch_norm=True)
log = cox_mlp.fit(df_train, 'time', 'event', df_val=df_val, epochs=10, verbose=True)

# Get survival predictions
surv_preds = cox_mlp.predict_survival_function(df_test)
```

For more detailed examples, see notebooks in [examples](./examples).

----------------


## Installation

The package only works for python 3.5+.

Before installing pycox, please install [pytorch](https://pytorch.org/). We now only support pytorch 0.4.

In addition, we require the following packages:

- numpy
- scipy
- pandas
- matplotlib
- scikit-learn
- lifelines
- [sklearn-pandas](https://github.com/scikit-learn-contrib/sklearn-pandas)


**Then install pycox from the GitHub source:**

First, clone pycox using `git`:

```sh
git clone https://github.com/havakv/pycox.git
```

 Then, `cd` to the pycox folder and run the install command:
```sh
cd pycox
sudo python setup.py install
```
------------------
## Citation

ADD BIBTEX

------------------

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage) project template.

