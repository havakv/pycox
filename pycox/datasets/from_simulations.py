"""Make dataset from the simulations, so we don't have to compute over again.
"""

import numpy as np
import pandas as pd
from pycox import simulations
from pycox.datasets._dataset_loader import _DatasetLoader

class _SimDataset(_DatasetLoader):
    col_duration = 'duration'
    col_event = 'event'
    cols_true = ['duration_true', 'censoring_true']

    def read_df(self, add_true=True):
        if not self.path.exists():
            print(f"Dataset '{self.name}' not created yet. Making dataset...")
            self._simulate_data()
            print(f"Done")
        df = super().read_df()
        if add_true is False:
            df = self._drop_true(df)
        return df

    def _simulate_data(self):
        raise NotImplementedError

    def _download(self):
        raise NotImplementedError("There is no `_download` for simulated data.")

    def _drop_true(self, df):
        return df.drop(columns=self.cols_true)


class _RRNLNPH(_SimDataset):
    """Dataset from simulation study in "Time-to-Event Prediction with Neural
    Networks and Cox Regression" [1].

    This is a continuous-time simulation study with event times drawn from a
    relative risk non-linear non-proportional hazards model (RRNLNPH).
    The full details are given in the paper [1].

    The dataset is created with `pycox.simulations.SimStudyNonLinearNonPH` (see
    `rr_nl_nph._simulate_data`).

    Variables:
        x0, x1, x2:
            numerical covariates.
        duration: (duration)
            the right-censored event-times.
        event: (event)
            event indicator {1: event, 0: censoring}.
        duration_true:
            the uncensored event times.
        censoring_true:
            the censoring times.

    To generate more data:
        >>> from pycox.simulations import SimStudyNonLinearNonPH
        >>> n = 10000
        >>> sim = SimStudyNonLinearNonPH()
        >>> data = sim.simulate(n)
        >>> df = sim.dict2df(data, True)

    References:
    [1] Håvard Kvamme, Ørnulf Borgan, and Ida Scheel.
        Time-to-event prediction with neural networks and Cox regression.
        Journal of Machine Learning Research, 20(129):1–30, 2019.
        http://jmlr.org/papers/v20/18-424.html
    """
    name = 'rr_nl_nph'
    _checksum = '68659bbb7d0320387fdc5584e647e288469eed86bfa75dac3369e36b237814ab'

    def _simulate_data(self):
        np.random.seed(1234)
        sim = simulations.SimStudyNonLinearNonPH()
        data = sim.simulate(25000)
        df = sim.dict2df(data, True)
        df.to_feather(self.path)


class _SAC3(_SimDataset):
    """Dataset from simulation study in [paper link].
    The dataset is created with `pycox.simulations.SimStudySACConstCensor` (see
    `sac3._simulate_data`).

    The full details are given in  Appendix A.1 in [paper link] for details.

    Variables:
        x0, ..., x44:
            numerical covariates.
        duration: (duration)
            the right-censored event-times.
        event: (event)
            event indicator {1: event, 0: censoring}.
        duration_true:
            the uncensored event times (only censored at max-time 100.)
        censoring_true:
            the censoring times.

    To generate more data:
        >>> from pycox.simulations import SimStudySACCensorConst
        >>> n = 10000
        >>> sim = SimStudySACCensorConst()
        >>> data = sim.simulate(n)
        >>> df = sim.dict2df(data, True, False)

    """
    name = 'sac3'
    _checksum = 'd5ec4153ba47e152383f3a1838cfaf2856ea2ad2dd198fe02c414c822524da20'

    def _simulate_data(self):
        np.random.seed(1234)
        sim = simulations.SimStudySACCensorConst()
        data = sim.simulate(100000)
        df = sim.dict2df(data, True, False)
        df.to_feather(self.path)
