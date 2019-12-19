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
        event_true:
            if `duration_true` is an event.
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
    _checksum = '4952a8712403f7222d1bec58e36cdbfcd46aa31ddf87c5fb2c455565fc3f7068'

    def _simulate_data(self):
        np.random.seed(1234)
        sim = simulations.SimStudyNonLinearNonPH()
        data = sim.simulate(25000)
        df = sim.dict2df(data, True)
        df.to_feather(self.path)


class _SAC3(_SimDataset):
    """Dataset from simulation study in "Continuous and Discrete-Time Survival Prediction
    with Neural Networks" [1].

    The dataset is created with `pycox.simulations.SimStudySACConstCensor`
    (see `sac3._simulate_data`).

    The full details are given in  Appendix A.1 in [1].

    Variables:
        x0, ..., x44:
            numerical covariates.
        duration: (duration)
            the right-censored event-times.
        event: (event)
            event indicator {1: event, 0: censoring}.
        duration_true:
            the uncensored event times (only censored at max-time 100.)
        event_true:
            if `duration_true` is an event.
        censoring_true:
            the censoring times.

    To generate more data:
        >>> from pycox.simulations import SimStudySACCensorConst
        >>> n = 10000
        >>> sim = SimStudySACCensorConst()
        >>> data = sim.simulate(n)
        >>> df = sim.dict2df(data, True, False)

    References:
    [1] Håvard Kvamme and Ørnulf Borgan. Continuous and Discrete-Time Survival Prediction
        with Neural Networks. arXiv preprint arXiv:1910.06724, 2019.
        https://arxiv.org/pdf/1910.06724.pdf
    """
    name = 'sac3'
    _checksum = '2941d46baf0fbae949933565dc88663adbf1d8f5a58f989baf915d6586641fea'

    def _simulate_data(self):
        np.random.seed(1234)
        sim = simulations.SimStudySACCensorConst()
        data = sim.simulate(100000)
        df = sim.dict2df(data, True, False)
        df.to_feather(self.path)


class _SACAdmin5(_SimDataset):
    """Dataset from simulation study in [1].
    The survival function is the same as in sac3, but the censoring is administrative 
    and determined by five covariates.

    Variables:
        x0, ..., x22:
            numerical covariates.
        duration: (duration)
            the right-censored event-times.
        event: (event)
            event indicator {1: event, 0: censoring}.
        duration_true:
            the uncensored event times (only censored at max-time 100.)
        event_true:
            if `duration_true` is an event or right-censored at time 100.
        censoring_true:
            the censoring times.

    To generate more data:
        >>> from pycox.simulations import SimStudySACAdmin
        >>> n = 10000
        >>> sim = SimStudySACAdmin()
        >>> data = sim.simulate(n)
        >>> df = sim.dict2df(data, True, True)

    References:
        [1] Håvard Kvamme and Ørnulf Borgan. The Brier Score under Administrative Censoring: Problems
            and Solutions. arXiv preprint arXiv:1912.08581, 2019.
            https://arxiv.org/pdf/1912.08581.pdf
    """
    name = 'sac_admin5'
    _checksum = '9882bc8651315bcd80cba20b5f11040d71e4a84865898d7c2ca7b82ccba56683'

    def _simulate_data(self):
        np.random.seed(1234)
        sim = simulations.SimStudySACAdmin()
        data = sim.simulate(50000)
        df = sim.dict2df(data, True, True)
        df.to_feather(self.path)
