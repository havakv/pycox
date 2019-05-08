import numpy as np
from sklearn.preprocessing import StandardScaler
from pycox.preprocessing.discretization import make_cuts, IdxDiscUnknownC, _values_if_series


class LabTransCoxTime:
    def __init__(self, log_duration=False, with_mean=True, with_std=True):
        self.log_duration = log_duration
        self.duration_scaler = StandardScaler(True, with_mean, with_std)

    @property
    def map_scaled_to_orig(self):
        if not hasattr(self, '_inverse_duration_map'):
            raise ValueError('Need to fit the models before you can call this method')
        return self._inverse_duration_map

    def fit(self, durations, events):
        self.fit_transform(durations, events)
        return self

    def fit_transform(self, durations, events):
        train_durations = durations
        duration = durations.astype('float32')
        events = events.astype('float32')
        if self.log_duration:
            durations = np.log1p(durations)
        durations = self.duration_scaler.fit_transform(durations.reshape(-1, 1)).flatten()
        self._inverse_duration_map = {scaled: orig for orig, scaled in zip(train_durations, durations)}
        self._inverse_duration_map = np.vectorize(self._inverse_duration_map.get)
        return durations, events

    def transform(self, durations, events):
        duration = durations.astype('float32')
        events = events.astype('float32')
        if self.log_duration:
            durations = np.log1p(durations)
        durations = self.duration_scaler.transform(durations.reshape(-1, 1)).flatten()
        return durations, events


class LabTransDiscreteSurv:
    """Works for both hazard and pmf parametrization."""
    def __init__(self, cuts, min_=0.):
        self._cuts = cuts
        self.min_ = min_

    def fit(self, durations, events):
        self.cuts = make_cuts(self._cuts, durations, events, self.min_)
        self.idu = IdxDiscUnknownC(self.cuts)
        return self

    def fit_transform(self, durations, events):
        # self.cuts = make_cuts(self._cuts, durations, events, self.min_)
        # self.idu = IdxDiscUnknownC(self.cuts)
        self.fit(durations, events)
        t_idx, events = self.transform(durations, events)
        return t_idx, events

    def transform(self, durations, events):
        durations = _values_if_series(durations)
        events = _values_if_series(events)
        t_idx, events = self.idu.transform(durations, events)
        return t_idx, events.astype('float32')

